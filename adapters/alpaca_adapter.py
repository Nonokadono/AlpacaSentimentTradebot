import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import alpaca_trade_api as tradeapi

logger = logging.getLogger("tradebot")


class AlpacaAdapter:
    """Thin adapter around Alpaca REST API, configured purely by env vars
    APCA_API_BASE_URL / APCA_API_KEY_ID / APCA_API_SECRET_KEY."""

    def __init__(self, env_mode: str):
        self.env_mode = env_mode
        base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        key_id = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            raise ValueError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set")
        self.rest = tradeapi.REST(
            key_id=key_id,
            secret_key=secret_key,
            base_url=base_url,
            api_version="v2",
        )

    # --- Account / positions ---

    def get_account(self) -> Any:
        return self.rest.get_account()

    def list_positions(self) -> List[Any]:
        return list(self.rest.list_positions()) or []

    def get_position(self, symbol: str) -> Optional[Any]:
        try:
            return self.rest.get_position(symbol)
        except Exception:
            return None

    # --- Market status ---

    def get_clock(self) -> Optional[Any]:
        """Return the raw Alpaca clock object (is_open, next_close, next_open).
        Returns None on error so callers must guard against None."""
        try:
            return self.rest.get_clock()
        except Exception as e:
            logger.warning(f"get_clock error: {e}")
            return None

    def get_market_open(self) -> bool:
        """Returns True if the US equity market is currently open, False otherwise.
        Uses Alpaca's v2/clock endpoint."""
        try:
            clock = self.rest.get_clock()
            return bool(clock.is_open)
        except Exception:
            return False

    def get_market_close_time(self) -> Optional[datetime]:
        """
        Return the next scheduled market close as a timezone-aware datetime
        (UTC), or None if unavailable.

        Implementation rules:
        - Calls self.get_clock() — does NOT call self.rest.get_clock() directly.
        - If get_clock() returns None, returns None immediately.
        - Parses clock.next_close (an ISO 8601 string) into a timezone-aware
          datetime using datetime.fromisoformat() or a compatible parser.
        - Ensures the result is UTC-aware. If the parsed datetime is naive,
          attaches UTC via .replace(tzinfo=timezone.utc).
        - Catches all exceptions, logs at WARNING level, and returns None.
        - Never raises.
        """
        try:
            clock = self.get_clock()
            if clock is None:
                return None
            raw_close = clock.next_close
            # next_close may be a string or already a datetime-like object.
            if isinstance(raw_close, datetime):
                dt = raw_close
            else:
                dt = datetime.fromisoformat(str(raw_close).replace("Z", "+00:00"))
            # Ensure offset-aware UTC.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception as e:
            logger.warning(f"get_market_close_time error: {e}")
            return None

    def is_pre_weekend_close(self, minutes_before: int = 12) -> bool:
        """
        Return True if and only if ALL of the following are true:
          1. The market is currently open (self.get_market_open() is True).
          2. The next scheduled close falls on a Friday (weekday() == 4),
             evaluated in the UTC timezone of the parsed close datetime.
          3. The current UTC time is within `minutes_before` minutes of
             that close.

        The default of 12 minutes is intentional: the main loop cycle is
        600 seconds (10 min). A 12-minute window guarantees this check
        fires at least once before close, even under scheduling jitter,
        without producing false positives during intra-week closes.

        Implementation rules:
        - Calls self.get_market_close_time() — does NOT call get_clock() again.
        - Uses datetime.utcnow().replace(tzinfo=timezone.utc) for the current
          time to ensure offset-aware arithmetic.
        - Catches all exceptions, logs at WARNING level, and returns False.
        - Never raises.
        """
        try:
            if not self.get_market_open():
                return False
            close_time = self.get_market_close_time()
            if close_time is None:
                return False
            # Check that the close falls on a Friday (weekday 4).
            if close_time.weekday() != 4:
                return False
            now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
            minutes_until_close = (close_time - now_utc).total_seconds() / 60.0
            return 0 <= minutes_until_close <= minutes_before
        except Exception as e:
            logger.warning(f"is_pre_weekend_close error: {e}")
            return False

    def is_pre_close_blackout(self, blackout_minutes: int = 180) -> bool:
        """
        Return True if the market is currently open AND the current UTC time
        is within `blackout_minutes` minutes of the next scheduled close,
        on ANY day of the week.

        Default is 180 minutes (3 hours). On a standard 09:30–16:00 ET
        session this suppresses new entries from 13:00 ET onward.

        Implementation rules:
        - Calls self.get_market_close_time() — does NOT call get_clock() again.
          get_market_close_time() is shared by both is_pre_weekend_close()
          and is_pre_close_blackout(). The clock API is never called more than
          once per logical check.
        - Uses datetime.utcnow().replace(tzinfo=timezone.utc) for the current
          time.
        - Emits a single logger.debug() message when returning True.
        - Catches all exceptions, logs at WARNING level, and returns False.
        - Never raises.
        """
        try:
            if not self.get_market_open():
                return False
            close_time = self.get_market_close_time()
            if close_time is None:
                return False
            now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
            minutes_until_close = (close_time - now_utc).total_seconds() / 60.0
            if 0 <= minutes_until_close <= blackout_minutes:
                logger.debug(
                    f"PRE-CLOSE BLACKOUT active: {minutes_until_close:.1f}m until market close."
                )
                return True
            return False
        except Exception as e:
            logger.warning(f"is_pre_close_blackout error: {e}")
            return False

    # --- Market data ---

    def get_last_quote(self, symbol: str) -> float:
        """Get a close/last price proxy via recent bars, fallback to latest trade."""
        end = datetime.utcnow()
        start = end - timedelta(minutes=60)
        try:
            bars = self.rest.get_bars(
                symbol, "5Min", start.isoformat() + "Z", end.isoformat() + "Z",
            )
        except Exception as e:
            print(f"get_bars error for {symbol}: {e}")
            bars = []
        if not bars:
            last = self.rest.get_latest_trade(symbol)
            return float(last.price)
        bar = bars[-1]
        close_price = getattr(bar, "c", None)
        if close_price is None:
            close_price = bar.c
        return float(close_price)

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        lookback_bars: int = 30,
    ) -> List[Any]:
        """Fetch up to lookback_bars recent bars for symbol.

        Fix 7: The start timestamp is now anchored 3 calendar days back
        (previously lookback_bars * 5 * 30 minutes).  This ensures the request
        window covers weekends, holidays, and pre/post-market gaps so the API
        always has enough trading bars to fill the limit.  The Alpaca API
        ``limit`` parameter caps the returned count at lookback_bars regardless
        of how wide the window is.  A WARNING is emitted if fewer bars than
        requested are returned, so under-fill is never silent downstream
        (RSI fallback to 50.0 etc.).
        """
        end = datetime.utcnow()
        # 3-day window survives any weekend / holiday combination.
        start = end - timedelta(days=3)
        try:
            bars = self.rest.get_bars(
                symbol,
                timeframe,
                start.isoformat() + "Z",
                end.isoformat() + "Z",
                limit=lookback_bars,
            )
        except Exception as e:
            logger.warning(
                f"get_recent_bars error for {symbol} "
                f"(timeframe={timeframe}, lookback={lookback_bars}): {e}"
            )
            bars = []
        result = list(bars) or []
        # Fix 7: warn explicitly when under-filled so issues are visible in logs.
        if len(result) < lookback_bars:
            logger.warning(
                f"get_recent_bars {symbol}: requested {lookback_bars} bars, "
                f"received {len(result)} — signal quality may be degraded. "
                f"RSI/momentum may fall back to neutral defaults."
            )
        return result

    # --- News sentiment inputs ---

    def get_news(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        try:
            kwargs: Dict[str, Any] = {"symbol": symbol, "limit": limit}
            if since is not None:
                kwargs["start"] = since.isoformat() + "Z"
            raw_items = self.rest.get_news(**kwargs)
        except Exception as e:
            print(f"get_news error for {symbol}: {e}")
            return []
        out: List[Dict[str, str]] = []
        for n in raw_items:
            headline = getattr(n, "headline", "") or ""
            summary = getattr(n, "summary", "") or ""
            out.append({"headline": headline, "summary": summary})
        return out

    # --- Orders ---

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> Any:
        return self.rest.submit_order(
            symbol=symbol,
            side=side,
            type="market",
            qty=qty,
            time_in_force=time_in_force,
        )

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        take_profit_price: float,
        time_in_force: str = "day",
    ) -> Any:
        """Submit an entry market order with an attached TP limit + fixed stop-loss
        as a single Alpaca bracket order (order_class='bracket').

        Alpaca manages the two exit legs as an internal OCO pair, so there is
        never a qty-reservation conflict between TP and stop orders.

        Direction is inferred automatically from ``side``:
        - buy  bracket: TP limit above entry, stop below entry.
        - sell bracket: TP limit below entry, stop above entry.

        Both prices are computed correctly by SignalEngine._decide_side_and_bands().
        """
        return self.rest.submit_order(
            symbol=symbol,
            side=side,
            type="market",
            qty=qty,
            time_in_force=time_in_force,
            order_class="bracket",
            take_profit={"limit_price": str(round(take_profit_price, 2))},
            stop_loss={"stop_price": str(round(stop_price, 2))},
        )

    def submit_take_profit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Any:
        return self.rest.submit_order(
            symbol=symbol,
            side=side,
            type="limit",
            qty=qty,
            limit_price=limit_price,
            time_in_force=time_in_force,
        )

    def submit_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        time_in_force: str = "day",
    ) -> Any:
        """Submit a plain stop-market order.  Used as the fallback single-exit
        path when only enable_trailing_stop is True (no TP configured).
        stop_price is the ATR-based level computed by SignalEngine."""
        return self.rest.submit_order(
            symbol=symbol,
            side=side,
            type="stop",
            qty=qty,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )

    def submit_trailing_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        trail_percent: float,
        time_in_force: str = "day",
    ) -> Any:
        return self.rest.submit_order(
            symbol=symbol,
            side=side,
            type="trailing_stop",
            qty=qty,
            trail_percent=trail_percent,
            time_in_force=time_in_force,
        )

    def cancel_order(self, order_id: str) -> None:
        self.rest.cancel_order(order_id)

    def cancel_all_orders(self) -> None:
        self.rest.cancel_all_orders()

    def close_all_positions(self) -> None:
        self.rest.close_all_positions()

    def list_orders(self, status: str = "open") -> List[Any]:
        return list(self.rest.list_orders(status=status)) or []
