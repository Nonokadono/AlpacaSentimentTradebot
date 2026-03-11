# IBKR Adapter — Drop-in replacement for the Alpaca adapter.
# Uses ib_async (successor to ib_insync) for TWS/Gateway socket connection.
# Uses Finnhub for news (replacing Alpaca's built-in news API).
# Uses exchange_calendars for NYSE market hours (replacing Alpaca's clock API).

import logging
import os
import time as _time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import finnhub
from ib_async import IB, Stock, MarketOrder, LimitOrder, StopOrder, Order, util

logger = logging.getLogger("tradebot")

# Start the ib_async event loop for synchronous-style usage
util.startLoop()


class IbkrAdapter:
    """Thin adapter around Interactive Brokers TWS/Gateway via ib_async,
    configured by env vars IB_HOST / IB_PORT / IB_CLIENT_ID / IB_ACCOUNT."""

    def __init__(self, env_mode: str):
        self.env_mode = env_mode
        self._host = os.getenv("IB_HOST", "127.0.0.1")
        self._port = int(os.getenv("IB_PORT", "7497"))  # 7497=paper, 7496=live
        self._client_id = int(os.getenv("IB_CLIENT_ID", "1"))
        self._account = os.getenv("IB_ACCOUNT", "")

        self.ib = IB()
        self._connect()

        # Clock cache (5-second TTL, same pattern as the old Alpaca adapter)
        self._clock_cache: Optional[dict] = None
        self._clock_cache_ts: float = 0.0

        # NYSE calendar for market hours
        import exchange_calendars as xcals
        self._nyse = xcals.get_calendar("XNYS")

        # Finnhub client for news
        self._finnhub = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY", ""))

    # --- Connection lifecycle ---

    def _connect(self):
        """Connect to TWS/Gateway. Retries 3 times with 5s backoff."""
        for attempt in range(3):
            try:
                self.ib.connect(
                    self._host, self._port, clientId=self._client_id,
                    timeout=20, readonly=False,
                )
                logger.info("Connected to IBKR at %s:%s (clientId=%s)",
                            self._host, self._port, self._client_id)
                return
            except Exception as e:
                logger.warning("IBKR connection attempt %d failed: %s", attempt + 1, e)
                _time.sleep(5)
        raise ConnectionError("Failed to connect to IBKR after 3 attempts")

    def _ensure_connected(self):
        """Reconnect if the socket is dead."""
        if not self.ib.isConnected():
            logger.warning("IBKR disconnected, attempting reconnect...")
            self._connect()

    def _make_contract(self, symbol: str) -> Stock:
        """Create a US equity contract."""
        return Stock(symbol, "SMART", "USD")

    def _qualify(self, contract: Stock) -> Stock:
        """Qualify a contract (resolve conId, etc.)."""
        self.ib.qualifyContracts(contract)
        return contract

    # --- Account / positions ---

    def get_account(self) -> Any:
        """Return an object with .equity, .cash, .buying_power, .portfolio_value,
        .day_trading_buying_power, .realized_pl, .unrealized_pl attributes."""
        self._ensure_connected()
        summary = self.ib.accountSummary(self._account) if self._account else self.ib.accountSummary()
        vals = {av.tag: av.value for av in summary}
        return SimpleNamespace(
            equity=float(vals.get("NetLiquidation", 0)),
            cash=float(vals.get("AvailableFunds", 0)),
            buying_power=float(vals.get("BuyingPower", 0)),
            portfolio_value=float(vals.get("NetLiquidation", 0)),
            day_trading_buying_power=float(vals.get("BuyingPower", 0)),
            realized_pl=0.0,
            unrealized_pl=float(vals.get("UnrealizedPnL", 0)),
            last_equity=float(vals.get("NetLiquidation", 0)),
        )

    def list_positions(self) -> List[Any]:
        self._ensure_connected()
        positions = self.ib.positions()
        if self._account:
            positions = [p for p in positions if p.account == self._account]

        results = []
        for p in positions:
            contract = p.contract
            qty = float(p.position)
            if abs(qty) < 1e-9:
                continue

            self._qualify(contract)
            # Use snapshot for current price, fall back to historical bars, then avgCost
            try:
                last = self.get_last_quote(contract.symbol)
            except Exception:
                last = float(p.avgCost)

            results.append(SimpleNamespace(
                symbol=contract.symbol,
                qty=str(abs(qty)),
                current_price=str(float(last)),
                side="long" if qty > 0 else "short",
                avg_entry_price=str(float(p.avgCost)),
            ))
        return results

    def get_position(self, symbol: str) -> Optional[Any]:
        try:
            positions = self.list_positions()
            for p in positions:
                if p.symbol == symbol:
                    return p
            return None
        except Exception:
            return None

    # --- Market status ---

    def get_clock(self) -> Optional[Any]:
        """Simulate Alpaca's clock object using exchange_calendars (NYSE).
        Returns object with .is_open, .next_close, .next_open.
        Cached for 5 seconds."""
        now = _time.monotonic()
        if self._clock_cache is not None and (now - self._clock_cache_ts) < 5.0:
            return SimpleNamespace(**self._clock_cache)
        try:
            now_utc = datetime.now(timezone.utc)
            now_naive = now_utc.replace(tzinfo=None)

            is_open = self._nyse.is_open_on_minute(now_naive)

            if is_open:
                session = self._nyse.minute_to_session(now_naive)
                next_close = self._nyse.session_close(session)
                try:
                    next_open = self._nyse.next_open(now_naive)
                except Exception:
                    next_open = next_close + timedelta(hours=17, minutes=30)
            else:
                try:
                    next_open = self._nyse.next_open(now_naive)
                except Exception:
                    next_open = now_naive + timedelta(days=1)
                try:
                    next_session = self._nyse.minute_to_session(next_open)
                    next_close = self._nyse.session_close(next_session)
                except Exception:
                    next_close = next_open + timedelta(hours=6, minutes=30)

            result = {
                "is_open": is_open,
                "next_close": next_close.isoformat(),
                "next_open": next_open.isoformat(),
            }
            self._clock_cache = result
            self._clock_cache_ts = now
            return SimpleNamespace(**result)
        except Exception as e:
            logger.warning("get_clock error: %s", e)
            return None

    def get_market_open(self) -> bool:
        clock = self.get_clock()
        if clock is None:
            return False
        return bool(clock.is_open)

    def get_market_close_time(self) -> Optional[datetime]:
        try:
            clock = self.get_clock()
            if clock is None:
                return None
            raw_close = clock.next_close
            if isinstance(raw_close, datetime):
                dt = raw_close
            else:
                dt = datetime.fromisoformat(str(raw_close).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception as e:
            logger.warning("get_market_close_time error: %s", e)
            return None

    def get_market_open_time(self) -> Optional[datetime]:
        try:
            clock = self.get_clock()
            if clock is None:
                return None
            raw_open = clock.next_open
            if isinstance(raw_open, datetime):
                dt = raw_open
            else:
                dt = datetime.fromisoformat(str(raw_open).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if clock.is_open:
                dt = dt - timedelta(days=1)
            return dt
        except Exception as e:
            logger.warning("get_market_open_time error: %s", e)
            return None

    def is_monday_open_blackout(self, blackout_minutes: int = 30) -> bool:
        try:
            if not self.get_market_open():
                return False
            open_time = self.get_market_open_time()
            if open_time is None:
                return False
            if open_time.weekday() != 0:
                return False
            now_utc = datetime.now(timezone.utc)
            minutes_since_open = (now_utc - open_time).total_seconds() / 60.0
            if 0 <= minutes_since_open <= blackout_minutes:
                logger.info(
                    "MONDAY OPEN BLACKOUT active: %.1fm since market open.",
                    minutes_since_open,
                )
                return True
            return False
        except Exception as e:
            logger.warning("is_monday_open_blackout error: %s", e)
            return False

    def is_pre_weekend_close(self, minutes_before: int = 12) -> bool:
        try:
            if not self.get_market_open():
                return False
            close_time = self.get_market_close_time()
            if close_time is None:
                return False
            if close_time.weekday() != 4:
                return False
            now_utc = datetime.now(timezone.utc)
            minutes_until_close = (close_time - now_utc).total_seconds() / 60.0
            return 0 <= minutes_until_close <= minutes_before
        except Exception as e:
            logger.warning("is_pre_weekend_close error: %s", e)
            return False

    def is_pre_close_blackout(self, blackout_minutes: int = 90) -> bool:
        try:
            if not self.get_market_open():
                return False
            close_time = self.get_market_close_time()
            if close_time is None:
                return False
            now_utc = datetime.now(timezone.utc)
            minutes_until_close = (close_time - now_utc).total_seconds() / 60.0
            if 0 <= minutes_until_close <= blackout_minutes:
                logger.debug(
                    "PRE-CLOSE BLACKOUT active: %.1fm until market close.",
                    minutes_until_close,
                )
                return True
            return False
        except Exception as e:
            logger.warning("is_pre_close_blackout error: %s", e)
            return False

    def is_pre_daily_close(self, minutes_before: int = 15) -> bool:
        try:
            if not self.get_market_open():
                return False
            close_time = self.get_market_close_time()
            if close_time is None:
                return False
            if close_time.weekday() not in (0, 1, 2, 3):
                return False
            now_utc = datetime.now(timezone.utc)
            minutes_until_close = (close_time - now_utc).total_seconds() / 60.0
            return 0 <= minutes_until_close <= minutes_before
        except Exception as e:
            logger.warning("is_pre_daily_close error: %s", e)
            return False

    # --- Market data ---

    def get_last_quote(self, symbol: str) -> float:
        """Get the last traded price for a symbol.
        Falls back to the last historical bar close if real-time snapshot
        data is unavailable (e.g. missing market data subscription)."""
        self._ensure_connected()
        contract = self._make_contract(symbol)
        self._qualify(contract)

        # Try real-time snapshot first
        try:
            ticker = self.ib.reqMktData(contract, "", True, False)
            self.ib.sleep(1)
            price = ticker.last
            if price != price:  # NaN
                price = ticker.close
            if price == price:  # not NaN
                return float(price)
        except Exception as e:
            logger.debug("get_last_quote snapshot failed for %s: %s", symbol, e)

        # Fallback: last close from historical bars (uses HMDS, no subscription needed)
        logger.debug("get_last_quote falling back to historical bars for %s", symbol)
        bars = self.get_recent_bars(symbol, timeframe="1Day", lookback_bars=1)
        if bars and hasattr(bars[-1], "c") and bars[-1].c > 0:
            return float(bars[-1].c)

        raise ValueError(f"No price data for {symbol}")

    def get_latest_quote(self, symbol: str) -> Any:
        """Return bid/ask quote as object with .bp and .ap attributes (for dashboard).
        Falls back to last historical close for both bid/ask if snapshot unavailable."""
        self._ensure_connected()
        contract = self._make_contract(symbol)
        self._qualify(contract)

        # Try real-time snapshot
        try:
            ticker = self.ib.reqMktData(contract, "", True, False)
            self.ib.sleep(0.5)
            bid = ticker.bid if ticker.bid == ticker.bid else None
            ask = ticker.ask if ticker.ask == ticker.ask else None
            if bid is not None and ask is not None and (bid > 0 or ask > 0):
                return SimpleNamespace(bp=bid, ap=ask)
        except Exception:
            pass

        # Fallback: use last close as both bid and ask
        try:
            last = self.get_last_quote(symbol)
            return SimpleNamespace(bp=last, ap=last)
        except Exception:
            return SimpleNamespace(bp=0.0, ap=0.0)

    def get_recent_bars(
        self,
        symbol: str,
        timeframe: str = "5Min",
        lookback_bars: int = 30,
    ) -> List[Any]:
        """Fetch up to lookback_bars recent bars for symbol."""
        self._ensure_connected()
        contract = self._make_contract(symbol)
        self._qualify(contract)

        # Map Alpaca-style timeframe strings to IBKR barSizeSetting
        tf_map = {
            "1Min": "1 min", "5Min": "5 mins", "15Min": "15 mins",
            "1Hour": "1 hour", "1Day": "1 day",
        }
        bar_size = tf_map.get(timeframe, "5 mins")

        # Request enough duration to cover lookback + weekends
        duration = "3 D" if bar_size in ("1 min", "5 mins", "15 mins") else "1 M"

        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",  # now
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        except Exception as e:
            logger.warning(
                "get_recent_bars error for %s (timeframe=%s, lookback=%d): %s",
                symbol, timeframe, lookback_bars, e,
            )
            return []

        # Map to Alpaca-compatible attribute names (.o, .h, .l, .c, .v)
        result = []
        for bar in bars[-lookback_bars:]:
            result.append(SimpleNamespace(
                o=bar.open, h=bar.high, l=bar.low, c=bar.close, v=bar.volume,
                t=bar.date if hasattr(bar, "date") else "",
            ))

        if len(result) < lookback_bars:
            logger.warning(
                "get_recent_bars %s: requested %d bars, received %d "
                "— signal quality may be degraded. "
                "RSI/momentum may fall back to neutral defaults.",
                symbol, lookback_bars, len(result),
            )
        return result

    # --- News (Finnhub) ---

    def get_news(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        """Fetch news from Finnhub."""
        try:
            if since is None:
                _from = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
            else:
                _from = since.strftime("%Y-%m-%d")
            _to = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            raw = self._finnhub.company_news(symbol, _from=_from, _to=_to)
            out: List[Dict[str, str]] = []
            for item in (raw or [])[:limit]:
                out.append({
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                })
            return out
        except Exception as e:
            logger.warning("get_news error for %s: %s", symbol, e)
            return []

    # --- Orders ---

    def _map_tif(self, tif: str) -> str:
        """Map Alpaca TIF strings to IBKR TIF strings."""
        mapping = {"day": "DAY", "gtc": "GTC", "ioc": "IOC"}
        return mapping.get(tif.lower(), "DAY")

    def _wrap_trade(self, trade) -> Any:
        """Wrap an ib_async Trade into a proxy with .id, .symbol, .side, .qty,
        .type, .status, .stop_price, .limit_price attributes."""
        order = trade.order if hasattr(trade, "order") else trade
        contract = trade.contract if hasattr(trade, "contract") else None

        return SimpleNamespace(
            id=str(order.orderId),
            symbol=contract.symbol if contract else "",
            side="buy" if order.action == "BUY" else "sell",
            qty=str(order.totalQuantity),
            type=self._map_order_type_reverse(order.orderType),
            status=trade.orderStatus.status if hasattr(trade, "orderStatus") else "submitted",
            stop_price=str(order.auxPrice) if order.auxPrice else None,
            limit_price=str(order.lmtPrice) if order.lmtPrice else None,
        )

    def _map_order_type_reverse(self, ibkr_type: str) -> str:
        """Map IBKR order types to Alpaca-style strings."""
        mapping = {
            "MKT": "market", "LMT": "limit", "STP": "stop",
            "STP LMT": "stop_limit", "TRAIL": "trailing_stop",
        }
        return mapping.get(ibkr_type, ibkr_type.lower())

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> Any:
        self._ensure_connected()
        contract = self._make_contract(symbol)
        self._qualify(contract)
        action = "BUY" if side == "buy" else "SELL"
        order = MarketOrder(action, float(qty))
        order.tif = self._map_tif(time_in_force)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        return self._wrap_trade(trade)

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        take_profit_price: float,
        time_in_force: str = "day",
    ) -> Any:
        """Submit bracket order as parent market + TP limit child + SL stop child.
        Uses parentId linking and transmit=False for atomic submission."""
        self._ensure_connected()
        contract = self._make_contract(symbol)
        self._qualify(contract)
        action = "BUY" if side == "buy" else "SELL"
        reverse = "SELL" if side == "buy" else "BUY"

        parent = Order(
            action=action, orderType="MKT", totalQuantity=float(qty),
            tif=self._map_tif(time_in_force), transmit=False,
        )
        tp = Order(
            action=reverse, orderType="LMT", totalQuantity=float(qty),
            lmtPrice=round(take_profit_price, 2),
            tif=self._map_tif(time_in_force), transmit=False,
        )
        sl = Order(
            action=reverse, orderType="STP", totalQuantity=float(qty),
            auxPrice=round(stop_price, 2),
            tif=self._map_tif(time_in_force), transmit=True,  # last child transmits all
        )

        parent_trade = self.ib.placeOrder(contract, parent)
        tp.parentId = parent_trade.order.orderId
        sl.parentId = parent_trade.order.orderId
        # OCA group links TP and SL so one cancels the other
        oca_group = f"OCA_{symbol}_{int(_time.time())}"
        tp.ocaGroup = oca_group
        sl.ocaGroup = oca_group
        tp.ocaType = 1  # Cancel remaining on fill
        sl.ocaType = 1

        self.ib.placeOrder(contract, tp)
        self.ib.placeOrder(contract, sl)
        self.ib.sleep(0.5)
        return self._wrap_trade(parent_trade)

    def submit_take_profit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Any:
        self._ensure_connected()
        contract = self._make_contract(symbol)
        self._qualify(contract)
        action = "BUY" if side == "buy" else "SELL"
        order = LimitOrder(action, float(qty), round(limit_price, 2))
        order.tif = self._map_tif(time_in_force)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        return self._wrap_trade(trade)

    def submit_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        time_in_force: str = "day",
    ) -> Any:
        self._ensure_connected()
        contract = self._make_contract(symbol)
        self._qualify(contract)
        action = "BUY" if side == "buy" else "SELL"
        order = StopOrder(action, float(qty), round(stop_price, 2))
        order.tif = self._map_tif(time_in_force)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        return self._wrap_trade(trade)

    def submit_trailing_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        trail_percent: float,
        time_in_force: str = "day",
    ) -> Any:
        self._ensure_connected()
        contract = self._make_contract(symbol)
        self._qualify(contract)
        action = "BUY" if side == "buy" else "SELL"
        order = Order(
            action=action, orderType="TRAIL", totalQuantity=float(qty),
            trailingPercent=float(trail_percent),
            tif=self._map_tif(time_in_force),
        )
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0.5)
        return self._wrap_trade(trade)

    def cancel_order(self, order_id: str) -> None:
        self._ensure_connected()
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == str(order_id):
                self.ib.cancelOrder(trade.order)
                return

    def cancel_all_orders(self) -> None:
        self._ensure_connected()
        self.ib.reqGlobalCancel()

    def close_all_positions(self) -> None:
        """Close all positions by submitting opposite market orders."""
        self._ensure_connected()
        for pos in self.ib.positions():
            if self._account and pos.account != self._account:
                continue
            qty = float(pos.position)
            if abs(qty) < 1e-9:
                continue
            contract = pos.contract
            self._qualify(contract)
            action = "SELL" if qty > 0 else "BUY"
            order = MarketOrder(action, abs(qty))
            self.ib.placeOrder(contract, order)

    def list_orders(self, status: str = "open") -> List[Any]:
        self._ensure_connected()
        if status == "open":
            trades = self.ib.openTrades()
        else:
            trades = self.ib.trades()
        return [self._wrap_trade(t) for t in trades]
