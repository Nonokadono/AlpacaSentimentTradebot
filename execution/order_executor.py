import logging
import time
from typing import Optional

from alpaca_trade_api.rest import APIError
from adapters.alpaca_adapter import AlpacaAdapter
from core.risk_engine import ProposedTrade, PositionInfo
from core.sentiment import SentimentResult
from monitoring.monitor import (
    log_proposed_trade,
    log_sentiment_close_decision,
)
from config.config import ExecutionConfig

logger = logging.getLogger("tradebot")


class OrderExecutor:
    def __init__(
        self,
        adapter: AlpacaAdapter,
        env_mode: str,
        live_trading_enabled: bool,
        execution_cfg: ExecutionConfig,
    ) -> None:
        self.adapter = adapter
        self.env_mode = env_mode
        self.live_trading_enabled = live_trading_enabled
        self.execution_cfg = execution_cfg

    def _can_place_orders(self) -> bool:
        """
        Returns True if it is allowed to place real orders in the current environment.
        PAPER: always True.
        LIVE: only if live_trading_enabled is True.
        """
        if self.env_mode == "LIVE" and not self.live_trading_enabled:
            return False
        return True

    def _wait_for_position(self, symbol: str, timeout_sec: int) -> Optional[any]:
        """
        Wait briefly for the entry to fill so we can submit exit orders safely.
        """
        deadline = time.time() + max(0, int(timeout_sec))
        while time.time() < deadline:
            pos = self.adapter.get_position(symbol)
            if pos is not None:
                return pos
            time.sleep(1)
        return None

    def _cancel_leg(self, order_id: Optional[str], label: str, symbol: str) -> None:
        """
        Cancel a single named exit leg by order_id.  Swallows errors so a
        failed cancel never aborts the main execution path.
        """
        if not order_id:
            return
        try:
            self.adapter.cancel_order(order_id)
            logger.info(
                f"OrderExecutor [{symbol}]: cancelled {label} leg order_id={order_id}"
            )
        except Exception as e:
            logger.warning(
                f"OrderExecutor [{symbol}]: failed to cancel {label} "
                f"leg order_id={order_id}: {e}"
            )

    def cancel_exit_legs(
        self,
        symbol: str,
        tp_order_id: Optional[str],
        ts_order_id: Optional[str],
    ) -> None:
        """
        Cancel both exit legs for a symbol.  Call this when one leg fills or
        when the position is being closed via a sentiment exit so orphaned
        orders do not persist.
        """
        self._cancel_leg(tp_order_id, "take-profit", symbol)
        self._cancel_leg(ts_order_id, "trailing-stop", symbol)

    def execute_proposed_trade(self, trade: ProposedTrade):
        # Log the proposed trade with environment mode
        log_proposed_trade(trade, self.env_mode)

        # Skip if rejected or zero qty
        if trade.rejected_reason is not None or trade.qty <= 0:
            return None

        # LIVE dry-run protection
        if not self._can_place_orders():
            return None

        side = trade.side
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        # Fix 8a: cancel any orphaned open orders for this symbol BEFORE
        # entering a new position.  This handles the race condition where a
        # prior cycle's trailing stop or take-profit is still resting on the
        # book after the position was flat.  Swallow errors so a stale-order
        # cleanup failure never blocks a new entry.
        try:
            existing_orders = self.adapter.list_orders(status="open")
            for o in existing_orders:
                if getattr(o, "symbol", None) == trade.symbol:
                    try:
                        self.adapter.cancel_order(o.id)
                        logger.info(
                            f"OrderExecutor [{trade.symbol}]: cancelled orphaned "
                            f"open order id={o.id} before new entry."
                        )
                    except Exception as ce:
                        logger.warning(
                            f"OrderExecutor [{trade.symbol}]: could not cancel "
                            f"orphaned order id={o.id}: {ce}"
                        )
        except Exception as e:
            logger.warning(
                f"OrderExecutor [{trade.symbol}]: failed to list open orders "
                f"for orphan cleanup: {e}"
            )

        try:
            # 1) Entry: market order
            entry_order = self.adapter.submit_market_order(
                symbol=trade.symbol,
                qty=trade.qty,
                side=side,
                time_in_force=self.execution_cfg.entry_time_in_force,
            )
        except APIError as e:
            logger.error(f"Entry order placement failed for {trade.symbol}: {e}")
            return None

        # 2) If configured, attempt to place trailing stop + take profit AFTER fill
        if not self.execution_cfg.enable_trailing_stop and not self.execution_cfg.enable_take_profit:
            return entry_order

        pos = self._wait_for_position(trade.symbol, self.execution_cfg.post_entry_fill_timeout_sec)
        if pos is None:
            # Position not visible yet; next loop will manage it via the
            # position sentinel exit logic.
            return entry_order

        try:
            pos_qty = abs(float(getattr(pos, "qty", 0.0)))
        except Exception:
            pos_qty = abs(trade.qty)

        if pos_qty <= 0:
            return entry_order

        # Determine exit side opposite of entry
        exit_side = "sell" if side == "buy" else "buy"

        # Attach exit order IDs to the entry_order object so main.py can
        # forward them to cancel_exit_legs() when a sentiment exit fires.
        tp_order_id: Optional[str] = None
        ts_order_id: Optional[str] = None

        # Place Take Profit limit (optional)
        if self.execution_cfg.enable_take_profit:
            tp_price = round(float(trade.take_profit_price), 2)
            try:
                tp_order = self.adapter.submit_take_profit_limit_order(
                    symbol=trade.symbol,
                    qty=pos_qty,
                    side=exit_side,
                    limit_price=tp_price,
                    time_in_force=self.execution_cfg.exit_time_in_force,
                )
                tp_order_id = getattr(tp_order, "id", None)
            except APIError as e:
                logger.error(f"Take profit placement failed for {trade.symbol}: {e}")

        # Place Trailing Stop (5% trail by default)
        if self.execution_cfg.enable_trailing_stop:
            try:
                ts_order = self.adapter.submit_trailing_stop_order(
                    symbol=trade.symbol,
                    qty=pos_qty,
                    side=exit_side,
                    trail_percent=float(self.execution_cfg.trailing_stop_percent),
                    time_in_force=self.execution_cfg.exit_time_in_force,
                )
                ts_order_id = getattr(ts_order, "id", None)
            except APIError as e:
                logger.error(f"Trailing stop placement failed for {trade.symbol}: {e}")

        # Attach exit order IDs to the entry order as plain attributes so
        # the caller can persist and later cancel the surviving leg.
        try:
            entry_order.tp_order_id = tp_order_id
            entry_order.ts_order_id = ts_order_id
        except Exception:
            # Alpaca REST objects may not support arbitrary attribute assignment
            # in all SDK versions.  This is best-effort; not fatal.
            pass

        return entry_order

    def close_position_due_to_sentiment(
        self,
        position: PositionInfo,
        sentiment: SentimentResult,
        reason: str,
    ):
        """
        Close an existing position immediately because sentiment has flipped strongly against it.

        For a long position, we submit a sell market order of full qty.
        For a short position, we submit a buy market order of full qty.

        Fix 8b: routes through self.adapter.submit_market_order() instead of
        self.adapter.rest.submit_order() directly — restores adapter abstraction
        and makes this path testable via adapter mocking.
        """
        log_sentiment_close_decision(
            symbol=position.symbol,
            side=position.side,
            qty=position.qty,
            sentiment_score=sentiment.score,
            confidence=sentiment.confidence,
            explanation=sentiment.explanation,
            env_mode=self.env_mode,
            reason=reason,
        )

        if not self._can_place_orders():
            return None

        if position.qty <= 0:
            return None

        if position.side == "long":
            side = "sell"
        elif position.side == "short":
            side = "buy"
        else:
            raise ValueError(f"Unexpected position side {position.side}")

        try:
            # Fix 8b: use adapter method, not bare REST call.
            order = self.adapter.submit_market_order(
                symbol=position.symbol,
                qty=abs(position.qty),
                side=side,
                time_in_force="day",
            )
            return order
        except APIError as e:
            logger.error(f"Close position failed for {position.symbol}: {e}")
            return None