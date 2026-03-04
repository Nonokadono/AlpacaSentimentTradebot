# CHANGES:
# FIX 3 — Added best-effort cancel attempt in execute_proposed_trade() when _wait_for_position() returns False.
#         Wrap in try/except so the cancel attempt never raises; log both success and failure cases.
# FIX 4D (FIX 9) — Deleted dead code line `sent_cfg = SentimentConfig(cfg.sentiment) if False else cfg.sentiment`.

import logging
import time
from typing import Dict, Optional

from alpaca_trade_api.rest import APIError

from adapters.alpaca_adapter import AlpacaAdapter
from config.config import BotConfig, ExecutionConfig, SentimentConfig
from core.risk_engine import PositionInfo, ProposedTrade
from core.sentiment import SentimentModule, SentimentResult
from monitoring.monitor import (
    log_proposed_trade,
    log_sentiment_close_decision,
    log_sentiment_position_check,
)

logger = logging.getLogger("tradebot")


def _check_and_exit_on_sentiment(
    positions: Dict[str, PositionInfo],
    adapter: AlpacaAdapter,
    sentiment_module: SentimentModule,
    executor: "OrderExecutor",
    cfg: BotConfig,
) -> None:
    """For every open position, force-rescore sentiment and compare the current
    compound score against the score recorded at entry time
    (PositionInfo.opening_compound).

    Three exit tiers from SentimentConfig:
      1. Hard exit (chaos): raw_discrete == -2 → close unconditionally, no delta check.
      2. Strong exit: delta > effective_strong_threshold AND
                      confidence > strong_exit_confidence_min
         Fires on large sentiment deterioration even at moderate confidence.
      3. Soft exit: delta > effective_soft_threshold AND
                    confidence > exit_confidence_min
         Catches partial deterioration (e.g. +0.7 → 0.0, delta=0.70).

    delta is defined side-aware (WRONG-2 FIX):
      long:  delta = opening_compound - current_score  (positive = sentiment worsened)
      short: delta = current_score - opening_compound  (positive = sentiment improved = exit)
    A positive delta in either case means the position should be exited.

    When pnl_exit_scale_enabled is True, effective thresholds are PnL-scaled:
      scale_adj               = unrealised_pnl_pct * pnl_exit_scale_factor
      effective_soft_threshold   = max(0.3, soft_exit_delta_threshold   + scale_adj)
      effective_strong_threshold = max(0.5, strong_exit_delta_threshold + scale_adj)
    where unrealised_pnl_pct is +ve for winning positions and -ve for losing ones.
    This widens thresholds for winners (harder to exit) and tightens them for
    losers (easier to exit).  The hard exit is NEVER gated by these thresholds.

    The effective_soft_threshold is used as display_threshold in the monitor log
    so the operator always sees the actual threshold that drove the decision.
    """
    # FIX 4D (FIX 9): Deleted the dead code line; only this line remains.
    sent_cfg = cfg.sentiment
    env_mode = str(cfg.env_mode)

    for symbol, pos in positions.items():
        news_items = adapter.get_news(symbol, limit=10)
        current_sentiment: SentimentResult = sentiment_module.force_rescore(symbol, news_items)
        opening_compound = float(pos.opening_compound)
        current_score = float(current_sentiment.score)

        # WRONG-2 FIX: side-aware delta so shorts exit on improving sentiment.
        # Long:  positive delta means sentiment has worsened since entry → exit.
        # Short: positive delta means sentiment has improved since entry → exit.
        if pos.side == "long":
            delta = float(opening_compound - current_score)
        else:  # short
            delta = float(current_score - opening_compound)

        # --- PnL-Coupled Sentiment Exit Threshold Scaling ---
        # Derive unrealised P&L percentage from PositionInfo fields.
        # avg_entry_price defaults to 0.0; when not populated the adjustment
        # resolves to 0.0 (no-op), preserving legacy behaviour completely.
        if pos.avg_entry_price > 0.0:
            raw_pnl_pct = (pos.market_price - pos.avg_entry_price) / pos.avg_entry_price
            # Short positions gain when price falls, so flip the sign.
            unrealised_pnl_pct = -raw_pnl_pct if pos.side == "short" else raw_pnl_pct
        else:
            unrealised_pnl_pct = 0.0

        if sent_cfg.pnl_exit_scale_enabled:
            scale_adj = unrealised_pnl_pct * sent_cfg.pnl_exit_scale_factor
            # Floor clamps prevent runaway losses being held indefinitely.
            effective_soft_threshold = max(
                0.3, sent_cfg.soft_exit_delta_threshold + scale_adj
            )
            effective_strong_threshold = max(
                0.5, sent_cfg.strong_exit_delta_threshold + scale_adj
            )
        else:
            # Scaling disabled: effective thresholds equal raw config values.
            effective_soft_threshold = sent_cfg.soft_exit_delta_threshold
            effective_strong_threshold = sent_cfg.strong_exit_delta_threshold
        # -----------------------------------------------------

        hard_exit = current_sentiment.raw_discrete == -2
        strong_exit = (
            not hard_exit
            and delta > effective_strong_threshold
            and current_sentiment.confidence > sent_cfg.strong_exit_confidence_min
        )
        soft_exit = (
            not hard_exit
            and not strong_exit
            and delta > effective_soft_threshold
            and current_sentiment.confidence > sent_cfg.exit_confidence_min
        )
        closing = hard_exit or strong_exit or soft_exit

        # Determine which (if any) exit tier fires
        if hard_exit:
            close_reason = "hard_exit_chaos"
        elif strong_exit:
            close_reason = "strong_exit"
        elif soft_exit:
            close_reason = "soft_exit"
        else:
            close_reason = "no_exit"

        # Use effective_soft_threshold as display_threshold so the log always
        # reflects the actual threshold applied to the decision (may differ
        # from the raw config value when PnL scaling is active).
        display_threshold = effective_soft_threshold

        log_sentiment_position_check(
            position=pos,
            entry_compound=opening_compound,
            current_sentiment=current_sentiment,
            delta=delta,
            delta_threshold=display_threshold,
            confidence_min=sent_cfg.exit_confidence_min if not hard_exit else 0.0,
            closing=closing,
            close_reason=close_reason,
            env_mode=env_mode,
            stop_price=None,
            take_profit_price=None,
            pnl_exit_scale_enabled=sent_cfg.pnl_exit_scale_enabled,
            pnl_exit_scale_factor=sent_cfg.pnl_exit_scale_factor,
        )

        if closing:
            executor.close_position_due_to_sentiment(
                position=pos,
                sentiment=current_sentiment,
                reason=close_reason,
                env_mode=env_mode,
            )


class OrderExecutor:
    """
    Submits entry and exit orders to Alpaca.

    Execution path (execute_proposed_trade):
      1. Log the proposed trade.
      2. If live_trading_enabled is False (paper mode guard), skip submission.
      3. Cancel any open orders for the symbol before entering.
      4. Submit a bracket order (entry + OCO TP/stop) when both
         enable_take_profit and enable_trailing_stop are True.
         AUDIT: bracket is a single atomic API payload; Alpaca's OMS activates
         the TP limit and stop-loss legs automatically on parent fill.  No
         post-fill confirmation is needed — the broker owns the state machine.
      5. Fall back to entry + standalone trailing-stop when only
         enable_trailing_stop is True.  The trailing stop is ONLY submitted
         after _wait_for_position() confirms the fill.  If confirmation times
         out, execute_proposed_trade() returns None — NOT the entry order —
         so main.py will NOT record _opening_compounds for an unconfirmed
         position.
      6. Fall back to plain market order when neither exit type is configured.
         AUDIT: no exit leg is attached, so no fill-confirmation requirement.

    Sentiment exit path (close_position_due_to_sentiment):
      AUDIT: _cancel_all_open_orders_for_symbol() is called first to clear any
      bracket/trailing-stop legs still live, then a single market order flattens
      the position.  No secondary order is submitted after the close order, so
      no fill-confirmation is needed for a flat-position-close.

    Weekend liquidation path (close_all_positions_for_weekend):
      AUDIT: cancel_all_orders() precedes close_all_positions() in the primary
      (atomic) path.  The broker handles the atomic close; no fill-confirmation
      is needed.  The per-symbol fallback delegates to close_position_due_to_sentiment()
      which already cancels orders internally.
    """

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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cancel_all_open_orders_for_symbol(self, symbol: str) -> None:
        """Cancel every open order for *symbol*.  Errors are logged and swallowed."""
        try:
            open_orders = self.adapter.list_orders(status="open")
            for order in open_orders:
                if getattr(order, "symbol", None) == symbol:
                    try:
                        self.adapter.cancel_order(order.id)
                        logger.info(
                            f"Cancelled open order {order.id} for {symbol} "
                            f"before new submission."
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not cancel order {order.id} for {symbol}: {e}"
                        )
        except Exception as e:
            logger.warning(f"Could not retrieve open orders for {symbol}: {e}")

    def _wait_for_position(self, symbol: str) -> bool:
        """
        Poll for up to post_entry_fill_poll_timeout_sec seconds (default 30s)
        to confirm that the position appears in the broker's positions list.

        Returns True if the position is found within the timeout, False otherwise.
        """
        timeout = self.execution_cfg.post_entry_fill_poll_timeout_sec
        interval = self.execution_cfg.post_entry_fill_poll_interval_sec
        elapsed = 0.0
        while elapsed < timeout:
            time.sleep(interval)
            elapsed += interval
            pos = self.adapter.get_position(symbol)
            if pos is not None:
                logger.info(
                    f"Position {symbol} confirmed after {elapsed:.1f}s."
                )
                return True
        logger.warning(
            f"Position {symbol} not confirmed within {timeout}s timeout."
        )
        return False

    # ── Entry paths ───────────────────────────────────────────────────────────

    def execute_proposed_trade(self, proposed: ProposedTrade) -> Optional[object]:
        """
        Submit entry order (and attached exit orders when configured).

        Returns the order object on success, None on failure or paper-mode skip.

        FIX 3: When _wait_for_position() times out in the trailing-stop path,
        we now attempt to cancel the entry order before returning None.
        """
        log_proposed_trade(proposed, self.env_mode)

        if not self.live_trading_enabled:
            logger.info(
                f"[PAPER MODE] Skipping order submission for {proposed.symbol}"
            )
            return None

        if proposed.qty <= 0 or proposed.rejected_reason is not None:
            return None

        self._cancel_all_open_orders_for_symbol(proposed.symbol)

        symbol = proposed.symbol
        qty = proposed.qty
        side = proposed.side
        tif = self.execution_cfg.entry_time_in_force

        try:
            # Path 1: Bracket order (atomic TP + stop)
            if (
                self.execution_cfg.enable_take_profit
                and self.execution_cfg.enable_trailing_stop
            ):
                order = self.adapter.submit_bracket_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    stop_price=proposed.stop_price,
                    take_profit_price=proposed.take_profit_price,
                    time_in_force=tif,
                )
                logger.info(
                    f"Submitted bracket order for {symbol}: {side} {qty} @ market, "
                    f"stop={proposed.stop_price:.2f}, tp={proposed.take_profit_price:.2f}"
                )
                return order

            # Path 2: Entry + trailing stop (requires fill confirmation)
            elif self.execution_cfg.enable_trailing_stop:
                order = self.adapter.submit_market_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                )
                logger.info(
                    f"Submitted entry market order for {symbol}: {side} {qty}"
                )

                filled = self._wait_for_position(symbol)
                if not filled:
                    # FIX 3: attempt to cancel the entry order before returning None
                    try:
                        self.adapter.cancel_order(getattr(order, "id", None) or "")
                        logger.warning(
                            f"Fill timeout for {symbol} — attempted cancellation of "
                            f"entry order {getattr(order, 'id', 'N/A')}."
                        )
                    except Exception as cancel_err:
                        logger.warning(
                            f"Fill timeout for {symbol} — cancellation attempt failed: "
                            f"{cancel_err}. Position may be open without a stop."
                        )
                    return None

                # Fill confirmed — submit trailing stop
                opposite_side = "sell" if side == "buy" else "buy"
                trail_pct = self.execution_cfg.trailing_stop_percent
                exit_tif = self.execution_cfg.exit_time_in_force
                self.adapter.submit_trailing_stop_order(
                    symbol=symbol,
                    qty=qty,
                    side=opposite_side,
                    trail_percent=trail_pct,
                    time_in_force=exit_tif,
                )
                logger.info(
                    f"Submitted trailing stop for {symbol}: {opposite_side} {qty} "
                    f"trail={trail_pct}%"
                )
                return order

            # Path 3: Plain market order (no exit legs)
            else:
                order = self.adapter.submit_market_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=tif,
                )
                logger.info(
                    f"Submitted plain market order for {symbol}: {side} {qty}"
                )
                return order

        except APIError as e:
            logger.error(f"API error executing trade for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error executing trade for {symbol}: {e}")
            return None

    # ── Exit paths ────────────────────────────────────────────────────────────

    def close_position_due_to_sentiment(
        self,
        position: PositionInfo,
        sentiment: SentimentResult,
        reason: str,
        env_mode: str,
    ) -> None:
        """Close a position due to sentiment deterioration."""
        if not self.live_trading_enabled:
            logger.info(
                f"[PAPER MODE] Would close {position.symbol} due to sentiment: {reason}"
            )
            return

        log_sentiment_close_decision(
            symbol=position.symbol,
            sentiment=sentiment,
            reason=reason,
            env_mode=env_mode,
        )

        self._cancel_all_open_orders_for_symbol(position.symbol)

        try:
            opposite_side = "sell" if position.side == "long" else "buy"
            self.adapter.submit_market_order(
                symbol=position.symbol,
                qty=abs(position.qty),
                side=opposite_side,
                time_in_force="day",
            )
            logger.info(
                f"Closed position {position.symbol} ({reason}): "
                f"{opposite_side} {abs(position.qty)} @ market"
            )
        except Exception as e:
            logger.error(
                f"Error closing position {position.symbol} due to sentiment: {e}"
            )

    def close_all_positions_for_weekend(
        self,
        positions: Dict[str, PositionInfo],
        env_mode: str,
    ) -> None:
        """Force-close all positions before weekend."""
        if not self.live_trading_enabled:
            logger.info("[PAPER MODE] Would close all positions for weekend.")
            return

        logger.warning("WEEKEND CLOSE: Closing all positions.")

        try:
            self.adapter.cancel_all_orders()
            self.adapter.close_all_positions()
            logger.info("All positions closed (atomic broker call).")
        except Exception as e:
            logger.error(
                f"Atomic close_all_positions failed: {e}. Falling back to per-symbol close."
            )
            for symbol, pos in positions.items():
                neutral_sentiment = SentimentResult(
                    score=0.0,
                    raw_discrete=0,
                    rawcompound=0.0,
                    ndocuments=0,
                    explanation="Weekend forced liquidation",
                    confidence=0.0,
                )
                self.close_position_due_to_sentiment(
                    position=pos,
                    sentiment=neutral_sentiment,
                    reason="weekend_forced_liquidation",
                    env_mode=env_mode,
                )
