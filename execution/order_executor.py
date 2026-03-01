# CHANGES:
# Feature 7A — Added close_all_positions_for_weekend() public method placed
#   after close_position_due_to_sentiment().
#
#   Execution sequence:
#     1. Logs a WARNING header with the total position count before any orders.
#     2. In paper mode (live_trading_enabled=False): logs a per-symbol INFO
#        line and returns immediately without submitting any orders.
#     3. PRIMARY PATH: calls self.adapter.cancel_all_orders() to clear every
#        open bracket/trailing-stop leg, then self.adapter.close_all_positions()
#        to flatten all positions atomically. Logs success and returns.
#     4. FALLBACK PATH (only if step 3 raises): logs the atomic-close failure,
#        then iterates over positions and calls close_position_due_to_sentiment()
#        per symbol with a synthetic SentimentResult and reason="weekend_forced_close".
#        Does NOT call _cancel_all_open_orders_for_symbol() in the loop —
#        close_position_due_to_sentiment() already calls it internally.
#        Per-symbol exceptions are caught individually so one failure never
#        blocks the rest.
#     5. Never raises under any circumstances.
#
# All prior changes (execute_proposed_trade, close_position_due_to_sentiment,
# _cancel_all_open_orders_for_symbol, _exit_side) are preserved unchanged.

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
      2. Strong exit: delta > strong_exit_delta_threshold AND
                      confidence > strong_exit_confidence_min
         Fires on large sentiment deterioration even at moderate confidence.
      3. Soft exit: delta > soft_exit_delta_threshold AND
                    confidence > exit_confidence_min
         Catches partial deterioration (e.g. +0.7 → 0.0, delta=0.70).

    delta is defined as:
      delta = opening_compound - current_sentiment.score
    A positive delta means sentiment has worsened since entry.

    The effective delta_threshold used for the monitor log is the lowest
    threshold that could have fired (soft wins if confidence qualifies, else
    strong). This keeps the log display consistent with the actual decision.
    """
    sent_cfg = SentimentConfig(cfg.sentiment) if False else cfg.sentiment  # type: ignore[arg-type]
    sent_cfg = cfg.sentiment
    env_mode = str(cfg.env_mode)

    for symbol, pos in positions.items():
        news_items = adapter.get_news(symbol, limit=10)
        current_sentiment: SentimentResult = sentiment_module.force_rescore(symbol, news_items)
        opening_compound = float(pos.opening_compound)
        current_score = float(current_sentiment.score)
        delta = float(opening_compound - current_score)

        hard_exit = current_sentiment.raw_discrete == -2
        strong_exit = (
            not hard_exit
            and delta > sent_cfg.strong_exit_delta_threshold
            and current_sentiment.confidence > sent_cfg.strong_exit_confidence_min
        )
        soft_exit = (
            not hard_exit
            and not strong_exit
            and delta > sent_cfg.soft_exit_delta_threshold
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

        display_threshold = sent_cfg.soft_exit_delta_threshold

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
      5. Fall back to entry + standalone trailing-stop when only
         enable_trailing_stop is True.
      6. Fall back to plain market order when neither exit type is configured.

    Sentiment exit path (close_position_due_to_sentiment):
      Submits a market order to flatten the position on the opposing side,
      cancelling open orders for the symbol first to avoid qty conflicts.

    Weekend liquidation path (close_all_positions_for_weekend):
      Atomically flattens all open positions before market close on Friday.
      Falls back to per-symbol sentiment-close if the atomic call fails.
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
            logger.warning(f"_cancel_all_open_orders_for_symbol({symbol}) error: {e}")

    def _exit_side(self, position_side: str) -> str:
        """Return the closing side for a given position side."""
        return "sell" if position_side == "long" else "buy"

    # ── Entry execution ───────────────────────────────────────────────────────

    def execute_proposed_trade(self, proposed: ProposedTrade) -> Optional[object]:
        """
        Execute a ProposedTrade.

        Returns the Alpaca order object on success, or None if skipped/failed.
        """
        log_proposed_trade(proposed, self.env_mode)

        if proposed.rejected_reason is not None or proposed.qty <= 0:
            return None

        if not self.live_trading_enabled:
            logger.info(
                f"[PAPER] Skipping live order submission for {proposed.symbol} "
                f"(live_trading_enabled=False)."
            )
            return None

        symbol = proposed.symbol
        qty = proposed.qty
        side = proposed.side
        stop_price = proposed.stop_price
        take_profit_price = proposed.take_profit_price

        self._cancel_all_open_orders_for_symbol(symbol)

        use_tp = self.execution_cfg.enable_take_profit
        use_ts = self.execution_cfg.enable_trailing_stop

        try:
            if use_tp and use_ts:
                # Bracket order: entry market + OCO (TP limit + stop-loss).
                order = self.adapter.submit_bracket_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    time_in_force=self.execution_cfg.entry_time_in_force,
                )
                logger.info(
                    f"Bracket order submitted for {symbol}: side={side} qty={qty} "
                    f"stop={stop_price:.2f} tp={take_profit_price:.2f} "
                    f"order_id={getattr(order, 'id', 'N/A')}"
                )
            elif use_ts:
                # Entry market order + separate trailing-stop exit.
                order = self.adapter.submit_market_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=self.execution_cfg.entry_time_in_force,
                )
                logger.info(
                    f"Market order submitted for {symbol}: side={side} qty={qty} "
                    f"order_id={getattr(order, 'id', 'N/A')}"
                )
                # Wait briefly for fill before attaching stop.
                time.sleep(self.execution_cfg.post_entry_fill_timeout_sec)

                exit_side = self._exit_side(side)
                try:
                    stop_order = self.adapter.submit_trailing_stop_order(
                        symbol=symbol,
                        qty=qty,
                        side=exit_side,
                        trail_percent=self.execution_cfg.trailing_stop_percent,
                        time_in_force=self.execution_cfg.exit_time_in_force,
                    )
                    logger.info(
                        f"Trailing stop attached for {symbol}: side={exit_side} qty={qty} "
                        f"trail%={self.execution_cfg.trailing_stop_percent} "
                        f"order_id={getattr(stop_order, 'id', 'N/A')}"
                    )
                except APIError as e:
                    logger.warning(
                        f"Trailing stop placement failed for {symbol}: {e}"
                    )
            else:
                # Plain market order, no automated exit leg.
                order = self.adapter.submit_market_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=self.execution_cfg.entry_time_in_force,
                )
                logger.info(
                    f"Market order (no exit leg) submitted for {symbol}: "
                    f"side={side} qty={qty} "
                    f"order_id={getattr(order, 'id', 'N/A')}"
                )
            return order
        except APIError as e:
            logger.error(f"execute_proposed_trade APIError for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"execute_proposed_trade unexpected error for {symbol}: {e}")
            return None

    # ── Sentiment exit ────────────────────────────────────────────────────────

    def close_position_due_to_sentiment(
        self,
        position: PositionInfo,
        sentiment: SentimentResult,
        reason: str,
        env_mode: str,
    ) -> None:
        """Flatten position with a market order on the opposing side.

        Steps:
          1. Cancel all open orders for the symbol (avoids qty-reservation
             conflicts with any bracket/trailing-stop legs still live).
          2. Submit a market order for the full position qty on the exit side.
          3. Log the closure via log_sentiment_close_decision.

        Errors are caught and logged — the method never raises so the main loop
        can continue processing other positions.
        """
        symbol = position.symbol
        qty = abs(position.qty)
        exit_side = self._exit_side(position.side)

        log_sentiment_close_decision(
            symbol=symbol,
            side=exit_side,
            qty=qty,
            sentiment_score=sentiment.score,
            confidence=sentiment.confidence,
            explanation=sentiment.explanation or "",
            env_mode=env_mode,
            reason=reason,
        )

        if not self.live_trading_enabled:
            logger.info(
                f"[PAPER] Skipping live sentiment-close for {symbol} "
                f"(live_trading_enabled=False)."
            )
            return

        self._cancel_all_open_orders_for_symbol(symbol)
        try:
            order = self.adapter.submit_market_order(
                symbol,
                qty,
                exit_side,
                time_in_force=self.execution_cfg.exit_time_in_force,
            )
            logger.info(
                f"Sentiment-close market order submitted for {symbol}: "
                f"side={exit_side} qty={qty} reason={reason} "
                f"order_id={getattr(order, 'id', 'N/A')}"
            )
        except APIError as e:
            logger.error(f"close_position_due_to_sentiment APIError for {symbol}: {e}")
        except Exception as e:
            logger.error(f"close_position_due_to_sentiment error for {symbol}: {e}")

    # ── Weekend forced liquidation ────────────────────────────────────────────

    def close_all_positions_for_weekend(
        self,
        positions: Dict[str, PositionInfo],
        envmode: str,
    ) -> None:
        """
        Flatten all open positions before the weekend market close.

        Execution sequence:
          1. Log a single WARNING-level header before any order activity.
          2. In paper mode: log a per-symbol INFO line and return immediately.
          3. PRIMARY PATH — atomic broker close:
             a. cancel_all_orders() to clear bracket/trailing-stop legs.
             b. close_all_positions() to flatten all positions atomically.
             c. Log success and return.
          4. FALLBACK PATH (only if step 3 raises):
             Log the failure, then iterate over positions calling
             close_position_due_to_sentiment() for each with a synthetic
             SentimentResult and reason="weekend_forced_close".
             Per-symbol exceptions are caught individually.
             Does NOT call _cancel_all_open_orders_for_symbol() in this loop —
             close_position_due_to_sentiment() already calls it internally.
          5. Never raises under any circumstances.
        """
        try:
            n = len(positions)
            logger.warning(
                f"WEEKEND CLOSE: Liquidating all {n} positions before market close."
            )

            if not self.live_trading_enabled:
                for symbol in positions:
                    logger.info(f"PAPER: Skipping live weekend-close for {symbol}.")
                return

            # PRIMARY PATH — atomic close
            try:
                self.adapter.cancel_all_orders()
                self.adapter.close_all_positions()
                logger.info("WEEKEND CLOSE: Atomic close_all_positions succeeded.")
                return
            except Exception as atomic_err:
                logger.warning(
                    f"WEEKEND CLOSE: Atomic close failed ({atomic_err}). "
                    f"Falling back to per-symbol sentiment-close."
                )

            # FALLBACK PATH — per-symbol close
            synthetic_sentiment = SentimentResult(
                score=0.0,
                raw_discrete=0,
                rawcompound=0.0,
                ndocuments=0,
                explanation="Weekend close — forced liquidation before market close.",
                confidence=1.0,
            )
            for symbol, position in positions.items():
                try:
                    self.close_position_due_to_sentiment(
                        position=position,
                        sentiment=synthetic_sentiment,
                        reason="weekend_forced_close",
                        env_mode=envmode,
                    )
                except Exception as sym_err:
                    logger.warning(
                        f"WEEKEND CLOSE: Fallback close failed for {symbol}: {sym_err}"
                    )
        except Exception as outer_err:
            logger.warning(
                f"close_all_positions_for_weekend unexpected error: {outer_err}"
            )
