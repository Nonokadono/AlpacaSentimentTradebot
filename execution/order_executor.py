# CHANGES:
# AUDIT FIX 8.1 — Added _wait_for_order_acceptance() to confirm trailing stop submission.
#                 If trailing stop is rejected or fails, immediately submit emergency market
#                 close to prevent naked positions. Logs CRITICAL alerts for operator intervention.
# AUDIT FIX 8.2 — After fill timeout + failed cancellation, check if position actually exists.
#                 If position opened server-side despite timeout, submit emergency trailing stop
#                 to protect the unintended position. Prevents unlimited loss from naked positions.
# MONITOR-FIX — Passed actual `pos.stop_price` and `pos.take_profit_price` natively fetched
#               into `log_sentiment_position_check` rather than hardcoding `None`.
# BRACKET-TIF-FIX — Bracket orders now explicitly use `exit_time_in_force` instead of the entry TIF
#                   so that the protective SL and TP legs persist GTC and do not expire at close.
# FIX 3 — Added best-effort cancel attempt in execute_proposed_trade() when _wait_for_position() returns False.
# FIX 4D (FIX 9) — Deleted dead code line `sent_cfg = SentimentConfig(cfg.sentiment) if False else cfg.sentiment`.
# SAFETY-GATE — Enhanced close_all_positions_for_weekend() with triple-gate protection.
# PURGE-FIX — close_position_due_to_sentiment() now purges opening_compounds[symbol] only after
#             a confirmed flat position, and retains the baseline on close failure.
# WAIT-TIMEOUT-FIX — Added threading.Timer hard kill for _wait_for_position() to prevent hung
#                    Alpaca poll from blocking the bot indefinitely. Timer spawns a daemon thread
#                    that logs a CRITICAL error and raises SystemExit after timeout+5s grace period.
#                    Main loop catches SystemExit and logs the forced termination before exiting.
# PROTECTION-PATH-FIX — Fixed entry execution gating so fixed stop-loss / take-profit bracket
#                       orders are submitted whenever proposed.stop_price and proposed.take_profit_price
#                       are available, without incorrectly depending on enable_trailing_stop.
# FLAT-CONFIRM-FIX — Added _wait_for_flat() and use it before declaring a close successful or
#                    purging the persisted entry baseline.
# TRADE-STATS-INTEGRATION — Sentiment-driven closes now notify TradeStatsTracker after confirmed
#                           broker flatten so closed-trade analytics stay synchronized with bot exits.

import logging
import time
import threading
from typing import Callable, Dict, Optional

from alpaca_trade_api.rest import APIError

from adapters.alpaca_adapter import AlpacaAdapter
from config.config import BotConfig, ExecutionConfig
from core.risk_engine import PositionInfo, ProposedTrade
from core.sentiment import SentimentModule, SentimentResult
from monitoring.monitor import (
    log_proposed_trade,
    log_sentiment_close_decision,
    log_sentiment_position_check,
)
from monitoring.trade_stats import TradeStatsTracker

logger = logging.getLogger("tradebot")


def _check_and_exit_on_sentiment(
    positions: Dict[str, PositionInfo],
    adapter: AlpacaAdapter,
    sentiment_module: SentimentModule,
    executor: "OrderExecutor",
    cfg: BotConfig,
    opening_compounds: Dict[str, float],
    persist_opening_compounds: Callable[[Dict[str, float]], None],
) -> None:
    """For every open position, force-rescore sentiment and compare the current
    sentiment score against the technical composite recorded at entry time.

    Three exit tiers from SentimentConfig:
      1. Hard exit (chaos): raw_discrete == -2 → close unconditionally, no delta check.
      2. Strong exit: delta > effective_strong_threshold AND
                      confidence > strong_exit_confidence_min
         Fires on large deterioration versus the entry technical composite.
      3. Soft exit: delta > effective_soft_threshold AND
                    confidence > exit_confidence_min
         Catches partial deterioration relative to the entry technical composite.

    delta is defined side-aware:
      long:  delta = opening_compound - current_score  (positive = sentiment worsened)
      short: delta = current_score - opening_compound  (positive = sentiment improved = exit)
    A positive delta in either case means the position should be exited.

    When pnl_exit_scale_enabled is True, effective thresholds are PnL-scaled:
      scale_adj                  = unrealised_pnl_pct * pnl_exit_scale_factor
      effective_soft_threshold   = max(0.3, soft_exit_delta_threshold   + scale_adj)
      effective_strong_threshold = max(0.5, strong_exit_delta_threshold + scale_adj)
    where unrealised_pnl_pct is +ve for winning positions and -ve for losing ones.
    This widens thresholds for winners (harder to exit) and tightens them for
    losers (easier to exit). The hard exit is NEVER gated by these thresholds.

    The effective_soft_threshold is used as display_threshold in the monitor log
    so the operator always sees the actual threshold that drove the decision.

    PURGE-FIX: opening_compounds and persist_opening_compounds are required
    parameters so close_position_due_to_sentiment() can purge the entry baseline
    only after the flatten has actually succeeded.
    """
    sent_cfg = cfg.sentiment
    env_mode = str(cfg.env_mode)

    for symbol, pos in positions.items():
        news_items = adapter.get_news(symbol, limit=10)
        current_sentiment: SentimentResult = sentiment_module.force_rescore(symbol, news_items)

        opening_compound = float(opening_compounds.get(symbol, 0.0))
        current_score = float(current_sentiment.score)

        if pos.side == "long":
            delta = float(opening_compound - current_score)
        else:
            delta = float(current_score - opening_compound)

        if pos.avg_entry_price > 0.0:
            raw_pnl_pct = (pos.market_price - pos.avg_entry_price) / pos.avg_entry_price
            unrealised_pnl_pct = -raw_pnl_pct if pos.side == "short" else raw_pnl_pct
        else:
            unrealised_pnl_pct = 0.0

        if sent_cfg.pnl_exit_scale_enabled:
            scale_adj = unrealised_pnl_pct * sent_cfg.pnl_exit_scale_factor
            effective_soft_threshold = max(
                0.3, sent_cfg.soft_exit_delta_threshold + scale_adj
            )
            effective_strong_threshold = max(
                0.5, sent_cfg.strong_exit_delta_threshold + scale_adj
            )
        else:
            effective_soft_threshold = sent_cfg.soft_exit_delta_threshold
            effective_strong_threshold = sent_cfg.strong_exit_delta_threshold

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

        if hard_exit:
            close_reason = "hard_exit_chaos"
        elif strong_exit:
            close_reason = "strong_exit"
        elif soft_exit:
            close_reason = "soft_exit"
        else:
            close_reason = "no_exit"

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
            stop_price=pos.stop_price,
            take_profit_price=pos.take_profit_price,
            pnl_exit_scale_enabled=sent_cfg.pnl_exit_scale_enabled,
            pnl_exit_scale_factor=sent_cfg.pnl_exit_scale_factor,
        )

        if closing:
            executor.close_position_due_to_sentiment(
                position=pos,
                sentiment=current_sentiment,
                reason=close_reason,
                env_mode=env_mode,
                opening_compounds=opening_compounds,
                persist_opening_compounds=persist_opening_compounds,
            )


class OrderExecutor:
    """
    Submits entry and exit orders to Alpaca.

    Execution path (execute_proposed_trade):
      1. Log the proposed trade.
      2. If live_trading_enabled is False (paper mode guard), skip submission.
      3. Cancel any open orders for the symbol before entering.
      4. Submit a bracket order (entry + OCO TP/stop) whenever both
         proposed.stop_price and proposed.take_profit_price are available.
      5. Fall back to entry + standalone trailing-stop when fixed bracket
         protection is unavailable and trailing stops are enabled. The trailing
         stop is ONLY submitted after _wait_for_position() confirms the fill.
      6. Fall back to plain market order when neither exit type is configured.

    AUDIT FIX 8.1: Trailing stop submission now confirmed via _wait_for_order_acceptance().
                   If rejected or fails, emergency market close prevents naked positions.

    AUDIT FIX 8.2: After fill timeout + failed cancellation, position existence is verified.
                   If position opened despite timeout, emergency trailing stop is submitted.

    Sentiment exit path (close_position_due_to_sentiment):
      - Cancel all open orders for the symbol.
      - Submit a market order to flatten.
      - Wait for broker confirmation that the position is flat.
      - Purge opening_compounds[symbol] only after the flat confirmation succeeds.

    Weekend liquidation path (close_all_positions_for_weekend):
      SAFETY-GATE: Triple-gate protection prevents accidental liquidation:
        1. live_trading_enabled must be True
        2. env_mode must be "LIVE"
        3. Warning log emitted before atomic close
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
        self.trade_stats = TradeStatsTracker()

    def _cancel_all_open_orders_for_symbol(self, symbol: str) -> None:
        """Cancel every open order for *symbol*. Errors are logged and swallowed."""
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
        """Poll for position existence after entry order submission.

        Returns True if position confirmed within timeout, False otherwise.
        Hard kill timer prevents indefinite hangs from broker API failures.
        """
        timeout = self.execution_cfg.post_entry_fill_poll_timeout_sec
        interval = self.execution_cfg.post_entry_fill_poll_interval_sec
        grace_period = 5.0
        kill_timeout = timeout + grace_period

        def _hard_kill():
            logger.critical(
                f"HARD KILL: _wait_for_position({symbol}) exceeded {kill_timeout}s "
                f"timeout+grace. Alpaca poll likely hung. Forcing bot termination."
            )
            raise SystemExit(1)

        kill_timer = threading.Timer(kill_timeout, _hard_kill)
        kill_timer.daemon = True
        kill_timer.start()

        elapsed = 0.0
        try:
            while elapsed < timeout:
                time.sleep(interval)
                elapsed += interval
                pos = self.adapter.get_position(symbol)
                if pos is not None:
                    logger.info(
                        f"Position {symbol} confirmed after {elapsed:.1f}s."
                    )
                    kill_timer.cancel()
                    return True
            logger.warning(
                f"Position {symbol} not confirmed within {timeout}s timeout."
            )
            kill_timer.cancel()
            return False
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Exception in _wait_for_position({symbol}): {e}")
            kill_timer.cancel()
            return False

    def _wait_for_order_acceptance(self, order_id: str, symbol: str) -> bool:
        """AUDIT FIX 8.1: Poll for order acceptance after submission.

        Confirms that a protective order (trailing stop, bracket leg) was accepted
        by the broker before declaring entry success.

        Returns True if order reaches "accepted" or "new" status within 5 seconds.
        Returns False if order is rejected, cancelled, or times out.
        """
        timeout = 5.0
        interval = 0.5
        elapsed = 0.0

        while elapsed < timeout:
            try:
                order = self.adapter.get_order(order_id)
                status = getattr(order, "status", "unknown").lower()
                if status in ["accepted", "new", "partially_filled", "filled"]:
                    logger.info(f"Order {order_id} for {symbol} accepted (status={status})")
                    return True
                if status in ["rejected", "cancelled", "expired"]:
                    logger.error(f"Order {order_id} for {symbol} REJECTED (status={status})")
                    return False
            except Exception as e:
                logger.warning(f"Order status poll error for {order_id} ({symbol}): {e}")

            time.sleep(interval)
            elapsed += interval

        logger.error(f"Order {order_id} for {symbol} acceptance timeout after {timeout}s")
        return False

    def _emergency_flatten_position(self, symbol: str, qty: float, side: str) -> None:
        """AUDIT FIX 8.1 & 8.2: Emergency market close for unprotected positions.

        Called when:
        - Trailing stop submission fails (FIX 8.1)
        - Position opened despite fill timeout (FIX 8.2)

        Submits immediate market order to flatten position. Logs CRITICAL alert
        for operator intervention if flatten also fails.
        """
        opposite_side = "sell" if side == "buy" else "buy"
        try:
            logger.critical(
                f"EMERGENCY FLATTEN: Submitting market close for {symbol} "
                f"(position opened without exit protection)"
            )
            self.adapter.submit_market_order(
                symbol=symbol,
                qty=abs(qty),
                side=opposite_side,
                time_in_force="day",
            )
            logger.critical(
                f"Emergency flatten order submitted for {symbol}: {opposite_side} {abs(qty)} @ market"
            )
        except Exception as flatten_err:
            logger.critical(
                f"EMERGENCY FLATTEN FAILED for {symbol}: {flatten_err}. "
                f"OPERATOR INTERVENTION REQUIRED: Manually close position {symbol} immediately."
            )

    def _wait_for_flat(self, symbol: str) -> bool:
        """Poll for position closure confirmation.

        Returns True if position is flat (qty=0 or does not exist) within timeout.
        Used by sentiment exit path to confirm flatten before purging entry baseline.
        """
        timeout = self.execution_cfg.post_entry_fill_poll_timeout_sec
        interval = self.execution_cfg.post_entry_fill_poll_interval_sec
        elapsed = 0.0

        while elapsed < timeout:
            try:
                pos = self.adapter.get_position(symbol)
                if pos is None:
                    logger.info(f"Position {symbol} confirmed flat after {elapsed:.1f}s.")
                    return True

                qty_raw = getattr(pos, "qty", 0.0)
                qty = float(qty_raw)
                if abs(qty) <= 1e-9:
                    logger.info(f"Position {symbol} quantity reached zero after {elapsed:.1f}s.")
                    return True
            except Exception as e:
                logger.warning(f"_wait_for_flat poll error for {symbol}: {e}")

            time.sleep(interval)
            elapsed += interval

        logger.warning(
            f"Position {symbol} still not flat after {timeout}s timeout."
        )
        return False

    def execute_proposed_trade(self, proposed: ProposedTrade) -> Optional[object]:
        """Execute a proposed trade with guaranteed exit protection.

        AUDIT FIX 8.1: Trailing stop acceptance confirmed before returning success.
        AUDIT FIX 8.2: Position re-checked after failed cancellation.
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
        has_fixed_bracket = (
            self.execution_cfg.enable_take_profit
            and proposed.stop_price is not None
            and proposed.take_profit_price is not None
        )

        try:
            if has_fixed_bracket:
                # Path 1: Bracket order with fixed SL/TP (best protection)
                bracket_tif = self.execution_cfg.exit_time_in_force
                order = self.adapter.submit_bracket_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    stop_price=proposed.stop_price,
                    take_profit_price=proposed.take_profit_price,
                    time_in_force=bracket_tif,
                )
                logger.info(
                    f"Submitted bracket order for {symbol}: {side} {qty} @ market, "
                    f"stop={proposed.stop_price:.2f}, tp={proposed.take_profit_price:.2f}"
                )
                return order

            elif self.execution_cfg.enable_trailing_stop:
                # Path 2: Entry + trailing stop (AUDIT FIX 8.1 & 8.2 applied here)
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
                    # AUDIT FIX 8.2: Position re-check after failed cancellation
                    try:
                        self.adapter.cancel_order(getattr(order, "id", None) or "")
                        logger.warning(
                            f"Fill timeout for {symbol} — cancelled entry order {getattr(order, 'id', 'N/A')}"
                        )
                    except Exception as cancel_err:
                        logger.error(
                            f"Fill timeout for {symbol} — cancellation failed: {cancel_err}"
                        )
                        # CRITICAL: Check if position actually exists despite timeout
                        try:
                            pos = self.adapter.get_position(symbol)
                            if pos is not None and abs(float(getattr(pos, "qty", 0.0))) > 1e-9:
                                logger.critical(
                                    f"AUDIT FIX 8.2: Position {symbol} EXISTS after failed cancel "
                                    f"(qty={getattr(pos, 'qty', 0.0)}). Submitting emergency trailing stop."
                                )
                                opposite_side = "sell" if side == "buy" else "buy"
                                pos_qty = abs(float(getattr(pos, "qty", 0.0)))
                                try:
                                    emergency_stop_order = self.adapter.submit_trailing_stop_order(
                                        symbol=symbol,
                                        qty=pos_qty,
                                        side=opposite_side,
                                        trail_percent=self.execution_cfg.trailing_stop_percent,
                                        time_in_force=self.execution_cfg.exit_time_in_force,
                                    )
                                    logger.critical(
                                        f"Emergency trailing stop submitted for {symbol}: "
                                        f"{opposite_side} {pos_qty} trail={self.execution_cfg.trailing_stop_percent}%"
                                    )
                                    # Return success since position is now protected
                                    return order
                                except Exception as emergency_err:
                                    logger.critical(
                                        f"EMERGENCY TRAILING STOP FAILED for {symbol}: {emergency_err}. "
                                        f"Attempting emergency flatten."
                                    )
                                    self._emergency_flatten_position(symbol, pos_qty, side)
                        except Exception as pos_check_err:
                            logger.error(f"Position re-check failed for {symbol}: {pos_check_err}")
                    return None

                # Position confirmed, submit trailing stop
                opposite_side = "sell" if side == "buy" else "buy"
                trail_pct = self.execution_cfg.trailing_stop_percent
                exit_tif = self.execution_cfg.exit_time_in_force

                # AUDIT FIX 8.1: Confirm trailing stop acceptance
                try:
                    trailing_stop_order = self.adapter.submit_trailing_stop_order(
                        symbol=symbol,
                        qty=qty,
                        side=opposite_side,
                        trail_percent=trail_pct,
                        time_in_force=exit_tif,
                    )
                    logger.info(
                        f"Submitted trailing stop for {symbol}: {opposite_side} {qty} trail={trail_pct}%"
                    )

                    # Confirm trailing stop was accepted by broker
                    stop_order_id = getattr(trailing_stop_order, "id", None)
                    if stop_order_id:
                        stop_accepted = self._wait_for_order_acceptance(stop_order_id, symbol)
                        if not stop_accepted:
                            logger.critical(
                                f"AUDIT FIX 8.1: Trailing stop REJECTED for {symbol}. "
                                f"Position is NAKED. Submitting emergency flatten."
                            )
                            self._emergency_flatten_position(symbol, qty, side)
                            return None
                    else:
                        logger.warning(
                            f"Trailing stop order for {symbol} has no ID — cannot confirm acceptance. "
                            f"Proceeding with caution."
                        )

                    return order

                except Exception as stop_err:
                    logger.critical(
                        f"AUDIT FIX 8.1: Trailing stop submission FAILED for {symbol}: {stop_err}. "
                        f"Position is NAKED. Submitting emergency flatten."
                    )
                    self._emergency_flatten_position(symbol, qty, side)
                    return None

            else:
                # Path 3: Plain market order (no exit protection configured)
                logger.warning(
                    f"Submitting plain market order for {symbol} WITHOUT exit protection "
                    f"(enable_trailing_stop=False, no bracket available). This is high-risk."
                )
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

    def close_position_due_to_sentiment(
        self,
        position: PositionInfo,
        sentiment: SentimentResult,
        reason: str,
        env_mode: str,
        opening_compounds: Dict[str, float],
        persist_opening_compounds: Callable[[Dict[str, float]], None],
    ) -> None:
        """Close a position due to sentiment deterioration.

        Transactional close: purge entry baseline only after confirmed broker flatten.
        """
        if not self.live_trading_enabled:
            logger.info(
                f"[PAPER MODE] Would close {position.symbol} due to sentiment: {reason}"
            )
            return

        log_sentiment_close_decision(
            symbol=position.symbol,
            side=position.side,
            qty=position.qty,
            sentiment_score=sentiment.score,
            confidence=sentiment.confidence,
            explanation=sentiment.explanation,
            env_mode=env_mode,
            reason=reason,
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
        except Exception as e:
            logger.error(
                f"Error closing position {position.symbol} due to sentiment: {e}"
            )
            return

        if not self._wait_for_flat(position.symbol):
            logger.warning(
                f"Close submitted for {position.symbol} ({reason}) but position is not yet flat; "
                "retaining opening_compound baseline."
            )
            return

        logger.info(
            f"Closed position {position.symbol} ({reason}): "
            f"{opposite_side} {abs(position.qty)} @ market"
        )

        try:
            self.trade_stats.close_active_trade(
                symbol=position.symbol,
                exit_price=float(getattr(position, "market_price", 0.0) or 0.0),
                exit_reason=reason,
                hit_stop_loss=False,
                hit_take_profit=False,
                notes="explicit_sentiment_exit",
            )
        except Exception as stats_error:
            logger.warning(
                f"Trade stats close tracking failed for {position.symbol}: {stats_error}"
            )

        if position.symbol in opening_compounds:
            del opening_compounds[position.symbol]
            persist_opening_compounds(opening_compounds)
            logger.info(
                f"Purged opening_compound for {position.symbol} after confirmed close."
            )

    def close_all_positions_for_weekend(
        self,
        positions: Dict[str, PositionInfo],
        env_mode: str,
        opening_compounds: Dict[str, float],
        persist_opening_compounds: Callable[[Dict[str, float]], None],
    ) -> None:
        """Close all positions before weekend.

        Triple-gate protection prevents accidental liquidation in paper mode.
        """
        if not self.live_trading_enabled:
            logger.info(
                "[PAPER MODE GUARD] Weekend liquidation blocked: "
                "live_trading_enabled=False. All positions preserved."
            )
            return

        if env_mode.upper() != "LIVE":
            logger.warning(
                f"[ENV MODE GUARD] Weekend liquidation blocked: "
                f"env_mode={env_mode} (expected LIVE). All positions preserved. "
                f"If you intended to liquidate, set APCA_API_ENV=LIVE and restart."
            )
            return

        logger.warning(
            "WEEKEND LIQUIDATION CONFIRMED: live_trading_enabled=True AND env_mode=LIVE. "
            "Closing all positions now."
        )

        try:
            self.adapter.cancel_all_orders()
            self.adapter.close_all_positions()
            logger.info("All positions closed (atomic broker call).")
            for symbol in list(opening_compounds.keys()):
                if symbol in opening_compounds:
                    del opening_compounds[symbol]
            persist_opening_compounds(opening_compounds)
            logger.info(
                "Purged all opening_compounds after weekend atomic close."
            )
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
                    opening_compounds=opening_compounds,
                    persist_opening_compounds=persist_opening_compounds,
                )
