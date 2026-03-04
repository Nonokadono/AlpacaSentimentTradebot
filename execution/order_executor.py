# CHANGES:
# FIX 3 — Added best-effort cancel attempt in execute_proposed_trade() when _wait_for_position() returns False.
#         Wrap in try/except so the cancel attempt never raises; log both success and failure cases.
# FIX 4D (FIX 9) — Deleted dead code line `sent_cfg = SentimentConfig(cfg.sentiment) if False else cfg.sentiment`.
# SAFETY-GATE — Enhanced close_all_positions_for_weekend() with triple-gate protection.
# PURGE-FIX — close_position_due_to_sentiment() now purges opening_compounds[symbol] immediately
#             inline with the close order, then persists the registry to disk before returning.
# WAIT-TIMEOUT-FIX — Added threading.Timer hard kill for _wait_for_position() to prevent hung
#                    Alpaca poll from blocking the bot indefinitely. Timer spawns a daemon thread
#                    that logs a CRITICAL error and raises SystemExit after timeout+5s grace period.
#                    Main loop catches SystemExit and logs the forced termination before exiting.

import logging
import time
import threading
from typing import Callable, Dict, Optional

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
    opening_compounds: Dict[str, tuple],
    persist_opening_compounds: Callable[[Dict[str, tuple]], None],
) -> None:
    """For every open position, force-rescore sentiment and compare the current
    compound score against the score recorded at entry time.

    Three exit tiers from SentimentConfig:
      1. Hard exit (chaos): raw_discrete == -2 → close unconditionally, no delta check.
      2. Strong exit: delta > effective_strong_threshold AND
                      confidence > strong_exit_confidence_min
         Fires on large sentiment deterioration even at moderate confidence.
      3. Soft exit: delta > effective_soft_threshold AND
                    confidence > exit_confidence_min
         Catches partial deterioration (e.g. +0.7 → 0.0, delta=0.70).

    delta is defined side-aware:
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

    CONFIDENCE-STORE: opening_compounds now stores {symbol: (compound, confidence)}
    tuples. The confidence value is reserved for potential future threshold scaling.
    Backward compatibility: legacy scalar entries are treated as (compound, 0.0).

    PURGE-FIX: opening_compounds and persist_opening_compounds are required
    parameters so close_position_due_to_sentiment() can purge the entry baseline
    immediately inline with the close order and flush it to disk before returning.
    """
    # FIX 4D (FIX 9): Deleted the dead code line; only this line remains.
    sent_cfg = cfg.sentiment
    env_mode = str(cfg.env_mode)

    for symbol, pos in positions.items():
        news_items = adapter.get_news(symbol, limit=10)
        current_sentiment: SentimentResult = sentiment_module.force_rescore(symbol, news_items)

        # CONFIDENCE-STORE: unpack tuple; fallback to (0.0, 0.0) if legacy scalar
        opening_data = opening_compounds.get(symbol, (0.0, 0.0))
        if isinstance(opening_data, tuple):
            opening_compound, opening_confidence = opening_data
        else:
            opening_compound = float(opening_data)
            opening_confidence = 0.0

        current_score = float(current_sentiment.score)

        # Side-aware delta so shorts exit on improving sentiment.
        if pos.side == "long":
            delta = float(opening_compound - current_score)
        else:  # short
            delta = float(current_score - opening_compound)

        # --- PnL-Coupled Sentiment Exit Threshold Scaling ---
        if pos.avg_entry_price > 0.0:
            raw_pnl_pct = (pos.market_price - pos.avg_entry_price) / pos.avg_entry_price
            # Short positions gain when price falls, so flip the sign.
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
      4. Submit a bracket order (entry + OCO TP/stop) when both
         enable_take_profit and enable_trailing_stop are True.
      5. Fall back to entry + standalone trailing-stop when only
         enable_trailing_stop is True.  The trailing stop is ONLY submitted
         after _wait_for_position() confirms the fill.
      6. Fall back to plain market order when neither exit type is configured.

    Sentiment exit path (close_position_due_to_sentiment):
      - Cancel all open orders for the symbol
      - Submit a market order to flatten
      - PURGE-FIX: delete opening_compounds[symbol] and persist registry inline.

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

        WAIT-TIMEOUT-FIX: Spawns a daemon threading.Timer that fires after
        timeout + 5s grace period. If the timer expires, it logs a CRITICAL
        error and raises SystemExit to force-terminate the bot. This prevents
        an indefinite hang if Alpaca's REST endpoint stops responding.

        The timer is cancelled immediately upon successful position confirmation
        so normal operation is unaffected.
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
            # Timer fired; re-raise to propagate to main loop
            raise
        except Exception as e:
            logger.error(f"Exception in _wait_for_position({symbol}): {e}")
            kill_timer.cancel()
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
        opening_compounds: Dict[str, tuple],
        persist_opening_compounds: Callable[[Dict[str, tuple]], None],
    ) -> None:
        """Close a position due to sentiment deterioration.

        PURGE-FIX: Immediately after closing the position (or after the paper-mode
        log), purge opening_compounds[symbol] and persist the updated registry to
        disk. This eliminates the race window between sentiment close and the next
        GAP-2 purge, and ensures crash-recovery never resurrects a stale baseline.
        """
        if not self.live_trading_enabled:
            logger.info(
                f"[PAPER MODE] Would close {position.symbol} due to sentiment: {reason}"
            )
            # PURGE-FIX: Even in paper mode, purge the baseline so the in-memory
            # registry stays consistent with the (hypothetical) flat position.
            if position.symbol in opening_compounds:
                del opening_compounds[position.symbol]
                persist_opening_compounds(opening_compounds)
                logger.info(
                    f"[PAPER MODE] Purged opening_compound for {position.symbol} "
                    f"(hypothetical close)."
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

        # PURGE-FIX: Delete the entry baseline immediately inline with the close
        # order and persist the updated registry to disk before returning.
        if position.symbol in opening_compounds:
            del opening_compounds[position.symbol]
            persist_opening_compounds(opening_compounds)
            logger.info(
                f"Purged opening_compound for {position.symbol} after sentiment close."
            )

    def close_all_positions_for_weekend(
        self,
        positions: Dict[str, PositionInfo],
        env_mode: str,
        opening_compounds: Dict[str, tuple],
        persist_opening_compounds: Callable[[Dict[str, tuple]], None],
    ) -> None:
        """
        Force-close all positions before weekend.

        SAFETY-GATE: Triple-gate protection prevents accidental liquidation:
          1. live_trading_enabled must be True (config-level gate).
          2. env_mode must be "LIVE" (environment-level gate).
          3. Confirmation log emitted at WARNING level before atomic close.

        PURGE-FIX: opening_compounds and persist_opening_compounds are required
        parameters so the fallback path can purge baselines inline with each close.
        """
        # SAFETY-GATE 1: live_trading_enabled check (config-level)
        if not self.live_trading_enabled:
            logger.info(
                "[PAPER MODE GUARD] Weekend liquidation blocked: "
                "live_trading_enabled=False. All positions preserved."
            )
            return

        # SAFETY-GATE 2: env_mode check (environment-level)
        if env_mode.upper() != "LIVE":
            logger.warning(
                f"[ENV MODE GUARD] Weekend liquidation blocked: "
                f"env_mode={env_mode} (expected LIVE). All positions preserved. "
                f"If you intended to liquidate, set APCA_API_ENV=LIVE and restart."
            )
            return

        # Both gates passed — emit confirmation log before proceeding
        logger.warning(
            "WEEKEND LIQUIDATION CONFIRMED: live_trading_enabled=True AND env_mode=LIVE. "
            "Closing all positions now."
        )

        try:
            self.adapter.cancel_all_orders()
            self.adapter.close_all_positions()
            logger.info("All positions closed (atomic broker call).")
            # PURGE-FIX: Atomic close succeeded — purge all baselines at once.
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
