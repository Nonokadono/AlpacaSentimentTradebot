# CHANGES:
# ─────────────────────────────────────────────────────────────────────────────
# WRONG-2 FIX — Side-aware delta in _check_and_exit_on_sentiment()
#
# Root cause:
#   The original formula `delta = float(opening_compound - current_score)` was
#   applied unconditionally to both sides.  For SHORT positions this was
#   inverted: improving sentiment (score rising) produced a NEGATIVE delta that
#   never reached the exit threshold, while worsening sentiment (score falling,
#   meaning the position is winning) produced a FALSE POSITIVE exit signal.
#
# Fix:
#   Compute delta in a side-aware branch immediately after deriving
#   opening_compound and current_score:
#
#     if pos.side == "long":
#         delta = float(opening_compound - current_score)   # +ve = worsened
#     else:  # short
#         delta = float(current_score - opening_compound)   # +ve = improved = exit
#
#   All three exit-tier comparisons (hard/strong/soft) and the
#   log_sentiment_position_check() call consume the corrected delta unchanged.
#   The hard_exit on raw_discrete == -2 remains unconditional for both sides.
#   The _opening_compounds registry is unaffected (stores proposed.sentiment_score
#   regardless of side — correct for both long and short entry compound storage).
#
# Four-case verification (soft_exit_delta_threshold = 0.60):
#   Case 1  LONG  opening=+0.70, current=+0.10 → delta = +0.70-0.10 = +0.60 → borderline soft_exit
#   Case 2  LONG  opening=+0.70, current=+0.70 → delta =  0.00               → no exit ✓
#   Case 3  SHORT opening=-0.30, current=+0.40 → delta = +0.40-(-0.30) = +0.70 → soft_exit ✓
#   Case 4  SHORT opening=-0.30, current=-0.50 → delta = -0.50-(-0.30) = -0.20 → no exit ✓
#
# No variables renamed.  No new dependencies introduced.
# The _check_and_exit_on_sentiment() docstring has been updated to reflect the
# corrected delta semantics.  All other code is unchanged.
# ─────────────────────────────────────────────────────────────────────────────

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
    sent_cfg = SentimentConfig(cfg.sentiment) if False else cfg.sentiment  # type: ignore[arg-type]
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
            logger.warning(f"_cancel_all_open_orders_for_symbol({symbol}) error: {e}")

    def _exit_side(self, position_side: str) -> str:
        """Return the closing side for a given position side."""
        return "sell" if position_side == "long" else "buy"

    def _wait_for_position(
        self,
        symbol: str,
        expected_qty: float,
        timeout_sec: int = 30,
        poll_interval_sec: float = 2.0,
    ) -> bool:
        """Poll adapter.list_positions() until the fill for *symbol* is confirmed.

        A position is "confirmed" when an entry for *symbol* exists in the
        broker's position list AND abs(qty) >= expected_qty * 0.95, allowing
        for a 5 % partial-fill tolerance before attaching the trailing stop.

        Args:
            symbol:            Ticker to watch.
            expected_qty:      Order quantity submitted to the entry market order.
            timeout_sec:       Hard deadline in seconds (default 30).
            poll_interval_sec: Sleep between successive list_positions() calls
                               (default 2.0 s).

        Returns:
            True  — position confirmed within timeout_sec.
            False — deadline expired without confirmation.  Emits a WARNING
                    that includes symbol, expected_qty, and elapsed time.

        Never raises — every exception inside the polling loop is caught and
        logged at WARNING level so the caller always gets a bool back.
        """
        start = time.monotonic()
        deadline = start + timeout_sec

        while time.monotonic() < deadline:
            try:
                positions = self.adapter.list_positions()
                for pos in positions:
                    pos_symbol = getattr(pos, "symbol", None)
                    pos_qty = getattr(pos, "qty", None)
                    if pos_symbol == symbol and pos_qty is not None:
                        if abs(float(pos_qty)) >= expected_qty * 0.95:
                            return True
            except Exception as e:
                logger.warning(
                    f"_wait_for_position poll error for {symbol}: {e}"
                )
            time.sleep(poll_interval_sec)

        elapsed = time.monotonic() - start
        logger.warning(
            f"_wait_for_position: position fill not confirmed for {symbol} "
            f"(expected_qty={expected_qty}, elapsed={elapsed:.1f}s)"
        )
        return False

    # ── Entry execution ───────────────────────────────────────────────────────

    def execute_proposed_trade(self, proposed: ProposedTrade) -> Optional[object]:
        """
        Execute a ProposedTrade.

        Returns:
            Alpaca order object — on a successful submission that main.py should
            record in _opening_compounds.  This covers:
              • Bracket orders (atomic OCO — no fill-confirmation needed).
              • Plain market orders with no exit leg.
              • Trailing-stop path where _wait_for_position() returned True.
            None — on any of the following conditions:
              • proposed.rejected_reason is not None or proposed.qty <= 0.
              • live_trading_enabled is False (paper mode guard).
              • APIError or unexpected exception during submission.
              • Trailing-stop path where _wait_for_position() returned False
                (fill not confirmed within timeout).  The entry order ID is
                logged before returning None so the operator retains an audit
                trail, but _opening_compounds is NOT updated for an unconfirmed
                position.
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
            # AUDIT 1 — BRACKET PATH ─────────────────────────────────────────
            # submit_bracket_order() sends a single atomic payload to Alpaca.
            # The broker creates the entry (market), TP (limit), and stop-loss
            # as a linked OCO group.  The TP/stop legs activate automatically
            # when the parent market order fills — the broker owns that state
            # machine.  No post-fill polling is needed here.
            if use_tp and use_ts:
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

            # AUDIT 2 — TRAILING-STOP-ONLY PATH ─────────────────────────────
            # Entry market order is submitted first.  The trailing stop is a
            # SEPARATE order and must only be submitted against a confirmed
            # position.  _wait_for_position() polls until the fill appears in
            # the broker's position list or the timeout expires.
            # FILL-CONFIRM FIX: on timeout, return None (not the entry order)
            # so main.py does NOT record _opening_compounds for an unconfirmed
            # position.  The entry order ID is logged for operator audit.
            elif use_ts:
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

                # Resolve timeout and interval from the new poll fields,
                # falling back to the deprecated legacy field so serialised
                # configs that only carry post_entry_fill_timeout_sec still work.
                poll_timeout = getattr(
                    self.execution_cfg,
                    "post_entry_fill_poll_timeout_sec",
                    getattr(self.execution_cfg, "post_entry_fill_timeout_sec", 30),
                )
                poll_interval = getattr(
                    self.execution_cfg, "post_entry_fill_poll_interval_sec", 2.0
                )

                # Actively poll for fill confirmation before attaching the stop.
                # Replaces the blind time.sleep() that previously allowed orphan
                # trailing stops to be submitted against unfilled/rejected orders.
                filled = self._wait_for_position(
                    symbol,
                    qty,
                    timeout_sec=poll_timeout,
                    poll_interval_sec=poll_interval,
                )
                if not filled:
                    # FILL-CONFIRM FIX: return None, not `order`.
                    # Returning the entry order object here caused main.py to
                    # record _opening_compounds[symbol] even when the position
                    # was unconfirmed, poisoning future sentiment-exit deltas.
                    # By returning None the `if order is not None` guard in
                    # main.py naturally suppresses _opening_compounds recording.
                    logger.warning(
                        f"Position fill not confirmed for {symbol} after "
                        f"{poll_timeout}s — returning None. "
                        f"Entry order {getattr(order, 'id', 'N/A')} was submitted "
                        f"but fill is unconfirmed. Trailing stop NOT attached. "
                        f"_opening_compounds[{symbol}] will NOT be recorded."
                    )
                    return None

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

            # AUDIT 3 — PLAIN MARKET ORDER PATH ─────────────────────────────
            # No exit leg is submitted here.  There is no secondary order that
            # could be orphaned, so no fill-confirmation requirement exists.
            else:
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

        AUDIT: _cancel_all_open_orders_for_symbol() is always called before
        the market close order (step 1 above).  No secondary order is submitted
        after the close order — the position is being flattened, not replaced —
        so no fill-confirmation is needed here.

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

        # AUDIT 4: cancel-then-close — orders cleared before market close order.
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

        AUDIT: cancel_all_orders() precedes close_all_positions() in the primary
        path (step 3a/3b).  The broker handles the atomic close; no fill-
        confirmation is needed.  The per-symbol fallback delegates to
        close_position_due_to_sentiment() which already cancels orders before
        its market order (see AUDIT 4 in that method's docstring).
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
            # AUDIT 5: cancel_all_orders() before close_all_positions() — confirmed.
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
