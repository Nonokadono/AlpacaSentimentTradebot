# CHANGES:
# - Added _load_opening_compounds() / _persist_opening_compounds() helpers that
#   read/write the opening_compounds sub-key of equity_state.json so entry-time
#   sentiment baselines survive bot restarts.
# - Added load_equity_state() / save_equity_state() for equity watermark
#   persistence.
# - get_equity_snapshot_from_account() uses the persisted state for
#   start_of_day_equity and high_watermark_equity.
# - _check_and_exit_on_sentiment() implements directional delta exit logic
#   (Change 1b): only adverse sentiment shifts fire exits.
# - main() loop integrates: kill switch → sentiment exit → portfolio build →
#   adaptive sleep with hysteresis (Change 5 / Improvement D).

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from adapters.alpaca_adapter import AlpacaAdapter
from config.config import load_config
from core.portfolio_builder import PortfolioBuilder
from core.risk_engine import RiskEngine, EquitySnapshot, PositionInfo
from core.sentiment import SentimentModule, SentimentResult
from core.signals import SignalEngine
from execution.order_executor import OrderExecutor
from execution.position_manager import PositionManager
from monitoring.kill_switch import KillSwitch
from monitoring.monitor import (
    log_equity_snapshot,
    log_environment_switch,
    log_kill_switch_state,
    log_portfolio_overview,
    log_sentiment_position_check,
    setup_logging,
)

logger = logging.getLogger("tradebot")

_EQUITY_STATE_PATH = Path("equity_state.json")


# ── Equity state persistence ───────────────────────────────────────────────────

def load_equity_state() -> dict:
    if _EQUITY_STATE_PATH.exists():
        try:
            with _EQUITY_STATE_PATH.open("r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"load_equity_state: could not read state file: {e}")
    return {}


def save_equity_state(state: dict) -> None:
    try:
        with _EQUITY_STATE_PATH.open("w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"save_equity_state: could not write state file: {e}")


# ── Opening compound persistence ───────────────────────────────────────────────

def _load_opening_compounds() -> Dict[str, float]:
    """Load the opening_compounds registry from equity_state.json."""
    state = load_equity_state()
    raw = state.get("opening_compounds", {})
    result: Dict[str, float] = {}
    for sym, val in raw.items():
        try:
            result[str(sym)] = float(val)
        except (TypeError, ValueError):
            pass
    return result


def _persist_opening_compounds(opening_compounds: Dict[str, float]) -> None:
    """
    Write the opening_compounds sub-key to equity_state.json.
    Does NOT overwrite unrelated keys (start_of_day_equity, etc.).
    """
    state = load_equity_state()
    state["opening_compounds"] = {sym: float(v) for sym, v in opening_compounds.items()}
    save_equity_state(state)


# ── Equity snapshot ────────────────────────────────────────────────────────────

def get_equity_snapshot_from_account(
    acct,
    positions: Dict[str, PositionInfo],
) -> EquitySnapshot:
    equity = float(acct.equity)
    cash = float(acct.cash)
    portfolio_value = float(acct.portfolio_value)

    state = load_equity_state()
    today_str = date.today().isoformat()
    last_day = state.get("last_trading_day")
    start_of_day_equity = float(state.get("start_of_day_equity", equity))
    high_watermark_equity = float(state.get("high_watermark_equity", equity))

    if last_day != today_str:
        start_of_day_equity = equity
        high_watermark_equity = equity
        state["last_trading_day"] = today_str

    if equity > high_watermark_equity:
        high_watermark_equity = equity

    if start_of_day_equity > 0:
        daily_loss_pct = (equity - start_of_day_equity) / start_of_day_equity
    else:
        daily_loss_pct = 0.0

    if high_watermark_equity > 0:
        drawdown_pct = (equity - high_watermark_equity) / high_watermark_equity
    else:
        drawdown_pct = 0.0

    state["start_of_day_equity"] = start_of_day_equity
    state["high_watermark_equity"] = high_watermark_equity
    # NOTE: do not write opening_compounds here — that key is managed exclusively
    # by main to avoid accidental overwrites during snapshot refreshes.
    save_equity_state(state)

    realized_pl_today = float(getattr(acct, "day_trade_pl", 0.0))
    unrealized_pl = float(getattr(acct, "unrealized_pl", 0.0))
    gross_exposure = sum(abs(p.notional) for p in positions.values())

    return EquitySnapshot(
        equity=equity,
        cash=cash,
        portfolio_value=portfolio_value,
        day_trading_buying_power=float(
            getattr(
                acct,
                "day_trading_buying_power",
                getattr(acct, "daytrading_buying_power", getattr(acct, "buying_power", 0.0)),
            )
        ),
        start_of_day_equity=start_of_day_equity,
        high_watermark_equity=high_watermark_equity,
        realized_pl_today=realized_pl_today,
        unrealized_pl=unrealized_pl,
        gross_exposure=gross_exposure,
        daily_loss_pct=daily_loss_pct,
        drawdown_pct=drawdown_pct,
    )


# ── Sentiment-exit check ───────────────────────────────────────────────────────

def _check_and_exit_on_sentiment(
    positions: Dict[str, PositionInfo],
    adapter: AlpacaAdapter,
    sentiment_module: SentimentModule,
    executor: OrderExecutor,
    cfg,
) -> None:
    """
    Iterate over ALL open positions.  For each one:
      1. Fetch fresh news; force_rescore() bypasses TTL cache and chaos cooldown.
      2. Hard exit unconditionally if raw_discrete == -2 (chaos event).
      3. Strong exit if delta > strong_exit_delta_threshold AND
         confidence > strong_exit_confidence_min.
      4. Soft exit if delta > soft_exit_delta_threshold AND
         confidence > exit_confidence_min.
      5. Emit a formatted per-instrument block via log_sentiment_position_check().

    opening_compound is sourced directly from PositionInfo.opening_compound,
    which main patches in from _opening_compounds after each entry fill.
    This survives TTL expiry and bot restarts because opening_compounds is
    persisted to equity_state.json.

    Change 1b: delta is now directional so only adverse sentiment shifts fire:
      LONG  position: delta = entry_sentiment - current_sentiment.score
                      A positive delta means sentiment has dropped since entry.
      SHORT position: delta = current_sentiment.score - entry_sentiment
                      A positive delta means sentiment has risen against the short.
    Absolute threshold comparisons are applied to this signed directional value.
    """
    soft_threshold = cfg.sentiment.soft_exit_delta_threshold
    strong_threshold = cfg.sentiment.strong_exit_delta_threshold
    confidence_min = cfg.sentiment.exit_confidence_min
    strong_confidence_min = cfg.sentiment.strong_exit_confidence_min

    for symbol, position in list(positions.items()):
        try:
            entry_sentiment = float(position.opening_compound)

            # Always fetch fresh news and force a real AI rescore.
            news_items = adapter.get_news(symbol, limit=10)
            current_sentiment = sentiment_module.force_rescore(symbol, news_items)
            current_confidence = float(current_sentiment.confidence)
            raw_discrete = int(current_sentiment.raw_discrete)

            # Change 1b: directional delta — only adverse shifts produce a
            # positive delta that can breach the exit thresholds.
            if position.side == "long":
                delta = entry_sentiment - current_sentiment.score
            else:
                delta = current_sentiment.score - entry_sentiment

            # ── Exit decision ─────────────────────────────────────────────────
            closing = False
            close_reason = ""

            # Tier 0: Hard exit — chaos / utterly unstable (raw_discrete == -2)
            if raw_discrete == -2:
                closing = True
                close_reason = (
                    f"raw_discrete=-2 chaos — utterly unstable; "
                    f"chaos cooldown timer applied."
                )

            # Tier 1: Strong exit — large delta with moderate confidence
            elif delta > strong_threshold and current_confidence > strong_confidence_min:
                closing = True
                close_reason = (
                    f"STRONG sentiment exit: "
                    f"Δ={delta:+.3f} > {strong_threshold} "
                    f"(entry={entry_sentiment:.3f} → current={current_sentiment.score:.3f}) "
                    f"against {position.side} position; "
                    f"confidence={current_confidence:.2f} > {strong_confidence_min}"
                )

            # Tier 2: Soft exit — moderate delta with higher confidence bar
            elif delta > soft_threshold and current_confidence > confidence_min:
                closing = True
                close_reason = (
                    f"SOFT sentiment exit: "
                    f"Δ={delta:+.3f} > {soft_threshold} "
                    f"(entry={entry_sentiment:.3f} → current={current_sentiment.score:.3f}) "
                    f"against {position.side} position; "
                    f"confidence={current_confidence:.2f} > {confidence_min}"
                )

            # Use the active threshold for the log display.
            display_threshold = strong_threshold if not closing or raw_discrete == -2 else (
                strong_threshold if delta > strong_threshold else soft_threshold
            )

            log_sentiment_position_check(
                position=position,
                entry_compound=entry_sentiment,
                current_sentiment=current_sentiment,
                delta=delta,
                delta_threshold=display_threshold,
                confidence_min=confidence_min,
                closing=closing,
                close_reason=close_reason,
                env_mode=cfg.env_mode,
            )

            if closing:
                executor.close_position_due_to_sentiment(
                    position=position,
                    sentiment=current_sentiment,
                    reason=close_reason,
                )

        except Exception as exc:
            logger.error(
                f"SentimentExit {symbol}: unexpected error during exit check: {exc}"
            )


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    cfg = load_config()
    setup_logging(cfg.env_mode)
    log_environment_switch(cfg.env_mode, user="manual_start")

    adapter           = AlpacaAdapter(cfg.env_mode)
    sentiment         = SentimentModule()
    signal_engine     = SignalEngine(adapter, sentiment, cfg.technical)
    risk_engine       = RiskEngine(cfg.risk_limits, cfg.sentiment, cfg.instruments)
    pm                = PositionManager(adapter)
    executor          = OrderExecutor(adapter, cfg.env_mode, cfg.live_trading_enabled, cfg.execution)
    kill_switch       = KillSwitch(cfg.risk_limits)
    portfolio_builder = PortfolioBuilder(cfg, adapter, sentiment, signal_engine, risk_engine)

    # Registry: symbol -> sentiment_score recorded at entry time.
    # Change 1a: stores proposed.sentiment_score (compound sentiment float) —
    # NOT proposed.signal_score (technical composite). The exit delta check
    # subtracts this from current sentiment.score; mixing a technical score here
    # produced a semantically invalid cross-space comparison.
    # Persisted to equity_state.json so it survives bot restarts.
    _opening_compounds: Dict[str, float] = _load_opening_compounds()

    # Improvement D: initialise adaptive sleep interval before the loop.
    # Starts at 600s (the neutral-band default); hysteresis prevents oscillation.
    _rescore_interval: int = 600

    while True:
        acct        = adapter.get_account()
        positions   = pm.get_positions(opening_compounds=_opening_compounds)
        snapshot    = get_equity_snapshot_from_account(acct, positions)
        market_open = adapter.get_market_open()
        log_equity_snapshot(snapshot, market_open=market_open)

        ks_state = kill_switch.check(snapshot)
        log_kill_switch_state(ks_state)
        if ks_state.halted:
            time.sleep(60)
            continue

        # ── STEP 1: SENTIMENT CHECK ON ALL OPEN POSITIONS ───────────────────
        if positions:
            _check_and_exit_on_sentiment(
                positions=positions,
                adapter=adapter,
                sentiment_module=sentiment,
                executor=executor,
                cfg=cfg,
            )
            # Refresh positions after potential closes.
            positions = pm.get_positions(opening_compounds=_opening_compounds)
            snapshot  = get_equity_snapshot_from_account(acct, positions)

            # Purge _opening_compounds for symbols no longer open, then persist.
            for sym in list(_opening_compounds.keys()):
                if sym not in positions:
                    del _opening_compounds[sym]
            _persist_opening_compounds(_opening_compounds)

        # ── STEP 2: EXPOSURE / POSITION-COUNT GUARD ─────────────────────────
        exposure_cap_notional = snapshot.equity * cfg.risk_limits.gross_exposure_cap_pct
        if (
            snapshot.gross_exposure >= exposure_cap_notional
            or len(positions) >= cfg.risk_limits.max_open_positions
        ):
            time.sleep(60)
            continue

        # ── *** ADD THIS HERE *** ────────────────────────────────────────────
        # Uncomment to gate new entries on market hours:
        # if not market_open:
        #     logger.info("Market closed — skipping portfolio build and new entries.")
        #     time.sleep(60)
        #     continue

        # ── STEP 3: BUILD AND EXECUTE NEW TRADES ────────────────────────────
        open_orders     = adapter.list_orders(status="open")
        proposed_trades = portfolio_builder.build_portfolio(snapshot, positions, open_orders)

        log_portfolio_overview(proposed_trades, cfg.env_mode)

        for proposed in proposed_trades:
            order = executor.execute_proposed_trade(proposed)
            if order is not None and proposed.rejected_reason is None and proposed.qty > 0:
                # Change 1a: record entry-time SENTIMENT score (proposed.sentiment_score)
                # as the opening compound baseline — NOT proposed.signal_score.
                _opening_compounds[proposed.symbol] = proposed.sentiment_score
                # Persist immediately so a crash/restart doesn't lose the entry record.
                _persist_opening_compounds(_opening_compounds)

        # ── STEP 4: ADAPTIVE SLEEP WITH HYSTERESIS ──────────────────────────
        # Change 5 / Improvement D: sleep duration is driven by the highest
        # |sentiment.score| across all currently open positions. High-conviction
        # positions rescore every 120s; neutral portfolios wait up to 900s,
        # reducing API cost. Hysteresis prevents rapid interval oscillation when
        # max_abs_s hovers near a boundary threshold.
        _max_abs_s = max(
            (
                abs(sentiment.get_cached_sentiment(sym).score)
                for sym in positions
                if sentiment.get_cached_sentiment(sym) is not None
            ),
            default=0.0,
        )
        # Improvement D: use hysteresis guard instead of bare adaptive call.
        _rescore_interval = sentiment.adaptive_rescore_interval_hysteresis(
            _max_abs_s, _rescore_interval
        )
        time.sleep(_rescore_interval)


if __name__ == "__main__":
    main()
