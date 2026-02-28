import logging
import json
from pathlib import Path
from datetime import datetime, date
import time
from typing import Dict

from config.config import load_config
from adapters.alpaca_adapter import AlpacaAdapter
from core.sentiment import SentimentModule
from core.signals import SignalEngine
from core.risk_engine import RiskEngine, EquitySnapshot, PositionInfo
from core.portfolio_builder import PortfolioBuilder
from execution.position_manager import PositionManager
from execution.order_executor import OrderExecutor
from monitoring.monitor import (
    setup_logging,
    log_equity_snapshot,
    log_environment_switch,
    log_kill_switch_state,
    log_sentiment_for_symbol,
    log_sentiment_position_check,
    log_portfolio_overview,
)
from monitoring.kill_switch import KillSwitch

logger = logging.getLogger("tradebot")

STATE_PATH = Path("data/equity_state.json")


def load_equity_state() -> Dict:
    if not STATE_PATH.exists():
        return {}
    try:
        with STATE_PATH.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_equity_state(state: Dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w") as f:
        json.dump(state, f)


def _load_opening_compounds() -> Dict[str, float]:
    """Load the opening-compound registry from the persisted equity state.
    Returns an empty dict if the state file is missing or the key is absent.
    This ensures PositionInfo.opening_compound is populated correctly even after
    a bot restart, so the sentiment-exit delta comparison is always meaningful.
    """
    state = load_equity_state()
    raw = state.get("opening_compounds", {})
    # Guard: only keep entries that are genuine floats.
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _persist_opening_compounds(opening_compounds: Dict[str, float]) -> None:
    """Write opening_compounds into equity_state.json without disturbing any
    other keys (start_of_day_equity, high_watermark_equity, last_trading_day)."""
    state = load_equity_state()
    state["opening_compounds"] = opening_compounds
    save_equity_state(state)


def get_equity_snapshot_from_account(acct, positions: Dict[str, PositionInfo]) -> EquitySnapshot:
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
        day_trading_buying_power=float(getattr(acct, "day_trading_buying_power",getattr(acct, "daytrading_buying_power",getattr(acct, "buying_power", 0.0)))),
        start_of_day_equity=start_of_day_equity,
        high_watermark_equity=high_watermark_equity,
        realized_pl_today=realized_pl_today,
        unrealized_pl=unrealized_pl,
        gross_exposure=gross_exposure,
        daily_loss_pct=daily_loss_pct,
        drawdown_pct=drawdown_pct,
    )


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
    """
    soft_threshold = cfg.sentiment.soft_exit_delta_threshold
    strong_threshold = cfg.sentiment.strong_exit_delta_threshold
    confidence_min = cfg.sentiment.exit_confidence_min
    strong_confidence_min = cfg.sentiment.strong_exit_confidence_min

    for symbol, position in list(positions.items()):
        try:
            entry_compound = float(position.opening_compound)

            # Always fetch fresh news and force a real AI rescore.
            news_items = adapter.get_news(symbol, limit=10)
            current_sentiment = sentiment_module.force_rescore(symbol, news_items)
            current_compound = float(current_sentiment.score)
            current_confidence = float(current_sentiment.confidence)
            raw_discrete = int(current_sentiment.raw_discrete)

            delta = abs(entry_compound - current_compound)

            # ── Exit decision ────────────────────────────────────────────────
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
                    f"(entry={entry_compound:.3f} → current={current_compound:.3f}) "
                    f"against {position.side} position; "
                    f"confidence={current_confidence:.2f} > {strong_confidence_min}"
                )

            # Tier 2: Soft exit — moderate delta with higher confidence bar
            elif delta > soft_threshold and current_confidence > confidence_min:
                closing = True
                close_reason = (
                    f"SOFT sentiment exit: "
                    f"Δ={delta:+.3f} > {soft_threshold} "
                    f"(entry={entry_compound:.3f} → current={current_compound:.3f}) "
                    f"against {position.side} position; "
                    f"confidence={current_confidence:.2f} > {confidence_min}"
                )

            # Use the active threshold for the log display (show whichever tier
            # was evaluated last — strong threshold takes display priority).
            display_threshold = strong_threshold if not closing or raw_discrete == -2 else (
                strong_threshold if delta > strong_threshold else soft_threshold
            )

            log_sentiment_position_check(
                position=position,
                entry_compound=entry_compound,
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
            logger.error(f"SentimentExit {symbol}: unexpected error during exit check: {exc}")


def main():
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

    # Registry: symbol -> signal_score recorded at entry time.
    # Persisted to equity_state.json so it survives bot restarts.
    _opening_compounds: Dict[str, float] = _load_opening_compounds()

    while True:
        acct      = adapter.get_account()
        positions = pm.get_positions(opening_compounds=_opening_compounds)
        snapshot  = get_equity_snapshot_from_account(acct, positions)
        market_open = adapter.get_market_open()
        log_equity_snapshot(snapshot, market_open=market_open)

        ks_state = kill_switch.check(snapshot)
        log_kill_switch_state(ks_state)
        if ks_state.halted:
            time.sleep(60)
            continue

        # ── STEP 1: SENTIMENT CHECK ON ALL OPEN POSITIONS ────────────────────
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

        # ── STEP 2: EXPOSURE / POSITION-COUNT GUARD ──────────────────────────
        exposure_cap_notional = snapshot.equity * cfg.risk_limits.gross_exposure_cap_pct
        if (
            snapshot.gross_exposure >= exposure_cap_notional
            or len(positions) >= cfg.risk_limits.max_open_positions
        ):
            time.sleep(60)
            continue

        # ── *** ADD THIS HERE *** ─────────────────────────────────────────────
        #if not market_open:
        #        logger.info("Market closed — skipping portfolio build and new entries.")
        #        time.sleep(60)
        #        continue

        # ── STEP 3: BUILD AND EXECUTE NEW TRADES ─────────────────────────────

        open_orders     = adapter.list_orders(status="open")
        proposed_trades = portfolio_builder.build_portfolio(snapshot, positions, open_orders)

        log_portfolio_overview(proposed_trades, cfg.env_mode)

        for proposed in proposed_trades:
            order = executor.execute_proposed_trade(proposed)
            if order is not None and proposed.rejected_reason is None and proposed.qty > 0:
                # Record entry-time signal_score as the opening compound baseline.
                # signal_score is the composite technical value that cleared the
                # long_threshold / short_threshold gate and caused this position
                # to be opened. The exit loop compares this against the current
                # sentiment.score each iteration to detect adverse drift.
                _opening_compounds[proposed.symbol] = proposed.signal_score
                # Persist immediately so a crash/restart doesn't lose the entry record.
                _persist_opening_compounds(_opening_compounds)

        time.sleep(600)


if __name__ == "__main__":
    main()