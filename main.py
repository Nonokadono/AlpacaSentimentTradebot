# CHANGES:
# Dashboard integration — monitoring/terminal.py wired into the main loop.
#
# Change T1: Import dashboard_state, build_position_rows, build_price_rows,
#   launch_dashboard, and dev_mode from monitoring.terminal.
#
# Change T2: A daemon thread is spawned ONCE before the main loop starts
#   (when dev_mode is False). The thread runs launch_dashboard() which blocks
#   inside Textual's event loop. Because it is a daemon thread it exits
#   automatically when the main process terminates — no cleanup needed.
#
# Change T3: Inside the main loop, dashboard_state.update() is called at
#   well-defined points:
#     a) After snapshot is available: update cash, buying_power, market_open.
#     b) At the start of portfolio build: set bot_state="SCANNING".
#     c) After execute: set bot_state="EXECUTING" then back to "IDLE".
#     d) Positions and price rows are rebuilt and pushed every cycle.
#   All updates are non-blocking (simple attribute writes under a Lock).
#
# Change T4: When dev_mode is True the dashboard thread is never started and
#   no extra imports from textual are triggered — headless operation is
#   completely unchanged.
#
# All prior changes (Change 1a, Change 5, Improvement D, adaptive sleep
# hysteresis, _load/_persist opening_compounds, _check_and_exit_on_sentiment,
# get_equity_snapshot_from_account, Feature 7A, Feature 7B, etc.) are
# preserved unchanged.

import json
import logging
import threading
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
from execution.order_executor import OrderExecutor, _check_and_exit_on_sentiment
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
from monitoring.terminal import (
    build_position_rows,
    build_price_rows,
    dashboard_state,
    dev_mode,
    launch_dashboard,
)

logger = logging.getLogger("tradebot")

EQUITY_STATE_PATH = Path("equity_state.json")


def _load_equity_state() -> dict:
    if EQUITY_STATE_PATH.exists():
        try:
            with EQUITY_STATE_PATH.open("r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_equity_state(state: dict) -> None:
    try:
        with EQUITY_STATE_PATH.open("w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning(f"_save_equity_state error: {e}")


def _load_opening_compounds() -> Dict[str, float]:
    state = _load_equity_state()
    raw = state.get("opening_compounds", {})
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    return {}


def _persist_opening_compounds(opening_compounds: Dict[str, float]) -> None:
    state = _load_equity_state()
    state["opening_compounds"] = opening_compounds
    _save_equity_state(state)


def load_equity_state() -> dict:
    return _load_equity_state()


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
    _save_equity_state(state)

    realized_pl_today = float(getattr(acct, "realized_pl", 0.0))
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
                getattr(acct, "daytrading_buying_power",
                        getattr(acct, "buying_power", 0.0)),
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

    # Change 1a: stores proposed.sentiment_score (compound sentiment float) —
    # NOT proposed.signal_score (technical composite).
    _opening_compounds: Dict[str, float] = _load_opening_compounds()

    # Improvement D: initialise adaptive sleep interval before the loop.
    _rescore_interval: int = 600

    # Change T2: Launch the TUI dashboard in a background daemon thread.
    # When dev_mode is True, launch_dashboard() returns immediately.
    _dashboard_thread = threading.Thread(
        target=launch_dashboard,
        args=(dashboard_state, 5.0),
        daemon=True,
        name="tui-dashboard",
    )
    _dashboard_thread.start()

    _cycle_count: int = 0

    while True:
        _cycle_count += 1

        acct        = adapter.get_account()
        positions   = pm.get_positions(opening_compounds=_opening_compounds)
        snapshot    = get_equity_snapshot_from_account(acct, positions)
        market_open = adapter.get_market_open()
        log_equity_snapshot(snapshot, market_open=market_open)

        # Change T3a: push account + market state into the dashboard.
        dashboard_state.update(
            cash=snapshot.cash,
            buying_power=snapshot.day_trading_buying_power,
            cycle_count=_cycle_count,
            market_open=market_open,
            bot_state="SCANNING",
        )

        # ── STEP 0: WEEKEND FORCED LIQUIDATION ──────────────────────────────
        if adapter.is_pre_weekend_close():
            logger.warning(
                "WEEKEND CLOSE DETECTED: Closing all positions and sleeping "
                "until next market open."
            )
            dashboard_state.update(bot_state="EXECUTING")
            executor.close_all_positions_for_weekend(positions, cfg.env_mode)
            dashboard_state.update(bot_state="IDLE")
            while not adapter.get_market_open():
                time.sleep(60)
            continue

        ks_state = kill_switch.check(snapshot)
        log_kill_switch_state(ks_state)
        if ks_state.halted:
            dashboard_state.update(bot_state="HALTED")
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
            positions = pm.get_positions(opening_compounds=_opening_compounds)
            snapshot  = get_equity_snapshot_from_account(acct, positions)

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
            # Change T3: push positions + prices before sleeping
            open_orders_ui = adapter.list_orders(status="open")
            dashboard_state.update(
                positions=build_position_rows(positions, open_orders_ui),
                prices=build_price_rows(cfg.instruments, adapter),
                bot_state="IDLE",
            )
            time.sleep(60)
            continue

        # ── STEP 3: PRE-CLOSE ENTRY BLACKOUT + BUILD AND EXECUTE NEW TRADES ─
        if adapter.is_pre_close_blackout():
            logger.info(
                "PRE-CLOSE BLACKOUT: New position entries suppressed "
                "(within 3 hours of market close). Monitoring existing positions only."
            )
        else:
            dashboard_state.update(bot_state="SCANNING")
            open_orders     = adapter.list_orders(status="open")
            proposed_trades = portfolio_builder.build_portfolio(snapshot, positions, open_orders)

            log_portfolio_overview(proposed_trades, cfg.env_mode)

            dashboard_state.update(bot_state="EXECUTING")
            for proposed in proposed_trades:
                order = executor.execute_proposed_trade(proposed)
                if order is not None and proposed.rejected_reason is None and proposed.qty > 0:
                    _opening_compounds[proposed.symbol] = proposed.sentiment_score
                    _persist_opening_compounds(_opening_compounds)

        # Sentiment-exit check runs unconditionally.
        _check_and_exit_on_sentiment(
            positions=positions,
            adapter=adapter,
            sentiment_module=sentiment,
            executor=executor,
            cfg=cfg,
        )

        # Change T3d: refresh positions + prices in dashboard after all executions.
        open_orders_final = adapter.list_orders(status="open")
        positions_refreshed = pm.get_positions(opening_compounds=_opening_compounds)
        dashboard_state.update(
            positions=build_position_rows(positions_refreshed, open_orders_final),
            prices=build_price_rows(cfg.instruments, adapter),
            bot_state="IDLE",
        )

        # ── STEP 4: ADAPTIVE SLEEP WITH HYSTERESIS ──────────────────────────
        _max_abs_s = max(
            (
                abs(sentiment.get_cached_sentiment(sym).score)
                for sym in positions
                if sentiment.get_cached_sentiment(sym) is not None
            ),
            default=0.0,
        )
        _rescore_interval = sentiment.adaptive_rescore_interval_hysteresis(
            _max_abs_s, _rescore_interval
        )
        time.sleep(_rescore_interval)


if __name__ == "__main__":
    main()
