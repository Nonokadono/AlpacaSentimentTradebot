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
)
from monitoring.kill_switch import KillSwitch


_STATE_PATH = Path("data/equity_state.json")


def _load_equity_state() -> Dict[str, float]:
    if not _STATE_PATH.exists():
        return {}
    try:
        with _STATE_PATH.open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_equity_state(state: Dict[str, float]) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_PATH.open("w") as f:
        json.dump(state, f)


def get_equity_snapshot_from_account(acct, positions: Dict[str, PositionInfo]) -> EquitySnapshot:
    equity = float(acct.equity)
    cash = float(acct.cash)
    portfolio_value = float(acct.portfolio_value)

    state = _load_equity_state()
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

    realized_pl_today = float(getattr(acct, "daytrade_pl", 0.0))
    unrealized_pl = float(getattr(acct, "unrealized_pl", 0.0))
    gross_exposure = sum(abs(p.notional) for p in positions.values())

    return EquitySnapshot(
        equity=equity,
        cash=cash,
        portfolio_value=portfolio_value,
        day_trading_buying_power=float(acct.daytrading_buying_power),
        start_of_day_equity=start_of_day_equity,
        high_watermark_equity=high_watermark_equity,
        realized_pl_today=realized_pl_today,
        unrealized_pl=unrealized_pl,
        gross_exposure=gross_exposure,
        daily_loss_pct=daily_loss_pct,
        drawdown_pct=drawdown_pct,
    )


def main():
    setup_logging()
    cfg = load_config()
    log_environment_switch(cfg.env_mode, user="manual_start")

    adapter = AlpacaAdapter(cfg.env_mode)
    sentiment = SentimentModule()
    signal_engine = SignalEngine(adapter, sentiment, cfg.technical)
    risk_engine = RiskEngine(cfg.risk_limits, cfg.sentiment, cfg.instruments)
    pm = PositionManager(adapter)
    executor = OrderExecutor(adapter, cfg.env_mode, cfg.live_trading_enabled, cfg.execution)
    kill_switch = KillSwitch(cfg.risk_limits)
    portfolio_builder = PortfolioBuilder(cfg, adapter, sentiment, signal_engine, risk_engine)

    while True:
        acct = adapter.get_account()
        positions = pm.get_positions()
        snapshot = get_equity_snapshot_from_account(acct, positions)
        market_open = adapter.get_market_open()
        log_equity_snapshot(snapshot, market_open=market_open)


        ks_state = kill_switch.check(snapshot)
        log_kill_switch_state(ks_state)
        if ks_state.halted:
            time.sleep(60)
            continue

        exposure_cap_notional = snapshot.equity * cfg.risk_limits.gross_exposure_cap_pct
        
        # --- NO EQUITY GUARD ---
        if snapshot.gross_exposure >= exposure_cap_notional or len(positions) >= cfg.risk_limits.max_open_positions:
            time.sleep(60)
            continue

        # Fetch open orders to prevent duplicates and pass to portfolio builder
        open_orders = adapter.list_orders(status="open")

        proposed_trades = portfolio_builder.build_portfolio(snapshot, positions, open_orders)

        from monitoring.monitor import log_portfolio_overview  # local import to avoid cycles
        log_portfolio_overview(proposed_trades, cfg.env_mode)

        for proposed in proposed_trades:
            executor.execute_proposed_trade(proposed)

        time.sleep(60)


if __name__ == "__main__":
    main()




