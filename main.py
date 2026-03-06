# CHANGES:
# AUDIT FIX 8.4 — Added main-level LIVE_TRADING_ENABLED gate before portfolio execution.
#                 Defense-in-depth: portfolio execution skipped entirely when
#                 cfg.live_trading_enabled is False, preventing config errors from
#                 causing accidental live trades.
# AUDIT FIX 8.5 — Added compute_pending_notional() to deduct pending order exposure from
#                 available gross exposure capacity. Portfolio builder now considers both
#                 existing positions and pending orders when checking gross exposure cap.
#                 Prevents silent breach of 90% cap when large orders are pending.
# AUDIT FIX 8.6 — Replaced naive date comparison with market calendar check for equity state
#                 reset. start_of_day_equity now resets on actual market open, not midnight.
#                 Fallback to date comparison if calendar API fails.
# FIX 4A — Moved _persist_opening_compounds() to immediately inside the fill-confirmation block.
# FIX 4B — Added positions refresh immediately before end-of-loop _check_and_exit_on_sentiment() call.
# FIX 4C — Deleted unused acct_startup assignment.
# OA-1 — Added OPERATOR ACTION REQUIRED comment for instrument_whitelist.yaml sector diversification.
# MONDAY-BLACKOUT — Integrated is_monday_open_blackout() check into main loop STEP 3 alongside
#                   is_pre_close_blackout(). New entries are suppressed for 30 minutes after
#                   Monday market open. Existing position exits are unaffected.
# ORPHAN-ORDER-FIX — Added adapter.cancel_all_orders() immediately before portfolio execution
#                    in STEP 3 to eliminate orphaned order risk from previous iterations.
# PURGE-FIX — Updated both _check_and_exit_on_sentiment() calls to pass opening_compounds and
#             _persist_opening_compounds as kwargs so the purge can execute atomically inline
#             with sentiment-driven position closes. Updated close_all_positions_for_weekend()
#             call to pass the same kwargs so weekend liquidation can purge baselines correctly.
# OPENING-COMPOSITE-FIX — Persist the technical entry composite via proposed.signal_score.
# NETWORK-GUARD-FIX — Wrapped broker bootstrap calls in the main loop with try/except so
#                     transient Alpaca connectivity failures log, back off, and continue
#                     instead of terminating the bot.
# TRADE-STATS-INTEGRATION — Added TradeStatsTracker lifecycle integration so entries are registered,
#                           open positions are reconciled, stop exits are inferred, and watched stop
#                           outcomes are updated from live price snapshots for debugging stop settings.

import json
import logging
import os
import tempfile
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[import]

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
from monitoring.dashboard import (
    build_position_rows,
    build_price_rows,
    dashboard_state,
    launch_dashboard,
    persist_state,
)
from monitoring.trade_stats import TradeStatsTracker

logger = logging.getLogger("tradebot")

EQUITY_STATE_PATH = Path("equity_state.json")
ET = ZoneInfo("America/New_York")


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
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=EQUITY_STATE_PATH.parent,
            prefix=".equity_state_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            os.unlink(tmp_path)
            raise
        os.replace(tmp_path, EQUITY_STATE_PATH)
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


def _persist_vol_and_sentiment(
    risk_engine: RiskEngine,
    sentiment_module: SentimentModule,
) -> None:
    state = _load_equity_state()
    state["vol_history"] = risk_engine.export_vol_history()
    state["sentiment_cache"] = sentiment_module.export_cache()
    _save_equity_state(state)


def load_equity_state() -> dict:
    return _load_equity_state()


def compute_pending_notional(open_orders: List) -> float:
    """AUDIT FIX 8.5: Calculate total notional exposure from pending orders.

    Pending orders lock capital and count against gross exposure cap even though
    they are not yet filled. This prevents the bot from over-allocating.

    Returns:
        Total notional value of all open orders (sum of qty * limit/stop price).
    """
    pending_notional = 0.0
    for order in open_orders:
        try:
            qty = abs(float(getattr(order, "qty", 0.0)))
            # Use limit_price if available, else stop_price, else skip
            price = float(
                getattr(order, "limit_price", None) or
                getattr(order, "stop_price", None) or
                0.0
            )
            if qty > 0 and price > 0:
                pending_notional += qty * price
        except Exception as e:
            logger.warning(f"Error computing notional for order {getattr(order, 'id', 'N/A')}: {e}")
    return pending_notional


def get_equity_snapshot_from_account(
    adapter: AlpacaAdapter,
    acct,
    positions: Dict[str, PositionInfo],
    open_orders: List,
) -> EquitySnapshot:
    """Compute equity snapshot with market calendar-aware state reset.

    AUDIT FIX 8.6: Uses market calendar API to determine actual market open
    date instead of naive date comparison. Prevents mid-session resets at
    timezone boundaries.

    AUDIT FIX 8.5: Includes pending order notional in gross exposure calculation.
    """
    equity = float(acct.equity)
    cash = float(acct.cash)
    portfolio_value = float(acct.portfolio_value)
    state = load_equity_state()
    today_str = datetime.now(tz=ET).date().isoformat()

    last_day = state.get("last_trading_day")
    start_of_day_equity = float(state.get("start_of_day_equity", equity))
    high_watermark_equity = float(state.get("high_watermark_equity", equity))

    # AUDIT FIX 8.6: Market calendar check for equity state reset
    should_reset = False
    try:
        # Query calendar for today's market session
        calendar = adapter.get_calendar(start=today_str, end=today_str)
        if calendar and len(calendar) > 0:
            market_date = str(calendar[0].date)
            if last_day != market_date:
                should_reset = True
                logger.info(
                    f"Market calendar reset: last_day={last_day} → market_date={market_date}"
                )
        else:
            # Fallback: no market session today (holiday/weekend)
            if last_day != today_str:
                logger.debug(
                    f"No market session today ({today_str}), preserving start_of_day_equity"
                )
    except Exception as e:
        # Fallback to naive date comparison if calendar API fails
        logger.warning(f"Market calendar API error: {e}. Falling back to date comparison.")
        if last_day != today_str:
            should_reset = True

    if should_reset:
        start_of_day_equity = equity
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

    # AUDIT FIX 8.5: Include pending order notional in gross exposure
    position_exposure = sum(abs(p.notional) for p in positions.values())
    pending_exposure = compute_pending_notional(open_orders)
    gross_exposure = position_exposure + pending_exposure

    if pending_exposure > 0:
        logger.debug(
            f"Gross exposure: positions={position_exposure:.2f} + pending={pending_exposure:.2f} = {gross_exposure:.2f}"
        )

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


def main() -> None:
    setup_logging()
    cfg = load_config()
    setup_logging(cfg.env_mode)
    log_environment_switch(cfg.env_mode, user="manual_start")

    adapter = AlpacaAdapter(cfg.env_mode)
    sentiment = SentimentModule()
    signal_engine = SignalEngine(adapter, sentiment, cfg.technical)
    risk_engine = RiskEngine(cfg.risk_limits, cfg.sentiment, cfg.instruments)
    pm = PositionManager(adapter)
    executor = OrderExecutor(adapter, cfg.env_mode, cfg.live_trading_enabled, cfg.execution)
    kill_switch = KillSwitch(cfg.risk_limits)
    portfolio_builder = PortfolioBuilder(cfg, adapter, sentiment, signal_engine, risk_engine)
    trade_stats = TradeStatsTracker()

    _opening_compounds: Dict[str, float] = _load_opening_compounds()

    _startup_state = _load_equity_state()
    risk_engine.import_vol_history(_startup_state.get("vol_history", {}))
    sentiment.import_cache(_startup_state.get("sentiment_cache", {}))
    logger.info(
        "Startup state restored: vol_history symbols=%d  sentiment_cache symbols=%d",
        len(risk_engine._vol_history),
        len(sentiment._cache),
    )

    positions_at_start = pm.get_positions(opening_compounds=_opening_compounds)
    stale_syms = [s for s in list(_opening_compounds) if s not in positions_at_start]
    if stale_syms:
        logger.info(
            f"Startup reconcile: purging stale opening_compounds for {stale_syms}"
        )
        for s in stale_syms:
            del _opening_compounds[s]
        _persist_opening_compounds(_opening_compounds)
    trade_stats.sync_open_positions(positions_at_start)

    _rescore_interval: int = 600
    launch_dashboard(refresh_seconds=5.0)
    _cycle_count: int = 0

    while True:
        _cycle_count += 1

        try:
            acct = adapter.get_account()
            positions = pm.get_positions(opening_compounds=_opening_compounds)
            open_orders = adapter.list_orders(status="open")
            market_open = adapter.get_market_open()
        except Exception as e:
            logger.warning(
                f"Broker connectivity issue during loop bootstrap: {e}. "
                "Sleeping 60s and continuing."
            )
            dashboard_state.update(
                cycle_count=_cycle_count,
                market_open=False,
                bot_state="IDLE",
            )
            persist_state(dashboard_state)
            time.sleep(60)
            continue

        trade_stats.sync_open_positions(positions)
        snapshot = get_equity_snapshot_from_account(adapter, acct, positions, open_orders)
        log_equity_snapshot(snapshot, market_open=market_open)

        dashboard_state.update(
            cash=snapshot.cash,
            buying_power=snapshot.day_trading_buying_power,
            cycle_count=_cycle_count,
            market_open=market_open,
            bot_state="SCANNING",
        )
        persist_state(dashboard_state)

        if adapter.is_pre_weekend_close():
            logger.warning(
                "WEEKEND CLOSE DETECTED: Closing all positions and sleeping "
                "until next market open."
            )
            dashboard_state.update(bot_state="EXECUTING")
            persist_state(dashboard_state)
            executor.close_all_positions_for_weekend(
                positions,
                cfg.env_mode,
                opening_compounds=_opening_compounds,
                persist_opening_compounds=_persist_opening_compounds,
            )
            dashboard_state.update(bot_state="IDLE")
            persist_state(dashboard_state)
            while not adapter.get_market_open():
                time.sleep(60)
            continue

        if adapter.is_pre_daily_close():
            logger.warning(
                "DAILY CLOSE DETECTED: Closing all positions before market close "
                "to avoid overnight gap risk."
            )
            executor.close_all_positions_for_weekend(
                positions,
                cfg.env_mode,
                opening_compounds=_opening_compounds,
                persist_opening_compounds=_persist_opening_compounds,
            )

        ks_state = kill_switch.check(snapshot)
        log_kill_switch_state(ks_state)
        if ks_state.halted:
            dashboard_state.update(bot_state="HALTED")
            persist_state(dashboard_state)
            time.sleep(60)
            continue

        if positions:
            _check_and_exit_on_sentiment(
                positions=positions,
                adapter=adapter,
                sentiment_module=sentiment,
                executor=executor,
                cfg=cfg,
                opening_compounds=_opening_compounds,
                persist_opening_compounds=_persist_opening_compounds,
            )
            positions = pm.get_positions(opening_compounds=_opening_compounds)
            trade_stats.sync_open_positions(positions)
            # Refresh open_orders after sentiment exits
            open_orders = adapter.list_orders(status="open")
            snapshot = get_equity_snapshot_from_account(adapter, acct, positions, open_orders)

        for sym in list(_opening_compounds.keys()):
            if sym not in positions:
                del _opening_compounds[sym]
        _persist_opening_compounds(_opening_compounds)

        exposure_cap_notional = snapshot.equity * cfg.risk_limits.gross_exposure_cap_pct
        if (
            snapshot.gross_exposure >= exposure_cap_notional
            or len(positions) >= cfg.risk_limits.max_open_positions
        ):
            open_orders_ui = adapter.list_orders(status="open")
            price_rows = build_price_rows(cfg.instruments, adapter)
            trade_stats.update_watched_trade_outcomes({row.symbol: row.last_price for row in price_rows})
            dashboard_state.update(
                positions=build_position_rows(positions, open_orders_ui),
                prices=price_rows,
                bot_state="IDLE",
            )
            persist_state(dashboard_state)
            time.sleep(60)
            continue

        if adapter.is_monday_open_blackout():
            logger.info(
                "MONDAY OPEN BLACKOUT: New position entries suppressed "
                "(within first 30 minutes of Monday market open). Monitoring existing positions only."
            )
        elif adapter.is_pre_close_blackout():
            logger.info(
                "PRE-CLOSE BLACKOUT: New position entries suppressed "
                "(within 3 hours of market close). Monitoring existing positions only."
            )
        elif cfg.live_trading_enabled:
            # AUDIT FIX 8.4: Main-level LIVE_TRADING_ENABLED gate
            # Defense-in-depth: portfolio execution only proceeds when live trading is enabled
            # This prevents accidental live trades if executor gate is bypassed or misconfigured
            dashboard_state.update(bot_state="SCANNING")
            persist_state(dashboard_state)

            logger.info("Canceling all open orders before portfolio execution.")
            adapter.cancel_all_orders()

            # Refresh open_orders after cancellation
            open_orders = adapter.list_orders(status="open")
            proposed_trades = portfolio_builder.build_portfolio(snapshot, positions, open_orders)

            log_portfolio_overview(proposed_trades, cfg.env_mode)

            dashboard_state.update(bot_state="EXECUTING")
            persist_state(dashboard_state)
            for proposed in proposed_trades:
                order = executor.execute_proposed_trade(proposed)
                if order is not None and proposed.rejected_reason is None and proposed.qty > 0:
                    _opening_compounds[proposed.symbol] = proposed.signal_score
                    _persist_opening_compounds(_opening_compounds)
                    trade_stats.register_entry_from_proposed(proposed)
        else:
            # AUDIT FIX 8.4: Log when live trading is disabled
            logger.info(
                "[PAPER MODE] Portfolio execution skipped: live_trading_enabled=False. "
                "Set LIVE_TRADING_ENABLED=true in .env to enable live trading."
            )

        positions = pm.get_positions(opening_compounds=_opening_compounds)
        trade_stats.sync_open_positions(positions)

        _check_and_exit_on_sentiment(
            positions=positions,
            adapter=adapter,
            sentiment_module=sentiment,
            executor=executor,
            cfg=cfg,
            opening_compounds=_opening_compounds,
            persist_opening_compounds=_persist_opening_compounds,
        )

        open_orders_final = adapter.list_orders(status="open")
        positions_refreshed = pm.get_positions(opening_compounds=_opening_compounds)
        trade_stats.sync_open_positions(positions_refreshed)
        price_rows = build_price_rows(cfg.instruments, adapter)
        trade_stats.update_watched_trade_outcomes({row.symbol: row.last_price for row in price_rows})
        dashboard_state.update(
            positions=build_position_rows(positions_refreshed, open_orders_final),
            prices=price_rows,
            bot_state="IDLE",
        )
        persist_state(dashboard_state)

        _persist_vol_and_sentiment(risk_engine, sentiment)

        _max_abs_s = max(
            (
                abs(sentiment.get_cached_sentiment(sym).score)
                for sym in positions_refreshed
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
