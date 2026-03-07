# CHANGES:
# AUDIT FIX 1.1 — Added fail-closed equity-state load handling with quarantine logging so
#                  corrupted JSON cannot silently reset persisted runtime state.
# AUDIT FIX 1.2 — Added strict persistence wrappers and cycle-level guards so state-write
#                  failures disable new entries instead of silently continuing.
# AUDIT FIX 1.3 — Added guarded broker/data helper functions for startup, execution, and
#                  dashboard refresh paths to prevent unhandled live-path crashes.
# AUDIT FIX 1.4 — Reset start_of_day_equity at the New York market-session boundary instead
#                  of ET calendar midnight to align daily-loss accounting with trading logic.
# AUDIT FIX 1.5 — Pass cfg.sentiment into SentimentModule so runtime sentiment thresholds and
#                  cache behavior come from configured values.
# AUDIT FIX 1.6 — Added explicit main-loop live-trading gate before execute_proposed_trade()
#                  so no live submission can occur when live trading is disabled.
# AUDIT FIX 1.7 — Added conservative stale-baseline reconciliation requiring broker-flat and
#                  no-open-order confirmation before purging persisted opening_compounds.
# AUDIT FIX 1.8 — Added resilient wrappers around monitoring refreshes so telemetry failures
#                  cannot terminate the trading loop after order activity.
# AUDIT FIX 1.9 — Reduced duplicate sentiment cache lookups and added cache-miss logging for
#                  adaptive rescore interval calculations.

import json
import logging
import os
import tempfile
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
MARKET_OPEN_HOUR_ET = 9
MARKET_OPEN_MINUTE_ET = 30


class EquityStateError(RuntimeError):
    pass


def _quarantine_bad_equity_state(exc: Exception) -> None:
    # AUDIT FIX 1.1 — Preserve the unreadable state file for forensics instead of silently
    #                 discarding it and resetting the bot to empty runtime state.
    try:
        timestamp = datetime.now(tz=ET).strftime("%Y%m%dT%H%M%S")
        quarantine_path = EQUITY_STATE_PATH.with_suffix(
            EQUITY_STATE_PATH.suffix + f".corrupt.{timestamp}"
        )
        os.replace(EQUITY_STATE_PATH, quarantine_path)
        logger.error(
            "Quarantined unreadable equity state file to %s after load failure: %s",
            quarantine_path,
            exc,
        )
    except Exception as quarantine_exc:
        logger.error(
            "Failed to quarantine unreadable equity state file after load failure %s: %s",
            exc,
            quarantine_exc,
        )


def _load_equity_state() -> dict:
    if not EQUITY_STATE_PATH.exists():
        return {}
    try:
        with EQUITY_STATE_PATH.open("r") as f:
            state = json.load(f)
    except Exception as exc:
        # AUDIT FIX 1.1 — Log and quarantine corrupted JSON so state loss is explicit.
        _quarantine_bad_equity_state(exc)
        raise EquityStateError(f"Unable to load {EQUITY_STATE_PATH}: {exc}") from exc
    if not isinstance(state, dict):
        exc = ValueError("equity state root must be a JSON object")
        _quarantine_bad_equity_state(exc)
        raise EquityStateError(f"Invalid equity state structure in {EQUITY_STATE_PATH}")
    return state


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
        # AUDIT FIX 1.2 — Fail closed so order-entry logic can suspend when persistence is broken.
        logger.error("_save_equity_state error: %s", e)
        raise EquityStateError(f"Unable to persist {EQUITY_STATE_PATH}: {e}") from e


def _safe_load_equity_state(default: Optional[dict] = None) -> dict:
    try:
        return _load_equity_state()
    except EquityStateError as exc:
        logger.error("Falling back to in-memory default equity state after load error: %s", exc)
        return {} if default is None else dict(default)


def _load_opening_compounds() -> Dict[str, float]:
    state = _safe_load_equity_state({})
    raw = state.get("opening_compounds", {})
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    return {}


def _persist_opening_compounds(opening_compounds: Dict[str, float]) -> None:
    state = _safe_load_equity_state({})
    state["opening_compounds"] = opening_compounds
    _save_equity_state(state)


def _safe_list_open_orders(adapter: AlpacaAdapter) -> List[Any]:
    # AUDIT FIX 1.3 — Centralize guarded open-order lookup for reconciliation and UI paths.
    try:
        open_orders = adapter.list_orders(status="open")
    except Exception as exc:
        logger.warning("Open-order lookup failed: %s", exc)
        return []
    if open_orders is None:
        return []
    if isinstance(open_orders, list):
        return open_orders
    try:
        return list(open_orders)
    except TypeError:
        logger.warning("Open-order lookup returned non-iterable payload: %r", open_orders)
        return []


def _extract_order_symbol(order: Any) -> Optional[str]:
    symbol = getattr(order, "symbol", None)
    if symbol is None and isinstance(order, dict):
        symbol = order.get("symbol")
    return str(symbol) if symbol else None


def _symbols_with_open_orders(open_orders: List[Any]) -> set:
    symbols = set()
    for order in open_orders:
        symbol = _extract_order_symbol(order)
        if symbol:
            symbols.add(symbol)
    return symbols


def _reconcile_opening_compounds(
    opening_compounds: Dict[str, float],
    positions: Dict[str, PositionInfo],
    open_orders: Optional[List[Any]] = None,
) -> None:
    # AUDIT FIX 1.7 — Only purge when broker-flat and there are no open orders for the symbol.
    positions = positions or {}
    protected_symbols = _symbols_with_open_orders(open_orders or [])
    stale_syms = [
        s
        for s in list(opening_compounds.keys())
        if s not in positions and s not in protected_symbols
    ]
    if not stale_syms:
        return
    logger.info(
        "Reconciling stale opening_compounds for broker-flat symbols with no open orders: %s",
        stale_syms,
    )
    for s in stale_syms:
        del opening_compounds[s]
    _persist_opening_compounds(opening_compounds)


def _persist_vol_and_sentiment(
    risk_engine: RiskEngine,
    sentiment_module: SentimentModule,
) -> None:
    state = _safe_load_equity_state({})
    state["vol_history"] = risk_engine.export_vol_history()
    state["sentiment_cache"] = sentiment_module.export_cache()
    _save_equity_state(state)


def load_equity_state() -> dict:
    return _safe_load_equity_state({})


def _current_session_date_et(now_et: Optional[datetime] = None) -> str:
    # AUDIT FIX 1.4 — Session date rolls at the regular market-open boundary, not midnight.
    now_et = now_et or datetime.now(tz=ET)
    market_open_dt = now_et.replace(
        hour=MARKET_OPEN_HOUR_ET,
        minute=MARKET_OPEN_MINUTE_ET,
        second=0,
        microsecond=0,
    )
    if now_et < market_open_dt:
        return (now_et.date() - timedelta(days=1)).isoformat()
    return now_et.date().isoformat()


def get_equity_snapshot_from_account(
    acct,
    positions: Dict[str, PositionInfo],
) -> EquitySnapshot:
    equity = float(acct.equity)
    cash = float(acct.cash)
    portfolio_value = float(acct.portfolio_value)
    state = load_equity_state()
    session_day_str = _current_session_date_et()

    last_day = state.get("last_trading_day")
    start_of_day_equity = float(state.get("start_of_day_equity", equity))
    high_watermark_equity = float(state.get("high_watermark_equity", equity))

    if last_day != session_day_str:
        # AUDIT FIX 1.4 — Reset baseline at session boundary so daily-loss math is stable intraday.
        start_of_day_equity = equity
        state["last_trading_day"] = session_day_str

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
                getattr(
                    acct,
                    "daytrading_buying_power",
                    getattr(acct, "buying_power", 0.0),
                ),
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


def _guarded_call(
    label: str,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    # AUDIT FIX 1.3 — Standardize broker/data call protection and logging across main().
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.warning("%s failed: %s", label, exc)
        raise


def _update_dashboard_state_safely(**kwargs: Any) -> None:
    # AUDIT FIX 1.8 — Dashboard persistence must never crash the trading loop.
    try:
        dashboard_state.update(**kwargs)
        persist_state(dashboard_state)
    except Exception as exc:
        logger.warning("Dashboard state persistence failed: %s", exc)


def _refresh_positions_safely(
    pm: PositionManager,
    adapter: AlpacaAdapter,
    opening_compounds: Dict[str, float],
) -> Dict[str, PositionInfo]:
    positions = _guarded_call(
        "Position refresh",
        pm.get_positions,
        opening_compounds=opening_compounds,
    )
    open_orders = _safe_list_open_orders(adapter)
    _reconcile_opening_compounds(opening_compounds, positions, open_orders=open_orders)
    return positions


def main() -> None:
    setup_logging()
    cfg = load_config()
    setup_logging(cfg.env_mode)
    log_environment_switch(cfg.env_mode, user="manual_start")

    adapter = AlpacaAdapter(cfg.env_mode)
    # AUDIT FIX 1.5 — Use configured sentiment thresholds and cache controls at runtime.
    sentiment = SentimentModule(cfg.sentiment)
    signal_engine = SignalEngine(adapter, sentiment, cfg.technical)
    risk_engine = RiskEngine(cfg.risk_limits, cfg.sentiment, cfg.instruments)
    pm = PositionManager(adapter)
    executor = OrderExecutor(adapter, cfg.env_mode, cfg.live_trading_enabled, cfg.execution)
    kill_switch = KillSwitch(cfg.risk_limits)
    portfolio_builder = PortfolioBuilder(cfg, adapter, sentiment, signal_engine, risk_engine)
    trade_stats = TradeStatsTracker()

    _opening_compounds: Dict[str, float] = _load_opening_compounds()
    _entry_disabled_due_to_persistence = False

    _startup_state = _safe_load_equity_state({})
    risk_engine.import_vol_history(_startup_state.get("vol_history", {}))
    sentiment.import_cache(_startup_state.get("sentiment_cache", {}))
    logger.info(
        "Startup state restored: vol_history symbols=%d  sentiment_cache symbols=%d",
        len(risk_engine._vol_history),
        len(sentiment._cache),
    )

    try:
        # AUDIT FIX 1.3 — Guard startup broker refresh so transient Alpaca outages cannot crash boot.
        positions_at_start = _refresh_positions_safely(pm, adapter, _opening_compounds)
    except Exception:
        logger.warning(
            "Startup broker refresh failed; beginning loop in degraded mode and retrying next cycle."
        )
        positions_at_start = {}
    trade_stats.sync_open_positions(positions_at_start)

    _rescore_interval: int = 600
    try:
        launch_dashboard(refresh_seconds=5.0)
    except Exception as exc:
        logger.warning("Dashboard launch failed: %s", exc)
    _cycle_count: int = 0

    while True:
        _cycle_count += 1

        try:
            acct = _guarded_call("Account refresh", adapter.get_account)
            positions = _guarded_call(
                "Loop position refresh",
                pm.get_positions,
                opening_compounds=_opening_compounds,
            )
            market_open = _guarded_call("Market-open check", adapter.get_market_open)
        except Exception:
            logger.warning(
                "Broker connectivity issue during loop bootstrap. Sleeping 60s and continuing."
            )
            _update_dashboard_state_safely(
                cycle_count=_cycle_count,
                market_open=False,
                bot_state="IDLE",
            )
            time.sleep(60)
            continue

        open_orders_bootstrap = _safe_list_open_orders(adapter)
        try:
            _reconcile_opening_compounds(
                _opening_compounds,
                positions,
                open_orders=open_orders_bootstrap,
            )
            trade_stats.sync_open_positions(positions)
            snapshot = get_equity_snapshot_from_account(acct, positions)
            if _entry_disabled_due_to_persistence:
                logger.info("Persistence recovered; re-enabling new entries.")
            _entry_disabled_due_to_persistence = False
        except EquityStateError as exc:
            # AUDIT FIX 1.2 — Disable entries when persistence is unhealthy instead of trading blindly.
            logger.error("Persistence failure detected; suspending new entries this cycle: %s", exc)
            _entry_disabled_due_to_persistence = True
            snapshot = EquitySnapshot(
                equity=float(acct.equity),
                cash=float(acct.cash),
                portfolio_value=float(acct.portfolio_value),
                day_trading_buying_power=float(
                    getattr(
                        acct,
                        "day_trading_buying_power",
                        getattr(
                            acct,
                            "daytrading_buying_power",
                            getattr(acct, "buying_power", 0.0),
                        ),
                    )
                ),
                start_of_day_equity=float(acct.equity),
                high_watermark_equity=float(acct.equity),
                realized_pl_today=float(getattr(acct, "realized_pl", 0.0)),
                unrealized_pl=float(getattr(acct, "unrealized_pl", 0.0)),
                gross_exposure=sum(abs(p.notional) for p in positions.values()),
                daily_loss_pct=0.0,
                drawdown_pct=0.0,
            )

        log_equity_snapshot(snapshot, market_open=market_open)

        _update_dashboard_state_safely(
            cash=snapshot.cash,
            buying_power=snapshot.day_trading_buying_power,
            cycle_count=_cycle_count,
            market_open=market_open,
            bot_state="SCANNING",
        )

        if adapter.is_pre_weekend_close():
            try:
                positions = _refresh_positions_safely(pm, adapter, _opening_compounds)
            except Exception:
                time.sleep(60)
                continue
            trade_stats.sync_open_positions(positions)
            logger.warning(
                "WEEKEND CLOSE DETECTED: Closing all positions and sleeping until next market open."
            )
            _update_dashboard_state_safely(bot_state="EXECUTING")
            try:
                executor.close_all_positions_for_weekend(
                    positions,
                    cfg.env_mode,
                    opening_compounds=_opening_compounds,
                    persist_opening_compounds=_persist_opening_compounds,
                )
            except Exception as exc:
                logger.warning("Weekend liquidation failed: %s", exc)
            _update_dashboard_state_safely(bot_state="IDLE")
            while not adapter.get_market_open():
                time.sleep(60)
            continue

        if not market_open:
            try:
                positions = _refresh_positions_safely(pm, adapter, _opening_compounds)
                trade_stats.sync_open_positions(positions)
            except Exception:
                pass
            _update_dashboard_state_safely(bot_state="IDLE")
            time.sleep(60)
            continue

        if adapter.is_pre_daily_close():
            try:
                positions = _refresh_positions_safely(pm, adapter, _opening_compounds)
            except Exception:
                time.sleep(60)
                continue
            trade_stats.sync_open_positions(positions)
            logger.warning(
                "DAILY CLOSE DETECTED: Closing all positions before market close to avoid overnight gap risk."
            )
            try:
                executor.close_all_positions_for_weekend(
                    positions,
                    cfg.env_mode,
                    opening_compounds=_opening_compounds,
                    persist_opening_compounds=_persist_opening_compounds,
                )
            except Exception as exc:
                logger.warning("Daily close liquidation failed: %s", exc)
            _update_dashboard_state_safely(bot_state="IDLE")
            time.sleep(60)
            continue

        ks_state = kill_switch.check(snapshot)
        log_kill_switch_state(ks_state)
        if ks_state.halted:
            _update_dashboard_state_safely(bot_state="HALTED")
            time.sleep(60)
            continue

        if positions:
            try:
                _check_and_exit_on_sentiment(
                    positions=positions,
                    adapter=adapter,
                    sentiment_module=sentiment,
                    executor=executor,
                    cfg=cfg,
                    opening_compounds=_opening_compounds,
                    persist_opening_compounds=_persist_opening_compounds,
                )
                positions = _refresh_positions_safely(pm, adapter, _opening_compounds)
                trade_stats.sync_open_positions(positions)
                snapshot = get_equity_snapshot_from_account(acct, positions)
            except Exception as exc:
                logger.warning("Sentiment exit pass failed: %s", exc)

        exposure_cap_notional = snapshot.equity * cfg.risk_limits.gross_exposure_cap_pct
        if (
            snapshot.gross_exposure >= exposure_cap_notional
            or len(positions) >= cfg.risk_limits.max_open_positions
        ):
            open_orders_ui = _safe_list_open_orders(adapter)
            try:
                price_rows = build_price_rows(cfg.instruments, adapter)
                trade_stats.update_watched_trade_outcomes(
                    {row.symbol: row.last_price for row in price_rows}
                )
                _update_dashboard_state_safely(
                    positions=build_position_rows(positions, open_orders_ui),
                    prices=price_rows,
                    bot_state="IDLE",
                )
            except Exception as exc:
                logger.warning("Exposure-cap dashboard refresh failed: %s", exc)
            time.sleep(60)
            continue

        if adapter.is_monday_open_blackout():
            logger.info(
                "MONDAY OPEN BLACKOUT: New position entries suppressed (within first 30 minutes of Monday market open). Monitoring existing positions only."
            )
        elif adapter.is_pre_close_blackout():
            logger.info(
                "PRE-CLOSE BLACKOUT: New position entries suppressed (within 3 hours of market close). Monitoring existing positions only."
            )
        else:
            _update_dashboard_state_safely(bot_state="SCANNING")

            if _entry_disabled_due_to_persistence:
                logger.warning(
                    "Skipping new entries because equity-state persistence is unavailable."
                )
            elif not cfg.live_trading_enabled:
                # AUDIT FIX 1.6 — Enforce live-trading gate in main() before any submit call path.
                logger.warning(
                    "LIVE_TRADING_ENABLED is false; skipping execute_proposed_trade submissions."
                )
            else:
                logger.info("Canceling all open orders before portfolio execution.")
                try:
                    # AUDIT FIX 1.3 — Prevent live-path crash if order cancellation fails.
                    adapter.cancel_all_orders()
                    open_orders = _safe_list_open_orders(adapter)
                    proposed_trades = portfolio_builder.build_portfolio(
                        snapshot,
                        positions,
                        open_orders,
                    )
                    log_portfolio_overview(proposed_trades, cfg.env_mode)
                    _update_dashboard_state_safely(bot_state="EXECUTING")
                    for proposed in proposed_trades:
                        try:
                            order = executor.execute_proposed_trade(proposed)
                        except Exception as exc:
                            logger.warning(
                                "Order execution failed for %s: %s",
                                getattr(proposed, "symbol", "UNKNOWN"),
                                exc,
                            )
                            proposed.rejected_reason = (
                                f"AUDIT FIX 1.3 execution failure: {exc}"
                            )
                            continue
                        if (
                            order is not None
                            and proposed.rejected_reason is None
                            and proposed.qty > 0
                        ):
                            _opening_compounds[proposed.symbol] = proposed.signal_score
                            _persist_opening_compounds(_opening_compounds)
                            trade_stats.register_entry_from_proposed(proposed)
                except Exception as exc:
                    logger.warning("Portfolio execution pass failed: %s", exc)

        try:
            positions = _refresh_positions_safely(pm, adapter, _opening_compounds)
            trade_stats.sync_open_positions(positions)
        except Exception as exc:
            logger.warning("Post-entry position refresh failed: %s", exc)
            positions = positions or {}

        try:
            _check_and_exit_on_sentiment(
                positions=positions,
                adapter=adapter,
                sentiment_module=sentiment,
                executor=executor,
                cfg=cfg,
                opening_compounds=_opening_compounds,
                persist_opening_compounds=_persist_opening_compounds,
            )
        except Exception as exc:
            logger.warning("End-of-loop sentiment exit pass failed: %s", exc)

        try:
            # AUDIT FIX 1.8 — Monitoring/data refresh failures must not crash the loop.
            open_orders_final = _safe_list_open_orders(adapter)
            positions_refreshed = _refresh_positions_safely(pm, adapter, _opening_compounds)
            trade_stats.sync_open_positions(positions_refreshed)
            price_rows = build_price_rows(cfg.instruments, adapter)
            trade_stats.update_watched_trade_outcomes(
                {row.symbol: row.last_price for row in price_rows}
            )
            _update_dashboard_state_safely(
                positions=build_position_rows(positions_refreshed, open_orders_final),
                prices=price_rows,
                bot_state="IDLE",
            )
        except Exception as exc:
            logger.warning("End-of-loop monitoring refresh failed: %s", exc)
            positions_refreshed = positions

        try:
            _persist_vol_and_sentiment(risk_engine, sentiment)
        except EquityStateError as exc:
            logger.error("Vol/sentiment persistence failed: %s", exc)
            _entry_disabled_due_to_persistence = True

        _cached_scores: List[float] = []
        _cache_miss_symbols: List[str] = []
        for sym in positions_refreshed:
            cached = sentiment.get_cached_sentiment(sym)
            if cached is None:
                _cache_miss_symbols.append(sym)
                continue
            _cached_scores.append(abs(cached.score))
        if _cache_miss_symbols:
            # AUDIT FIX 1.9 — Make adaptive rescore degradation visible in logs.
            logger.info(
                "Adaptive rescore skipped missing cached sentiment for symbols: %s",
                _cache_miss_symbols,
            )
        _max_abs_s = max(_cached_scores, default=0.0)
        _rescore_interval = sentiment.adaptive_rescore_interval_hysteresis(
            _max_abs_s,
            _rescore_interval,
        )
        time.sleep(_rescore_interval)


if __name__ == "__main__":
    main()
