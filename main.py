# CHANGES:
# FIX H2 (PROD-READINESS) — Added kill switch auto-restart logic that resets the halt
#                            state at the start of each new trading day. When kill switch
#                            is triggered (daily loss or drawdown breach), bot now checks
#                            if a new trading day has begun (today_str != last_halt_day).
#                            If so, logs auto-reset, clears halt state, and allows normal
#                            execution to resume. Prevents multi-day unintended halt that
#                            would cause missed opportunities. last_halt_day persisted in
#                            equity_state.json for restart resilience.
# TASK 5.1 — Integrated TradeStatsTracker into main() loop.
#            1. Construct tracker after loading config, pass risk & sentiment config.
#            2. Register each executed entry via tracker.register_entry().
#            3. Call tracker.register_exit() in sentiment-exit path (order_executor.py).
#            4. Persist tracker state via export_state() at end of each loop.
#            5. Import state from equity_state.json on bot startup.
# FIX ORPHAN-ORDER — Call adapter.cancel_all_orders() before portfolio execution to prevent
#                    duplicate entries and stale limit orders from prior loops.
# FIX 8 — Replaced datetime.utcnow() with datetime.now(timezone.utc) throughout to eliminate
#         DeprecationWarning. Added timezone import.
# FIX PERSIST-SENTIMENT — Added sentiment cache import/export to _persist_vol_and_sentiment()
#                         and _load_vol_and_sentiment() so sentiment scores survive bot restart.
# CHANGE CYCLE-TIME — Increased main loop cycle from 600s (10 min) to 300s (5 min) to reduce
#                     latency for sentiment-exit checks while still respecting Alpaca rate limits.
#                     Adaptive sentiment rescore interval (120s–900s) provides higher-frequency
#                     checks for high-conviction positions without burning API quota.

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pytz

from adapters.alpaca_adapter import AlpacaAdapter
from ai_client import AIConfig
from config.config import load_config, BotConfig
from core.risk_engine import (
    RiskEngine,
    EquitySnapshot,
    PositionInfo,
)
from core.sentiment import SentimentModule
from core.signals import SignalEngine
from core.portfolio_builder import PortfolioBuilder
from execution.order_executor import (
    OrderExecutor,
    _check_and_exit_on_sentiment,
)
from monitoring.kill_switch import KillSwitch
from monitoring.dashboard import DashboardState
from monitoring.trade_stats import TradeStatsTracker

# US Eastern Time for market-relative date checks.
ET = pytz.timezone("America/New_York")

logger = logging.getLogger("tradebot")

# Equity state file path (relative to repo root).
EQUITY_STATE_PATH = Path("equity_state.json")


def setup_logging():
    """Configure logging: INFO to console, DEBUG to rotating file."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(log_format)
    console.setFormatter(console_fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console)

    file_handler = logging.FileHandler("tradebot.log")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(log_format)
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)


def _load_equity_state() -> dict:
    """Load equity state from JSON file, or return empty dict if not found."""
    if not EQUITY_STATE_PATH.exists():
        return {}
    try:
        with EQUITY_STATE_PATH.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"_load_equity_state error: {e}")
        return {}


def _save_equity_state(state: dict) -> None:
    """Atomically write equity state to JSON file using temp file + rename.
    
    This ensures that equity_state.json is never partially written or corrupted.
    If the process crashes mid-write, the temp file is orphaned and the original
    state file remains intact.
    """
    try:
        # 1. Create temp file in same directory as target file.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=EQUITY_STATE_PATH.parent,
            prefix=".equity_state_",
            suffix=".tmp",
        )
        try:
            # 2. Write to temp file and fsync.
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            # Clean up temp file on write failure.
            os.unlink(tmp_path)
            raise
        # 3. Atomic rename (POSIX guarantee).
        os.replace(tmp_path, EQUITY_STATE_PATH)
    except Exception as e:
        logger.warning(f"_save_equity_state error: {e}")


def _load_opening_compounds() -> Dict[str, float]:
    """Load opening compound scores from equity state.
    
    Opening compounds are the technical composite scores at entry time,
    used by sentiment-exit logic to compute delta (sentiment deterioration).
    """
    state = _load_equity_state()
    raw = state.get("opening_compounds", {})
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    return {}


def _persist_opening_compounds(compounds: Dict[str, float]) -> None:
    """Persist opening compound scores to equity state."""
    state = _load_equity_state()
    state["opening_compounds"] = compounds
    _save_equity_state(state)


def _load_vol_and_sentiment(risk_engine: RiskEngine, sentiment: SentimentModule) -> None:
    """Import vol history and sentiment cache from equity state.
    
    FIX PERSIST-SENTIMENT: Now imports sentiment cache so TTL-valid scores
    survive bot restart, reducing API calls on warm start.
    """
    state = _load_equity_state()
    vol_hist = state.get("vol_history", {})
    if vol_hist:
        try:
            risk_engine.import_vol_history(vol_hist)
            logger.info(f"Imported vol history for {len(vol_hist)} symbols.")
        except Exception as e:
            logger.warning(f"import_vol_history error: {e}")
    
    # FIX PERSIST-SENTIMENT: import sentiment cache.
    sent_cache = state.get("sentiment_cache", {})
    if sent_cache:
        try:
            sentiment.import_cache(sent_cache)
            logger.info(f"Imported sentiment cache for {len(sent_cache)} symbols.")
        except Exception as e:
            logger.warning(f"import_cache error: {e}")


def _persist_vol_and_sentiment(risk_engine: RiskEngine, sentiment: SentimentModule) -> None:
    """Export vol history and sentiment cache to equity state.
    
    FIX PERSIST-SENTIMENT: Now exports sentiment cache so TTL-valid scores
    survive bot restart.
    """
    state = _load_equity_state()
    state["vol_history"] = risk_engine.export_vol_history()
    # FIX PERSIST-SENTIMENT: export sentiment cache.
    state["sentiment_cache"] = sentiment.export_cache()
    _save_equity_state(state)


def get_equity_snapshot_from_account(
    acct,
    positions: Dict[str, PositionInfo],
) -> EquitySnapshot:
    """
    Compute the equity snapshot from Alpaca account object and positions.
    
    This function manages start-of-day equity reset and high-watermark tracking.
    Both values are persisted in equity_state.json and loaded at bot startup.
    
    FIX 8: Uses datetime.now(timezone.utc) instead of datetime.utcnow().
    """
    state = _load_equity_state()
    equity = float(acct.equity)

    # Start-of-day equity reset.
    today_str = datetime.now(tz=ET).date().isoformat()
    last_day = state.get("last_trading_day")
    start_of_day_equity = float(state.get("start_of_day_equity", equity))
    if last_day != today_str:
        start_of_day_equity = equity
        state["last_trading_day"] = today_str
        logger.info(f"New trading day {today_str}: start_of_day_equity reset to {equity:.2f}")

    # High-watermark tracking (monotonic increase).
    high_watermark_equity = float(state.get("high_watermark_equity", equity))
    if equity > high_watermark_equity:
        high_watermark_equity = equity

    # Persist updated state.
    state["start_of_day_equity"] = start_of_day_equity
    state["high_watermark_equity"] = high_watermark_equity
    _save_equity_state(state)

    # Compute daily loss and drawdown.
    daily_pnl = equity - start_of_day_equity
    daily_loss_pct = daily_pnl / start_of_day_equity if start_of_day_equity != 0 else 0.0
    drawdown_pct = (
        (equity - high_watermark_equity) / high_watermark_equity
        if high_watermark_equity != 0
        else 0.0
    )

    # Gross exposure from positions.
    gross_exposure = sum(abs(pos.market_value) for pos in positions.values())

    return EquitySnapshot(
        equity=equity,
        gross_exposure=gross_exposure,
        start_of_day_equity=start_of_day_equity,
        high_watermark_equity=high_watermark_equity,
        daily_loss_pct=daily_loss_pct,
        drawdown_pct=drawdown_pct,
    )


def _get_positions_from_adapter(adapter: AlpacaAdapter) -> Dict[str, PositionInfo]:
    """Fetch positions from Alpaca and convert to PositionInfo dict."""
    from execution.position_manager import get_positions_from_adapter
    return get_positions_from_adapter(adapter)


def persist_state(dashboard_state: DashboardState) -> None:
    """Persist dashboard state to equity_state.json.
    
    This is a placeholder for future dashboard state persistence.
    Currently a no-op to avoid cluttering equity_state.json.
    """
    pass


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Alpaca Sentiment Tradebot — Production Readiness Fixes Applied")
    logger.info("FIX H3: Rate limit handling with retry-after logic")
    logger.info("FIX H2: Kill switch auto-restart on new trading day")
    logger.info("FIX H1: Pre-close blackout reduced to 90 minutes")
    logger.info("=" * 60)

    cfg: BotConfig = load_config()
    logger.info(f"Loaded config: {len(cfg.instruments)} instruments, env={cfg.env_mode}")

    adapter = AlpacaAdapter(cfg.env_mode)
    sentiment = SentimentModule(cfg.sentiment)
    signal_engine = SignalEngine(cfg.technical, cfg.execution, adapter, sentiment)
    risk_engine = RiskEngine(cfg.risk_limits)
    portfolio_builder = PortfolioBuilder(cfg, adapter, sentiment, signal_engine, risk_engine)
    kill_switch = KillSwitch(cfg.risk_limits)
    
    # TASK 5.1: Construct TradeStatsTracker.
    tracker = TradeStatsTracker(cfg.risk_limits, cfg.sentiment)

    executor = OrderExecutor(
        adapter=adapter,
        execution_cfg=cfg.execution,
        live_trading_enabled=cfg.live_trading_enabled,
        env_mode=cfg.env_mode,
        tracker=tracker,
    )

    # Load persisted state.
    _opening_compounds = _load_opening_compounds()
    logger.info(f"Loaded {len(_opening_compounds)} opening compounds from equity state.")
    
    _load_vol_and_sentiment(risk_engine, sentiment)
    
    # TASK 5.1: Import tracker state from equity_state.json.
    state = _load_equity_state()
    tracker_state = state.get("trade_stats", {})
    if tracker_state:
        try:
            tracker.import_state(tracker_state)
            logger.info(f"Imported trade stats: {len(tracker_state.get('closed_trades', []))} closed trades.")
        except Exception as e:
            logger.warning(f"import_trade_stats error: {e}")

    dashboard_state = DashboardState()
    _cycle_count = 0
    _rescore_interval = 600  # Initial sentiment rescore interval (seconds).

    logger.info("Entering main loop...")
    while True:
        _cycle_count += 1
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CYCLE {_cycle_count}")
        logger.info(f"{'=' * 60}")

        # --- Bootstrap: fetch account, positions, market status with network resilience ---
        try:
            acct = adapter.get_account()
            positions = _get_positions_from_adapter(adapter)
            market_open = adapter.get_market_open()
        except Exception as e:
            logger.warning(f"Bootstrap error (account/positions/market_open): {e}")
            logger.warning("Sleeping 60s before retry...")
            time.sleep(60)
            continue

        if not market_open:
            logger.info("Market closed. Sleeping 60s...")
            time.sleep(60)
            continue

        snapshot = get_equity_snapshot_from_account(acct, positions)
        logger.info(
            f"Equity: ${snapshot.equity:.2f} | "
            f"Gross Exposure: ${snapshot.gross_exposure:.2f} | "
            f"Daily P&L: {snapshot.daily_loss_pct:.2%} | "
            f"Drawdown: {snapshot.drawdown_pct:.2%}"
        )

        # --- Weekend liquidation ---
        if adapter.is_pre_weekend_close():
            logger.warning("WEEKEND LIQUIDATION TRIGGERED: Closing all positions before Friday close...")
            executor.close_all_positions_for_weekend(
                positions,
                cfg.env_mode,
                opening_compounds=_opening_compounds,
                persist_opening_compounds=_persist_opening_compounds,
            )
            # Refresh positions after liquidation.
            try:
                positions = _get_positions_from_adapter(adapter)
            except Exception as e:
                logger.warning(f"Position refresh after weekend liquidation error: {e}")
            dashboard_state.update(bot_state="WEEKEND_LIQUIDATION_COMPLETE")
            persist_state(dashboard_state)
            logger.info("Weekend liquidation complete. Sleeping 60s...")
            time.sleep(60)
            continue

        # --- Daily close liquidation ---
        if adapter.is_pre_daily_close():
            logger.warning("DAILY CLOSE DETECTED: Closing all positions before market close...")
            executor.close_all_positions_for_weekend(
                positions,
                cfg.env_mode,
                opening_compounds=_opening_compounds,
                persist_opening_compounds=_persist_opening_compounds,
            )
            try:
                positions = _get_positions_from_adapter(adapter)
            except Exception as e:
                logger.warning(f"Position refresh after daily close error: {e}")
            dashboard_state.update(bot_state="DAILY_CLOSE_COMPLETE")
            persist_state(dashboard_state)
            logger.info("Daily close complete. Sleeping 60s...")
            time.sleep(60)
            continue

        # --- Kill switch check with auto-restart on new trading day (FIX H2) ---
        ks_state = kill_switch.check(snapshot)
        if ks_state.halted:
            # FIX H2: Check if new trading day.
            state = _load_equity_state()
            today_str = datetime.now(tz=ET).date().isoformat()
            last_halt_day = state.get("last_halt_day")
            
            if last_halt_day != today_str:
                # New trading day detected, auto-reset kill switch.
                logger.warning(
                    f"KILL SWITCH AUTO-RESET: New trading day {today_str} detected. "
                    f"Previous halt on {last_halt_day}. Resuming normal operation."
                )
                state["last_halt_day"] = today_str
                _save_equity_state(state)
                # Continue to normal execution instead of sleeping.
            else:
                # Same day, remain halted.
                logger.warning(f"Kill switch halted (same day): {ks_state.reason}")
                dashboard_state.update(bot_state="HALTED", halt_reason=ks_state.reason)
                persist_state(dashboard_state)
                time.sleep(60)
                continue
        else:
            # Not halted, clear last_halt_day if it exists.
            state = _load_equity_state()
            if "last_halt_day" in state:
                del state["last_halt_day"]
                _save_equity_state(state)

        # --- Sentiment-driven position exits (first check) ---
        _check_and_exit_on_sentiment(
            positions=positions,
            adapter=adapter,
            sentiment_module=sentiment,
            executor=executor,
            cfg=cfg,
            opening_compounds=_opening_compounds,
            persist_opening_compounds=_persist_opening_compounds,
        )
        # Refresh positions after sentiment exits.
        try:
            positions = _get_positions_from_adapter(adapter)
        except Exception as e:
            logger.warning(f"Position refresh after sentiment exit error: {e}")

        # Clean up orphaned opening_compounds (symbols no longer in positions).
        orphaned = [sym for sym in _opening_compounds if sym not in positions]
        if orphaned:
            for sym in orphaned:
                del _opening_compounds[sym]
            _persist_opening_compounds(_opening_compounds)
            logger.info(f"Purged {len(orphaned)} orphaned opening compounds: {orphaned}")

        # --- Exposure cap guard ---
        exposure_cap_notional = snapshot.equity * cfg.risk_limits.gross_exposure_cap_pct
        if snapshot.gross_exposure >= exposure_cap_notional or len(positions) >= cfg.risk_limits.max_open_positions:
            logger.info(
                f"Exposure cap reached: gross={snapshot.gross_exposure:.2f} >= "
                f"cap={exposure_cap_notional:.2f} OR positions={len(positions)} >= "
                f"max={cfg.risk_limits.max_open_positions}. Monitoring only."
            )
            dashboard_state.update(
                bot_state="MONITORING",
                equity=snapshot.equity,
                gross_exposure=snapshot.gross_exposure,
                num_positions=len(positions),
            )
            persist_state(dashboard_state)
            time.sleep(60)
            continue

        # --- Blackout windows ---
        if adapter.is_monday_open_blackout():
            logger.info("Monday open blackout active. Monitoring existing positions only.")
            time.sleep(60)
            continue

        if adapter.is_pre_close_blackout():
            logger.info("Pre-close blackout active. Monitoring existing positions only.")
            time.sleep(60)
            continue

        # --- Portfolio execution ---
        logger.info("Building portfolio...")
        open_orders = adapter.list_orders(status="open")
        
        # FIX ORPHAN-ORDER: Cancel all open orders before portfolio execution.
        logger.info("Canceling all open orders before portfolio execution.")
        adapter.cancel_all_orders()
        
        proposed_trades = portfolio_builder.build_portfolio(snapshot, positions, open_orders)
        logger.info(f"Portfolio builder proposed {len(proposed_trades)} trades.")

        for proposed in proposed_trades:
            order = executor.execute_proposed_trade(proposed)
            if order is not None and proposed.rejected_reason is None and proposed.qty > 0:
                # Persist opening compound (technical composite at entry).
                _opening_compounds[proposed.symbol] = proposed.signal_score
                _persist_opening_compounds(_opening_compounds)
                # TASK 5.1: Register entry with tracker.
                tracker.register_entry(proposed)

        # --- Second sentiment exit check after execution ---
        _check_and_exit_on_sentiment(
            positions=positions,
            adapter=adapter,
            sentiment_module=sentiment,
            executor=executor,
            cfg=cfg,
            opening_compounds=_opening_compounds,
            persist_opening_compounds=_persist_opening_compounds,
        )
        # Refresh positions after second sentiment exit.
        try:
            positions = _get_positions_from_adapter(adapter)
        except Exception as e:
            logger.warning(f"Position refresh after second sentiment exit error: {e}")

        # --- Update dashboard ---
        dashboard_state.update(
            bot_state="ACTIVE",
            equity=snapshot.equity,
            gross_exposure=snapshot.gross_exposure,
            num_positions=len(positions),
        )
        persist_state(dashboard_state)

        # --- Persist vol history, sentiment cache, and trade stats ---
        _persist_vol_and_sentiment(risk_engine, sentiment)
        
        # TASK 5.1: Persist tracker state.
        state = _load_equity_state()
        state["trade_stats"] = tracker.export_state()
        _save_equity_state(state)

        # --- Adaptive sentiment rescore interval ---
        max_abs_s = max((abs(float(s)) for s in _opening_compounds.values()), default=0.0)
        _rescore_interval = sentiment.adaptive_rescore_interval_hysteresis(
            max_abs_s, _rescore_interval
        )
        logger.info(
            f"Adaptive rescore interval: {_rescore_interval}s "
            f"(max_abs_sentiment={max_abs_s:.2f})"
        )

        # CHANGE CYCLE-TIME: Main loop cycle reduced from 600s to 300s.
        # Sentiment rescore interval is adaptive (120s–900s) and decoupled from main cycle.
        logger.info("Cycle complete. Sleeping 300s...")
        time.sleep(300)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C).")
        sys.exit(0)
    except SystemExit as e:
        logger.critical(f"Bot terminated by SystemExit: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {e}")
        sys.exit(1)
