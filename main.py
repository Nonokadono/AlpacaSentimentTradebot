# CHANGES:
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
# OPENING-COMPOSITE-FIX — Modified _opening_compounds entry to save proposed.signal_score
#                         instead of proposed.sentiment_score to check sentiment deterioration
#                         against the technical composite used to open the position.

import json
import logging
import os
import tempfile
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

# zoneinfo is stdlib in Python 3.9+.  The backports fallback covers Python 3.8;
# the # type: ignore[import] suppresses the Pylance/mypy "unresolved import"
# warning that fires on 3.9+ environments where backports.zoneinfo is not
# installed — the except branch is simply never reached there at runtime.
try:
    from zoneinfo import ZoneInfo                           # Python 3.9+ stdlib
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[import]  # Python 3.8

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

logger = logging.getLogger("tradebot")

EQUITY_STATE_PATH = Path("equity_state.json")

# US Eastern timezone constant — used to derive today_str regardless of the
# server's local timezone, ensuring start_of_day_equity resets at 00:00 ET.
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
    """Write equity state atomically via temp file + os.replace().

    The write target is a sibling temp file inside EQUITY_STATE_PATH.parent so
    that os.replace() is guaranteed to stay on the same filesystem (rename(2)
    is only atomic within one filesystem).  equity_state.json is never opened
    for writing and therefore remains uncorrupted under any failure mode that
    occurs before os.replace() completes.
    """
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
                os.fsync(f.fileno())   # flush OS page cache → persistent storage
        except Exception:
            os.unlink(tmp_path)       # clean up orphaned temp file on write failure
            raise
        os.replace(tmp_path, EQUITY_STATE_PATH)  # atomic on POSIX; near-atomic on Win
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


# ── PERSIST-FIX: Combined vol-history + sentiment-cache persistence ───────────

def _persist_vol_and_sentiment(
    risk_engine: RiskEngine,
    sentiment_module: SentimentModule,
) -> None:
    """Persist vol_history and sentiment_cache into equity_state.json.

    Single read-modify-write: both keys are updated atomically in one
    _save_equity_state() call to minimise I/O and avoid a window where one
    key is written but the other is not (partial persistence).

    Called once per full loop iteration, immediately before time.sleep().
    Early-continue paths (kill-switch, weekend close, exposure cap) correctly
    skip this call because no new vol samples or sentiment entries are produced
    on those paths.
    """
    state = _load_equity_state()
    state["vol_history"]      = risk_engine.export_vol_history()
    state["sentiment_cache"]  = sentiment_module.export_cache()
    _save_equity_state(state)

# ── END PERSIST-FIX ──────────────────────────────────────────────────────────


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

    # TIMEZONE FIX: derive today_str in US Eastern time so that the daily
    # reset fires at 00:00 ET on every server regardless of its local timezone.
    # date.today() (old code) used the server's local clock — a CET host rolled
    # over at 17:00 ET (30 min before market close), resetting start_of_day_equity
    # mid-session and silently blinding the daily_loss_limit_pct guard.
    today_str = datetime.now(tz=ET).date().isoformat()

    last_day = state.get("last_trading_day")
    start_of_day_equity = float(state.get("start_of_day_equity", equity))
    high_watermark_equity = float(state.get("high_watermark_equity", equity))

    # WRONG-3 FIX: only start_of_day_equity resets on a new calendar day.
    # high_watermark_equity is intentionally NOT reset here — doing so would
    # restart the max-drawdown clock every morning, turning a portfolio-level
    # protection into an intraday-only guard.
    #
    # Detailed impact example:
    #   Day 1 peak : equity=$11,000 → watermark persisted as $11,000
    #   Day 2 open : equity=$10,200
    #     BEFORE fix → watermark clobbered to $10,200; drawdown = 0.00%
    #     AFTER  fix → watermark stays at $11,000;     drawdown = −7.27%
    #   Day 2 intraday: equity=$9,800
    #     BEFORE fix → drawdown = −3.92%  → kill switch silent  (dangerous!)
    #     AFTER  fix → drawdown = −10.91% → kill switch fires   (correct)
    if last_day != today_str:
        start_of_day_equity = equity
        state["last_trading_day"] = today_str
    # high_watermark_equity is NOT reset here — it is only ever raised.

    # Legitimate new all-time peak — monotonically update the watermark.
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


# ── Main loop ────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    cfg = load_config()
    setup_logging(cfg.env_mode)
    log_environment_switch(cfg.env_mode, user="manual_start")

    # OPERATOR ACTION REQUIRED (OA-1):
    # config/instrument_whitelist.yaml — all non-TECH symbols are commented out.
    # The portfolio is structurally 100% mega-cap tech. Before going live,
    # uncomment at least one ETF_INDEX (SPY or QQQ) and one non-TECH equity
    # (BAC or COST) to achieve meaningful sector diversification. Without this,
    # risk parameters calibrated for 15 diverse positions are applied to 3
    # correlated tech names.

    adapter           = AlpacaAdapter(cfg.env_mode)
    sentiment         = SentimentModule()
    signal_engine     = SignalEngine(adapter, sentiment, cfg.technical)
    risk_engine       = RiskEngine(cfg.risk_limits, cfg.sentiment, cfg.instruments)
    pm                = PositionManager(adapter)
    executor          = OrderExecutor(adapter, cfg.env_mode, cfg.live_trading_enabled, cfg.execution)
    kill_switch       = KillSwitch(cfg.risk_limits)
    portfolio_builder = PortfolioBuilder(cfg, adapter, sentiment, signal_engine, risk_engine)

    # Registry: symbol -> sentiment_score recorded at entry time.
    # OPENING-COMPOSITE-FIX: We are now saving proposed.signal_score instead of
    # proposed.sentiment_score to check sentiment deterioration against the technical composite.
    # Persisted to equity_state.json so it survives bot restarts.
    _opening_compounds: Dict[str, float] = _load_opening_compounds()

    # ── PERSIST-FIX: Restore vol history and sentiment cache from disk ────────
    # Both import methods are fully safe: missing keys fall back to empty dicts,
    # and malformed entries are silently skipped inside each method.
    # import_vol_history() called before any trades so _kelly_fraction can use
    # the restored per-symbol deques immediately on the first loop iteration.
    # import_cache() called before the main loop so get_cached_sentiment() hits
    # the warm cache on the very first sentiment check — no API burst on restart.
    _startup_state = _load_equity_state()
    risk_engine.import_vol_history(_startup_state.get("vol_history", {}))
    sentiment.import_cache(_startup_state.get("sentiment_cache", {}))
    logger.info(
        "Startup state restored: vol_history symbols=%d  sentiment_cache symbols=%d",
        len(risk_engine._vol_history),
        len(sentiment._cache),
    )
    # ── END PERSIST-FIX ──────────────────────────────────────────────────────

    # ── GAP-1 FIX: Startup reconciliation ───────────────────────────────────
    # After loading from disk, validate the registry against LIVE Alpaca
    # positions. While the bot was offline, broker bracket-orders or trailing-
    # stops may have closed positions. Those symbols must be purged NOW —
    # before the main loop begins — so no stale baseline contaminates the
    # delta check on a future re-entry of the same symbol.
    #
    # Scenario example:
    #   disk:  {"AAPL": 0.65, "NVDA": 0.30}
    #   live:  {"AAPL": PositionInfo(...)}         ← NVDA was bracket-stopped
    #
    #   stale_syms = ["NVDA"]
    #   → del _opening_compounds["NVDA"]
    #   → _opening_compounds is now {"AAPL": 0.65}
    #   → _persist_opening_compounds() called exactly once (only if stale found)
    # FIX 4C: Deleted acct_startup assignment; it was never used after this block.
    positions_at_start = pm.get_positions(opening_compounds=_opening_compounds)
    stale_syms = [s for s in list(_opening_compounds) if s not in positions_at_start]
    if stale_syms:
        logger.info(
            f"Startup reconcile: purging stale opening_compounds for {stale_syms}"
        )
        for s in stale_syms:
            del _opening_compounds[s]
        _persist_opening_compounds(_opening_compounds)
    # ── END GAP-1 FIX ───────────────────────────────────────────────────────

    # Improvement D: initialise adaptive sleep interval before the loop.
    # Starts at 600s (the neutral-band default); hysteresis prevents oscillation.
    _rescore_interval: int = 600

    # Change D2: Launch the PyQt5 desktop GUI in a separate daemon thread.
    # When dev_mode is True or PyQt5 is missing, launch_dashboard() returns immediately.
    launch_dashboard(refresh_seconds=5.0)

    # Change D4: Cycle counter for the GUI status bar.
    _cycle_count: int = 0

    while True:
        _cycle_count += 1

        acct        = adapter.get_account()
        positions   = pm.get_positions(opening_compounds=_opening_compounds)
        snapshot    = get_equity_snapshot_from_account(acct, positions)
        market_open = adapter.get_market_open()
        log_equity_snapshot(snapshot, market_open=market_open)

        # Change D3: Push account + market state into the dashboard.
        dashboard_state.update(
            cash=snapshot.cash,
            buying_power=snapshot.day_trading_buying_power,
            cycle_count=_cycle_count,
            market_open=market_open,
            bot_state="SCANNING",
        )
        persist_state(dashboard_state)

        # ── STEP 0: WEEKEND FORCED LIQUIDATION ──────────────────────────────
        if adapter.is_pre_weekend_close():
            logger.warning(
                "WEEKEND CLOSE DETECTED: Closing all positions and sleeping "
                "until next market open."
            )
            dashboard_state.update(bot_state="EXECUTING")
            persist_state(dashboard_state)
            # PURGE-FIX: Pass opening_compounds and _persist_opening_compounds
            # so weekend liquidation can purge baselines inline with closes.
            executor.close_all_positions_for_weekend(
                positions,
                cfg.env_mode,
                opening_compounds=_opening_compounds,
                persist_opening_compounds=_persist_opening_compounds,
            )
            dashboard_state.update(bot_state="IDLE")
            persist_state(dashboard_state)
            # Park the bot over the weekend — poll every 60s until Monday open.
            while not adapter.get_market_open():
                time.sleep(60)
            continue  # Re-enter the loop from a clean state on Monday open.

        ks_state = kill_switch.check(snapshot)
        log_kill_switch_state(ks_state)
        if ks_state.halted:
            dashboard_state.update(bot_state="HALTED")
            persist_state(dashboard_state)
            time.sleep(60)
            continue

        # ── STEP 1: SENTIMENT CHECK ON ALL OPEN POSITIONS ───────────────────
        if positions:
            # PURGE-FIX: Pass opening_compounds and _persist_opening_compounds
            # so sentiment exits can purge baselines inline with closes.
            _check_and_exit_on_sentiment(
                positions=positions,
                adapter=adapter,
                sentiment_module=sentiment,
                executor=executor,
                cfg=cfg,
                opening_compounds=_opening_compounds,
                persist_opening_compounds=_persist_opening_compounds,
            )
            # Refresh positions after potential closes.
            positions = pm.get_positions(opening_compounds=_opening_compounds)
            snapshot  = get_equity_snapshot_from_account(acct, positions)

        # ── GAP-2 FIX: Purge moved OUTSIDE `if positions:` ──────────────────
        # Previously this block was indented inside `if positions:`, meaning it
        # was skipped entirely when no positions were open. Stale entries from a
        # prior session (or from the GAP-1 scenario where all positions were
        # closed overnight) would then persist in _opening_compounds indefinitely.
        #
        # By moving the purge here it runs unconditionally on every loop cycle,
        # ensuring the registry stays accurate even when positions is empty {}.
        #
        # PURGE-FIX NOTE: This loop-level purge now acts as a safety net only.
        # The primary purge happens inline inside close_position_due_to_sentiment(),
        # so this block should typically find nothing to delete. It remains here
        # as a defensive guard against any edge cases where a position closes via
        # an external path (e.g. manual liquidation, broker-side stop execution).
        for sym in list(_opening_compounds.keys()):
            if sym not in positions:
                del _opening_compounds[sym]
        _persist_opening_compounds(_opening_compounds)
        # ── END GAP-2 FIX ───────────────────────────────────────────────────

        # ── STEP 2: EXPOSURE / POSITION-COUNT GUARD ─────────────────────────
        exposure_cap_notional = snapshot.equity * cfg.risk_limits.gross_exposure_cap_pct
        if (
            snapshot.gross_exposure >= exposure_cap_notional
            or len(positions) >= cfg.risk_limits.max_open_positions
        ):
            # Update GUI with current positions/prices before sleeping
            open_orders_ui = adapter.list_orders(status="open")
            dashboard_state.update(
                positions=build_position_rows(positions, open_orders_ui),
                prices=build_price_rows(cfg.instruments, adapter),
                bot_state="IDLE",
            )
            persist_state(dashboard_state)
            time.sleep(60)
            continue

        if not market_open:
            logger.info("MARKET CLOSED: Skipping trading logic.")
            time.sleep(60)
            continue


        # ── STEP 3: PRE-CLOSE + MONDAY-OPEN ENTRY BLACKOUTS + BUILD AND EXECUTE NEW TRADES ─
        # MONDAY-BLACKOUT: Monday open blackout takes precedence over pre-close blackout
        # because both can theoretically be true simultaneously (rare but possible if
        # Monday is a short session). The Monday blackout message is more specific.
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
        else:
            dashboard_state.update(bot_state="SCANNING")
            persist_state(dashboard_state)
            
            # ORPHAN-ORDER-FIX: Cancel all open orders before building new portfolio.
            # This eliminates orphaned orders from previous iterations that may have
            # become stale or conflicting. Clean slate ensures execution integrity.
            logger.info("Canceling all open orders before portfolio execution.")
            adapter.cancel_all_orders()
            
            open_orders     = adapter.list_orders(status="open")
            proposed_trades = portfolio_builder.build_portfolio(snapshot, positions, open_orders)

            log_portfolio_overview(proposed_trades, cfg.env_mode)

            dashboard_state.update(bot_state="EXECUTING")
            persist_state(dashboard_state)
            for proposed in proposed_trades:
                order = executor.execute_proposed_trade(proposed)
                # FILL-CONFIRM FIX invariant — `order is not None` is a safe
                # proxy for "fill confirmed (or confirmation not required)":
                #
                #   Bracket path:     atomic OCO submitted; broker activates
                #                     TP/stop on fill automatically. order ≠ None
                #                     ↔ the API call succeeded.
                #   Plain market:     no exit leg; order ≠ None ↔ submitted OK.
                #   Trailing-stop:    order_executor returns None when
                #                     _wait_for_position() times out, so
                #                     order ≠ None ↔ fill confirmed by polling.
                #   Errors / paper:   all code paths return None already.
                #
                # Therefore recording _opening_compounds only when order is not
                # None guarantees we never store a sentiment baseline for a
                # position whose fill we could not confirm.
                if order is not None and proposed.rejected_reason is None and proposed.qty > 0:
                    # OPENING-COMPOSITE-FIX: record entry-time technical composite (proposed.signal_score)
                    # as the opening compound baseline — NOT proposed.sentiment_score.
                    _opening_compounds[proposed.symbol] = proposed.signal_score
                    # FIX 4A: Persist immediately so a crash/restart doesn't lose the entry record.
                    _persist_opening_compounds(_opening_compounds)

        # FIX 4B: Refresh positions immediately before the end-of-loop sentiment check
        # so newly entered positions are included in the current cycle's exit evaluation.
        positions = pm.get_positions(opening_compounds=_opening_compounds)

        # Sentiment-exit check runs unconditionally — blackouts do NOT affect exits.
        # PURGE-FIX: Pass opening_compounds and _persist_opening_compounds
        # so sentiment exits can purge baselines inline with closes.
        _check_and_exit_on_sentiment(
            positions=positions,
            adapter=adapter,
            sentiment_module=sentiment,
            executor=executor,
            cfg=cfg,
            opening_compounds=_opening_compounds,
            persist_opening_compounds=_persist_opening_compounds,
        )

        # Change D3: Refresh positions + prices in dashboard after all executions.
        open_orders_final = adapter.list_orders(status="open")
        positions_refreshed = pm.get_positions(opening_compounds=_opening_compounds)
        dashboard_state.update(
            positions=build_position_rows(positions_refreshed, open_orders_final),
            prices=build_price_rows(cfg.instruments, adapter),
            bot_state="IDLE",
        )
        persist_state(dashboard_state)

        # ── PERSIST-FIX: Save vol_history and sentiment_cache ─────────────────
        # Called here — after all trade execution and position refreshes are done
        # for this iteration — so the persisted state always reflects the freshest
        # vol samples (appended inside _kelly_fraction during pre_trade_checks) and
        # the freshest sentiment entries (updated inside _call_ai / force_rescore).
        # One atomic read-modify-write; see _persist_vol_and_sentiment() docstring.
        _persist_vol_and_sentiment(risk_engine, sentiment)
        # ── END PERSIST-FIX ───────────────────────────────────────────────────

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
