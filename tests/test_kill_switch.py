"""
Tests for monitoring/kill_switch.py — KillSwitch halt logic.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from monitoring.kill_switch import KillSwitch, KillSwitchState
from core.risk_engine import EquitySnapshot
from config.config import RiskLimits


def _make_snapshot(daily_loss_pct=0.0, drawdown_pct=0.0, equity=100000.0):
    return EquitySnapshot(
        equity=equity, cash=equity * 0.5, portfolio_value=equity,
        day_trading_buying_power=equity * 4,
        start_of_day_equity=equity, high_watermark_equity=equity,
        realized_pl_today=0.0, unrealized_pl=0.0,
        gross_exposure=20000.0,
        daily_loss_pct=daily_loss_pct, drawdown_pct=drawdown_pct,
    )


class TestKillSwitch:

    def test_no_halt_healthy_state(self):
        """Normal conditions → no halt."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(daily_loss_pct=-0.01, drawdown_pct=-0.02)
        state = ks.check(snap)
        assert not state.halted
        assert state.reason == ""

    def test_halt_on_daily_loss_breach(self):
        """Daily loss > 4% → halt."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(daily_loss_pct=-0.05)
        state = ks.check(snap)
        assert state.halted
        assert "daily loss" in state.reason.lower()

    def test_halt_on_exact_daily_loss_boundary(self):
        """Exactly at -4% daily loss → halt."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(daily_loss_pct=-0.04)
        state = ks.check(snap)
        assert state.halted

    def test_no_halt_just_below_daily_loss(self):
        """-3.99% daily loss → no halt."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(daily_loss_pct=-0.0399)
        state = ks.check(snap)
        assert not state.halted

    def test_halt_on_drawdown_breach(self):
        """Drawdown > 9% → halt."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(drawdown_pct=-0.10)
        state = ks.check(snap)
        assert state.halted
        assert "drawdown" in state.reason.lower()

    def test_halt_on_exact_drawdown_boundary(self):
        """Exactly at -9% drawdown → halt."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(drawdown_pct=-0.09)
        state = ks.check(snap)
        assert state.halted

    def test_no_halt_just_below_drawdown(self):
        """-8.99% drawdown → no halt."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(drawdown_pct=-0.0899)
        state = ks.check(snap)
        assert not state.halted

    def test_daily_loss_checked_before_drawdown(self):
        """When both conditions are met, daily loss should trigger first."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(daily_loss_pct=-0.05, drawdown_pct=-0.10)
        state = ks.check(snap)
        assert state.halted
        assert "daily loss" in state.reason.lower()

    def test_state_carries_metrics(self):
        """KillSwitchState should carry daily_loss_pct and drawdown_pct."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(daily_loss_pct=-0.02, drawdown_pct=-0.05)
        state = ks.check(snap)
        assert state.daily_loss_pct == pytest.approx(-0.02)
        assert state.drawdown_pct == pytest.approx(-0.05)

    def test_custom_limits(self):
        """Custom risk limits should be respected."""
        custom = RiskLimits(daily_loss_limit_pct=0.02, max_drawdown_pct=0.05)
        ks = KillSwitch(custom)
        # -2.5% exceeds 2% custom limit
        snap = _make_snapshot(daily_loss_pct=-0.025)
        state = ks.check(snap)
        assert state.halted

    def test_zero_loss_not_halted(self):
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(daily_loss_pct=0.0, drawdown_pct=0.0)
        state = ks.check(snap)
        assert not state.halted

    def test_positive_pnl_not_halted(self):
        """Positive P&L should never trigger halt."""
        ks = KillSwitch(RiskLimits())
        snap = _make_snapshot(daily_loss_pct=0.05, drawdown_pct=0.0)
        state = ks.check(snap)
        assert not state.halted
