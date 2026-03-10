"""
Tests for main.py helper functions — equity state persistence,
opening compounds, session dates, reconciliation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import tempfile
import pytest
from datetime import datetime, timedelta, date
from unittest.mock import MagicMock, patch

# We need to patch environment variables before importing main
os.environ.setdefault("APCA_API_KEY_ID", "test_key")
os.environ.setdefault("APCA_API_SECRET_KEY", "test_secret")
os.environ.setdefault("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
os.environ.setdefault("APCA_API_ENV", "PAPER")

from core.risk_engine import PositionInfo


# ═══════════════════════════════════════════════════════════════════════════
#  EQUITY STATE PERSISTENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEquityStatePersistence:

    def test_load_empty_state(self, tmp_path):
        """Non-existent file returns empty dict."""
        with patch("main.EQUITY_STATE_PATH", tmp_path / "equity_state.json"):
            from main import _load_equity_state
            assert _load_equity_state() == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import _save_equity_state, _load_equity_state
            data = {"start_of_day_equity": 100000.0, "high_watermark_equity": 105000.0}
            _save_equity_state(data)
            loaded = _load_equity_state()
            assert loaded["start_of_day_equity"] == 100000.0
            assert loaded["high_watermark_equity"] == 105000.0

    def test_load_corrupted_state_raises(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state_path.write_text("NOT VALID JSON{{{")
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import _load_equity_state, EquityStateError
            with pytest.raises(EquityStateError):
                _load_equity_state()

    def test_load_non_dict_raises(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state_path.write_text('"just a string"')
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import _load_equity_state, EquityStateError
            with pytest.raises(EquityStateError):
                _load_equity_state()

    def test_safe_load_returns_default_on_error(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state_path.write_text("CORRUPT")
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import _safe_load_equity_state
            result = _safe_load_equity_state({"fallback": True})
            assert result == {"fallback": True}

    def test_quarantine_corrupted_file(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state_path.write_text("BAD DATA")
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import _load_equity_state, EquityStateError
            with pytest.raises(EquityStateError):
                _load_equity_state()
            # Original file should be renamed
            assert not state_path.exists()
            quarantined = list(tmp_path.glob("*.corrupt.*"))
            assert len(quarantined) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  OPENING COMPOUNDS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestOpeningCompounds:

    def test_load_empty(self, tmp_path):
        with patch("main.EQUITY_STATE_PATH", tmp_path / "equity_state.json"):
            from main import _load_opening_compounds
            result = _load_opening_compounds()
            assert result == {}

    def test_load_from_state(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state = {"opening_compounds": {"AAPL": 0.5, "MSFT": -0.3}}
        state_path.write_text(json.dumps(state))
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import _load_opening_compounds
            result = _load_opening_compounds()
            assert result["AAPL"] == pytest.approx(0.5)
            assert result["MSFT"] == pytest.approx(-0.3)

    def test_persist_and_reload(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state_path.write_text("{}")
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import _persist_opening_compounds, _load_opening_compounds
            _persist_opening_compounds({"AAPL": 0.7})
            result = _load_opening_compounds()
            assert result["AAPL"] == pytest.approx(0.7)


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION DATE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionDate:

    def test_after_market_open_uses_today(self):
        from main import _current_session_date_et
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        # 2:00 PM ET = after 9:30 AM open
        dt = datetime(2026, 3, 6, 14, 0, 0, tzinfo=ET)
        result = _current_session_date_et(dt)
        assert result == "2026-03-06"

    def test_before_market_open_uses_yesterday(self):
        from main import _current_session_date_et
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        # 8:00 AM ET = before 9:30 AM open
        dt = datetime(2026, 3, 6, 8, 0, 0, tzinfo=ET)
        result = _current_session_date_et(dt)
        assert result == "2026-03-05"

    def test_at_exact_market_open(self):
        from main import _current_session_date_et
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        # 9:30 AM ET = exactly at open
        dt = datetime(2026, 3, 6, 9, 30, 0, tzinfo=ET)
        result = _current_session_date_et(dt)
        assert result == "2026-03-06"


# ═══════════════════════════════════════════════════════════════════════════
#  RECONCILIATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestReconciliation:

    def test_purge_stale_symbols(self):
        from main import _reconcile_opening_compounds
        compounds = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.4}
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL", qty=10, market_price=150,
                side="long", notional=1500,
            ),
        }
        _reconcile_opening_compounds(compounds, positions, open_orders=[])
        assert "AAPL" in compounds
        assert "MSFT" not in compounds
        assert "GOOGL" not in compounds

    def test_preserve_with_open_orders(self):
        """Symbols with open orders should not be purged."""
        from main import _reconcile_opening_compounds
        compounds = {"AAPL": 0.5, "MSFT": 0.3}
        positions = {}  # No positions held
        open_orders = [MagicMock(symbol="MSFT")]
        _reconcile_opening_compounds(compounds, positions, open_orders=open_orders)
        assert "MSFT" in compounds  # Protected by open order
        assert "AAPL" not in compounds

    def test_no_purge_when_all_active(self):
        from main import _reconcile_opening_compounds
        compounds = {"AAPL": 0.5}
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL", qty=10, market_price=150,
                side="long", notional=1500,
            ),
        }
        _reconcile_opening_compounds(compounds, positions)
        assert "AAPL" in compounds

    def test_empty_compounds_noop(self):
        from main import _reconcile_opening_compounds
        compounds = {}
        _reconcile_opening_compounds(compounds, {})
        assert compounds == {}


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestHelperFunctions:

    def test_extract_order_symbol_object(self):
        from main import _extract_order_symbol
        order = MagicMock(symbol="AAPL")
        assert _extract_order_symbol(order) == "AAPL"

    def test_extract_order_symbol_dict(self):
        from main import _extract_order_symbol
        order = {"symbol": "MSFT"}
        assert _extract_order_symbol(order) == "MSFT"

    def test_extract_order_symbol_none(self):
        from main import _extract_order_symbol
        order = MagicMock(spec=[])
        assert _extract_order_symbol(order) is None

    def test_symbols_with_open_orders(self):
        from main import _symbols_with_open_orders
        orders = [
            MagicMock(symbol="AAPL"),
            MagicMock(symbol="MSFT"),
            MagicMock(symbol="AAPL"),
        ]
        result = _symbols_with_open_orders(orders)
        assert result == {"AAPL", "MSFT"}

    def test_symbols_with_open_orders_empty(self):
        from main import _symbols_with_open_orders
        assert _symbols_with_open_orders([]) == set()


# ═══════════════════════════════════════════════════════════════════════════
#  EQUITY SNAPSHOT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEquitySnapshot:

    def test_basic_snapshot(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state_path.write_text("{}")
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import get_equity_snapshot_from_account
            from tests.conftest import MockAccount
            acct = MockAccount()
            positions = {}
            snapshot = get_equity_snapshot_from_account(acct, positions)
            assert snapshot.equity == 100000.0
            assert snapshot.cash == 50000.0
            assert snapshot.gross_exposure == 0.0

    def test_gross_exposure_calculated(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state_path.write_text("{}")
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import get_equity_snapshot_from_account
            from tests.conftest import MockAccount
            acct = MockAccount()
            positions = {
                "AAPL": PositionInfo(
                    symbol="AAPL", qty=10, market_price=150,
                    side="long", notional=1500.0,
                ),
                "MSFT": PositionInfo(
                    symbol="MSFT", qty=5, market_price=300,
                    side="long", notional=1500.0,
                ),
            }
            snapshot = get_equity_snapshot_from_account(acct, positions)
            assert snapshot.gross_exposure == 3000.0

    def test_daily_loss_pct_calculation(self, tmp_path):
        state_path = tmp_path / "equity_state.json"
        state = {"start_of_day_equity": 100000.0, "last_trading_day": "2026-03-08"}
        state_path.write_text(json.dumps(state))
        with patch("main.EQUITY_STATE_PATH", state_path):
            from main import get_equity_snapshot_from_account, _current_session_date_et
            with patch("main._current_session_date_et", return_value="2026-03-08"):
                from tests.conftest import MockAccount
                acct = MockAccount(equity="96000.0")
                snapshot = get_equity_snapshot_from_account(acct, {})
                # (96000 - 100000) / 100000 = -0.04
                assert snapshot.daily_loss_pct == pytest.approx(-0.04)
