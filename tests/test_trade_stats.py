"""
Tests for monitoring/trade_stats.py — TradeStatsTracker lifecycle tracking.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from monitoring.trade_stats import (
    TradeStatsTracker, TradeStat, ActiveTrade, SummaryStats,
    _pnl, _return_pct, _best_price_for_side, _worst_price_for_side,
    _infer_stop_hit, _infer_take_profit_hit, _better_price, _worse_price,
)


# ═══════════════════════════════════════════════════════════════════════════
#  PNL & RETURN HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class TestPnlHelpers:

    def test_long_profit(self):
        assert _pnl("long", 10.0, 100.0, 110.0) == pytest.approx(100.0)

    def test_long_loss(self):
        assert _pnl("long", 10.0, 100.0, 90.0) == pytest.approx(-100.0)

    def test_short_profit(self):
        assert _pnl("short", 10.0, 100.0, 90.0) == pytest.approx(100.0)

    def test_short_loss(self):
        assert _pnl("short", 10.0, 100.0, 110.0) == pytest.approx(-100.0)

    def test_return_pct_long(self):
        assert _return_pct("long", 100.0, 110.0) == pytest.approx(0.1)

    def test_return_pct_short(self):
        assert _return_pct("short", 100.0, 90.0) == pytest.approx(0.1)

    def test_return_pct_zero_entry(self):
        assert _return_pct("long", 0.0, 100.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  PRICE HELPER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPriceHelpers:

    def test_best_price_long(self):
        assert _best_price_for_side("long", [90, 100, 110, 95]) == 110.0

    def test_best_price_short(self):
        assert _best_price_for_side("short", [90, 100, 110, 95]) == 90.0

    def test_worst_price_long(self):
        assert _worst_price_for_side("long", [90, 100, 110, 95]) == 90.0

    def test_worst_price_short(self):
        assert _worst_price_for_side("short", [90, 100, 110, 95]) == 110.0

    def test_better_price_long(self):
        assert _better_price("long", 100.0, 110.0) == 110.0
        assert _better_price("long", 100.0, 90.0) == 100.0

    def test_better_price_short(self):
        assert _better_price("short", 100.0, 90.0) == 90.0
        assert _better_price("short", 100.0, 110.0) == 100.0

    def test_worse_price_long(self):
        assert _worse_price("long", 100.0, 90.0) == 90.0

    def test_worse_price_short(self):
        assert _worse_price("short", 100.0, 110.0) == 110.0

    def test_better_price_zero_existing(self):
        assert _better_price("long", 0.0, 50.0) == 50.0

    def test_worse_price_zero_existing(self):
        assert _worse_price("long", 0.0, 50.0) == 50.0


# ═══════════════════════════════════════════════════════════════════════════
#  STOP/TP INFERENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestStopTpInference:

    def test_stop_hit_long(self):
        assert _infer_stop_hit("long", 95.0, 95.0) is True

    def test_stop_not_hit_long(self):
        assert _infer_stop_hit("long", 105.0, 95.0) is False

    def test_stop_hit_short(self):
        assert _infer_stop_hit("short", 105.0, 105.0) is True

    def test_stop_not_hit_short(self):
        assert _infer_stop_hit("short", 95.0, 105.0) is False

    def test_stop_zero_never_hit(self):
        assert _infer_stop_hit("long", 50.0, 0.0) is False

    def test_tp_hit_long(self):
        assert _infer_take_profit_hit("long", 110.0, 110.0) is True

    def test_tp_not_hit_long(self):
        assert _infer_take_profit_hit("long", 105.0, 110.0) is False

    def test_tp_hit_short(self):
        assert _infer_take_profit_hit("short", 90.0, 90.0) is True

    def test_tp_zero_never_hit(self):
        assert _infer_take_profit_hit("long", 110.0, 0.0) is False


# ═══════════════════════════════════════════════════════════════════════════
#  TRADE RECORDING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestTradeRecording:

    def test_record_long_trade(self, tmp_path):
        tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
        trade = tracker.record_trade(
            symbol="AAPL", side="long", qty=10.0,
            entry_price=150.0, exit_price=160.0,
            stop_price=145.0, take_profit_price=165.0,
            exit_reason="take_profit",
        )
        assert trade.symbol == "AAPL"
        assert trade.realized_pnl == pytest.approx(100.0)
        assert trade.realized_return_pct == pytest.approx(0.0667, abs=0.01)

    def test_record_short_trade(self, tmp_path):
        tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
        trade = tracker.record_trade(
            symbol="AAPL", side="short", qty=10.0,
            entry_price=150.0, exit_price=140.0,
            exit_reason="take_profit",
        )
        assert trade.realized_pnl == pytest.approx(100.0)

    def test_side_normalization(self, tmp_path):
        tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
        trade = tracker.record_trade(
            symbol="AAPL", side="buy", qty=10.0,
            entry_price=150.0, exit_price=160.0,
        )
        assert trade.side == "long"

    def test_invalid_side_raises(self, tmp_path):
        tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
        with pytest.raises(ValueError, match="Unsupported side"):
            tracker.record_trade(
                symbol="AAPL", side="invalid", qty=10.0,
                entry_price=150.0, exit_price=160.0,
            )

    def test_load_trades(self, tmp_path):
        tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
        tracker.record_trade(
            symbol="AAPL", side="long", qty=10.0,
            entry_price=150.0, exit_price=160.0,
        )
        tracker.record_trade(
            symbol="MSFT", side="short", qty=5.0,
            entry_price=300.0, exit_price=290.0,
        )
        trades = tracker.load_trades()
        assert len(trades) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  ACTIVE TRADE LIFECYCLE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestActiveTradeLifecycle:

    def test_register_from_proposed(self, tmp_path):
        with patch("monitoring.trade_stats.ACTIVE_TRADES_PATH", tmp_path / "active.json"):
            tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
            proposed = MagicMock()
            proposed.symbol = "AAPL"
            proposed.side = "buy"
            proposed.qty = 10.0
            proposed.entry_price = 150.0
            proposed.stop_price = 145.0
            proposed.take_profit_price = 165.0

            active = tracker.register_entry_from_proposed(proposed)
            assert active.symbol == "AAPL"
            assert active.side == "long"
            assert active.qty == 10.0

    def test_sync_open_positions_detects_closure(self, tmp_path):
        with patch("monitoring.trade_stats.ACTIVE_TRADES_PATH", tmp_path / "active.json"), \
             patch("monitoring.trade_stats.WATCHED_STOP_EXITS_PATH", tmp_path / "watched.json"):
            tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")

            # Register an active trade
            proposed = MagicMock()
            proposed.symbol = "AAPL"
            proposed.side = "buy"
            proposed.qty = 10.0
            proposed.entry_price = 150.0
            proposed.stop_price = 145.0
            proposed.take_profit_price = 165.0
            tracker.register_entry_from_proposed(proposed)

            # Sync with empty positions → should detect closure
            closed = tracker.sync_open_positions({})
            assert len(closed) == 1
            assert closed[0].symbol == "AAPL"

    def test_close_active_trade(self, tmp_path):
        with patch("monitoring.trade_stats.ACTIVE_TRADES_PATH", tmp_path / "active.json"), \
             patch("monitoring.trade_stats.WATCHED_STOP_EXITS_PATH", tmp_path / "watched.json"):
            tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")

            proposed = MagicMock()
            proposed.symbol = "AAPL"
            proposed.side = "buy"
            proposed.qty = 10.0
            proposed.entry_price = 150.0
            proposed.stop_price = 145.0
            proposed.take_profit_price = 165.0
            tracker.register_entry_from_proposed(proposed)

            result = tracker.close_active_trade(
                symbol="AAPL", exit_price=160.0,
                exit_reason="take_profit",
            )
            assert result is not None
            assert result.symbol == "AAPL"
            assert result.realized_pnl > 0

    def test_close_nonexistent_returns_none(self, tmp_path):
        with patch("monitoring.trade_stats.ACTIVE_TRADES_PATH", tmp_path / "active.json"), \
             patch("monitoring.trade_stats.WATCHED_STOP_EXITS_PATH", tmp_path / "watched.json"):
            tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
            result = tracker.close_active_trade(
                symbol="FAKE", exit_price=100.0, exit_reason="test",
            )
            assert result is None


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY STATS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSummaryStats:

    def test_empty_summary(self, tmp_path):
        tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
        summary = tracker.summary()
        assert summary.total_trades == 0
        assert summary.hit_rate == 0.0
        assert summary.profit_factor == 0.0

    def test_summary_with_trades(self, tmp_path):
        tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
        tracker.record_trade(
            symbol="AAPL", side="long", qty=10.0,
            entry_price=150.0, exit_price=160.0,
        )
        tracker.record_trade(
            symbol="MSFT", side="long", qty=5.0,
            entry_price=300.0, exit_price=290.0,
        )
        summary = tracker.summary()
        assert summary.total_trades == 2
        assert summary.winners == 1
        assert summary.losers == 1
        assert summary.hit_rate == pytest.approx(0.5)

    def test_text_report(self, tmp_path):
        tracker = TradeStatsTracker(path=tmp_path / "trades.jsonl")
        tracker.record_trade(
            symbol="AAPL", side="long", qty=10.0,
            entry_price=150.0, exit_price=160.0,
        )
        report = tracker.render_text_report()
        assert "Trade Stats Report" in report
        assert "AAPL" in report
