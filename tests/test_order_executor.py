"""
Tests for execution/order_executor.py — OrderExecutor, _check_and_exit_on_sentiment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import MagicMock, patch, call

from execution.order_executor import OrderExecutor, _check_and_exit_on_sentiment
from core.risk_engine import PositionInfo, ProposedTrade
from core.sentiment import SentimentResult
from config.config import ExecutionConfig, BotConfig, RiskLimits, SentimentConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_executor(live=True, env_mode="LIVE", cfg=None):
    adapter = MagicMock()
    cfg = cfg or ExecutionConfig()
    return OrderExecutor(adapter, env_mode, live, cfg)


def _make_proposed(symbol="AAPL", side="buy", qty=10.0, entry_price=150.0,
                   stop_price=145.0, take_profit_price=160.0,
                   rejected_reason=None, signal_score=0.5):
    return ProposedTrade(
        symbol=symbol, side=side, qty=qty,
        entry_price=entry_price, stop_price=stop_price,
        take_profit_price=take_profit_price,
        risk_amount=500.0, risk_pct_of_equity=0.005,
        sentiment_score=0.5, sentiment_scale=0.8,
        signal_score=signal_score, rationale="test",
        rejected_reason=rejected_reason,
    )


def _make_position(symbol="AAPL", qty=10.0, side="long", market_price=155.0):
    return PositionInfo(
        symbol=symbol, qty=qty, market_price=market_price,
        side=side, notional=qty * market_price,
        opening_compound=0.5, avg_entry_price=150.0,
    )


def _make_sentiment(score=0.5, raw_discrete=1, confidence=0.7):
    return SentimentResult(
        score=score, raw_discrete=raw_discrete, rawcompound=score,
        ndocuments=3, explanation="test", confidence=confidence,
        raw_model_score=score,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  EXECUTE PROPOSED TRADE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestExecuteProposedTrade:

    def test_paper_mode_skips_submission(self):
        """Paper mode should log but not submit orders."""
        executor = _make_executor(live=False)
        proposed = _make_proposed()
        result = executor.execute_proposed_trade(proposed)
        assert result is None
        executor.adapter.submit_bracket_order.assert_not_called()

    def test_rejected_trade_returns_none(self):
        executor = _make_executor()
        proposed = _make_proposed(rejected_reason="Too risky")
        result = executor.execute_proposed_trade(proposed)
        assert result is None

    def test_zero_qty_returns_none(self):
        executor = _make_executor()
        proposed = _make_proposed(qty=0.0)
        result = executor.execute_proposed_trade(proposed)
        assert result is None

    def test_bracket_order_submitted(self):
        """Valid trade with stop/TP should submit bracket order."""
        executor = _make_executor()
        executor.adapter.submit_bracket_order.return_value = MagicMock(id="order1")
        proposed = _make_proposed()
        result = executor.execute_proposed_trade(proposed)
        assert result is not None
        executor.adapter.submit_bracket_order.assert_called_once()
        call_kwargs = executor.adapter.submit_bracket_order.call_args
        assert call_kwargs[1]["symbol"] == "AAPL" or call_kwargs.kwargs["symbol"] == "AAPL"

    def test_cancels_existing_orders_before_entry(self):
        executor = _make_executor()
        executor.adapter.list_orders.return_value = []
        executor.adapter.submit_bracket_order.return_value = MagicMock()
        proposed = _make_proposed()
        executor.execute_proposed_trade(proposed)
        executor.adapter.list_orders.assert_called()

    @patch("execution.order_executor.log_proposed_trade")
    def test_trailing_stop_fallback(self, mock_log):
        """When no fixed bracket, should submit market + trailing stop."""
        cfg = ExecutionConfig(enable_take_profit=False, enable_trailing_stop=True)
        executor = _make_executor(cfg=cfg)
        proposed = _make_proposed(stop_price=0.0, take_profit_price=0.0)
        executor.adapter.submit_market_order.return_value = MagicMock(id="o1")
        executor.adapter.get_position.return_value = MagicMock(qty=10)
        executor.execute_proposed_trade(proposed)
        executor.adapter.submit_market_order.assert_called()

    def test_broker_error_returns_none(self):
        executor = _make_executor()
        executor.adapter.submit_bracket_order.side_effect = Exception("broker error")
        proposed = _make_proposed()
        result = executor.execute_proposed_trade(proposed)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
#  WAIT FOR FLAT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestWaitForFlat:

    def test_immediate_flat(self):
        executor = _make_executor()
        executor.adapter.get_position.return_value = None
        assert executor._wait_for_flat("AAPL") is True

    def test_flat_after_delay(self):
        executor = _make_executor()
        executor.adapter.get_position.side_effect = [
            MagicMock(qty="10"),  # Still open
            None,  # Now flat
        ]
        assert executor._wait_for_flat("AAPL") is True

    def test_timeout_not_flat(self):
        cfg = ExecutionConfig(post_entry_fill_poll_timeout_sec=2,
                              post_entry_fill_poll_interval_sec=0.5)
        executor = _make_executor(cfg=cfg)
        executor.adapter.get_position.return_value = MagicMock(qty="10")
        assert executor._wait_for_flat("AAPL") is False


# ═══════════════════════════════════════════════════════════════════════════
#  CLOSE POSITION DUE TO SENTIMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestClosePositionDueToSentiment:

    def test_paper_mode_skips_close(self):
        executor = _make_executor(live=False)
        pos = _make_position()
        sentiment = _make_sentiment(score=-0.5)
        compounds = {"AAPL": 0.5}
        executor.close_position_due_to_sentiment(
            position=pos, sentiment=sentiment, reason="soft_exit",
            env_mode="PAPER", opening_compounds=compounds,
            persist_opening_compounds=MagicMock(),
        )
        executor.adapter.submit_market_order.assert_not_called()

    def test_successful_close_purges_compounds(self):
        executor = _make_executor()
        executor.adapter.get_position.return_value = None  # flat immediately
        executor.adapter.list_orders.return_value = []
        pos = _make_position()
        sentiment = _make_sentiment(score=-0.5)
        compounds = {"AAPL": 0.5}
        persist = MagicMock()
        executor.close_position_due_to_sentiment(
            position=pos, sentiment=sentiment, reason="hard_exit_chaos",
            env_mode="LIVE", opening_compounds=compounds,
            persist_opening_compounds=persist,
        )
        assert "AAPL" not in compounds
        persist.assert_called()

    def test_failed_close_retains_compounds(self):
        executor = _make_executor()
        executor.adapter.submit_market_order.side_effect = Exception("fail")
        executor.adapter.list_orders.return_value = []
        pos = _make_position()
        sentiment = _make_sentiment()
        compounds = {"AAPL": 0.5}
        persist = MagicMock()
        executor.close_position_due_to_sentiment(
            position=pos, sentiment=sentiment, reason="soft_exit",
            env_mode="LIVE", opening_compounds=compounds,
            persist_opening_compounds=persist,
        )
        assert "AAPL" in compounds

    def test_not_flat_retains_compounds(self):
        cfg = ExecutionConfig(post_entry_fill_poll_timeout_sec=1,
                              post_entry_fill_poll_interval_sec=0.3)
        executor = _make_executor(cfg=cfg)
        executor.adapter.get_position.return_value = MagicMock(qty="10")  # never flat
        executor.adapter.list_orders.return_value = []
        pos = _make_position()
        sentiment = _make_sentiment()
        compounds = {"AAPL": 0.5}
        persist = MagicMock()
        executor.close_position_due_to_sentiment(
            position=pos, sentiment=sentiment, reason="soft_exit",
            env_mode="LIVE", opening_compounds=compounds,
            persist_opening_compounds=persist,
        )
        assert "AAPL" in compounds


# ═══════════════════════════════════════════════════════════════════════════
#  WEEKEND CLOSE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCloseAllPositionsForWeekend:

    def test_paper_mode_blocks(self):
        executor = _make_executor(live=False)
        positions = {"AAPL": _make_position()}
        executor.close_all_positions_for_weekend(
            positions, "PAPER", {}, MagicMock(),
        )
        executor.adapter.close_all_positions.assert_not_called()

    def test_non_live_env_blocks(self):
        executor = _make_executor(live=True, env_mode="PAPER")
        positions = {"AAPL": _make_position()}
        executor.close_all_positions_for_weekend(
            positions, "PAPER", {}, MagicMock(),
        )
        executor.adapter.close_all_positions.assert_not_called()

    def test_live_mode_closes(self):
        executor = _make_executor(live=True, env_mode="LIVE")
        executor.adapter.get_position.return_value = None  # flat
        positions = {"AAPL": _make_position()}
        compounds = {"AAPL": 0.5}
        persist = MagicMock()
        executor.close_all_positions_for_weekend(
            positions, "LIVE", compounds, persist,
        )
        executor.adapter.cancel_all_orders.assert_called()
        executor.adapter.close_all_positions.assert_called()

    def test_b2_fix_waits_for_flat_before_purge(self):
        """B2 FIX: Should wait for each position to confirm flat."""
        executor = _make_executor(live=True, env_mode="LIVE")
        # AAPL goes flat, MSFT doesn't
        def get_pos_side_effect(symbol):
            if symbol == "AAPL":
                return None  # flat
            return MagicMock(qty="10")  # not flat

        executor.adapter.get_position.side_effect = get_pos_side_effect
        positions = {
            "AAPL": _make_position("AAPL"),
            "MSFT": _make_position("MSFT"),
        }
        compounds = {"AAPL": 0.5, "MSFT": 0.3}
        persist = MagicMock()
        executor.close_all_positions_for_weekend(
            positions, "LIVE", compounds, persist,
        )
        # AAPL should be purged, MSFT retained
        assert "AAPL" not in compounds
        assert "MSFT" in compounds


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK AND EXIT ON SENTIMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckAndExitOnSentiment:

    def _make_cfg(self):
        from config.config import (
            TechnicalSignalConfig, ExecutionConfig, PortfolioConfig,
            InstrumentMeta,
        )
        return BotConfig(
            env_mode="LIVE",
            live_trading_enabled=True,
            risk_limits=RiskLimits(),
            sentiment=SentimentConfig(),
            technical=TechnicalSignalConfig(),
            execution=ExecutionConfig(),
            instruments={},
            portfolio=PortfolioConfig(),
        )

    def test_hard_exit_on_chaos(self):
        """raw_discrete == -2 should trigger hard exit."""
        adapter = MagicMock()
        sentiment_module = MagicMock()
        executor = MagicMock()
        cfg = self._make_cfg()

        chaos = SentimentResult(
            score=-1.0, raw_discrete=-2, rawcompound=-1.0,
            ndocuments=3, confidence=0.9, raw_model_score=-2.0,
        )
        adapter.get_news.return_value = [{"headline": "Test"}]
        sentiment_module.force_rescore.return_value = chaos

        positions = {"AAPL": _make_position()}
        compounds = {"AAPL": 0.5}

        _check_and_exit_on_sentiment(
            positions=positions, adapter=adapter,
            sentiment_module=sentiment_module, executor=executor,
            cfg=cfg, opening_compounds=compounds,
            persist_opening_compounds=MagicMock(),
        )
        executor.close_position_due_to_sentiment.assert_called_once()
        call_kwargs = executor.close_position_due_to_sentiment.call_args
        assert call_kwargs.kwargs.get("reason") == "hard_exit_chaos" or \
               (call_kwargs[1] if len(call_kwargs) > 1 else {}).get("reason") == "hard_exit_chaos"

    def test_no_exit_when_delta_small(self):
        """Small delta should not trigger any exit."""
        adapter = MagicMock()
        sentiment_module = MagicMock()
        executor = MagicMock()
        cfg = self._make_cfg()

        # opening_compound=0.5, current_score=0.4, delta=0.1 < 0.6
        current = SentimentResult(
            score=0.4, raw_discrete=1, rawcompound=0.4,
            ndocuments=3, confidence=0.7, raw_model_score=0.4,
        )
        adapter.get_news.return_value = [{"headline": "Test"}]
        sentiment_module.force_rescore.return_value = current

        positions = {"AAPL": _make_position()}
        compounds = {"AAPL": 0.5}

        _check_and_exit_on_sentiment(
            positions=positions, adapter=adapter,
            sentiment_module=sentiment_module, executor=executor,
            cfg=cfg, opening_compounds=compounds,
            persist_opening_compounds=MagicMock(),
        )
        executor.close_position_due_to_sentiment.assert_not_called()

    def test_soft_exit_on_large_delta(self):
        """Delta > soft_exit_delta_threshold (0.6) with sufficient confidence → soft exit."""
        adapter = MagicMock()
        sentiment_module = MagicMock()
        executor = MagicMock()
        cfg = self._make_cfg()

        # opening_compound=0.5, current_score=-0.2, delta=0.7 > 0.6
        current = SentimentResult(
            score=-0.2, raw_discrete=-1, rawcompound=-0.2,
            ndocuments=3, confidence=0.6, raw_model_score=-0.2,
        )
        adapter.get_news.return_value = [{"headline": "Bad news"}]
        sentiment_module.force_rescore.return_value = current

        positions = {"AAPL": _make_position()}
        compounds = {"AAPL": 0.5}

        _check_and_exit_on_sentiment(
            positions=positions, adapter=adapter,
            sentiment_module=sentiment_module, executor=executor,
            cfg=cfg, opening_compounds=compounds,
            persist_opening_compounds=MagicMock(),
        )
        executor.close_position_due_to_sentiment.assert_called_once()

    def test_h4_fix_adopts_current_when_missing(self):
        """H4 FIX: When opening_compound is missing, adopt current sentiment."""
        adapter = MagicMock()
        sentiment_module = MagicMock()
        executor = MagicMock()
        cfg = self._make_cfg()

        current = SentimentResult(
            score=0.3, raw_discrete=1, rawcompound=0.3,
            ndocuments=3, confidence=0.5, raw_model_score=0.3,
        )
        adapter.get_news.return_value = [{"headline": "Test"}]
        sentiment_module.force_rescore.return_value = current

        positions = {"AAPL": _make_position()}
        compounds = {}  # Missing!
        persist = MagicMock()

        _check_and_exit_on_sentiment(
            positions=positions, adapter=adapter,
            sentiment_module=sentiment_module, executor=executor,
            cfg=cfg, opening_compounds=compounds,
            persist_opening_compounds=persist,
        )
        # Should have adopted current sentiment as baseline
        assert compounds.get("AAPL") == pytest.approx(0.3)
        persist.assert_called()
        # Delta should be 0 → no exit
        executor.close_position_due_to_sentiment.assert_not_called()

    def test_short_position_delta_inversion(self):
        """For short positions, delta = current - opening (inverted)."""
        adapter = MagicMock()
        sentiment_module = MagicMock()
        executor = MagicMock()
        cfg = self._make_cfg()

        # Short position: opening=-0.5, current=0.3
        # delta = current - opening = 0.3 - (-0.5) = 0.8 > 0.6
        current = SentimentResult(
            score=0.3, raw_discrete=1, rawcompound=0.3,
            ndocuments=3, confidence=0.6, raw_model_score=0.3,
        )
        adapter.get_news.return_value = [{"headline": "Test"}]
        sentiment_module.force_rescore.return_value = current

        pos = _make_position(side="short")
        positions = {"AAPL": pos}
        compounds = {"AAPL": -0.5}

        _check_and_exit_on_sentiment(
            positions=positions, adapter=adapter,
            sentiment_module=sentiment_module, executor=executor,
            cfg=cfg, opening_compounds=compounds,
            persist_opening_compounds=MagicMock(),
        )
        executor.close_position_due_to_sentiment.assert_called_once()
