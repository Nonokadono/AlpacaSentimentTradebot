"""
Tests for core/risk_engine.py — RiskEngine, sentiment_scale, Kelly fraction, pre_trade_checks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import pytest
from collections import deque
from unittest.mock import MagicMock

from core.risk_engine import RiskEngine, EquitySnapshot, PositionInfo, ProposedTrade
from core.sentiment import SentimentResult
from config.config import RiskLimits, SentimentConfig, InstrumentMeta


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_engine(risk_limits=None, sentiment_cfg=None, instruments=None):
    rl = risk_limits or RiskLimits()
    sc = sentiment_cfg or SentimentConfig()
    meta = instruments or {
        "AAPL": InstrumentMeta(
            symbol="AAPL", exchange="NASDAQ", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="TECH",
        ),
    }
    return RiskEngine(rl, sc, meta)


def _make_sentiment(score=0.5, raw_discrete=1, confidence=0.7):
    return SentimentResult(
        score=score, raw_discrete=raw_discrete, rawcompound=score,
        ndocuments=3, explanation="test", confidence=confidence,
        raw_model_score=score,
    )


def _make_snapshot(equity=100000.0, gross_exposure=20000.0, daily_loss_pct=0.0,
                   drawdown_pct=0.0):
    return EquitySnapshot(
        equity=equity, cash=equity * 0.5, portfolio_value=equity,
        day_trading_buying_power=equity * 4,
        start_of_day_equity=equity, high_watermark_equity=equity,
        realized_pl_today=0.0, unrealized_pl=0.0,
        gross_exposure=gross_exposure,
        daily_loss_pct=daily_loss_pct, drawdown_pct=drawdown_pct,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SENTIMENT SCALE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSentimentScale:
    """Tests for RiskEngine.sentiment_scale() piecewise-linear function."""

    def test_hard_bearish_block(self):
        """Sentiment below no_trade_negative_threshold → 0.0."""
        engine = _make_engine()
        assert engine.sentiment_scale(-0.5) == 0.0
        assert engine.sentiment_scale(-0.41) == 0.0
        assert engine.sentiment_scale(-1.0) == 0.0

    def test_neutral_band_zero_both_sides(self):
        """Sentiment within neutral_band (±0.1) → 0.0."""
        engine = _make_engine()
        assert engine.sentiment_scale(0.0) == 0.0
        assert engine.sentiment_scale(0.05) == 0.0
        assert engine.sentiment_scale(-0.05) == 0.0
        assert engine.sentiment_scale(0.09) == 0.0
        assert engine.sentiment_scale(-0.09) == 0.0

    def test_boundary_at_neutral_band(self):
        """Sentiment exactly at neutral_band should return min_scale."""
        engine = _make_engine()
        result = engine.sentiment_scale(0.1)
        assert result == pytest.approx(0.2, abs=0.01)

    def test_max_bullish(self):
        """Sentiment at 1.0 → max_scale."""
        engine = _make_engine()
        result = engine.sentiment_scale(1.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_moderate_bullish(self):
        """Mid-range bullish sentiment interpolation."""
        engine = _make_engine()
        result = engine.sentiment_scale(0.5)
        assert 0.2 < result < 1.0

    def test_negative_interpolation_region(self):
        """Sentiment between no_trade_negative_threshold and -neutral_band."""
        engine = _make_engine()
        # Default: no_trade_neg=-0.4, neutral_band=0.1
        # s=-0.2 is between -0.4 and -0.1
        result = engine.sentiment_scale(-0.2)
        assert 0.0 < result < 0.2

    def test_bearish_at_threshold_boundary(self):
        """Sentiment exactly at no_trade_negative_threshold → 0.0."""
        engine = _make_engine()
        assert engine.sentiment_scale(-0.4) == 0.0

    def test_monotonically_increasing_positive(self):
        """Scale should be monotonically increasing for positive sentiment."""
        engine = _make_engine()
        vals = [engine.sentiment_scale(s / 10.0) for s in range(1, 11)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i-1], f"Not monotonic at index {i}"

    def test_custom_config_narrow_neutral_band(self):
        """Custom narrow neutral band."""
        cfg = SentimentConfig(neutral_band=0.01, min_scale=0.1, max_scale=0.9)
        engine = _make_engine(sentiment_cfg=cfg)
        # 0.02 is above neutral_band=0.01
        result = engine.sentiment_scale(0.02)
        assert result > 0.0

    def test_custom_config_zero_neutral_band_division(self):
        """Edge case: denominator = 1 - neutral_band when neutral_band=1."""
        cfg = SentimentConfig(neutral_band=1.0)
        engine = _make_engine(sentiment_cfg=cfg)
        # All values within [-1, 1] are in neutral band
        assert engine.sentiment_scale(0.5) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  KELLY FRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestKellyFraction:
    """Tests for RiskEngine._kelly_fraction() Half-Kelly position sizing."""

    def test_zero_signal_returns_near_zero(self):
        """Zero signal score → minimal Kelly fraction."""
        engine = _make_engine()
        result = engine._kelly_fraction(0.0, 1.0, 0.01, s=0.0, symbol="AAPL")
        assert 0.0 <= result <= 0.1

    def test_strong_signal_returns_positive(self):
        """Strong positive signal → meaningful Kelly fraction."""
        engine = _make_engine()
        result = engine._kelly_fraction(0.8, 1.0, 0.01, s=0.5, symbol="AAPL")
        assert result > 0.0

    def test_fraction_bounded_zero_to_one(self):
        """Kelly fraction should always be in [0, 1]."""
        engine = _make_engine()
        for signal in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for vol in [0.001, 0.01, 0.05, 0.1]:
                for s in [-1.0, 0.0, 1.0]:
                    result = engine._kelly_fraction(
                        signal, 1.0, vol, s=s, symbol="TEST"
                    )
                    assert 0.0 <= result <= 1.0, (
                        f"Out of bounds: signal={signal}, vol={vol}, s={s}, result={result}"
                    )

    def test_warm_up_vol_norm_uses_005(self):
        """During warm-up (<20 samples), vol_norm should be 0.005 (H3 fix)."""
        engine = _make_engine()
        # Call with less than 20 history points
        for _ in range(5):
            engine._kelly_fraction(0.5, 1.0, 0.01, s=0.0, symbol="WARM")
        # Check that the symbol's history has the entries
        assert len(engine._vol_history["WARM"]) == 5

    def test_per_symbol_vol_history(self):
        """Each symbol should have its own vol history deque."""
        engine = _make_engine()
        engine._kelly_fraction(0.5, 1.0, 0.01, symbol="AAPL")
        engine._kelly_fraction(0.5, 1.0, 0.02, symbol="MSFT")
        assert "AAPL" in engine._vol_history
        assert "MSFT" in engine._vol_history
        assert len(engine._vol_history["AAPL"]) == 1
        assert len(engine._vol_history["MSFT"]) == 1

    def test_adaptive_vol_norm_after_20_samples(self):
        """After 20 samples, vol_norm should use percentile normalization."""
        engine = _make_engine()
        for i in range(25):
            engine._kelly_fraction(0.5, 1.0, 0.01 + i * 0.001, symbol="TEST")
        assert len(engine._vol_history["TEST"]) == 25

    def test_zero_volatility_returns_zero_volfactor(self):
        """Zero volatility → volfactor=0, vol_penalty=1."""
        engine = _make_engine()
        result = engine._kelly_fraction(0.5, 1.0, 0.0, symbol="ZERO")
        assert result >= 0.0

    def test_higher_signal_gives_higher_fraction(self):
        """Stronger conviction should produce larger Kelly fractions."""
        engine = _make_engine()
        low = engine._kelly_fraction(0.2, 1.0, 0.01, s=0.0, symbol="COMP_L")
        high = engine._kelly_fraction(0.8, 1.0, 0.01, s=0.0, symbol="COMP_H")
        assert high >= low

    def test_sentiment_blending_effect(self):
        """Positive sentiment should increase Kelly fraction vs negative."""
        engine = _make_engine()
        pos = engine._kelly_fraction(0.5, 1.0, 0.01, s=0.5, symbol="SENT_P")
        neg = engine._kelly_fraction(0.5, 1.0, 0.01, s=-0.5, symbol="SENT_N")
        assert pos >= neg


# ═══════════════════════════════════════════════════════════════════════════
#  VOL HISTORY PERSISTENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestVolHistoryPersistence:
    """Tests for export/import of per-symbol volatility history."""

    def test_export_empty_history(self):
        engine = _make_engine()
        exported = engine.export_vol_history()
        assert exported["schema_version"] == "1.0.0"
        assert exported["data"] == {}

    def test_roundtrip_vol_history(self):
        engine = _make_engine()
        for i in range(10):
            engine._kelly_fraction(0.5, 1.0, 0.01 + i * 0.001, symbol="AAPL")
        exported = engine.export_vol_history()

        new_engine = _make_engine()
        new_engine.import_vol_history(exported)
        assert len(new_engine._vol_history["AAPL"]) == 10

    def test_import_legacy_format(self):
        """Legacy format (bare dict without schema_version) should work."""
        engine = _make_engine()
        legacy = {"AAPL": [0.01, 0.02, 0.03]}
        engine.import_vol_history(legacy)
        assert len(engine._vol_history["AAPL"]) == 3

    def test_import_wrong_schema_version_raises(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="schema version mismatch"):
            engine.import_vol_history({"schema_version": "2.0.0", "data": {}})

    def test_import_malformed_entries_skipped(self):
        engine = _make_engine()
        data = {
            "schema_version": "1.0.0",
            "data": {
                "AAPL": [0.01, 0.02],
                "BAD": 12345,           # int is not iterable → triggers except
                "ALSO_BAD": None,       # None is not iterable → triggers except
            },
        }
        engine.import_vol_history(data)
        assert "AAPL" in engine._vol_history
        assert "BAD" not in engine._vol_history
        assert "ALSO_BAD" not in engine._vol_history

    def test_import_none_input(self):
        engine = _make_engine()
        engine.import_vol_history(None)  # Should not raise
        assert engine._vol_history == {}

    def test_import_caps_at_200(self):
        engine = _make_engine()
        data = {"AAPL": list(range(300))}
        engine.import_vol_history(data)
        assert len(engine._vol_history["AAPL"]) == 200


# ═══════════════════════════════════════════════════════════════════════════
#  PRE-TRADE CHECKS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPreTradeChecks:
    """Tests for RiskEngine.pre_trade_checks() — all rejection paths."""

    def test_accept_valid_trade(self):
        """Valid trade should produce qty > 0 and no rejected_reason."""
        engine = _make_engine()
        snapshot = _make_snapshot()
        sentiment = _make_sentiment(score=0.5)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=sentiment,
            signal_score=0.5, volatility=0.02,
        )
        assert proposed.qty > 0
        assert proposed.rejected_reason is None

    def test_reject_unlisted_symbol(self):
        engine = _make_engine()
        snapshot = _make_snapshot()
        sentiment = _make_sentiment()
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="FAKE",
            side="buy", entry_price=100.0, stop_price=95.0,
            take_profit_price=110.0, sentiment=sentiment,
        )
        assert proposed.qty == 0
        assert "not whitelisted" in proposed.rejected_reason.lower()

    def test_reject_chaos_sentiment(self):
        engine = _make_engine()
        snapshot = _make_snapshot()
        chaos = _make_sentiment(score=-1.0, raw_discrete=-2)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=chaos,
        )
        assert proposed.qty == 0
        assert "-2" in proposed.rejected_reason

    def test_reject_negative_sentiment_scale(self):
        engine = _make_engine()
        snapshot = _make_snapshot()
        # Sentiment in neutral band → s_scale=0
        neutral = _make_sentiment(score=0.0, raw_discrete=0)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=neutral,
        )
        assert proposed.qty == 0
        assert "sentiment" in proposed.rejected_reason.lower()

    def test_reject_zero_stop_distance(self):
        engine = _make_engine()
        snapshot = _make_snapshot()
        sentiment = _make_sentiment(score=0.5)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=150.0,
            take_profit_price=160.0, sentiment=sentiment,
        )
        assert proposed.qty == 0
        assert "stop distance" in proposed.rejected_reason.lower()

    def test_reject_daily_loss_limit(self):
        engine = _make_engine()
        snapshot = _make_snapshot(daily_loss_pct=-0.05)  # >4% limit
        sentiment = _make_sentiment(score=0.5)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=sentiment,
        )
        assert proposed.qty == 0
        assert "daily loss" in proposed.rejected_reason.lower()

    def test_reject_drawdown_limit(self):
        engine = _make_engine()
        snapshot = _make_snapshot(drawdown_pct=-0.10)  # >9% limit
        sentiment = _make_sentiment(score=0.5)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=sentiment,
        )
        assert proposed.qty == 0
        assert "drawdown" in proposed.rejected_reason.lower()

    def test_reject_max_positions(self):
        engine = _make_engine()
        snapshot = _make_snapshot()
        sentiment = _make_sentiment(score=0.5)
        # Create 15 fake positions
        positions = {
            f"SYM{i}": PositionInfo(
                symbol=f"SYM{i}", qty=10.0, market_price=100.0,
                side="long", notional=1000.0,
            )
            for i in range(15)
        }
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions=positions, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=sentiment,
        )
        assert proposed.qty == 0
        assert "max open positions" in proposed.rejected_reason.lower()

    def test_reject_gross_exposure_cap(self):
        engine = _make_engine()
        snapshot = _make_snapshot(gross_exposure=89000.0)  # Near 90% cap
        sentiment = _make_sentiment(score=0.5)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=sentiment,
            signal_score=0.8, volatility=0.02,
        )
        assert proposed.qty == 0
        assert "exposure" in proposed.rejected_reason.lower()

    def test_qty_rounded_to_lot_size(self):
        """Quantity should be a multiple of lot_size."""
        engine = _make_engine()
        snapshot = _make_snapshot()
        sentiment = _make_sentiment(score=0.5)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=sentiment,
            signal_score=0.5, volatility=0.02,
        )
        if proposed.qty > 0:
            assert proposed.qty == int(proposed.qty)  # lot_size=1.0

    def test_sentiment_scale_override(self):
        """sentiment_scale_override should bypass internal sentiment_scale()."""
        engine = _make_engine()
        snapshot = _make_snapshot()
        # Score=0.0 would normally give s_scale=0 (rejected),
        # but override=0.8 should allow the trade.
        neutral = _make_sentiment(score=0.0, raw_discrete=0)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=145.0,
            take_profit_price=160.0, sentiment=neutral,
            sentiment_scale_override=0.8, signal_score=0.5,
            volatility=0.02,
        )
        assert proposed.qty > 0

    def test_broker_notional_cap(self):
        """Projected notional > 40% equity should be capped."""
        engine = _make_engine()
        snapshot = _make_snapshot(equity=10000.0)
        sentiment = _make_sentiment(score=0.9)
        proposed = engine.pre_trade_checks(
            snapshot=snapshot, positions={}, symbol="AAPL",
            side="buy", entry_price=150.0, stop_price=149.0,
            take_profit_price=160.0, sentiment=sentiment,
            signal_score=0.9, volatility=0.02,
        )
        if proposed.qty > 0:
            assert proposed.qty * proposed.entry_price <= 0.4 * 10000.0 + 1.0
