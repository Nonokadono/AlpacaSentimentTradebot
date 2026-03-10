"""
Tests for core/sentiment.py — SentimentModule: caching, TTL, chaos cooldown,
force_rescore, adaptive intervals.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from core.sentiment import SentimentModule, SentimentResult
from config.config import SentimentConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_module(cfg=None):
    with patch("core.sentiment.NewsReasoner") as MockReasoner:
        mock_reasoner = MagicMock()
        MockReasoner.return_value = mock_reasoner
        module = SentimentModule(cfg or SentimentConfig())
        module.reasoner = mock_reasoner
    return module


def _make_result(score=0.5, raw_discrete=1, confidence=0.7, raw_model_score=None):
    return SentimentResult(
        score=score, raw_discrete=raw_discrete, rawcompound=score,
        ndocuments=3, explanation="test", confidence=confidence,
        raw_model_score=raw_model_score if raw_model_score is not None else score,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  COERCE MODEL SENTIMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCoerceModelSentiment:

    def test_normal_positive(self):
        module = _make_module()
        score, disc = module._coerce_model_sentiment(0.6)
        assert score == pytest.approx(0.6)
        assert disc == 1

    def test_normal_negative(self):
        module = _make_module()
        score, disc = module._coerce_model_sentiment(-0.4)
        assert score == pytest.approx(-0.4)
        assert disc == -1

    def test_zero_neutral(self):
        module = _make_module()
        score, disc = module._coerce_model_sentiment(0.0)
        assert score == 0.0
        assert disc == 0

    def test_chaos_sentinel(self):
        module = _make_module()
        score, disc = module._coerce_model_sentiment(-2.0)
        assert score == -2.0
        assert disc == -2

    def test_clamp_above_1(self):
        module = _make_module()
        score, disc = module._coerce_model_sentiment(1.5)
        assert score == 1.0
        assert disc == 1

    def test_clamp_below_minus_1(self):
        module = _make_module()
        score, disc = module._coerce_model_sentiment(-1.5)
        assert score == -1.0
        assert disc == -1

    def test_non_numeric_returns_neutral(self):
        module = _make_module()
        score, disc = module._coerce_model_sentiment("bad")
        assert score == 0.0
        assert disc == 0

    def test_none_returns_neutral(self):
        module = _make_module()
        score, disc = module._coerce_model_sentiment(None)
        assert score == 0.0
        assert disc == 0


# ═══════════════════════════════════════════════════════════════════════════
#  SCORE FROM MODEL SENTIMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreFromModelSentiment:

    def test_normal_passthrough(self):
        module = _make_module()
        assert module._score_from_model_sentiment(0.5) == 0.5

    def test_chaos_bounded(self):
        """Chaos score (-2) should be bounded to -1.0."""
        module = _make_module()
        assert module._score_from_model_sentiment(-2.0) == -1.0

    def test_clamp_positive(self):
        module = _make_module()
        assert module._score_from_model_sentiment(1.5) == 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  CACHING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSentimentCaching:

    def test_cache_hit_within_ttl(self):
        module = _make_module()
        result = _make_result(0.5)
        module._set_last_known("AAPL", result)
        cached = module.get_cached_sentiment("AAPL")
        assert cached is not None
        assert cached.score == 0.5

    def test_cache_miss_no_entry(self):
        module = _make_module()
        assert module.get_cached_sentiment("AAPL") is None

    def test_cache_miss_expired(self):
        module = _make_module()
        result = _make_result(0.5)
        # Manually insert with old timestamp
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        module._cache["AAPL"] = (result, old_time)
        assert module.get_cached_sentiment("AAPL") is None

    def test_set_and_get_last_known(self):
        module = _make_module()
        result = _make_result(0.3)
        module._set_last_known("MSFT", result)
        last = module._get_last_known("MSFT")
        assert last is not None
        assert last[0].score == 0.3


# ═══════════════════════════════════════════════════════════════════════════
#  SCORENEWSITEMS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestScoreNewsItems:

    def test_no_news_no_cache_returns_neutral(self):
        module = _make_module()
        result = module.scorenewsitems("AAPL", [])
        assert result.score == 0.0
        assert "no" in result.explanation.lower()

    def test_no_news_with_cache_reuses(self):
        module = _make_module()
        cached = _make_result(0.6)
        module._set_last_known("AAPL", cached)
        result = module.scorenewsitems("AAPL", [])
        assert result.score == 0.6

    def test_fresh_cache_skips_ai(self):
        module = _make_module()
        cached = _make_result(0.4)
        module._set_last_known("AAPL", cached)
        news = [{"headline": "Test"}]
        result = module.scorenewsitems("AAPL", news)
        assert result.score == 0.4
        module.reasoner.scorenews.assert_not_called()

    def test_calls_ai_when_cache_expired(self):
        module = _make_module()
        # Put stale entry
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        module._cache["AAPL"] = (_make_result(0.3), old_time)
        module.reasoner.scorenews.return_value = {
            "sentiment": 0.7, "confidence": 0.8, "explanation": "Good"
        }
        news = [{"headline": "Good news"}]
        result = module.scorenewsitems("AAPL", news)
        assert result.score == pytest.approx(0.7)
        module.reasoner.scorenews.assert_called_once()

    def test_chaos_cooldown_reuses_cached(self):
        module = _make_module()
        chaos_result = _make_result(score=-1.0, raw_discrete=-2, raw_model_score=-2.0)
        module._set_last_known("AAPL", chaos_result)
        news = [{"headline": "Some news"}]
        result = module.scorenewsitems("AAPL", news)
        assert result.raw_discrete == -2
        module.reasoner.scorenews.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
#  FORCE RESCORE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestForceRescore:

    def test_no_news_returns_cached(self):
        module = _make_module()
        cached = _make_result(0.6)
        module._set_last_known("AAPL", cached)
        result = module.force_rescore("AAPL", [])
        assert result.score == 0.6

    def test_no_news_no_cache_returns_neutral(self):
        module = _make_module()
        result = module.force_rescore("AAPL", [])
        assert result.score == 0.0

    def test_calls_ai_bypasses_cache(self):
        module = _make_module()
        cached = _make_result(0.4)
        module._set_last_known("AAPL", cached)
        module.reasoner.scorenews.return_value = {
            "sentiment": 0.8, "confidence": 0.9, "explanation": "Very good"
        }
        news = [{"headline": "Breaking news"}]
        result = module.force_rescore("AAPL", news)
        assert result.score == pytest.approx(0.8)
        module.reasoner.scorenews.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
#  CACHE PERSISTENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCachePersistence:

    def test_export_import_roundtrip(self):
        module = _make_module()
        result = _make_result(0.5, raw_model_score=0.5)
        module._set_last_known("AAPL", result)
        exported = module.export_cache()

        new_module = _make_module()
        new_module.import_cache(exported)
        cached = new_module.get_cached_sentiment("AAPL")
        assert cached is not None
        assert cached.score == pytest.approx(0.5)

    def test_import_skips_stale_entries(self):
        module = _make_module()
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        exported = {
            "AAPL": {
                "score": 0.5, "raw_discrete": 1, "rawcompound": 0.5,
                "ndocuments": 3, "confidence": 0.7,
                "ts_utc": old_time.isoformat(),
            }
        }
        module.import_cache(exported)
        assert module.get_cached_sentiment("AAPL") is None

    def test_import_malformed_entry_skipped(self):
        module = _make_module()
        exported = {
            "AAPL": {"score": "not_a_number"},
        }
        module.import_cache(exported)
        assert module.get_cached_sentiment("AAPL") is None


# ═══════════════════════════════════════════════════════════════════════════
#  ADAPTIVE RESCORE INTERVAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveRescoreInterval:

    def test_high_conviction(self):
        module = _make_module()
        assert module.adaptive_rescore_interval(0.9) == 60

    def test_strong_signal(self):
        module = _make_module()
        assert module.adaptive_rescore_interval(0.6) == 120

    def test_moderate_signal(self):
        module = _make_module()
        assert module.adaptive_rescore_interval(0.3) == 180

    def test_neutral_signal(self):
        module = _make_module()
        assert module.adaptive_rescore_interval(0.1) == 300

    def test_boundary_08(self):
        module = _make_module()
        assert module.adaptive_rescore_interval(0.8) == 60

    def test_boundary_05(self):
        module = _make_module()
        assert module.adaptive_rescore_interval(0.5) == 120

    def test_boundary_02(self):
        module = _make_module()
        assert module.adaptive_rescore_interval(0.2) == 180


class TestAdaptiveRescoreHysteresis:

    def test_same_interval_no_change(self):
        module = _make_module()
        assert module.adaptive_rescore_interval_hysteresis(0.9, 60) == 60

    def test_large_change_transitions(self):
        module = _make_module()
        result = module.adaptive_rescore_interval_hysteresis(0.1, 60)
        assert result == 300  # Large gap → transition

    def test_borderline_stays(self):
        module = _make_module()
        # midpoint between 0.8 and 0.5 is 0.65
        # If max_abs_s is close to midpoint, hysteresis should hold
        result = module.adaptive_rescore_interval_hysteresis(0.65, 60)
        # 0.65 is exactly at midpoint, abs(0.65-0.65)=0 < 0.05 → stays
        assert result == 60
