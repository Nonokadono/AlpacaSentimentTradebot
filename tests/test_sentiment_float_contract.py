# CHANGES:
# Added regression tests for the direct float sentiment contract and chaos sentinel.

import unittest

from ai_client import _normalize_model_sentiment
from core.sentiment import SentimentModule


class _FakeReasoner:
    def __init__(self, sentiment, confidence=0.0, explanation="ok"):
        self.sentiment = sentiment
        self.confidence = confidence
        self.explanation = explanation

    def scorenews(self, symbol: str, newsitems):
        return {
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


class SentimentFloatContractTests(unittest.TestCase):
    def test_normalize_model_sentiment_preserves_float(self):
        self.assertAlmostEqual(_normalize_model_sentiment("0.78"), 0.78)
        self.assertAlmostEqual(_normalize_model_sentiment(-0.37), -0.37)

    def test_normalize_model_sentiment_preserves_chaos(self):
        self.assertEqual(_normalize_model_sentiment(-2), -2)
        self.assertEqual(_normalize_model_sentiment("-2"), -2)

    def test_sentiment_module_uses_direct_float_score(self):
        module = SentimentModule()
        module.reasoner = _FakeReasoner(sentiment=0.78, confidence=0.64)

        result = module.scorenewsitems("AAPL", [{"headline": "Positive update"}])

        self.assertAlmostEqual(result.score, 0.78)
        self.assertAlmostEqual(result.raw_model_score, 0.78)
        self.assertEqual(result.raw_discrete, 1)
        self.assertAlmostEqual(result.confidence, 0.64)

    def test_confidence_does_not_recompute_score(self):
        module_low = SentimentModule()
        module_low.reasoner = _FakeReasoner(sentiment=0.41, confidence=0.10)
        low = module_low.scorenewsitems("MSFT", [{"headline": "Update"}])

        module_high = SentimentModule()
        module_high.reasoner = _FakeReasoner(sentiment=0.41, confidence=0.95)
        high = module_high.scorenewsitems("NVDA", [{"headline": "Update"}])

        self.assertAlmostEqual(low.score, 0.41)
        self.assertAlmostEqual(high.score, 0.41)

    def test_chaos_keeps_exact_raw_sentinel(self):
        module = SentimentModule()
        module.reasoner = _FakeReasoner(sentiment=-2, confidence=0.22)

        result = module.scorenewsitems("TSLA", [{"headline": "Extreme uncertainty"}])

        self.assertEqual(result.raw_model_score, -2.0)
        self.assertEqual(result.raw_discrete, -2)
        self.assertEqual(result.score, -1.0)


if __name__ == "__main__":
    unittest.main()
