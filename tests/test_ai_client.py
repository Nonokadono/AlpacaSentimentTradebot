"""
Tests for ai_client.py — _normalize_model_sentiment, NewsReasoner.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pytest
from unittest.mock import patch, MagicMock

from ai_client import _normalize_model_sentiment, NewsReasoner


# ═══════════════════════════════════════════════════════════════════════════
#  NORMALIZE MODEL SENTIMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestNormalizeModelSentiment:

    def test_positive_float(self):
        assert _normalize_model_sentiment(0.5) == pytest.approx(0.5)

    def test_negative_float(self):
        assert _normalize_model_sentiment(-0.3) == pytest.approx(-0.3)

    def test_zero(self):
        assert _normalize_model_sentiment(0.0) == 0.0

    def test_chaos_sentinel_exact(self):
        result = _normalize_model_sentiment(-2.0)
        assert result == -2

    def test_chaos_sentinel_int(self):
        result = _normalize_model_sentiment(-2)
        assert result == -2

    def test_clamp_above_1(self):
        assert _normalize_model_sentiment(1.5) == 1.0

    def test_clamp_below_minus_1(self):
        assert _normalize_model_sentiment(-1.5) == -1.0

    def test_boundary_1(self):
        assert _normalize_model_sentiment(1.0) == 1.0

    def test_boundary_minus_1(self):
        assert _normalize_model_sentiment(-1.0) == -1.0

    def test_string_float(self):
        assert _normalize_model_sentiment("0.7") == pytest.approx(0.7)

    def test_invalid_string(self):
        assert _normalize_model_sentiment("bad") == 0.0

    def test_none(self):
        assert _normalize_model_sentiment(None) == 0.0

    def test_list_input(self):
        assert _normalize_model_sentiment([1, 2]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  NEWS REASONER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestNewsReasoner:

    def test_disabled_without_api_key(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
            reasoner = NewsReasoner()
            assert reasoner.disabled is True

    def test_disabled_returns_neutral(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Test"}])
            assert result["sentiment"] == 0.0
            assert result["confidence"] == 0.0

    def test_no_news_returns_neutral(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [])
            assert result["sentiment"] == 0.0

    def test_no_usable_text_returns_neutral(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "", "summary": ""}])
            assert result["sentiment"] == 0.0

    @patch("ai_client.requests.post")
    @patch("ai_client.time.sleep")
    def test_successful_scoring(self, mock_sleep, mock_post):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "sentiment": 0.6,
                            "confidence": 0.8,
                            "explanation": "Positive outlook",
                        })
                    }
                }]
            }
            mock_post.return_value = mock_response

            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Good earnings"}])
            assert result["sentiment"] == pytest.approx(0.6)
            assert result["confidence"] == pytest.approx(0.8)

    @patch("ai_client.requests.post")
    @patch("ai_client.time.sleep")
    def test_rate_limit_returns_neutral(self, mock_sleep, mock_post):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = 429
            mock_post.return_value = mock_response

            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Test"}])
            assert result["sentiment"] == 0.0

    @patch("ai_client.requests.post")
    @patch("ai_client.time.sleep")
    def test_api_error_returns_neutral(self, mock_sleep, mock_post):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = 500
            mock_post.return_value = mock_response

            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Test"}])
            assert result["sentiment"] == 0.0

    @patch("ai_client.requests.post")
    @patch("ai_client.time.sleep")
    def test_malformed_json_returns_neutral(self, mock_sleep, mock_post):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "NOT JSON AT ALL"}
                }]
            }
            mock_post.return_value = mock_response

            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Test"}])
            assert result["sentiment"] == 0.0

    @patch("ai_client.requests.post")
    @patch("ai_client.time.sleep")
    def test_code_fenced_json_parsed(self, mock_sleep, mock_post):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            fenced = '```json\n{"sentiment": 0.4, "confidence": 0.6, "explanation": "ok"}\n```'
            mock_response.json.return_value = {
                "choices": [{"message": {"content": fenced}}]
            }
            mock_post.return_value = mock_response

            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Test"}])
            assert result["sentiment"] == pytest.approx(0.4)

    @patch("ai_client.requests.post")
    @patch("ai_client.time.sleep")
    def test_chaos_sentiment_preserved(self, mock_sleep, mock_post):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "sentiment": -2,
                            "confidence": 0.9,
                            "explanation": "Chaos event",
                        })
                    }
                }]
            }
            mock_post.return_value = mock_response

            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Major crash"}])
            assert result["sentiment"] == -2

    @patch("ai_client.requests.post")
    @patch("ai_client.time.sleep")
    def test_exception_returns_neutral(self, mock_sleep, mock_post):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            mock_post.side_effect = Exception("Network error")

            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Test"}])
            assert result["sentiment"] == 0.0

    @patch("ai_client.requests.post")
    @patch("ai_client.time.sleep")
    def test_confidence_clamped(self, mock_sleep, mock_post):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test_key"}, clear=False):
            mock_response = MagicMock()
            mock_response.ok = True
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "sentiment": 0.5,
                            "confidence": 1.5,  # Over 1.0
                            "explanation": "test",
                        })
                    }
                }]
            }
            mock_post.return_value = mock_response

            reasoner = NewsReasoner()
            result = reasoner.scorenews("AAPL", [{"headline": "Test"}])
            assert result["confidence"] == 1.0
