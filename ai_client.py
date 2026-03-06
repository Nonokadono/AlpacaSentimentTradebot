# CHANGES:
# FIX SENTIMENT-FLOAT-CONTRACT — Preserve model-provided float sentiment in [-1, 1]
# rather than rounding to {-1, 0, 1} and then recomputing score from confidence.
# Chaos remains the only sentinel and is represented by the exact integer -2.
# Added _normalize_model_sentiment() helper and updated prompts / docstrings to
# request a direct float sentiment contract from the model.

import json
import logging
import os
import time
import requests

logger = logging.getLogger("tradebot")

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "llama-3.1-8b-instant"
# Groq free tier is 30 RPM. 2s spacing = max 30 calls/min with headroom.
_INTER_CALL_DELAY_SEC = 2.0


def _normalize_model_sentiment(raw_sentiment):
    """
    Preserve the model's continuous sentiment contract.

    Valid outputs:
      - exact -2 for chaos / do-not-trade
      - otherwise a float clamped to [-1.0, 1.0]

    Any malformed value falls back to neutral 0.0.
    """
    try:
        sentiment = float(raw_sentiment)
    except (TypeError, ValueError):
        return 0.0

    if sentiment == -2.0:
        return -2

    return max(-1.0, min(1.0, sentiment))


class NewsReasoner:
    """
    Uses Groq (llama-3.1-8b-instant) via the OpenAI-compatible endpoint
    to score short-term news sentiment.

    Returns a dict with keys:
        - sentiment: float in [-1, 1], or exact integer -2 for chaos
        - confidence: float in [0, 1]
        - explanation: str
    """

    def __init__(self) -> None:
        self.apiurl = _GROQ_URL
        self.apikey = os.getenv("GROQ_API_KEY")
        if not self.apikey:
            logger.warning(
                "GROQ_API_KEY is not set — NewsReasoner disabled. "
                "All sentiment scores will be neutral (0.0) until the key is provided."
            )
            self.disabled = True
        else:
            self.disabled = False

    def scorenews(self, symbol: str, newsitems):
        """
        Input:
            symbol: string ticker, e.g. "AAPL"
            newsitems: list of dicts from Alpaca news API

        Output dict:
            - sentiment: float in [-1, 1], or exact integer -2 for chaos
            - confidence: float 0-1
            - explanation: str
        """
        if self.disabled:
            return {
                "sentiment": 0.0,
                "confidence": 0.0,
                "explanation": "NewsReasoner disabled (GROQ_API_KEY not set).",
            }

        # Rate-limit guard: enforce minimum spacing between API calls to stay
        # under the 30 RPM free-tier cap regardless of how fast the loop runs.
        time.sleep(_INTER_CALL_DELAY_SEC)

        try:
            # No news -> neutral, low confidence
            if not newsitems:
                return {
                    "sentiment": 0.0,
                    "confidence": 0.0,
                    "explanation": "No recent news.",
                }

            # Build compact headlines + summaries for up to 10 news items
            summaries = []
            for n in newsitems[:10]:
                title = n.get("headline") or n.get("title") or ""
                summary = n.get("summary") or ""
                text = f"{title} {summary}".strip()
                if not text:
                    continue
                summaries.append(text[:300])
            if not summaries:
                return {
                    "sentiment": 0.0,
                    "confidence": 0.0,
                    "explanation": "No usable news text.",
                }

            userprompt = (
                f"You are a professional equity analyst.\n"
                f"Evaluate the SHORT-TERM (next few trading days) impact of the following news "
                f"on {symbol} stock.\n"
                f"Return a single JSON object with keys:\n"
                f'  sentiment: exact integer -2 for chaos / do-not-trade, otherwise a float between -1 and 1\n'
                f'    -2 means extremely unstable / utterly undesirable to trade now (e.g. chaotic, '\
                f'        very high uncertainty, extreme event risk),\n'
                f'    values near -1 are strongly negative,\n'
                f'    values near 0 are neutral or mixed,\n'
                f'    values near 1 are strongly positive.\n'
                f'  confidence: a number between 0 and 1\n'
                f'  explanation: short textual explanation (1-3 sentences).\n\n'
                f"News:\n- " + "\n- ".join(summaries)
            )

            headers = {
                "Authorization": f"Bearer {self.apikey}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": _GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise financial sentiment classifier. "
                            "You only output strict JSON with the requested keys. "
                            "Do not wrap your response in markdown code fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": userprompt,
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
            }

            resp = requests.post(self.apiurl, headers=headers, json=payload, timeout=30)

            if resp.status_code == 429:
                logger.warning(
                    f"Groq API rate-limited (429) for {symbol} — treating sentiment as neutral. "
                    f"Check free-tier RPM/RPD quota."
                )
                return {
                    "sentiment": 0.0,
                    "confidence": 0.0,
                    "explanation": "Groq API rate limit hit; treating sentiment as neutral.",
                }

            if not resp.ok:
                logger.warning(
                    f"Groq API error {resp.status_code} for {symbol} — treating sentiment as neutral."
                )
                return {
                    "sentiment": 0.0,
                    "confidence": 0.0,
                    "explanation": "Error from AI API, treating sentiment as neutral.",
                }

            data = resp.json()
            rawcontent = data["choices"][0]["message"]["content"].strip()

            logger.debug(f"Groq raw response for {symbol}: {rawcontent!r}")

            # Strip markdown code fences if model wraps response despite instructions.
            if rawcontent.startswith("```"):
                rawcontent = rawcontent.strip("`")
                if rawcontent.startswith("json"):
                    rawcontent = rawcontent[4:]
                rawcontent = rawcontent.strip()

            # Expect JSON, fallback to extracting JSON substring if needed
            try:
                result = json.loads(rawcontent)
            except json.JSONDecodeError:
                try:
                    start = rawcontent.index("{")
                    end = rawcontent.rindex("}") + 1
                    result = json.loads(rawcontent[start:end])
                except Exception:
                    return {
                        "sentiment": 0.0,
                        "confidence": 0.0,
                        "explanation": "Could not parse model output, treating as neutral.",
                    }

            sentiment = _normalize_model_sentiment(result.get("sentiment", 0.0))

            try:
                confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            explanation = result.get("explanation", "")

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "explanation": explanation,
            }

        except Exception as e:
            # Normalize fields on any unexpected error
            logger.warning(f"scorenews error for {symbol}: {e}")
            return {
                "sentiment": 0.0,
                "confidence": 0.0,
                "explanation": "Exception in sentiment analysis, treating as neutral.",
            }
