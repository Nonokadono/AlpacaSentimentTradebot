# CHANGES:
# GROQ MIGRATION — Replace Gemini with Groq (llama-3.1-8b-instant).
#   - _GROQ_URL points to Groq's OpenAI-compat endpoint.
#   - apikey reads GROQ_API_KEY env var.
#   - model changed to "llama-3.1-8b-instant" (30 RPM / 14,400 RPD free tier).
#   - _INTER_CALL_DELAY_SEC reduced from 6.0 to 2.0 — Groq allows 30 RPM so
#     2s spacing gives comfortable headroom for 8 symbols/loop.
#   - max_tokens kept at 1024 to prevent truncation.
#   - All error handling, markdown fence stripping, and parse logic unchanged.

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


class NewsReasoner:
    """
    Uses Groq (llama-3.1-8b-instant) via the OpenAI-compatible endpoint
    to score short-term news sentiment.

    Returns a dict with keys:
        - sentiment: int in {-2, -1, 0, 1}
        - confidence: float in [0, 1]
        - explanation: str
    """

    def __init__(self) -> None:
        self.apiurl = _GROQ_URL
        self.apikey = os.getenv("GROQ_API_KEY")
        if not self.apikey:
            logger.warning(
                "GROQ_API_KEY is not set — NewsReasoner disabled. "
                "All sentiment scores will be neutral (0) until the key is provided."
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
            - sentiment: -2, -1, 0, or 1
            - confidence: float 0-1
            - explanation: str
        """
        if self.disabled:
            return {
                "sentiment": 0,
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
                    "sentiment": 0,
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
                    "sentiment": 0,
                    "confidence": 0.0,
                    "explanation": "No usable news text.",
                }

            userprompt = (
                f"You are a professional equity analyst.\n"
                f"Evaluate the SHORT-TERM (next few trading days) impact of the following news "
                f"on {symbol} stock.\n"
                f"Return a single JSON object with keys:\n"
                f'  sentiment: -2, -1, 0, or 1\n'
                f'    -2 for extremely unstable / utterly undesirable to trade now (e.g. chaotic, '
                f'        very high uncertainty, extreme event risk),\n'
                f'    -1 for clearly negative,\n'
                f'     0 for neutral or mixed,\n'
                f'     1 for clearly positive.\n'
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
                    "sentiment": 0,
                    "confidence": 0.0,
                    "explanation": "Groq API rate limit hit; treating sentiment as neutral.",
                }

            if not resp.ok:
                logger.warning(
                    f"Groq API error {resp.status_code} for {symbol} — treating sentiment as neutral."
                )
                return {
                    "sentiment": 0,
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
                        "sentiment": 0,
                        "confidence": 0.0,
                        "explanation": "Could not parse model output, treating as neutral.",
                    }

            sentiment = result.get("sentiment", 0)
            # Fix M6: coerce to int before membership check so float strings
            # like "-1.0" or bare floats from the model are handled correctly.
            try:
                sentiment = int(round(float(sentiment)))
            except (TypeError, ValueError):
                sentiment = 0
            # Normalize sentiment to allowed set {-2, -1, 0, 1}
            if sentiment not in (-2, -1, 0, 1):
                sentiment = 0

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
                "sentiment": 0,
                "confidence": 0.0,
                "explanation": "Exception in sentiment analysis, treating as neutral.",
            }
