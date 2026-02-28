# CHANGES:
# Fix M6 — In scorenews(), added type-coercion for the sentiment field before
#   the membership check.  After sentiment = result.get("sentiment", 0), we now
#   apply: try: sentiment = int(round(float(sentiment))) except (TypeError,
#   ValueError): sentiment = 0.  This handles model responses that return the
#   value as a float string (e.g. "-1.0") or a bare float, which would
#   previously fall through the `if sentiment not in (-2,-1,0,1)` guard and
#   silently become 0.  No variable renames.

# aiclient.py
import json
import os
import requests


class NewsReasoner:
    """
    Uses Perplexity's Chat Completions API (OpenAI-compatible with the sonar model)
    to score short-term news sentiment.

    Returns a dict with keys:
        - sentiment: int in {-2, -1, 0, 1}
        - confidence: float in [0, 1]
        - explanation: str
    """

    def __init__(self) -> None:
        # Read directly from environment variables
        self.apiurl = os.getenv("AI_API_URL", "https://api.perplexity.ai/chat/completions")
        self.apikey = os.getenv("AI_API_KEY")
        if not self.apikey:
            raise RuntimeError("AI_API_KEY is not set. Please export your Perplexity API key.")

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
        try:
            # No news -> neutral, low confidence
            if not newsitems:
                return {
                    "sentiment": 0,
                    "confidence": 0.0,
                    "explanation": "No recent news."
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
                    "explanation": "No usable news text."
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
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise financial sentiment classifier. "
                            "You only output strict JSON with the requested keys."
                        ),
                    },
                    {
                        "role": "user",
                        "content": userprompt,
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            }

            resp = requests.post(self.apiurl, headers=headers, json=payload, timeout=30)
            if not resp.ok:
                # Log error and degrade gracefully to neutral sentiment
                print("Perplexity error", resp.status_code, resp.text)
                return {
                    "sentiment": 0,
                    "confidence": 0.0,
                    "explanation": "Error from AI API, treating sentiment as neutral.",
                }

            data = resp.json()
            rawcontent = data["choices"][0]["message"]["content"].strip()

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
            print("scorenews error", e)
            return {
                "sentiment": 0,
                "confidence": 0.0,
                "explanation": "Exception in sentiment analysis, treating as neutral.",
            }
