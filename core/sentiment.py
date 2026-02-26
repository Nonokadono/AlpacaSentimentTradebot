# CHANGES:
#   - Added force_rescore(symbol, newsitems) method.
#     This bypasses the TTL cache and chaos cooldown to always call the AI for
#     open-position sentiment-exit checks. It still writes the result back to the
#     cache so subsequent normal scoring benefits from it.
#     The existing scorenewsitems(), get_cached_sentiment(), _neutral(),
#     _map_discrete_to_score(), _get_last_known(), _set_last_known() methods are
#     completely untouched. No field renames.

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ai_client import NewsReasoner


@dataclass
class SentimentResult:
    """
    score: continuous sentiment score in [-1, 1] for risk sizing.
           -1 strongly negative, +1 strongly positive.
           For discrete -2 (utterly undesirable / unstable), score is fixed at -1
           and should trigger no-trade / forced exit logic at the risk engine level.
    raw_discrete: the raw discrete value from the model in {-2, -1, 0, 1}
    rawcompound: legacy field; kept for compatibility, here = score
    ndocuments: number of news items used
    explanation: optional short explanation
    confidence: model-reported confidence in [0, 1]
    """
    score: float
    raw_discrete: int
    rawcompound: float
    ndocuments: int
    explanation: Optional[str] = None
    confidence: float = 0.0


class SentimentModule:
    """
    Sentiment engine backed by Perplexity Sonar via NewsReasoner.

    Implements cost controls:
      (1) TTL-based per-symbol cache to reduce AI calls.
      (2) If no *new* news arrives for a symbol, reuse last-known sentiment (no AI call).
      (6) If last-known raw_discrete == -2, apply a cooldown window during which we do not
          call the AI again for that symbol (even if new news arrives).
    """

    def __init__(self) -> None:
        self.reasoner = NewsReasoner()

        ttl_min = int(os.getenv("SENTIMENT_CACHE_TTL_MIN", "30"))
        ttl_min = max(1, ttl_min)
        self.cache_ttl = timedelta(minutes=ttl_min)

        chaos_cd_min = int(os.getenv("SENTIMENT_CHAOS_COOLDOWN_MIN", "120"))
        chaos_cd_min = max(0, chaos_cd_min)
        self.chaos_cooldown = timedelta(minutes=chaos_cd_min)

        # Cache is the single source of truth for last-known sentiment.
        # symbol -> (SentimentResult, timestamp_utc)
        self._cache: Dict[str, Tuple[SentimentResult, datetime]] = {}

    def _neutral(self, reason: str, ndocs: int = 0) -> SentimentResult:
        return SentimentResult(
            score=0.0,
            raw_discrete=0,
            rawcompound=0.0,
            ndocuments=ndocs,
            explanation=reason,
            confidence=0.0,
        )

    def _map_discrete_to_score(self, sdisc: int, confidence: float) -> float:
        """
        Map discrete sentiment {-2, -1, 0, 1} plus confidence into a continuous score in [-1, 1].
        """
        confidence = max(0.0, min(1.0, confidence))

        if sdisc == -2:
            # Hard floor at -1 to signal "utterly undesirable / unstable".
            return -1.0

        if sdisc == -1:
            base = -1.0
        elif sdisc == 0:
            base = 0.0
        elif sdisc == 1:
            base = 1.0
        else:
            base = 0.0

        return max(-1.0, min(1.0, base * confidence))

    def get_cached_sentiment(self, symbol: str) -> Optional[SentimentResult]:
        """
        TTL cache getter.
        Returns a cached sentiment only if it is within the TTL window.
        """
        now = datetime.utcnow()
        cached = self._cache.get(symbol)
        if not cached:
            return None
        result, ts = cached
        if now - ts <= self.cache_ttl:
            return result
        return None

    def _get_last_known(self, symbol: str) -> Optional[Tuple[SentimentResult, datetime]]:
        return self._cache.get(symbol)

    def _set_last_known(self, symbol: str, result: SentimentResult) -> None:
        self._cache[symbol] = (result, datetime.utcnow())

    def _call_ai(self, symbol: str, newsitems: List[dict]) -> SentimentResult:
        """
        Internal: call the AI, parse the result, update the cache, and return a
        SentimentResult. Used by both scorenewsitems() and force_rescore().
        """
        res = self.reasoner.scorenews(symbol, newsitems)

        try:
            sdisc = int(res.get("sentiment", 0))
        except (TypeError, ValueError):
            sdisc = 0
        if sdisc not in (-2, -1, 0, 1):
            sdisc = 0

        try:
            confidence = float(res.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        explanation = res.get("explanation", "") or ""
        score = self._map_discrete_to_score(sdisc, confidence)

        result = SentimentResult(
            score=score,
            raw_discrete=sdisc,
            rawcompound=score,
            ndocuments=len(newsitems),
            explanation=explanation,
            confidence=confidence,
        )
        self._set_last_known(symbol, result)
        return result

    def scorenewsitems(self, symbol: str, newsitems: List[dict]) -> SentimentResult:
        """
        newsitems: list of dicts (typically *new since last check* from Alpaca news API),
                  each with at least 'headline' and/or 'summary'.

        Cost controls:
          - If there are no new news items and we have a last-known sentiment, reuse it. (2)
          - If cached sentiment is still fresh (TTL), reuse it. (1)
          - If last-known sentiment is -2 and still within cooldown, reuse it. (6)
        """
        now = datetime.utcnow()
        last_known = self._get_last_known(symbol)

        # (6) Chaos cooldown: if we recently deemed the symbol "unstable / -2", don't rescore.
        if last_known:
            last_res, last_ts = last_known
            if last_res.raw_discrete == -2 and (now - last_ts) <= self.chaos_cooldown:
                return last_res

        # (2) No new news -> do not call AI; just reuse last-known sentiment if available.
        if not newsitems:
            if last_known:
                return last_known[0]
            return self._neutral("No recent news (no prior sentiment cached).", ndocs=0)

        # (1) TTL cache: if within TTL, reuse cached sentiment even if new news exists.
        # Rationale: avoids frequent rescores when headlines trickle in; TTL bounds staleness.
        cached_fresh = self.get_cached_sentiment(symbol)
        if cached_fresh is not None:
            return cached_fresh

        return self._call_ai(symbol, newsitems)

    def force_rescore(self, symbol: str, newsitems: List[dict]) -> SentimentResult:
        """
        Unconditional AI rescore — bypasses TTL cache and chaos cooldown.

        Use this exclusively for open-position sentiment-exit checks, where stale
        cached data would cause the exit logic to silently produce delta = 0 and
        never fire.

        If newsitems is empty the AI cannot reason about new information; in that
        case we fall back to the last-known cached result (if any) or neutral.
        We do NOT want to exit a position solely because news is thin — the caller
        must decide what to do with a low-confidence neutral result.
        """
        if not newsitems:
            last_known = self._get_last_known(symbol)
            if last_known:
                return last_known[0]
            return self._neutral("No recent news for forced rescore.", ndocs=0)

        return self._call_ai(symbol, newsitems)
