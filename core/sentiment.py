# CHANGES:
# FIX 8 — Replaced datetime.utcnow() with datetime.now(timezone.utc) throughout to eliminate DeprecationWarning.
#         Added timezone import; ensured naive persisted timestamps are compatible via .replace(tzinfo=None) guard.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from ai_client import NewsReasoner
from config.config import SentimentConfig


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

    def __init__(self, cfg: Optional[SentimentConfig] = None) -> None:
        self.reasoner = NewsReasoner()
        # Fix C2: store SentimentConfig so _map_discrete_to_score can read
        # confidence_gamma.  Default to a fresh SentimentConfig() if not supplied
        # so all existing call sites (which pass no argument) remain unchanged.
        self.cfg: SentimentConfig = cfg if cfg is not None else SentimentConfig()

        ttl_min = int(os.getenv("SENTIMENT_CACHE_TTL_MIN", "30"))
        ttl_min = max(1, ttl_min)
        self.cache_ttl = timedelta(minutes=ttl_min)

        chaos_cd_min = int(os.getenv("SENTIMENT_CHAOS_COOLDOWN_MIN", "120"))
        chaos_cd_min = max(0, chaos_cd_min)
        self.chaos_cooldown = timedelta(minutes=chaos_cd_min)

        # Cache is the single source of truth for last-known sentiment.
        # symbol -> (SentimentResult, timestamp_utc)
        self._cache: Dict[str, Tuple[SentimentResult, datetime]] = {}

    # ── PERSIST-FIX: Cache serialisation ─────────────────────────────────────

    def export_cache(self) -> Dict[str, dict]:
        """Serialise the in-memory sentiment cache to a JSON-safe dict.

        Each entry is a plain dict with all SentimentResult fields plus
        ts_utc (ISO-format string of the naive UTC timestamp).
        SentimentResult fields are serialised manually because the dataclass
        is not natively JSON-serialisable (Optional[str] explanation field).

        Called by main._persist_vol_and_sentiment() after each loop iteration.
        """
        out: Dict[str, dict] = {}
        for sym, (result, ts) in self._cache.items():
            out[sym] = {
                "score":        result.score,
                "raw_discrete": result.raw_discrete,
                "rawcompound":  result.rawcompound,
                "ndocuments":   result.ndocuments,
                "explanation":  result.explanation,
                "confidence":   result.confidence,
                "ts_utc":       ts.isoformat(),
            }
        return out

    def import_cache(self, data: Dict[str, dict]) -> None:
        """Deserialise a previously exported cache dict back into _cache.

        TTL filtering: only entries whose age (now - ts_utc) is still within
        self.cache_ttl are imported. Stale entries are silently discarded so
        that a long outage or a very large cache_ttl cannot resurrect outdated
        sentiment scores as "fresh" after a restart.

        Malformed entries (missing keys, bad types, unparseable timestamps) are
        silently skipped — a corrupted equity_state.json must not crash the bot.

        FIX 8: naive comparison guard — if the loaded ts is naive, compare against
        a naive now. This ensures backward compatibility with equity_state.json
        files that stored naive datetimes before FIX 8 was applied.

        Called by main.main() immediately after SentimentModule is constructed,
        before the main loop starts.
        """
        now = datetime.now(timezone.utc)
        for sym, entry in data.items():
            try:
                ts = datetime.fromisoformat(entry["ts_utc"])
                # FIX 8: naive compatibility guard
                if ts.tzinfo is None:
                    # Loaded timestamp is naive — compare against naive now
                    now_cmp = now.replace(tzinfo=None)
                else:
                    now_cmp = now
                # Discard expired entries at load time — never serve stale data.
                if (now_cmp - ts) > self.cache_ttl:
                    continue
                result = SentimentResult(
                    score=float(entry["score"]),
                    raw_discrete=int(entry["raw_discrete"]),
                    rawcompound=float(entry["rawcompound"]),
                    ndocuments=int(entry["ndocuments"]),
                    explanation=entry.get("explanation"),
                    confidence=float(entry.get("confidence", 0.0)),
                )
                self._cache[sym] = (result, ts)
            except Exception:
                continue   # silently skip malformed entries

    # ── END PERSIST-FIX ──────────────────────────────────────────────────────

    def _neutral(self, reason: str, ndocs: int = 0) -> SentimentResult:
        return SentimentResult(
            score=0.0,
            raw_discrete=0,
            rawcompound=0.0,
            ndocuments=ndocs,
            explanation=reason,
            confidence=0.0,
        )

    def _map_discrete_to_score(self, s_disc: int, confidence: float) -> float:
        """
        Map discrete sentiment {-2, -1, 0, 1} plus confidence into a continuous score in [-1, 1].

        Fix C2: applies confidence_gamma from SentimentConfig.
          gamma = clamp(self.cfg.confidence_gamma, 1.0, 4.0)
          score = base * (confidence ** gamma)
        gamma=1.0 reproduces the previous linear behaviour.
        gamma=2.0 (default) applies convex weighting: high-confidence scores are
        amplified relative to low-confidence ones.

        Change 6: s_disc == -2 returns -1.0 explicitly and unconditionally.
        Confidence is irrelevant for the chaos case — the score is used as a
        sizing input and -1.0 is already the semantic floor.
        """
        confidence = max(0.0, min(1.0, confidence))

        if s_disc == -2:
            return -1.0          # chaos: always floor, confidence irrelevant

        if s_disc == -1:
            base = -1.0
        elif s_disc == 0:
            base = 0.0
        elif s_disc == 1:
            base = 1.0
        else:
            base = 0.0

        # Fix C2: power-law confidence weighting.
        gamma = max(1.0, min(4.0, float(self.cfg.confidence_gamma)))
        return max(-1.0, min(1.0, base * (confidence ** gamma)))

    def get_cached_sentiment(self, symbol: str) -> Optional[SentimentResult]:
        """
        TTL cache getter.
        Returns a cached sentiment only if it is within the TTL window.
        """
        now = datetime.now(timezone.utc)
        cached = self._cache.get(symbol)
        if not cached:
            return None
        result, ts = cached
        # FIX 8: naive compatibility guard for comparison
        if ts.tzinfo is None:
            now_cmp = now.replace(tzinfo=None)
        else:
            now_cmp = now
        if (now_cmp - ts) <= self.cache_ttl:
            return result
        return None

    def _get_last_known(self, symbol: str) -> Optional[Tuple[SentimentResult, datetime]]:
        return self._cache.get(symbol)

    def _set_last_known(self, symbol: str, result: SentimentResult) -> None:
        self._cache[symbol] = (result, datetime.now(timezone.utc))

    def _call_ai(self, symbol: str, newsitems: List[dict]) -> SentimentResult:
        """
        Internal: call the AI, parse the result, update the cache, and return a
        SentimentResult. Used by both scorenewsitems() and force_rescore().
        """
        res = self.reasoner.scorenews(symbol, newsitems)

        # Fix M6: coerce raw sentiment to int before membership check.
        try:
            sdisc = int(round(float(res.get("sentiment", 0))))
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
        now = datetime.now(timezone.utc)
        last_known = self._get_last_known(symbol)

        # (6) Chaos cooldown: if we recently deemed the symbol "unstable / -2", don't rescore.
        if last_known:
            last_res, last_ts = last_known
            # FIX 8: naive compatibility guard
            if last_ts.tzinfo is None:
                now_cmp = now.replace(tzinfo=None)
            else:
                now_cmp = now
            if last_res.raw_discrete == -2 and ((now_cmp - last_ts) <= self.chaos_cooldown):
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

    def adaptive_rescore_interval(self, max_abs_s: float) -> int:
        """
        Return a rescore sleep interval in seconds based on the highest
        absolute sentiment score across open positions.
        Thresholds:
            |s| >= 0.8  -> 120s   (high conviction / high risk)
            |s| >= 0.5  -> 300s   (strong signal)
            |s| >= 0.2  -> 600s   (current default)
            |s| <  0.2  -> 900s   (neutral band, minimal alpha)
        """
        if max_abs_s >= 0.8:
            return 120
        if max_abs_s >= 0.5:
            return 300
        if max_abs_s >= 0.2:
            return 600
        return 900

    def adaptive_rescore_interval_hysteresis(self, max_abs_s: float, current_interval: int) -> int:
        target = self.adaptive_rescore_interval(max_abs_s)
        if target == current_interval:
            return current_interval
        boundaries = {120: 0.8, 300: 0.5, 600: 0.2, 900: 0.0}
        current_boundary = boundaries.get(current_interval, 0.0)
        target_boundary  = boundaries.get(target, 0.0)
        midpoint = (current_boundary + target_boundary) / 2.0
        if abs(max_abs_s - midpoint) > 0.05:
            return target
        return current_interval
