# CHANGES:
# FIX SENTIMENT-FLOAT-CONTRACT — Preserve direct float sentiment from the model
# instead of reconstructing score from a rounded discrete class and confidence.
# Added raw_model_score to SentimentResult so the exact model sentiment is stored.
# Chaos remains an explicit sentinel (-2) via raw_model_score / raw_discrete while
# score stays bounded for risk sizing and downstream safety.
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
    score: authoritative continuous sentiment score used by the bot.
           For normal operation this is the direct model-provided float in [-1, 1].
           For chaos, score is bounded at -1.0 for risk sizing safety while
           raw_model_score preserves the exact chaos sentinel -2.0.
    raw_discrete: legacy compatibility field.
           -2 for chaos, -1 for negative float sentiment, 0 for neutral, 1 for positive.
    rawcompound: legacy field; kept for compatibility, here = score.
    raw_model_score: exact model-provided sentiment, either -2.0 for chaos or a float in [-1, 1].
    ndocuments: number of news items used.
    explanation: optional short explanation.
    confidence: model-reported confidence in [0, 1]; retained as metadata only.
    """
    score: float
    raw_discrete: int
    rawcompound: float
    ndocuments: int
    explanation: Optional[str] = None
    confidence: float = 0.0
    raw_model_score: float = 0.0


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
                "score": result.score,
                "raw_discrete": result.raw_discrete,
                "rawcompound": result.rawcompound,
                "raw_model_score": result.raw_model_score,
                "ndocuments": result.ndocuments,
                "explanation": result.explanation,
                "confidence": result.confidence,
                "ts_utc": ts.isoformat(),
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
                if ts.tzinfo is None:
                    now_cmp = now.replace(tzinfo=None)
                else:
                    now_cmp = now
                if (now_cmp - ts) > self.cache_ttl:
                    continue
                raw_model_score = float(entry.get("raw_model_score", entry.get("score", 0.0)))
                result = SentimentResult(
                    score=float(entry["score"]),
                    raw_discrete=int(entry["raw_discrete"]),
                    rawcompound=float(entry["rawcompound"]),
                    ndocuments=int(entry["ndocuments"]),
                    explanation=entry.get("explanation"),
                    confidence=float(entry.get("confidence", 0.0)),
                    raw_model_score=raw_model_score,
                )
                self._cache[sym] = (result, ts)
            except Exception:
                continue

    # ── END PERSIST-FIX ──────────────────────────────────────────────────────

    def _neutral(self, reason: str, ndocs: int = 0) -> SentimentResult:
        return SentimentResult(
            score=0.0,
            raw_discrete=0,
            rawcompound=0.0,
            raw_model_score=0.0,
            ndocuments=ndocs,
            explanation=reason,
            confidence=0.0,
        )

    def _coerce_model_sentiment(self, raw_sentiment) -> Tuple[float, int]:
        """
        Preserve the direct model sentiment contract.

        Returns:
          - raw_model_score: -2.0 for chaos, otherwise a float in [-1.0, 1.0]
          - raw_discrete: legacy compatibility bucket (-2, -1, 0, 1)
        """
        try:
            model_score = float(raw_sentiment)
        except (TypeError, ValueError):
            return 0.0, 0

        if model_score == -2.0:
            return -2.0, -2

        model_score = max(-1.0, min(1.0, model_score))
        if abs(model_score) < 1e-12:
            model_score = 0.0

        if model_score > 0.0:
            raw_discrete = 1
        elif model_score < 0.0:
            raw_discrete = -1
        else:
            raw_discrete = 0
        return model_score, raw_discrete

    def _score_from_model_sentiment(self, raw_model_score: float) -> float:
        """
        The bot now uses the direct model sentiment as its score.

        For chaos, keep score bounded at -1.0 so downstream risk sizing and
        interval logic continue to operate on a safe range while raw_model_score
        retains the exact -2 sentinel.
        """
        if raw_model_score == -2.0:
            return -1.0
        return max(-1.0, min(1.0, raw_model_score))

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

        raw_model_score, sdisc = self._coerce_model_sentiment(res.get("sentiment", 0.0))

        try:
            confidence = float(res.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        explanation = res.get("explanation", "") or ""
        score = self._score_from_model_sentiment(raw_model_score)

        result = SentimentResult(
            score=score,
            raw_discrete=sdisc,
            rawcompound=score,
            raw_model_score=raw_model_score,
            ndocuments=len(newsitems),
            explanation=explanation,
            confidence=confidence,
        )
        self._set_last_known(symbol, result)
        return result

    def scorenewsitems(self, symbol: str, newsitems: List[dict]) -> SentimentResult:
        """
        newsitems: list of dicts (typically *new since last check* from news API),
                  each with at least 'headline' and/or 'summary'.

        Cost controls:
          - If there are no new news items and we have a last-known sentiment, reuse it. (2)
          - If cached sentiment is still fresh (TTL), reuse it. (1)
          - If last-known sentiment is -2 and still within cooldown, reuse it. (6)
        """
        now = datetime.now(timezone.utc)
        last_known = self._get_last_known(symbol)

        if last_known:
            last_res, last_ts = last_known
            if last_ts.tzinfo is None:
                now_cmp = now.replace(tzinfo=None)
            else:
                now_cmp = now
            if last_res.raw_discrete == -2 and ((now_cmp - last_ts) <= self.chaos_cooldown):
                return last_res

        if not newsitems:
            if last_known:
                return last_known[0]
            return self._neutral("No recent news (no prior sentiment cached).", ndocs=0)

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
            |s| >= 0.8  ->  60s   (high conviction / high risk — once per minute)
            |s| >= 0.5  -> 120s   (strong signal)
            |s| >= 0.2  -> 180s   (moderate signal)
            |s| <  0.2  -> 300s   (neutral band, minimal alpha)
        """
        if max_abs_s >= 0.8:
            return 60
        if max_abs_s >= 0.5:
            return 120
        if max_abs_s >= 0.2:
            return 180
        return 300

    def adaptive_rescore_interval_hysteresis(self, max_abs_s: float, current_interval: int) -> int:
        target = self.adaptive_rescore_interval(max_abs_s)
        if target == current_interval:
            return current_interval
        boundaries = {60: 0.8, 120: 0.5, 180: 0.2, 300: 0.0}
        current_boundary = boundaries.get(current_interval, 0.0)
        target_boundary = boundaries.get(target, 0.0)
        midpoint = (current_boundary + target_boundary) / 2.0
        if abs(max_abs_s - midpoint) > 0.05:
            return target
        return current_interval
