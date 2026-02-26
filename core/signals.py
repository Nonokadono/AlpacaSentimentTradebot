# core/signals.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import math

from adapters.alpaca_adapter import AlpacaAdapter
from config.config import ENV_MODE, TechnicalSignalConfig
from monitoring.monitor import log_sentiment_for_symbol, log_signal_score

from .sentiment import SentimentModule, SentimentResult


@dataclass
class Signal:
    symbol: str
    side: str  # "buy", "sell", or "skip"
    rationale: str
    sentiment_result: SentimentResult
    stop_price: float
    take_profit_price: float
    signal_score: float
    momentum_score: float
    mean_reversion_score: float
    price_action_score: float


class SignalEngine:
    """
    Composite technical signal engine:
      - Momentum/trend
      - Mean reversion (RSI / MA distance proxy)
      - Price action structure

    AI cost controls implemented here:
      (3) Compute technical signal first; if side == "skip", do not fetch news or call AI.
    Suggestion (2) is enforced in SentimentModule.scorenewsitems via reuse when no new news arrives.
    """

    def __init__(
        self,
        adapter: AlpacaAdapter,
        sentiment: SentimentModule,
        technicalcfg: TechnicalSignalConfig,
    ) -> None:
        self.adapter = adapter
        self.sentiment = sentiment
        self.technicalcfg = technicalcfg

        # Per-symbol cursor for incremental news pulls.
        self.last_news_timestamp: Dict[str, datetime] = {}

    def _get_news_items(self, symbol: str) -> List[dict]:
        now = datetime.utcnow()
        last_ts = self.last_news_timestamp.get(symbol)
        if last_ts is None:
            since = now - timedelta(hours=6)
        else:
            since = last_ts

        items = self.adapter.get_news(symbol, since=since, limit=20)
        # Always advance cursor to now; "no new news" will be items == [].
        self.last_news_timestamp[symbol] = now
        return items

    def _compute_simple_momentum_raw(self, bars: List) -> float:
        if not bars or len(bars) < 2:
            return 0.0
        first = float(getattr(bars[0], "c", bars[0].c))
        last = float(getattr(bars[-1], "c", bars[-1].c))
        if first <= 0:
            return 0.0
        return (last - first) / first

    def _compute_trend_signal_raw(self, bars: List) -> float:
        # Simple slope proxy: last close vs mean close
        if not bars:
            return 0.0
        closes = [float(getattr(b, "c", b.c)) for b in bars]
        mean_c = sum(closes) / max(1, len(closes))
        last_c = closes[-1]
        if mean_c <= 0:
            return 0.0
        return (last_c - mean_c) / mean_c

    def _compute_volatility(self, bars: List) -> float:
        # Stddev of simple returns (close-to-close)
        if not bars or len(bars) < 3:
            return 0.0
        closes = [float(getattr(b, "c", b.c)) for b in bars]
        rets = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            cur = closes[i]
            if prev > 0:
                rets.append((cur - prev) / prev)
        if len(rets) < 2:
            return 0.0
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(max(0.0, var))

    def _compute_rsi(self, bars: List, period: int = 14) -> Optional[float]:
        if not bars or len(bars) < period + 1:
            return None
        closes = [float(getattr(b, "c", b.c)) for b in bars]
        gains = 0.0
        losses = 0.0
        for i in range(-period, 0):
            delta = closes[i] - closes[i - 1]
            if delta >= 0:
                gains += delta
            else:
                losses += -delta
        if gains == 0 and losses == 0:
            return 50.0
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1.0 + rs))

    def _normalize_momentum_trend(self, momentum_raw: float, trend_raw: float) -> float:
        # Scale and clip to [-1, 1]
        scale = float(self.technicalcfg.momentum_norm_scale)
        if scale <= 0:
            scale = 0.05
        x = (momentum_raw + trend_raw) / scale
        return max(-1.0, min(1.0, x))

    def _normalize_mean_reversion(self, last_price: float, bars: List, rsi: Optional[float]) -> float:
        if not bars:
            return 0.0
        closes = [float(getattr(b, "c", b.c)) for b in bars]
        ma = sum(closes) / max(1, len(closes))
        if ma <= 0:
            return 0.0

        ma_dist = (last_price - ma) / ma  # positive => above MA
        scale = float(self.technicalcfg.ma_distance_norm_scale)
        if scale <= 0:
            scale = 0.05

        # Mean reversion prefers: below MA and oversold => positive score (buy),
        # above MA and overbought => negative score (sell).
        score = -(ma_dist / scale)

        if rsi is not None:
            if rsi >= float(self.technicalcfg.rsi_overbought):
                score -= 0.3
            elif rsi <= float(self.technicalcfg.rsi_oversold):
                score += 0.3

        return max(-1.0, min(1.0, score))

    def _compute_price_action_score(self, bars: List, last_price: float) -> float:
        if not bars:
            return 0.0

        lookback = int(self.technicalcfg.breakout_lookback_bars)
        lookback = max(5, lookback)
        window = bars[-lookback:] if len(bars) >= lookback else bars

        highs = [float(getattr(b, "h", b.h)) for b in window]
        lows = [float(getattr(b, "l", b.l)) for b in window]
        hi = max(highs) if highs else last_price
        lo = min(lows) if lows else last_price

        breakout_strength = float(self.technicalcfg.breakout_strength)
        if breakout_strength <= 0:
            breakout_strength = 1.0

        # Simple breakout score: above range => positive, below range => negative.
        if last_price > hi:
            raw = (last_price - hi) / max(1e-9, hi)
            score = breakout_strength * raw * 10.0
        elif last_price < lo:
            raw = (lo - last_price) / max(1e-9, lo)
            score = -breakout_strength * raw * 10.0
        else:
            score = 0.0

        return max(-1.0, min(1.0, score))

    def _combine_technical_scores(self, momentum: float, mean_reversion: float, price_action: float) -> float:
        w_m = float(self.technicalcfg.weight_momentum_trend)
        w_r = float(self.technicalcfg.weight_mean_reversion)
        w_p = float(self.technicalcfg.weight_price_action)
        denom = max(1e-9, (abs(w_m) + abs(w_r) + abs(w_p)))
        s = (w_m * momentum + w_r * mean_reversion + w_p * price_action) / denom
        return max(-1.0, min(1.0, s))

    def _decide_side_and_bands(self, last_price: float, volatility: float, signal_score: float):
        long_th = float(self.technicalcfg.long_threshold)
        short_th = float(self.technicalcfg.short_threshold)

        side = "skip"
        if signal_score >= long_th:
            side = "buy"
        elif signal_score <= short_th:
            side = "sell"

        # Use price-based volatility proxy for bands.
        # If volatility is tiny, fall back to a minimal band.
        vol_px = max(0.0025 * last_price, volatility * last_price)

        stop_mult = float(self.technicalcfg.base_stop_vol_mult)
        tp_mult = float(self.technicalcfg.base_tp_vol_mult)

        tp_scale = 1.0
        # Scale TP slightly with conviction but clamp.
        min_tp_scale = float(self.technicalcfg.min_tp_scale_from_signal)
        max_tp_scale = float(self.technicalcfg.max_tp_scale_from_signal)
        tp_scale = max(min_tp_scale, min(max_tp_scale, 1.0 + 0.3 * abs(signal_score)))

        if side == "buy":
            stop = last_price - stop_mult * vol_px
            tp = last_price + tp_mult * vol_px * tp_scale
        elif side == "sell":
            stop = last_price + stop_mult * vol_px
            tp = last_price - tp_mult * vol_px * tp_scale
        else:
            # Still return bands, but they won't be used.
            stop = last_price
            tp = last_price

        return side, float(stop), float(tp)

    def generate_signal_for_symbol(self, symbol: str) -> Signal:
        last_trade = self.adapter.get_last_quote(symbol)
        bars = self.adapter.get_recent_bars(symbol, timeframe="5Min", lookback_bars=30)

        momentum_raw = self._compute_simple_momentum_raw(bars)
        trend_raw = self._compute_trend_signal_raw(bars)
        volatility = self._compute_volatility(bars)
        rsi = self._compute_rsi(bars)

        momentum_score = self._normalize_momentum_trend(momentum_raw, trend_raw)
        mean_reversion_score = self._normalize_mean_reversion(last_trade, bars, rsi)
        price_action_score = self._compute_price_action_score(bars, last_trade)

        signal_score = self._combine_technical_scores(
            momentum_score, mean_reversion_score, price_action_score
        )

        side, stop_price, take_profit_price = self._decide_side_and_bands(
            last_price=last_trade,
            volatility=volatility,
            signal_score=signal_score,
        )

        if side == "skip":
            rationale = (
                f"Composite technical signal_score={signal_score:.3f} "
                f"within neutral band; no trade."
            )
        elif side == "buy":
            rationale = (
                f"Long bias from composite technicals: signal_score={signal_score:.3f}, "
                f"mom={momentum_score:.3f}, mr={mean_reversion_score:.3f}, "
                f"pa={price_action_score:.3f}."
            )
        else:
            rationale = (
                f"Short bias from composite technicals: signal_score={signal_score:.3f}, "
                f"mom={momentum_score:.3f}, mr={mean_reversion_score:.3f}, "
                f"pa={price_action_score:.3f}."
            )

        # Always log composite and factor scores, regardless of trade decision.
        log_signal_score(
            symbol=symbol,
            signal_score=signal_score,
            momentum_score=momentum_score,
            mean_reversion_score=mean_reversion_score,
            price_action_score=price_action_score,
            env_mode=ENV_MODE,
        )

        # (Suggestion 3) If we have no technical setup, do NOT fetch news and do NOT call AI.
        if side == "skip":
            s_result = SentimentResult(
                score=0.0,
                raw_discrete=0,
                rawcompound=0.0,
                ndocuments=0,
                explanation="Skipped AI sentiment due to neutral technical signal.",
                confidence=0.0,
            )
            return Signal(
                symbol=symbol,
                side=side,
                rationale=rationale,
                sentiment_result=s_result,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                signal_score=signal_score,
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion_score,
                price_action_score=price_action_score,
            )

        # News → sentiment (Suggestions 1, 2, 6 are enforced inside SentimentModule)
        news_items = self._get_news_items(symbol)
        s_result = self.sentiment.scorenewsitems(symbol, news_items)
        log_sentiment_for_symbol(symbol, s_result, ENV_MODE)

        return Signal(
            symbol=symbol,
            side=side,
            rationale=rationale,
            sentiment_result=s_result,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            signal_score=signal_score,
            momentum_score=momentum_score,
            mean_reversion_score=mean_reversion_score,
            price_action_score=price_action_score,
        )

