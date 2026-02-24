# core/signals.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import math

from adapters.alpaca_adapter import AlpacaAdapter
from .sentiment import SentimentModule, SentimentResult
from monitoring.monitor import log_sentiment_for_symbol, log_signal_score
from config.config import ENV_MODE, TechnicalSignalConfig


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
    - Mean-reversion/overbought
    - Price action / structure
    """

    def __init__(
        self,
        adapter: AlpacaAdapter,
        sentiment: SentimentModule,
        technical_cfg: TechnicalSignalConfig,
    ):
        self.adapter = adapter
        self.sentiment = sentiment
        self.technical_cfg = technical_cfg
        self._last_news_timestamp: Dict[str, datetime] = {}

    # --- News integration ---

    def _get_news_items(self, symbol: str) -> List[dict]:
        now = datetime.utcnow()
        last_ts = self._last_news_timestamp.get(symbol)
        if last_ts is None:
            since = now - timedelta(hours=6)
        else:
            since = last_ts

        items = self.adapter.get_news(symbol, since=since, limit=20)
        self._last_news_timestamp[symbol] = now
        return items

    # --- Technical helpers ---

    def _compute_simple_momentum_raw(self, bars: List) -> float:
        if len(bars) < 2:
            return 0.0
        first = float(getattr(bars[0], "c"))
        last = float(getattr(bars[-1], "c"))
        if first <= 0:
            return 0.0
        return (last / first) - 1.0

    def _compute_trend_signal_raw(self, bars: List) -> float:
        if len(bars) < 10:
            return 0.0
        closes = [float(getattr(b, "c")) for b in bars]
        short_n = min(5, len(closes))
        long_n = min(15, len(closes))
        short_ma = sum(closes[-short_n:]) / short_n
        long_ma = sum(closes[-long_n:]) / long_n
        if long_ma <= 0:
            return 0.0
        return (short_ma / long_ma) - 1.0

    def _compute_volatility(self, bars: List) -> float:
        if len(bars) < 5:
            return 0.0
        closes = [float(getattr(b, "c")) for b in bars]
        rets: List[float] = []
        for i in range(1, len(closes)):
            if closes[i - 1] <= 0:
                continue
            rets.append(math.log(closes[i] / closes[i - 1]))
        if not rets:
            return 0.0
        mean_ret = sum(rets) / len(rets)
        var = sum((r - mean_ret) ** 2 for r in rets) / len(rets)
        return math.sqrt(var)

    def _compute_rsi(self, bars: List, period: int = 14) -> Optional[float]:
        if len(bars) < period + 1:
            return None
        closes = [float(getattr(b, "c")) for b in bars]
        gains: List[float] = []
        losses: List[float] = []
        for i in range(1, period + 1):
            diff = closes[-i] - closes[-i - 1]
            if diff >= 0:
                gains.append(diff)
            else:
                losses.append(-diff)
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_loss == 0:
            if avg_gain == 0:
                return 50.0
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1 + rs))

    def _normalize_momentum_trend(
        self,
        momentum_raw: float,
        trend_raw: float,
    ) -> float:
        scale = self.technical_cfg.momentum_norm_scale
        composite = momentum_raw + trend_raw
        if scale <= 0:
            return 0.0
        x = composite / scale
        return max(-1.0, min(1.0, x))

    def _normalize_mean_reversion(
        self,
        last_price: float,
        bars: List,
        rsi: Optional[float],
    ) -> float:
        if not bars or last_price <= 0:
            return 0.0

        closes = [float(getattr(b, "c")) for b in bars]
        ma_period = min(20, len(closes))
        ma = sum(closes[-ma_period:]) / ma_period if ma_period > 0 else last_price

        distance = (last_price - ma) / ma if ma > 0 else 0.0
        dist_scaled = 0.0
        if self.technical_cfg.ma_distance_norm_scale > 0:
            dist_scaled = distance / self.technical_cfg.ma_distance_norm_scale
            dist_scaled = max(-1.0, min(1.0, dist_scaled))

        rsi_scaled = 0.0
        if rsi is not None:
            rsi_scaled = 1.0 - 2.0 * (rsi / 100.0)
            rsi_scaled = max(-1.0, min(1.0, rsi_scaled))

        mr_raw = 0.5 * rsi_scaled + 0.5 * (-dist_scaled)
        return max(-1.0, min(1.0, mr_raw))

    def _compute_price_action_score(
        self,
        bars: List,
        last_price: float,
    ) -> float:
        if len(bars) < 5:
            return 0.0

        closes = [float(getattr(b, "c")) for b in bars]
        highs = [float(getattr(b, "h", getattr(b, "c"))) for b in bars]
        lows = [float(getattr(b, "l", getattr(b, "c"))) for b in bars]

        lookback = min(self.technical_cfg.breakout_lookback_bars, len(bars))
        recent_high = max(highs[-lookback:])
        recent_low = min(lows[-lookback:])

        breakout_score = 0.0
        if last_price > recent_high:
            breakout_score = 1.0
        elif last_price < recent_low:
            breakout_score = -1.0

        last_bar = bars[-1]
        o = float(getattr(last_bar, "o", closes[-1]))
        h = float(getattr(last_bar, "h", closes[-1]))
        l = float(getattr(last_bar, "l", closes[-1]))
        c = float(getattr(last_bar, "c", closes[-1]))

        body = abs(c - o)
        range_total = max(h - l, 1e-6)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        candle_score = 0.0
        if lower_wick / range_total > 0.5 and body / range_total < 0.3:
            candle_score = 0.5
        elif upper_wick / range_total > 0.5 and body / range_total < 0.3:
            candle_score = -0.5
        elif body / range_total > 0.6:
            candle_score = 0.5 if c > o else -0.5

        pa_raw = breakout_score + candle_score
        return max(-1.0, min(1.0, pa_raw))

    def _combine_technical_scores(
        self,
        mom_score: float,
        mr_score: float,
        pa_score: float,
    ) -> float:
        w_mom = self.technical_cfg.weight_momentum_trend
        w_mr = self.technical_cfg.weight_mean_reversion
        w_pa = self.technical_cfg.weight_price_action

        numerator = w_mom * mom_score + w_mr * mr_score + w_pa * pa_score
        denom = abs(w_mom) + abs(w_mr) + abs(w_pa)
        if denom == 0:
            return 0.0
        score = numerator / denom
        return max(-1.0, min(1.0, score))

    def _decide_side_and_bands(
        self,
        last_price: float,
        volatility: float,
        signal_score: float,
    ) -> Tuple[str, float, float]:
        if last_price <= 0:
            return "skip", last_price, last_price

        if signal_score >= self.technical_cfg.long_threshold:
            side = "buy"
        elif signal_score <= self.technical_cfg.short_threshold:
            side = "sell"
        else:
            return "skip", last_price, last_price

        vol_floor = 0.001
        vol_cap = 0.05
        v = max(vol_floor, min(vol_cap, volatility))

        stop_mult = self.technical_cfg.base_stop_vol_mult
        tp_mult_base = self.technical_cfg.base_tp_vol_mult

        tp_scale_from_signal = 1.0 + 0.3 * signal_score
        tp_scale_from_signal = max(
            self.technical_cfg.min_tp_scale_from_signal,
            min(self.technical_cfg.max_tp_scale_from_signal, tp_scale_from_signal),
        )
        tp_mult = tp_mult_base * tp_scale_from_signal

        stop_dist = stop_mult * v * last_price
        tp_dist = tp_mult * v * last_price

        if side == "buy":
            stop_price = last_price - stop_dist
            tp_price = last_price + tp_dist
        else:
            stop_price = last_price + stop_dist
            tp_price = last_price - tp_dist

        return side, stop_price, tp_price

    # --- Public API ---

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

        # News → sentiment
        news_items = self._get_news_items(symbol)
        s_result = self.sentiment.scorenewsitems(symbol, news_items)
        log_sentiment_for_symbol(symbol, s_result, ENV_MODE)

        # Always log composite and factor scores, regardless of trade decision
        log_signal_score(
            symbol=symbol,
            signal_score=signal_score,
            momentum_score=momentum_score,
            mean_reversion_score=mean_reversion_score,
            price_action_score=price_action_score,
            env_mode=ENV_MODE,
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
