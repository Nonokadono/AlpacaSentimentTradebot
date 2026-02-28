# CHANGES:
# Change 3 — Sentiment-adjusted dynamic entry threshold:
#   _decide_side_and_bands gains optional keyword argument symbol="" and computes
#   long_th_adj / short_th_adj from the most recently cached sentiment score for
#   the symbol. Gated behind TechnicalSignalConfig.enable_dynamic_threshold
#   (default False) — when False, long_th_adj == long_th and short_th_adj ==
#   short_th, preserving existing behaviour exactly.
#   generate_signal_for_symbol() threads symbol through to the call site.
#   No existing variable names renamed.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import math

from adapters.alpaca_adapter import AlpacaAdapter
from config.config import ENV_MODE, TechnicalSignalConfig
from monitoring.monitor import log_instrument_report

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
    # Feature 3: expose per-symbol volatility so PortfolioBuilder can forward
    # it to RiskEngine.pre_trade_checks() for Half-Kelly sizing.
    volatility: float = 0.0


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
    ):
        self.adapter = adapter
        self.sentiment = sentiment
        self.technicalcfg = technicalcfg

    def _compute_simple_momentum_raw(self, bars: List) -> float:
        """
        Simple momentum: (Current Close - Close N bars ago) / Close N bars ago
        """
        if not bars or len(bars) < 10:
            return 0.0

        # fallback if not enough bars for full lookback, just use what we have
        lookback = min(len(bars) - 1, 10)
        current = bars[-1].c
        past = bars[-1 - lookback].c

        if past == 0:
            return 0.0
        return (current - past) / past

    def _compute_trend_signal_raw(self, bars: List) -> float:
        """
        Simple trend proxy: is price above SMA(20)?
        Returns +1 if > SMA, -1 if < SMA, 0 otherwise.
        """
        if len(bars) < 20:
            return 0.0

        closes = [b.c for b in bars]
        # last 20 bars
        window = closes[-20:]
        sma = sum(window) / len(window)
        current = closes[-1]

        if current > sma:
            return 1.0
        elif current < sma:
            return -1.0
        return 0.0

    def _compute_volatility(self, bars: List) -> float:
        """
        ATR-like proxy or simple std dev of last N closes.
        Let's use a percentage volatility proxy: Stdev(returns) * sqrt(bars)
        """
        if len(bars) < 5:
            return 0.01  # fallback

        closes = [b.c for b in bars]
        returns = []
        for i in range(1, len(closes)):
            r = (closes[i] - closes[i-1]) / closes[i-1]
            returns.append(r)

        if not returns:
            return 0.01

        mean_ret = sum(returns) / len(returns)
        sq_diffs = [(r - mean_ret)**2 for r in returns]
        variance = sum(sq_diffs) / len(returns)
        std_dev = math.sqrt(variance)

        # Annualized-ish or per-bar? Just use per-bar std dev as vol proxy
        return std_dev

    def _compute_rsi(self, bars: List, period: int = 14) -> float:
        """
        Wilder's EMA-smoothed RSI.

        Algorithm:
          1. Require at least period+1 closes (unchanged guard — returns 50.0
             when insufficient data).
          2. Compute per-bar price changes for ALL available bars.
          3. Seed avg_gain / avg_loss with a plain SMA of the FIRST `period`
             deltas — this is the standard Wilder initialisation.
          4. Roll Wilder's EMA forward over all remaining deltas:
               avg_gain = (avg_gain * (period - 1) + gain) / period
               avg_loss = (avg_loss * (period - 1) + loss) / period
          5. Compute RS and RSI from the final smoothed averages.

        This produces the same values as TradingView / most professional
        charting libraries, unlike the previous simple-average approach which
        used all gains/losses pooled indiscriminately and returned a biased
        result.
        """
        if len(bars) < period + 1:
            return 50.0

        closes = [b.c for b in bars]
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        # Step 3: seed with plain SMA of first `period` deltas
        seed_gains = [max(d, 0.0) for d in deltas[:period]]
        seed_losses = [abs(min(d, 0.0)) for d in deltas[:period]]
        avg_gain = sum(seed_gains) / period
        avg_loss = sum(seed_losses) / period

        # Step 4: Wilder's EMA smoothing over all remaining deltas
        for delta in deltas[period:]:
            gain = max(delta, 0.0)
            loss = abs(min(delta, 0.0))
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        # Step 5: RSI
        if avg_loss == 0.0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _normalize_momentum_trend(self, mom_raw: float, trend_raw: float) -> float:
        # Scale momentum: if mom is 1%, that's huge in 5min bars.
        # standardizing roughly: mom / 0.005 -> clamped
        mom_score = max(-1.0, min(1.0, mom_raw / self.technicalcfg.momentum_norm_scale))

        # Combine with trend (-1 or 1)
        # 70% momentum value, 30% trend direction
        combined = 0.7 * mom_score + 0.3 * trend_raw
        return combined

    def _normalize_mean_reversion(self, current_price: float, bars: List, rsi: float) -> float:
        """
        Mean reversion score:
        High RSI (>70) -> Negative score (expect pullback)
        Low RSI (<30) -> Positive score (expect bounce)

        Also distance from MA: if price >> MA, revert down (-).
        """
        # RSI component
        rsi_score = 0.0
        if rsi > self.technicalcfg.rsi_overbought:
            # e.g. 75 -> -0.5
            rsi_score = -1.0 * (rsi - 70) / 30.0
        elif rsi < self.technicalcfg.rsi_oversold:
            # e.g. 25 -> +0.5
            rsi_score = 1.0 * (30 - rsi) / 30.0

        # MA distance component
        if len(bars) < 20:
            ma_score = 0.0
        else:
            closes = [b.c for b in bars[-20:]]
            sma = sum(closes) / len(closes)
            # (price - sma) / sma
            dist = (current_price - sma) / sma
            # if dist is +1%, score is negative (revert)
            # scale: 0.05 (5%) -> full -1.0
            ma_score = -1.0 * (dist / self.technicalcfg.ma_distance_norm_scale)

        return max(-1.0, min(1.0, 0.5 * rsi_score + 0.5 * ma_score))

    def _compute_price_action_score(self, bars: List, current_price: float) -> float:
        """
        Simple breakout detection:
        If current price > highest of last N bars -> +1 (Breakout)
        If current price < lowest of last N bars -> -1 (Breakdown)
        """
        if len(bars) < self.technicalcfg.breakout_lookback_bars:
            return 0.0

        window = bars[-self.technicalcfg.breakout_lookback_bars:-1]  # exclude current
        highs = [b.h for b in window]
        lows = [b.l for b in window]

        recent_high = max(highs)
        recent_low = min(lows)

        if current_price > recent_high:
            return 1.0
        elif current_price < recent_low:
            return -1.0

        return 0.0

    def _combine_technical_scores(
        self, mom: float, mr: float, pa: float
    ) -> Tuple[float, bool]:
        """
        Compute the weighted composite signal score and flag whether the
        momentum/mean-reversion conflict dampener was applied.

        Weights from config (unchanged):
            weight_momentum_trend  * mom
            weight_mean_reversion  * mr
            weight_price_action    * pa

        Conflict dampener:
            When momentum_score (mom) and mean_reversion_score (mr) have
            OPPOSITE signs their product is negative, indicating the move is
            already extended (momentum maxed) while mean-reversion pushes back.
            In that case the raw weighted sum is multiplied by
            `conflict_dampener_penalty` (default 0.6), reducing apparent
            conviction by 40%.

            Same-sign or zero sub-signals (product >= 0) are NOT penalised —
            existing behaviour for clean / aligned setups is fully preserved.

        Returns:
            (signal_score: float, dampened: bool)
        """
        raw = (
            self.technicalcfg.weight_momentum_trend * mom
            + self.technicalcfg.weight_mean_reversion * mr
            + self.technicalcfg.weight_price_action * pa
        )

        # Apply dampener only when momentum and mean-reversion actively conflict.
        dampened = False
        if mom * mr < 0:
            penalty = float(getattr(self.technicalcfg, "conflict_dampener_penalty", 0.6))
            raw = raw * penalty
            dampened = True

        return max(-1.0, min(1.0, raw)), dampened

    def _decide_side_and_bands(
        self,
        last_price: float,
        volatility: float,
        signal_score: float,
        symbol: str = "",
    ) -> Tuple[str, float, float]:
        """
        Determine the trade side and compute stop/take-profit price bands.

        Change 3: When TechnicalSignalConfig.enable_dynamic_threshold is True,
        the raw long_th and short_th values are adjusted by the most recently
        cached sentiment for `symbol` before the comparison against signal_score.
        When enable_dynamic_threshold is False (the default), long_th_adj ==
        long_th and short_th_adj == short_th, preserving existing behaviour
        exactly.

        The config object is NEVER mutated — adjustments are local variables only.
        """
        long_th = float(self.technicalcfg.long_threshold)
        short_th = float(self.technicalcfg.short_threshold)

        # Change 3: sentiment-adjusted thresholds (local only, config not mutated).
        # When enable_dynamic_threshold is False (default), adj == raw threshold.
        if getattr(self.technicalcfg, "enable_dynamic_threshold", False) and symbol:
            cached_s_result = self.sentiment.get_cached_sentiment(symbol)
            cached_s = cached_s_result.score if cached_s_result is not None else 0.0
            s_adj = max(-0.5, min(0.5, cached_s))
            long_th_adj  = long_th  * (1.0 - 0.35 * s_adj / 0.5)
            short_th_adj = short_th * (1.0 - 0.35 * s_adj / 0.5)
        else:
            long_th_adj  = long_th
            short_th_adj = short_th

        side = "skip"
        if signal_score >= long_th_adj:
            side = "buy"
        elif signal_score <= short_th_adj:
            side = "sell"

        # Use price-based volatility proxy for bands.
        # If volatility is tiny, fall back to a minimal band.
        vol_px = max(0.0025 * last_price, volatility * last_price)

        stop_mult = float(self.technicalcfg.base_stop_vol_mult)
        tp_mult = float(self.technicalcfg.base_tp_vol_mult)

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

    def _get_news_items(self, symbol: str) -> List[Dict]:
        """
        Fetch news from Alpaca adapter.
        Here we just ask for latest 10 items.
        """
        return self.adapter.get_news(symbol, limit=10)

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

        signal_score, dampened = self._combine_technical_scores(
            momentum_score, mean_reversion_score, price_action_score
        )

        # Change 3: thread symbol through so _decide_side_and_bands can look up
        # the cached sentiment for the dynamic threshold adjustment.
        side, stop_price, take_profit_price = self._decide_side_and_bands(
            last_price=last_trade,
            volatility=volatility,
            signal_score=signal_score,
            symbol=symbol,
        )

        dampener_tag = " [CONFLICT DAMPENED]" if dampened else ""

        if side == "skip":
            rationale = (
                f"Composite technical signal_score={signal_score:.3f}{dampener_tag} "
                f"within neutral band; no trade."
            )
        elif side == "buy":
            rationale = (
                f"Long bias from composite technicals: signal_score={signal_score:.3f}{dampener_tag}, "
                f"momentum={momentum_score:.3f}, mean_reversion={mean_reversion_score:.3f}, "
                f"price_action={price_action_score:.3f}."
            )
        else:
            rationale = (
                f"Short bias from composite technicals: signal_score={signal_score:.3f}{dampener_tag}, "
                f"momentum={momentum_score:.3f}, mean_reversion={mean_reversion_score:.3f}, "
                f"price_action={price_action_score:.3f}."
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
            log_instrument_report(
                symbol=symbol,
                signal_score=signal_score,
                sentiment=s_result,
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
                volatility=volatility,
            )

        # News -> sentiment (cost controls 1, 2, 6 are enforced inside SentimentModule)
        news_items = self._get_news_items(symbol)
        s_result = self.sentiment.scorenewsitems(symbol, news_items)

        # Single unified report — includes live sentiment + all four technical scores.
        log_instrument_report(
            symbol=symbol,
            signal_score=signal_score,
            sentiment=s_result,
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
            volatility=volatility,
        )
