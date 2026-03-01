# CHANGES:
# FIX 5 — generate_signal_for_symbol(): lookback_bars changed from 30 to 45.
#   With period=14, 30 bars gave only 15 smoothing steps (minimum viable).
#   45 bars gives 44 deltas: 14 for the Wilder seed + 30 for smoothing,
#   meeting the conventional 3x-period stability threshold.
#
# FIX 6 — Added _compute_atr() private method that computes the standard
#   Average True Range using bar.h, bar.l, and bars[i-1].c.
#   TR(i) = max(h-l, |h - prev_c|, |l - prev_c|).  ATR = simple mean of
#   last `period` TRs.  Requires period+1 bars; returns 0.0 if insufficient.
#   Returns a price distance (not a ratio).
#   In _decide_side_and_bands(): vol_px now uses atr (with 0.25% fallback)
#   instead of the std-dev-based proxy.  The `volatility` parameter is
#   unchanged in the signature and still used by Kelly sizing via
#   _compute_volatility() -> pre_trade_checks().
#   _decide_side_and_bands() now also accepts `bars` as a parameter so it can
#   call _compute_atr().  generate_signal_for_symbol() passes bars through.
#
# FIX 7 — _compute_simple_momentum_raw(): EMA crossover normalisation now uses
#   ema_crossover_norm_scale (new TechnicalSignalConfig field, default 0.10)
#   instead of momentum_norm_scale (0.05).  Prevents saturation at ±1 for
#   high-momentum equities where EMA divergence exceeds 5%.  getattr fallback
#   ensures backward compatibility if the field is absent on older configs.
#   momentum_norm_scale is still used by _normalize_momentum_trend() unchanged.
#
# All prior changes (Fix M1, Fix M2, Fix M3, Fix L3, Change 3, Improvement B,
# Wilder's RSI) are preserved unchanged.

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
        EMA-8 vs EMA-21 crossover signal, normalised to [-1, 1].

        Fix M1: replaced the raw 10-bar return with a fast/slow EMA crossover.
        - Requires at least 22 bars; returns 0.0 if fewer are available.
        - EMA seeded with simple average of first `period` closes.
        - Recursive formula: ema = price * k + ema_prev * (1 - k), k = 2/(period+1).
        - Signal = (ema_fast - ema_slow) / ema_slow, then / ema_crossover_norm_scale, clamped.

        FIX 7: normalisation now uses ema_crossover_norm_scale (default 0.10) via
        getattr fallback, replacing momentum_norm_scale (0.05) which caused ±1
        saturation for high-momentum equities where EMA divergence exceeds 5%.
        Method name and signature are unchanged.
        """
        fast_period = 8
        slow_period = 21
        min_bars = slow_period + 1  # need at least 22 closes

        if not bars or len(bars) < min_bars:
            return 0.0

        closes = [b.c for b in bars]

        k_fast = 2.0 / (fast_period + 1)
        k_slow = 2.0 / (slow_period + 1)

        # Seed fast EMA with SMA of first fast_period bars
        ema_fast = sum(closes[:fast_period]) / fast_period
        for price in closes[fast_period:]:
            ema_fast = price * k_fast + ema_fast * (1.0 - k_fast)

        # Seed slow EMA with SMA of first slow_period bars
        ema_slow = sum(closes[:slow_period]) / slow_period
        for price in closes[slow_period:]:
            ema_slow = price * k_slow + ema_slow * (1.0 - k_slow)

        if ema_slow == 0.0:
            return 0.0

        raw_crossover = (ema_fast - ema_slow) / ema_slow
        # FIX 7: use ema_crossover_norm_scale (wider range) for graded signal.
        # getattr fallback to momentum_norm_scale ensures backward compatibility.
        norm_scale = getattr(self.technicalcfg, "ema_crossover_norm_scale",
                             self.technicalcfg.momentum_norm_scale)
        norm = raw_crossover / norm_scale
        return max(-1.0, min(1.0, norm))

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
        Uses percentage volatility proxy: Stdev(returns) with Bessel's correction.

        Fix M2: replaced population variance (/ n) with sample variance (/ (n-1)).
        Added guard: if len(returns) < 2, return 0.01 default.
        """
        if len(bars) < 5:
            return 0.01  # fallback

        closes = [b.c for b in bars]
        returns = []
        for i in range(1, len(closes)):
            r = (closes[i] - closes[i-1]) / closes[i-1]
            returns.append(r)

        # Fix M2: guard for insufficient data
        if len(returns) < 2:
            return 0.01

        mean_ret = sum(returns) / len(returns)
        sq_diffs = [(r - mean_ret)**2 for r in returns]
        # Fix M2: Bessel's correction — divide by (n-1) not n
        variance = sum(sq_diffs) / (len(returns) - 1)
        std_dev = math.sqrt(variance)

        # Per-bar std dev as vol proxy
        return std_dev

    def _compute_atr(self, bars: List, period: int = 14) -> float:
        """
        FIX 6: Standard Average True Range computed from bar.h, bar.l, and
        the previous bar's bar.c.

        True Range for bar i:
            TR(i) = max(bar.h - bar.l,
                        abs(bar.h - bars[i-1].c),
                        abs(bar.l - bars[i-1].c))

        ATR = simple average of TR over the last `period` bars.

        Requires at least period+1 bars to compute one TR per bar; returns
        0.0 if insufficient data.

        Returns ATR as a price distance (not a ratio).
        """
        if len(bars) < period + 1:
            return 0.0

        trs = []
        for i in range(1, len(bars)):
            h = bars[i].h
            l = bars[i].l
            prev_c = bars[i - 1].c
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)

        # Use the last `period` true ranges for the ATR average.
        recent_trs = trs[-period:]
        if not recent_trs:
            return 0.0
        return sum(recent_trs) / len(recent_trs)

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

        Fix M3: returns 0.0 immediately when the market is closed to prevent
        the permanent -1.0 artifact that appears outside regular trading hours
        (when the current bar's price equals the most recent close and sits
        below the historical high of the lookback window).
        """
        # Fix M3: suppress price action score when market is closed.
        if not self.adapter.get_market_open():
            return 0.0

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
            # Fix L3: direct attribute access — field is now explicit in TechnicalSignalConfig.
            penalty = float(self.technicalcfg.conflict_dampener_penalty)
            raw = raw * penalty
            dampened = True

        return max(-1.0, min(1.0, raw)), dampened

    def _decide_side_and_bands(
        self,
        last_price: float,
        volatility: float,
        signal_score: float,
        symbol: str = "",
        bars: List = None,
    ) -> Tuple[str, float, float]:
        """
        Determine the trade side and compute stop/take-profit price bands.

        FIX 6: vol_px is now derived from ATR (_compute_atr()) instead of the
        std-dev-based proxy.  ATR uses bar.h, bar.l, and bars[i-1].c for a
        proper measure of true daily range.  If ATR is zero or bars are
        unavailable, falls back to 0.25% of last_price.
        The `volatility` parameter (std-dev) remains in the signature and is
        still used downstream by _compute_volatility() -> pre_trade_checks() ->
        _kelly_fraction() for Kelly sizing.

        Improvement B: When TechnicalSignalConfig.enable_dynamic_threshold is
        True, thresholds are adjusted asymmetrically using sentiment_th_scale:
          s_adj = clamp(cached_s, -0.5, 0.5)
          long_th_adj  = long_th  * (1.0 - sentiment_th_scale * s_adj / 0.5)
          short_th_adj = short_th * (1.0 + sentiment_th_scale * s_adj / 0.5)

        Sign logic (CONFIRMED CORRECT):
          short_th is negative.
          Positive s_adj (bullish sentiment):
            long_th_adj  < long_th   -> easier to go long  (threshold shrinks)
            short_th_adj > short_th  -> harder to go short (negative * (1+x) is
                                        more negative, i.e. a stricter bar)
          Negative s_adj (bearish sentiment):
            long_th_adj  > long_th   -> harder to go long
            short_th_adj < short_th  -> easier to go short

        The config object is NEVER mutated — adjustments are local variables only.
        When enable_dynamic_threshold is False (the default), long_th_adj ==
        long_th and short_th_adj == short_th, preserving existing behaviour.
        """
        long_th = float(self.technicalcfg.long_threshold)
        short_th = float(self.technicalcfg.short_threshold)

        # Improvement B: asymmetric sentiment-adjusted thresholds (local only).
        # When enable_dynamic_threshold is False (default), adj == raw threshold.
        if getattr(self.technicalcfg, "enable_dynamic_threshold", False) and symbol:
            cached_s_result = self.sentiment.get_cached_sentiment(symbol)
            cached_s = cached_s_result.score if cached_s_result is not None else 0.0
            sentiment_th_scale = float(
                getattr(self.technicalcfg, "sentiment_th_scale", 0.25)
            )
            s_adj = max(-0.5, min(0.5, cached_s))
            # Asymmetric adjustment:
            # long_th_adj  shrinks when sentiment positive (easier to go long).
            # short_th_adj becomes more negative when sentiment positive
            # (harder to go short) because short_th is negative and
            # (1.0 + positive_value) > 1.0, making the product more negative.
            long_th_adj  = long_th  * (1.0 - sentiment_th_scale * s_adj / 0.5)
            short_th_adj = short_th * (1.0 + sentiment_th_scale * s_adj / 0.5)
        else:
            long_th_adj  = long_th
            short_th_adj = short_th

        side = "skip"
        if signal_score >= long_th_adj:
            side = "buy"
        elif signal_score <= short_th_adj:
            side = "sell"

        # FIX 6: use ATR-based vol_px for stop/TP band calculation.
        # ATR gives a proper true-range measure; std-dev proxy underestimates.
        if bars:
            atr = self._compute_atr(bars)
        else:
            atr = 0.0
        if atr <= 0.0:
            atr = 0.0025 * last_price   # fallback: 0.25% of price
        vol_px = atr

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
        # FIX 5: increased from 30 to 45 bars (3x period=14) for stable Wilder RSI.
        bars = self.adapter.get_recent_bars(symbol, timeframe="5Min", lookback_bars=45)

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

        # Change 3: thread symbol and bars through so _decide_side_and_bands can
        # look up the cached sentiment for dynamic threshold adjustment (Improvement B)
        # and compute ATR from bar.h / bar.l / bar.c (FIX 6).
        side, stop_price, take_profit_price = self._decide_side_and_bands(
            last_price=last_trade,
            volatility=volatility,
            signal_score=signal_score,
            symbol=symbol,
            bars=bars,
        )

        dampener_tag = " [CONFLICT DAMPENED]" if dampened else ""

        # AI cost control (3): skip sentiment entirely when the technical signal
        # is neutral — saves an API call.
        if side == "skip":
            neutral_sentiment = SentimentResult(
                score=0.0,
                raw_discrete=0,
                rawcompound=0.0,
                ndocuments=0,
                explanation="Technical signal neutral; sentiment not evaluated.",
                confidence=0.0,
            )
            return Signal(
                symbol=symbol,
                side="skip",
                rationale=f"Technical signal neutral{dampener_tag}",
                sentiment_result=neutral_sentiment,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                signal_score=signal_score,
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion_score,
                price_action_score=price_action_score,
                volatility=volatility,
            )

        news_items = self._get_news_items(symbol)
        sentiment_result = self.sentiment.scorenewsitems(symbol, news_items)

        # Sentiment no-trade block
        if sentiment_result.score < self.sentiment.cfg.no_trade_negative_threshold:
            return Signal(
                symbol=symbol,
                side="skip",
                rationale=(
                    f"Sentiment too negative: {sentiment_result.score:.3f} "
                    f"< threshold {self.sentiment.cfg.no_trade_negative_threshold}"
                    f"{dampener_tag}"
                ),
                sentiment_result=sentiment_result,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                signal_score=signal_score,
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion_score,
                price_action_score=price_action_score,
                volatility=volatility,
            )

        rationale = (
            f"side={side} signal={signal_score:.3f} "
            f"mom={momentum_score:.3f} mr={mean_reversion_score:.3f} "
            f"pa={price_action_score:.3f} "
            f"sentiment={sentiment_result.score:.3f}{dampener_tag}"
        )

        log_instrument_report(
            symbol=symbol,
            signal_score=signal_score,
            sentiment=sentiment_result,
            momentum_score=momentum_score,
            mean_reversion_score=mean_reversion_score,
            price_action_score=price_action_score,
            env_mode=ENV_MODE,
        )

        return Signal(
            symbol=symbol,
            side=side,
            rationale=rationale,
            sentiment_result=sentiment_result,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            signal_score=signal_score,
            momentum_score=momentum_score,
            mean_reversion_score=mean_reversion_score,
            price_action_score=price_action_score,
            volatility=volatility,
        )
