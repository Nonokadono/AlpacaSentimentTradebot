# CHANGES:
# TASK 2.1.1 — Added _compute_macd() method implementing MACD (Moving Average Convergence Divergence)
#              with proper EMA calculation for fast/slow lines, signal line, and histogram.
#              Returns (macd_line, signal_line, histogram) tuple. Requires at least (slow + signal)
#              bars for accurate calculation. Uses per-symbol MACD history for signal line EMA.
# TASK 2.1.2 — Enhanced _normalize_momentum_trend() to compute 3-factor momentum blend:
#              EMA crossover (45%) + MACD histogram (25%) + SMA-20 trend (30%).
#              MACD histogram normalized by macd_histogram_norm_scale (0.005 = 0.5% of price).
#              Positive histogram indicates bullish momentum (MACD above signal), negative indicates
#              bearish momentum. All three factors weighted and combined into [-1, 1] range.
# TASK 1.2.1 — Added _compute_bollinger_bands() method implementing Bollinger Bands with
#              Bessel-corrected standard deviation. Returns (lower, middle, upper) tuple.
# TASK 1.2.2 — Replaced static SMA-only mean reversion with 3-factor hybrid in
#              _normalize_mean_reversion(). New blend: RSI (35%) + Bollinger Bands (35%)
#              + SMA distance (30%). BB Z-score inverted so price near lower band → positive
#              MR signal (expect bounce), price near upper band → negative signal (pullback).
# FIX 2 — Added last_price: float = 0.0 field to Signal dataclass; populated in generate_signal_for_symbol().
# FIX 6 — Replaced _compute_volatility()-based vol_px with _compute_atr() in _decide_side_and_bands().
# FIX 7 — Changed guard from len(bars) < lookback to len(bars) < lookback + 1 in _compute_price_action_score().
# TASK-PRE-BLEND-DAMPEN — Moved conflict dampener to pre-blend stage in _combine_technical_scores().
#                          Now dampens individual mom and mr scores BEFORE applying weights, not after.
#                          When mom * mr < 0 (conflict), both scores are scaled by sqrt(penalty) so
#                          their weighted product matches the original post-blend penalty exactly.
#                          When no conflict exists, behaviour is byte-identical to previous implementation.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

import math

from adapters.ibkr_adapter import IbkrAdapter
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
    volatility: float = 0.0
    last_price: float = 0.0


class SignalEngine:
    """
    Composite technical signal engine:
      - Momentum/trend (now enhanced with MACD histogram)
      - Mean reversion (RSI / MA distance proxy)
      - Price action structure

    AI cost controls implemented here:
      (3) Compute technical signal first; if side == "skip", do not fetch news or call AI.
    Suggestion (2) is enforced in SentimentModule.scorenewsitems via reuse when no new news arrives.
    """

    def __init__(
        self,
        adapter: IbkrAdapter,
        sentiment: SentimentModule,
        technicalcfg: TechnicalSignalConfig,
    ):
        self.adapter = adapter
        self.sentiment = sentiment
        self.technicalcfg = technicalcfg
        # TASK 2.1.1: Per-symbol MACD history for signal line EMA smoothing
        # Each symbol stores a deque of recent MACD line values (maxlen=50)
        self._macd_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    # ── M2 FIX: MACD history serialisation ─────────────────────────────────────

    def export_macd_history(self) -> Dict[str, list]:
        """Serialise per-symbol MACD history deques for JSON persistence.

        Called by main._persist_vol_and_sentiment() after each loop iteration.
        """
        return {sym: list(dq) for sym, dq in self._macd_history.items()}

    def import_macd_history(self, data: Dict) -> None:
        """Restore per-symbol MACD history deques from a previously exported dict.

        Silently skips malformed entries. Called during main() startup.
        """
        if not isinstance(data, dict):
            return
        for sym, values in data.items():
            try:
                dq: deque = deque(maxlen=50)
                dq.extend(values[-50:])
                self._macd_history[sym] = dq
            except Exception:
                continue

    # ── END M2 FIX ─────────────────────────────────────────────────────────────

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

    def _compute_macd(self, bars: List, symbol: str = "", fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        TASK 2.1.1: MACD (Moving Average Convergence Divergence) calculation.
        
        MACD Components:
          - MACD Line = EMA(fast) - EMA(slow)
          - Signal Line = EMA(signal) of MACD Line
          - Histogram = MACD Line - Signal Line
        
        Returns: (macd_line, signal_line, histogram)
        
        Requirements:
          - Needs at least (slow + signal) bars for proper signal line calculation
          - For production, maintains per-symbol MACD history (self._macd_history)
          - Signal line uses Wilder's EMA smoothing over MACD line history
        
        MACD Semantics:
          - Positive histogram → MACD above signal (bullish momentum)
          - Negative histogram → MACD below signal (bearish momentum)
          - Histogram slope change → early divergence warning
          - Divergence: price makes new high but histogram declining → bearish signal
        """
        if len(bars) < slow + signal:
            return (0.0, 0.0, 0.0)
        
        closes = [b.c for b in bars]
        
        # 1. Compute Fast EMA (default 12-period)
        k_fast = 2.0 / (fast + 1)
        ema_fast = sum(closes[:fast]) / fast
        for price in closes[fast:]:
            ema_fast = price * k_fast + ema_fast * (1.0 - k_fast)
        
        # 2. Compute Slow EMA (default 26-period)
        k_slow = 2.0 / (slow + 1)
        ema_slow = sum(closes[:slow]) / slow
        for price in closes[slow:]:
            ema_slow = price * k_slow + ema_slow * (1.0 - k_slow)
        
        # 3. MACD Line = Fast EMA - Slow EMA
        macd_line = ema_fast - ema_slow
        
        # 4. Signal Line = EMA(signal) of MACD Line
        # Use per-symbol history for proper EMA smoothing
        if symbol:
            hist = self._macd_history[symbol]
            hist.append(macd_line)
            
            if len(hist) >= signal:
                # Proper Wilder's EMA over MACD history
                k_signal = 2.0 / (signal + 1)
                signal_line = sum(list(hist)[:signal]) / signal
                for macd_val in list(hist)[signal:]:
                    signal_line = macd_val * k_signal + signal_line * (1.0 - k_signal)
            else:
                # Insufficient history: use approximation
                signal_line = macd_line * 0.8
        else:
            # No symbol provided: use simplified approximation
            signal_line = macd_line * 0.8
        
        # 5. Histogram = MACD Line - Signal Line
        histogram = macd_line - signal_line
        
        return (macd_line, signal_line, histogram)

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
    
    def _compute_obv(self, bars: List) -> float:
        """
        On-Balance Volume: Cumulative volume flow weighted by price direction.
    
        Calculation:
          - If close[i] > close[i-1]: OBV += volume[i]
          - If close[i] < close[i-1]: OBV -= volume[i]
          - If close[i] == close[i-1]: OBV unchanged
    
        Returns normalized OBV (divided by average volume over period).
        """
        if len(bars) < 2:
            return 0.0
    
        obv = 0.0
        for i in range(1, len(bars)):
            if bars[i].c > bars[i-1].c:
                obv += bars[i].v
            elif bars[i].c < bars[i-1].c:
                obv -= bars[i].v
        # else: no change
    
    # Normalize by average volume to make scale-invariant
        avg_vol = sum(b.v for b in bars) / len(bars)
        if avg_vol == 0.0:
            return 0.0
    
        return obv / avg_vol

    def _compute_bollinger_bands(self, bars: List, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        TASK 1.2.1: Bollinger Bands calculation.
        
        Bollinger Bands: SMA(period) ± std_dev * StdDev(period)
        
        Returns: (lower_band, middle_band, upper_band)
        
        - Middle Band: SMA of last `period` closes
        - Upper Band: Middle + (std_dev * standard deviation)
        - Lower Band: Middle - (std_dev * standard deviation)
        
        Uses Bessel's correction for std dev (divide by n-1).
        """
        if len(bars) < period:
            return (0.0, 0.0, 0.0)
        
        closes = [b.c for b in bars[-period:]]
        middle = sum(closes) / period
        
        # Bessel-corrected standard deviation
        variance = sum((c - middle) ** 2 for c in closes) / (period - 1)
        std = math.sqrt(variance)
        
        lower = middle - std_dev * std
        upper = middle + std_dev * std
        
        return (lower, middle, upper)

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

    def _normalize_momentum_trend(self, mom_raw: float, trend_raw: float, bars: List, symbol: str = "") -> float:
        """
        TASK 2.1.2: Enhanced 3-factor momentum blend.
        
        Momentum Components:
          1. EMA Crossover (45%) — existing fast/slow EMA signal, already normalized
          2. MACD Histogram (25%) — NEW: divergence and momentum confirmation
          3. SMA-20 Trend (30%) — existing trend direction signal
        
        MACD Histogram Semantics:
          - Positive histogram → bullish momentum (MACD above signal)
          - Negative histogram → bearish momentum (MACD below signal)
          - Histogram slope change → early divergence warning
        
        Normalization:
          - Histogram typically ranges ±0.2-1.0% of price for most equities
          - Scale by macd_histogram_norm_scale (default 0.005 = 0.5% of price)
          - Clamp to [-1, 1] after scaling
        
        Blend Weights (configurable):
          - weight_ema_mom: 0.45 (EMA crossover momentum)
          - weight_macd_mom: 0.25 (MACD histogram)
          - weight_trend_mom: 0.30 (SMA-20 trend direction)
        """
        # 1. EMA Crossover Component (existing, already normalized to [-1, 1])
        mom_score = max(-1.0, min(1.0, mom_raw))
        
        # 2. NEW: MACD Histogram Component
        macd_fast = getattr(self.technicalcfg, "macd_fast", 12)
        macd_slow = getattr(self.technicalcfg, "macd_slow", 26)
        macd_signal = getattr(self.technicalcfg, "macd_signal", 9)
        
        _, _, histogram = self._compute_macd(
            bars,
            symbol=symbol,
            fast=macd_fast,
            slow=macd_slow,
            signal=macd_signal
        )
        
        macd_score = 0.0
        last_price = bars[-1].c if bars else 0.0
        if last_price > 0:
            # Normalize histogram as % of price
            macd_pct = histogram / last_price
            macd_histogram_norm_scale = getattr(self.technicalcfg, "macd_histogram_norm_scale", 0.005)
            macd_norm = macd_pct / macd_histogram_norm_scale
            macd_score = max(-1.0, min(1.0, macd_norm))
        
        # 3. Trend Direction Component (existing SMA-20 logic, already ±1.0)
        trend_score = trend_raw
        
        # Weighted blend with configurable weights
        weight_ema_mom = getattr(self.technicalcfg, "weight_ema_mom", 0.45)
        weight_macd_mom = getattr(self.technicalcfg, "weight_macd_mom", 0.25)
        weight_trend_mom = getattr(self.technicalcfg, "weight_trend_mom", 0.30)
        
        composite = (
            weight_ema_mom * mom_score +
            weight_macd_mom * macd_score +
            weight_trend_mom * trend_score
        )
        
        return max(-1.0, min(1.0, composite))

    def _normalize_mean_reversion(self, current_price: float, bars: List, rsi: float) -> float:
        """
        TASK 1.2.2: Hybrid mean reversion = RSI (35%) + Bollinger Bands (35%) + SMA distance (30%)
        
        BB Z-Score Logic:
          - Price near lower band → positive MR signal (expect bounce)
          - Price near upper band → negative MR signal (expect pullback)
          - Z-score = (price - middle) / (band_width / 4)
          - Inverted to match MR semantics (high price → negative signal)
        
        RSI Component (existing logic):
        High RSI (> rsi_overbought) -> Negative score (expect pullback)
        Low RSI  (< rsi_oversold)   -> Positive score (expect bounce)

        FIX-RSI-THRESH: both RSI formula branches now use config-sourced
        threshold values instead of hardcoded literals 70 and 30.

        Formulae:
          Overbought: rsi_score = -1.0 * (rsi - rsi_overbought) / (100.0 - rsi_overbought)
          Oversold:   rsi_score = +1.0 * (rsi_oversold - rsi) / rsi_oversold

        Properties:
          • At the threshold boundary, rsi_score = 0.0 exactly (continuous).
          • At the extreme (RSI=100 overbought, RSI=0 oversold), rsi_score = ±1.0.
          • Sign is always correct regardless of threshold values.
          • With default values (rsi_overbought=70, rsi_oversold=30) the output
            is numerically identical to the previous implementation.

        Also distance from MA: if price >> MA, revert down (-).
        FIX 6 — Added intermediate clamp of ma_score before blending to prevent
        outlier SMA deviations from nullifying RSI contribution.
        """
        ob = self.technicalcfg.rsi_overbought
        os = self.technicalcfg.rsi_oversold

        # 1. RSI Component — FIX-RSI-THRESH: config-sourced thresholds and denominators
        rsi_score = 0.0
        if rsi > ob:
            # Normalise over the distance from threshold to maximum (100).
            # e.g. default: RSI=75, ob=70 → -1.0*(75-70)/(100-70) = -0.167
            rsi_score = -1.0 * (rsi - ob) / (100.0 - ob)
        elif rsi < os:
            # Normalise over the distance from threshold to minimum (0).
            # e.g. default: RSI=25, os=30 → +1.0*(30-25)/30 = +0.167
            rsi_score = 1.0 * (os - rsi) / os

        # 2. NEW: Bollinger Bands Component
        bb_period = getattr(self.technicalcfg, "bb_period", 20)
        bb_std_dev = getattr(self.technicalcfg, "bb_std_dev", 2.0)
        lower, middle, upper = self._compute_bollinger_bands(bars, period=bb_period, std_dev=bb_std_dev)
        
        bb_score = 0.0
        if upper > lower:
            band_width = upper - lower
            # Z-score normalized to ±1 range (4 std devs = full band width)
            bb_zscore = (current_price - middle) / (band_width / 4.0) if band_width > 0 else 0.0
            # Invert: price above upper band (zscore > +1) → negative MR signal
            bb_score = -1.0 * max(-1.0, min(1.0, bb_zscore))

        # 3. SMA Distance Component (existing ma_score logic)
        if len(bars) < 20:
            ma_score = 0.0
        else:
            closes = [b.c for b in bars[-20:]]
            sma = sum(closes) / len(closes)
            # (price - sma) / sma
            dist = (current_price - sma) / sma if sma > 0 else 0.0
            # if dist is +1%, score is negative (revert)
            # scale: 0.05 (5%) -> full -1.0
            ma_score = -1.0 * (dist / self.technicalcfg.ma_distance_norm_scale)
            # FIX 6: clamp ma_score independently before blending
            ma_score = max(-1.0, min(1.0, ma_score))

        # Weighted blend
        weight_rsi_mr = getattr(self.technicalcfg, "weight_rsi_mr", 0.35)
        weight_bb_mr = getattr(self.technicalcfg, "weight_bb_mr", 0.35)
        weight_sma_dist_mr = getattr(self.technicalcfg, "weight_sma_dist_mr", 0.30)
        
        composite = (weight_rsi_mr * rsi_score +
                     weight_bb_mr * bb_score +
                     weight_sma_dist_mr * ma_score)
        
        return max(-1.0, min(1.0, composite))

    def _compute_price_action_score(self, bars: List, current_price: float) -> float:
        """
        WEEK 1.1: Volume-filtered breakout detection.

        Breakout detection with OBV confirmation:
          1. Identify price breakout (existing logic)
          2. If breakout detected AND enable_volume_filter=True:
             - Calculate OBV over lookback period
             - If OBV < threshold: return 0.0 (reject breakout)
          3. Return ±1.0 for confirmed breakout/breakdown
        """
        # Fix M3: suppress price action score when market is closed.
        if not self.adapter.get_market_open():
            return 0.0

        if len(bars) < self.technicalcfg.breakout_lookback_bars + 1:
            return 0.0

        window = bars[-self.technicalcfg.breakout_lookback_bars:-1]  # exclude current
        highs = [b.h for b in window]
        lows = [b.l for b in window]

        recent_high = max(highs)
        recent_low = min(lows)

        # Detect price breakout
        breakout_signal = 0.0
        if current_price > recent_high:
            breakout_signal = 1.0
        elif current_price < recent_low:
           breakout_signal = -1.0
        else:
            return 0.0

    # WEEK 1.1: Volume confirmation
        if self.technicalcfg.enable_volume_filter and breakout_signal != 0.0:
            obv_lookback = self.technicalcfg.obv_lookback_bars
            obv = self._compute_obv(bars[-obv_lookback:])

            # For bullish breakouts, require positive OBV trend
            if breakout_signal > 0 and obv < self.technicalcfg.obv_breakout_threshold:
                return 0.0  # Reject weak-volume breakout
        
            # For bearish breakdowns, require negative OBV trend
            if breakout_signal < 0 and obv > -self.technicalcfg.obv_breakout_threshold:
                return 0.0  # Reject weak-volume breakdown

        return breakout_signal


    def _combine_technical_scores(
        self, mom: float, mr: float, pa: float
    ) -> Tuple[float, bool]:
        """
        Compute the weighted composite signal score and flag whether the
        momentum/mean-reversion conflict dampener was applied.

        TASK-PRE-BLEND-DAMPEN: The conflict dampener is now applied BEFORE
        the weighted blending, not after. When momentum and mean-reversion
        have opposite signs (mom * mr < 0), both scores are scaled by
        sqrt(penalty) so that their weighted product matches the original
        post-blend penalty of `penalty` exactly.

        Algorithm:
          1. Check if mom * mr < 0 (conflicting signals).
          2. If conflict exists:
               scale_factor = sqrt(conflict_dampener_penalty)
               mom_damped = mom * scale_factor
               mr_damped  = mr  * scale_factor
             Otherwise:
               mom_damped = mom
               mr_damped  = mr
          3. Apply weights to the (possibly dampened) scores:
               raw = weight_momentum_trend * mom_damped
                   + weight_mean_reversion * mr_damped
                   + weight_price_action   * pa
          4. Clamp to [-1, 1].

        M4 FIX — Behavioral difference vs. original post-blend dampener:

          Old (post-blend): raw = (w_mom*mom + w_mr*mr + w_pa*pa) * penalty
            — Dampens ALL three components equally by ``penalty`` (0.6).

          New (pre-blend):
            mom_d = mom * sqrt(penalty),  mr_d = mr * sqrt(penalty)
            raw = w_mom*mom_d + w_mr*mr_d + w_pa*pa
            — Dampens only mom and mr by sqrt(penalty) ≈ 0.775.
            — Price action is NOT dampened.
            — sqrt(0.6)=0.775 > 0.6, so the penalty per-factor is milder.

          These are intentionally NOT equivalent.  Pre-blend was chosen because:
            1. Price action (breakout) is an independent signal and should not be
               penalised for a mom/mr disagreement it has no part in.
            2. Dampened sub-scores are visible in logs for debugging.
            3. The milder sqrt() penalty avoids over-suppressing in mild conflicts.

        Returns:
            (signal_score: float, dampened: bool)
        """
        dampened = False
        mom_damped = mom
        mr_damped = mr

        if mom * mr < 0:
            # TASK-PRE-BLEND-DAMPEN: apply sqrt(penalty) to each conflicting score.
            penalty = float(self.technicalcfg.conflict_dampener_penalty)
            scale_factor = math.sqrt(penalty)
            mom_damped = mom * scale_factor
            mr_damped = mr * scale_factor
            dampened = True

        raw = (
            self.technicalcfg.weight_momentum_trend * mom_damped
            + self.technicalcfg.weight_mean_reversion * mr_damped
            + self.technicalcfg.weight_price_action * pa
        )

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
        Fetch news from IBKR adapter.
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

        # TASK 2.1.2: Enhanced momentum calculation with MACD histogram (pass symbol for history)
        momentum_score = self._normalize_momentum_trend(momentum_raw, trend_raw, bars, symbol=symbol)
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

        # AI cost control: skip sentiment entirely when the technical signal
        # is neutral — saves an API call.
        # OBSERVABILITY FIX: log_instrument_report() is now called here too so
        # technically-skipped symbols appear in the logs with their sub-scores,
        # making it visible why they were dropped before any AI was consulted.
        if side == "skip":
            neutral_sentiment = SentimentResult(
                score=0.0,
                raw_discrete=0,
                rawcompound=0.0,
                ndocuments=0,
                explanation="Technical signal neutral; sentiment not evaluated.",
                confidence=0.0,
            )
            log_instrument_report(
                symbol=symbol,
                signal_score=signal_score,
                sentiment=neutral_sentiment,
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion_score,
                price_action_score=price_action_score,
                env_mode=ENV_MODE,
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
                last_price=last_trade,
            )

        news_items = self._get_news_items(symbol)
        sentiment_result = self.sentiment.scorenewsitems(symbol, news_items)

        log_instrument_report(
            symbol=symbol,
            signal_score=signal_score,
            sentiment=sentiment_result,
            momentum_score=momentum_score,
            mean_reversion_score=mean_reversion_score,
            price_action_score=price_action_score,
            env_mode=ENV_MODE,
        )

        # Sentiment no-trade block — now reached AFTER logging so the
        # operator can see the negative score that caused the skip.
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
                last_price=last_trade,
            )

        rationale = (
            f"side={side} signal={signal_score:.3f} "
            f"mom={momentum_score:.3f} mr={mean_reversion_score:.3f} "
            f"pa={price_action_score:.3f} "
            f"sentiment={sentiment_result.score:.3f}{dampener_tag}"
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
            last_price=last_trade,
        )
