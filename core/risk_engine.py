# CHANGES:
# FIX 3 — Kelly path in pre_trade_checks() now applies min_risk_per_trade_pct as
#   the floor for risk_pct, matching the fixed-fractional path.  Changed:
#     max(0.0, raw_risk_pct)
#   to:
#     max(self.limits.min_risk_per_trade_pct, raw_risk_pct)
#   This prevents a very low kelly_f from collapsing risk_pct to zero and
#   producing qty=0 (silently wasted candidate evaluation).
#
# FIX 4 — _vol_history is now a per-symbol Dict[str, deque] instead of a single
#   shared deque.  A single shared deque caused cross-symbol vol contamination:
#   one high-vol symbol (TSLA, NVDA) shifted the median upward and under-sized
#   all normal-vol symbols.
#   In __init__: self._vol_history: Dict[str, deque] = {}
#   In _kelly_fraction(): added symbol: str = "" parameter (default preserves all
#   existing call sites).  Inside the method, per-symbol deque is created on
#   first use (maxlen=200) and all percentile logic uses that deque exclusively.
#   In pre_trade_checks(): _kelly_fraction() call now passes symbol=symbol.
#
# All prior changes (Fix M5, Change 2, Change 4, Improvement A, Improvement C)
# are preserved unchanged.

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict

from .sentiment import SentimentResult
from config.config import RiskLimits, SentimentConfig, InstrumentMeta


@dataclass
class EquitySnapshot:
    equity: float
    cash: float
    portfolio_value: float
    day_trading_buying_power: float
    start_of_day_equity: float
    high_watermark_equity: float
    realized_pl_today: float
    unrealized_pl: float
    gross_exposure: float
    daily_loss_pct: float
    drawdown_pct: float


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    market_price: float
    side: str
    notional: float
    opening_compound: float = 0.0


@dataclass
class ProposedTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    risk_amount: float
    risk_pct_of_equity: float
    sentiment_score: float
    sentiment_scale: float
    signal_score: float = 0.0
    rationale: Optional[str] = None
    rejected_reason: Optional[str] = None


class RiskEngine:
    def __init__(
        self,
        risk_limits: RiskLimits,
        sentiment_cfg: SentimentConfig,
        instrument_meta: Dict[str, InstrumentMeta],
    ) -> None:
        self.limits = risk_limits
        self.sentiment_cfg = sentiment_cfg
        self.instrument_meta = instrument_meta
        # FIX 4: per-symbol vol history dict replaces the single shared deque.
        # Key = symbol str, value = deque(maxlen=200).
        # Previously self._vol_history = deque(maxlen=200) was a single pool
        # that mixed all symbols' volatilities, biasing the percentile normaliser.
        self._vol_history: Dict[str, deque] = {}

    def sentiment_scale(self, s: float) -> float:
        """
        Continuous piecewise-linear sentiment scale with no discontinuities.
        [unchanged — see original docstring]
        """
        no_trade_neg = self.sentiment_cfg.no_trade_negative_threshold
        min_sc = self.sentiment_cfg.min_scale
        max_sc = self.sentiment_cfg.max_scale

        if s < no_trade_neg:
            return 0.0
        if s <= 0.0:
            band_width = 0.0 - no_trade_neg
            return min_sc * (s - no_trade_neg) / band_width
        span = max_sc - min_sc
        return min(max_sc, min_sc + span * s)

    def _kelly_fraction(
        self,
        signal_score: float,
        s_scale: float,
        volatility: float,
        s: float = 0.0,
        symbol: str = "",
    ) -> float:
        """
        Compute a Half-Kelly multiplier in [0.0, 1.0].

        FIX 4: added symbol: str = "" parameter.  Volatility history is now
          maintained per-symbol in self._vol_history[symbol] (deque maxlen=200).
          The percentile normaliser therefore reflects each symbol's own vol
          distribution rather than the cross-symbol pool.
          Existing call sites that do not pass symbol still work (default "").

        Improvement A: adaptive vol normalisation.
          volatility is appended to the symbol's deque on every call.
          If len >= 20, vol_norm = percentile(history, kelly_vol_norm_percentile).
          Otherwise vol_norm = 0.002 (warm-up fallback).
          volfactor = volatility / vol_norm if volatility > 0 else 0.0.

        Improvement C: log-odds sentiment blending.
          p_tech = 0.5 + 0.15 * tech_conviction
          log_odds = log(p_tech / (1 - p_tech)) + kelly_sentiment_weight * clamp(s, -1, 1)
          p = clamp(sigmoid(log_odds), 0.35, 0.75)

        Change 4: b uses only tech_conviction.
        s_scale is accepted for backward compatibility but not used here.
        """
        tech_conviction = min(abs(signal_score), 1.0)

        # Improvement C: log-odds blend for p
        p_tech = 0.5 + 0.15 * tech_conviction
        # Guard: p_tech must be strictly in (0, 1) for log to be defined.
        p_tech = max(1e-6, min(1.0 - 1e-6, p_tech))
        log_odds = (
            math.log(p_tech / (1.0 - p_tech))
            + self.limits.kelly_sentiment_weight * max(-1.0, min(1.0, s))
        )
        p = max(0.35, min(0.75, 1.0 / (1.0 + math.exp(-log_odds))))
        q = 1.0 - p
        b = max(1.0, 1.0 + 1.5 * tech_conviction)

        if b <= 0:
            return 0.0

        full_kelly = (p * b - q) / b
        half_kelly = full_kelly * 0.5

        # FIX 4: per-symbol deque — create on first use.
        if symbol not in self._vol_history:
            self._vol_history[symbol] = deque(maxlen=200)
        hist = self._vol_history[symbol]
        hist.append(volatility)

        # Improvement A: adaptive percentile normalisation using symbol-local history.
        if len(hist) >= 20:
            sorted_hist = sorted(hist)
            pct = self.limits.kelly_vol_norm_percentile
            # Linear interpolation for the given percentile.
            idx_f = pct * (len(sorted_hist) - 1)
            idx_lo = int(idx_f)
            idx_hi = min(idx_lo + 1, len(sorted_hist) - 1)
            frac = idx_f - idx_lo
            vol_norm = sorted_hist[idx_lo] + frac * (sorted_hist[idx_hi] - sorted_hist[idx_lo])
            # Guard: never divide by zero even if all history entries are 0.
            if vol_norm <= 0.0:
                vol_norm = 0.002
        else:
            # Warm-up fallback (same as the prior fixed constant).
            vol_norm = 0.002

        volfactor = volatility / vol_norm if volatility > 0 else 0.0
        vol_penalty = 1.0 / (1.0 + volfactor)

        kelly = half_kelly * vol_penalty

        return max(0.0, min(1.0, kelly))

    def pre_trade_checks(
        self,
        snapshot: EquitySnapshot,
        positions: Dict[str, "PositionInfo"],
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        sentiment: SentimentResult,
        signal_score: float = 0.0,
        rationale: Optional[str] = None,
        volatility: float = 0.0,
        sentiment_scale_override: float = -1.0,
    ) -> ProposedTrade:
        """
        Run all pre-trade checks and compute position size.
        [unchanged — see original docstring]
        """
        # 0) Instrument whitelist
        if symbol not in self.instrument_meta:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=0.0, risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score, sentiment_scale=0.0,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Instrument not whitelisted",
            )

        # 1) Hard block for unstable / utterly undesirable sentiment (-2)
        if getattr(sentiment, "raw_discrete", 0) == -2:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=0.0, risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score, sentiment_scale=0.0,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Sentiment -2 (unstable / do not trade)",
            )

        meta = self.instrument_meta[symbol]

        # 2) Sentiment-based sizing scale
        if sentiment_scale_override >= 0.0:
            s_scale = sentiment_scale_override
        else:
            s_scale = self.sentiment_scale(sentiment.score)

        if s_scale == 0.0:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=0.0, risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score, sentiment_scale=0.0,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Sentiment too negative for new trade",
            )

        # 3) Validate stop distance
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=0.0, risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score, sentiment_scale=s_scale,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Invalid stop distance",
            )

        # 4) Risk per trade — fixed-fractional OR Half-Kelly
        if self.limits.enable_kelly_sizing:
            # Change 4 / Improvement C: pass s=sentiment.score to separate tech
            # from sentiment and apply log-odds blending.
            # FIX 4: pass symbol=symbol so _kelly_fraction uses per-symbol vol history.
            kelly_f = self._kelly_fraction(signal_score, s_scale, volatility,
                                           s=sentiment.score, symbol=symbol)
            # Fix M5: apply s_scale as a multiplier so sentiment strength
            # differentiates sizing within the Kelly path.
            kelly_f = kelly_f * s_scale
            raw_risk_pct = kelly_f * self.limits.max_risk_per_trade_pct
            # FIX 3: apply min_risk_per_trade_pct floor (was max(0.0, ...)).
            # Matches the fixed-fractional path and prevents qty=0 from a tiny kelly_f.
            risk_pct = min(
                self.limits.max_risk_per_trade_pct,
                max(self.limits.min_risk_per_trade_pct, raw_risk_pct),
            )
        else:
            raw_risk_pct = self.limits.max_risk_per_trade_pct * s_scale
            risk_pct = min(
                self.limits.max_risk_per_trade_pct,
                max(self.limits.min_risk_per_trade_pct, raw_risk_pct),
            )

        risk_amount = snapshot.equity * risk_pct

        # 5) Size by risk and force whole-lot qty
        qty = risk_amount / stop_distance
        qty = int(qty / meta.lot_size) * meta.lot_size

        if qty <= 0:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=0.0, risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score, sentiment_scale=s_scale,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Calculated quantity is zero",
            )

        # 6) Broker-aware cap
        max_notional_broker = 0.4 * snapshot.equity
        projected_notional = qty * entry_price
        if projected_notional > max_notional_broker:
            max_qty_broker = int(max_notional_broker / entry_price / meta.lot_size) * meta.lot_size
            if max_qty_broker <= 0:
                return ProposedTrade(
                    symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                    stop_price=stop_price, take_profit_price=take_profit_price,
                    risk_amount=risk_amount, risk_pct_of_equity=risk_pct,
                    sentiment_score=sentiment.score, sentiment_scale=s_scale,
                    signal_score=signal_score, rationale=rationale,
                    rejected_reason="Broker buying power cap per trade",
                )
            qty = max_qty_broker
            projected_notional = qty * entry_price

        # 7) Gross exposure / loss / drawdown / position-count rules
        projected_gross = snapshot.gross_exposure + abs(projected_notional)
        if projected_gross > snapshot.equity * self.limits.gross_exposure_cap_pct:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=risk_amount, risk_pct_of_equity=risk_pct,
                sentiment_score=sentiment.score, sentiment_scale=s_scale,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Gross exposure cap breached",
            )

        if snapshot.daily_loss_pct <= -self.limits.daily_loss_limit_pct:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=risk_amount, risk_pct_of_equity=risk_pct,
                sentiment_score=sentiment.score, sentiment_scale=s_scale,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Daily loss limit breached",
            )

        if snapshot.drawdown_pct <= -self.limits.max_drawdown_pct:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=risk_amount, risk_pct_of_equity=risk_pct,
                sentiment_score=sentiment.score, sentiment_scale=s_scale,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Max drawdown limit breached",
            )

        if len(positions) >= self.limits.max_open_positions and symbol not in positions:
            return ProposedTrade(
                symbol=symbol, side=side, qty=0.0, entry_price=entry_price,
                stop_price=stop_price, take_profit_price=take_profit_price,
                risk_amount=risk_amount, risk_pct_of_equity=risk_pct,
                sentiment_score=sentiment.score, sentiment_scale=s_scale,
                signal_score=signal_score, rationale=rationale,
                rejected_reason="Max open positions exceeded",
            )

        return ProposedTrade(
            symbol=symbol, side=side, qty=qty,
            entry_price=entry_price, stop_price=stop_price,
            take_profit_price=take_profit_price,
            risk_amount=risk_amount, risk_pct_of_equity=risk_pct,
            sentiment_score=sentiment.score, sentiment_scale=s_scale,
            signal_score=signal_score, rationale=rationale,
            rejected_reason=None,
        )
