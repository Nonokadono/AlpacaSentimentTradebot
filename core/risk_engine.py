# CHANGES:
# Fix M5 — In pre_trade_checks(), after computing kelly_f via _kelly_fraction(),
#   multiply the result by s_scale before computing raw_risk_pct:
#   kelly_f = kelly_f * s_scale.  This ensures a neutral-sentiment trade
#   (s_scale ≈ 1.0) is unaffected, a high-confidence positive trade
#   (s_scale = 1.3) receives proportionally larger sizing, and a no-trade
#   negative threshold (s_scale = 0.0) zeroes out Kelly sizing without relying
#   on the upstream s_scale == 0.0 guard alone.  No variable renames.
#
# All prior changes (Change 2, Change 4, Improvement A, Improvement C) are
# preserved unchanged.

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
        # Improvement A: rolling vol history for adaptive percentile normalisation.
        # maxlen=200 keeps roughly the last 200 position-sizing calls (~3-4 hours
        # at 600s sleep with a moderate symbol count).
        self._vol_history: deque = deque(maxlen=200)

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
    ) -> float:
        """
        Compute a Half-Kelly multiplier in [0.0, 1.0].

        Improvement A: adaptive vol normalisation.
          volatility is appended to self._vol_history on every call.
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

        # Improvement A: append to history and compute adaptive vol_norm.
        self._vol_history.append(volatility)
        if len(self._vol_history) >= 20:
            sorted_hist = sorted(self._vol_history)
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
            kelly_f = self._kelly_fraction(signal_score, s_scale, volatility,
                                           s=sentiment.score)
            # Fix M5: apply s_scale as a multiplier so sentiment strength
            # differentiates sizing within the Kelly path.
            kelly_f = kelly_f * s_scale
            raw_risk_pct = kelly_f * self.limits.max_risk_per_trade_pct
            risk_pct = min(
                self.limits.max_risk_per_trade_pct,
                max(0.0, raw_risk_pct),
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
