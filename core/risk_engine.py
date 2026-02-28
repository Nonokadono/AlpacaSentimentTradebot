# CHANGES:
# Change 2 — Kelly volfactor constant: 0.005 → 0.002 so that typical intraday
#   per-bar volatility (0.001–0.003) produces a vol_penalty in a useful range
#   rather than squashing kelly to near-zero.
# Change 4 — Separate technical p from sentiment p in kelly_fraction: add
#   optional parameter s: float = 0.0 (raw sentiment score in [-1,+1]). p is
#   now computed as 0.5 + 0.15*tech_conviction + 0.05*s, separating the two
#   signals. b uses only tech_conviction. The old conviction variable is removed
#   from kelly_fraction; s_scale is kept as a parameter for backward compat but
#   used only in the fixed-fractional path. Call site in pre_trade_checks passes
#   s=sentiment.score. No renames.

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

        Change 4: p and b are now derived solely from tech_conviction = min(|signal_score|, 1.0).
          p = max(0.35, min(0.75, 0.5 + 0.15*tech_conviction + 0.05*clamp(s, -1, 1)))
          q = 1 - p
          b = max(1.0, 1.0 + 1.5*tech_conviction)
        s_scale is accepted for backward compatibility but not used here.

        Change 2: volfactor = volatility / 0.002 (was 0.005).
        """
        tech_conviction = min(abs(signal_score), 1.0)

        p = max(0.35, min(0.75, 0.5 + 0.15 * tech_conviction + 0.05 * max(-1.0, min(1.0, s))))
        q = 1.0 - p
        b = max(1.0, 1.0 + 1.5 * tech_conviction)

        if b <= 0:
            return 0.0

        full_kelly = (p * b - q) / b
        half_kelly = full_kelly * 0.5

        # Change 2: constant 0.002 (was 0.005)
        volfactor = volatility / 0.002 if volatility > 0 else 0.0
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
            # Change 4: pass s=sentiment.score to separate tech from sentiment
            kelly_f = self._kelly_fraction(signal_score, s_scale, volatility,
                                           s=sentiment.score)
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
