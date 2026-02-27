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
    gross_exposure: float  # sum |position_notional|
    daily_loss_pct: float
    drawdown_pct: float


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    market_price: float
    side: str  # long/short
    notional: float
    # Sentiment compound score recorded at entry time (rawcompound from SentimentResult).
    # Defaults to 0.0 so existing callsites that build PositionInfo without this
    # field continue to work — main.py patches it in after entry is confirmed.
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
        if s < self.sentiment_cfg.no_trade_negative_threshold:
            return 0.0
        if abs(s) <= self.sentiment_cfg.neutral_band:
            return 1.0
        span = self.sentiment_cfg.max_scale - self.sentiment_cfg.min_scale
        scaled = self.sentiment_cfg.min_scale + (s + 1) / 2 * span
        return max(self.sentiment_cfg.min_scale, min(self.sentiment_cfg.max_scale, scaled))

    # ── Feature 3: Half-Kelly helper ──────────────────────────────────────────

    def _kelly_fraction(
        self,
        signal_score: float,
        s_scale: float,
        volatility: float,
    ) -> float:
        """
        Compute a Half-Kelly multiplier in [0.0, 1.0].

        Kelly formula: f* = (p*b - q) / b   where
          b  = reward/risk ratio proxy derived from |signal_score| and s_scale
          p  = estimated win probability  (mapped from |signal_score| and s_scale)
          q  = 1 - p

        Both b and p are estimated from the composite signal rather than from
        historical trade outcomes (which we do not track here).  The estimates
        are intentionally conservative:

          conviction = |signal_score| * s_scale  ∈ [0, 1.3]
            — combines technical edge (|signal_score|) with sentiment
              confirmation (s_scale ≥ 1 is favourable, < 1 is cautious).

          p = 0.5 + 0.15 * min(conviction, 1.0)
            — neutral market → p = 0.5; max conviction → p = 0.65.
            — Keeps us inside the "realistic" win-rate band for short-term equity.

          b = max(1.0, 1.5 * min(conviction, 1.0) + 1.0)
            — Minimum reward/risk of 1.0; scales with conviction up to ~2.5.

          vol_penalty = 1.0 / (1.0 + vol_factor)
            — Higher volatility shrinks the Kelly fraction; acts as a
              volatility-adjusted position sizing damper.
            — vol_factor normalises raw per-bar std-dev (typically 0.001–0.02)
              to a meaningful scale.

        The raw Full Kelly is then halved (Half-Kelly) and clamped to [0, 1].

        This approach is intentionally conservative and opinionated.
        Reference: https://pyquantnews.com/the-pyquant-newsletter/use-kelly-criterion-optimal-position-sizing
        """
        conviction = abs(signal_score) * s_scale                   # [0, ~1.3]
        conviction = min(conviction, 1.0)                          # cap at 1

        p = 0.5 + 0.15 * conviction                                # [0.50, 0.65]
        q = 1.0 - p
        b = 1.0 + 1.5 * conviction                                 # [1.0, 2.5]

        if b <= 0:
            return 0.0

        full_kelly = (p * b - q) / b                               # Kelly criterion
        half_kelly = full_kelly * 0.5                              # Half-Kelly

        # Volatility penalty: normalise raw per-bar std-dev.
        # A typical per-bar std-dev of ~0.005 (0.5%) maps to vol_factor ≈ 1.0.
        vol_factor = volatility / 0.005 if volatility > 0 else 0.0
        vol_penalty = 1.0 / (1.0 + vol_factor)

        kelly = half_kelly * vol_penalty

        return max(0.0, min(1.0, kelly))

    # ── Main pre-trade check ──────────────────────────────────────────────────

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

        New keyword args (both have safe defaults so existing callers are
        unaffected):
          volatility: per-symbol return std-dev from SignalEngine._compute_volatility().
                      Used only when enable_kelly_sizing is True.
          sentiment_scale_override: if >= 0 the value is used directly as s_scale
                      (avoids re-computing when the caller already has it).
                      Default -1.0 triggers the existing internal computation.
        """
        # 0) Instrument whitelist
        if symbol not in self.instrument_meta:
            return ProposedTrade(
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score,
                sentiment_scale=0.0,
                signal_score=signal_score,
                rationale=rationale,
                rejected_reason="Instrument not whitelisted",
            )

        # 1) Hard block for unstable / utterly undesirable sentiment (-2)
        if getattr(sentiment, "raw_discrete", 0) == -2:
            return ProposedTrade(
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score,
                sentiment_scale=0.0,
                signal_score=signal_score,
                rationale=rationale,
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
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score,
                sentiment_scale=0.0,
                signal_score=signal_score,
                rationale=rationale,
                rejected_reason="Sentiment too negative for new trade",
            )

        # 3) Validate stop distance
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return ProposedTrade(
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score,
                sentiment_scale=s_scale,
                signal_score=signal_score,
                rationale=rationale,
                rejected_reason="Invalid stop distance",
            )

        # 4) Risk per trade — fixed-fractional OR Half-Kelly
        if self.limits.enable_kelly_sizing:
            # Feature 3: Half-Kelly fraction scales the max allowable risk %.
            # Fix 5: use 0.0 as the lower clamp bound (not min_risk_per_trade_pct)
            # so a genuinely weak Kelly signal can produce qty == 0 and be
            # rejected in step 5, rather than being silently forced up to the
            # minimum floor — which would defeat the purpose of Kelly sizing.
            kelly_f = self._kelly_fraction(signal_score, s_scale, volatility)
            raw_risk_pct = kelly_f * self.limits.max_risk_per_trade_pct
            risk_pct = min(
                self.limits.max_risk_per_trade_pct,
                max(0.0, raw_risk_pct),
            )
        else:
            # Original fixed-fractional path (unchanged)
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
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                sentiment_score=sentiment.score,
                sentiment_scale=s_scale,
                signal_score=signal_score,
                rationale=rationale,
                rejected_reason="Calculated quantity is zero",
            )

        # 6) Broker-aware cap: fraction of equity per trade
        max_notional_broker = 0.4 * snapshot.equity
        projected_notional = qty * entry_price
        if projected_notional > max_notional_broker:
            max_qty_broker = int(max_notional_broker / entry_price / meta.lot_size) * meta.lot_size
            if max_qty_broker <= 0:
                return ProposedTrade(
                    symbol=symbol,
                    side=side,
                    qty=0.0,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    risk_amount=risk_amount,
                    risk_pct_of_equity=risk_pct,
                    sentiment_score=sentiment.score,
                    sentiment_scale=s_scale,
                    signal_score=signal_score,
                    rationale=rationale,
                    rejected_reason="Broker buying power cap per trade",
                )
            qty = max_qty_broker
            projected_notional = qty * entry_price

        # 7) Gross exposure / loss / drawdown / position-count rules
        projected_gross = snapshot.gross_exposure + abs(projected_notional)
        if projected_gross > snapshot.equity * self.limits.gross_exposure_cap_pct:
            return ProposedTrade(
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=risk_amount,
                risk_pct_of_equity=risk_pct,
                sentiment_score=sentiment.score,
                sentiment_scale=s_scale,
                signal_score=signal_score,
                rationale=rationale,
                rejected_reason="Gross exposure cap breached",
            )

        if snapshot.daily_loss_pct <= -self.limits.daily_loss_limit_pct:
            return ProposedTrade(
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=risk_amount,
                risk_pct_of_equity=risk_pct,
                sentiment_score=sentiment.score,
                sentiment_scale=s_scale,
                signal_score=signal_score,
                rationale=rationale,
                rejected_reason="Daily loss limit breached",
            )

        if snapshot.drawdown_pct <= -self.limits.max_drawdown_pct:
            return ProposedTrade(
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=risk_amount,
                risk_pct_of_equity=risk_pct,
                sentiment_score=sentiment.score,
                sentiment_scale=s_scale,
                signal_score=signal_score,
                rationale=rationale,
                rejected_reason="Max drawdown limit breached",
            )

        if len(positions) >= self.limits.max_open_positions and symbol not in positions:
            return ProposedTrade(
                symbol=symbol,
                side=side,
                qty=0.0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=risk_amount,
                risk_pct_of_equity=risk_pct,
                sentiment_score=sentiment.score,
                sentiment_scale=s_scale,
                signal_score=signal_score,
                rationale=rationale,
                rejected_reason="Max open positions exceeded",
            )

        return ProposedTrade(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            risk_amount=risk_amount,
            risk_pct_of_equity=risk_pct,
            sentiment_score=sentiment.score,
            sentiment_scale=s_scale,
            signal_score=signal_score,
            rationale=rationale,
            rejected_reason=None,
        )