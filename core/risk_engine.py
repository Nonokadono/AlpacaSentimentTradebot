# CHANGES:
# MONITOR-FIX — Added stop_price and take_profit_price optional fields to PositionInfo dataclass.
#               These fields map active protective order levels into the position state so they
#               can be consumed by log_sentiment_position_check in the monitor.
#               Both default to None to maintain backwards compatibility.

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
    # CRASH-2 FIX: avg_entry_price was accessed in order_executor.py but did
    # not exist in this dataclass, raising AttributeError on every position check.
    # Default 0.0 is safe: the guarding condition `if pos.avg_entry_price > 0.0`
    # in _check_and_exit_on_sentiment() resolves to False, so unrealised_pnl_pct
    # stays 0.0 and PnL scaling is a no-op for any position where this field
    # is not explicitly populated by PositionManager.
    avg_entry_price: float = 0.0
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None


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

    # ── PERSIST-FIX + CHANGE 3: Vol history serialisation with schema versioning ──

    def export_vol_history(self) -> Dict:
        """Serialise per-symbol volatility deques to a JSON-safe dict with schema versioning.

        CHANGE 3: Returns a wrapper dict with two keys:
          - "schema_version": "1.0.0" (semver string for format validation)
          - "data": {symbol: [float, ...]} (the actual vol history mapping)

        Each list preserves insertion order so import_vol_history can reconstruct
        the deque in the same order, maintaining rolling percentile calculation integrity.

        The schema_version field allows import_vol_history to detect incompatible
        format changes (e.g. if a future version changes the data structure from
        per-symbol lists to per-symbol dicts with metadata). When the structure
        evolves, increment the minor or major version and update the import logic
        to handle both old and new formats gracefully.

        Called by main._persist_vol_and_sentiment() after each loop iteration.
        """
        return {
            "schema_version": "1.0.0",
            "data": {sym: list(dq) for sym, dq in self._vol_history.items()},
        }

    def import_vol_history(self, raw: Dict) -> None:
        """Restore per-symbol volatility deques from a previously exported dict.

        CHANGE 3: Validates schema_version on import. Accepts two formats:
          1. New format (v1.0.0):  {"schema_version": "1.0.0", "data": {...}}
          2. Legacy bare dict:     {symbol: [float, ...], ...}

        The legacy format (no schema_version key) is treated as implicit v1.0.0
        for backward compatibility with equity_state.json files written before
        this change.

        When schema_version is present and differs from "1.0.0", raises ValueError
        with an actionable error message listing:
          - The expected version(s)
          - The detected version
          - The action required (manual upgrade, re-initialise, or contact maintainer)

        Each value list is sliced to the last 200 entries before insertion so we
        never exceed maxlen=200 even if the persisted list was somehow longer.

        Malformed entries (wrong type, unparseable values) are silently skipped per
        symbol so that partial corruption in equity_state.json cannot crash the bot
        on startup. A fully corrupt file will result in an empty _vol_history dict,
        which is safe (Kelly sizing falls back to warm-up behaviour with vol_norm=0.002).

        Called by main.main() immediately after RiskEngine is constructed.
        """
        if not isinstance(raw, dict):
            return  # Silently ignore non-dict input (e.g. None, [], etc.)

        # Detect format: new (with schema_version key) or legacy (bare dict).
        if "schema_version" in raw:
            # New format — validate version
            version = raw.get("schema_version", "")
            if version != "1.0.0":
                raise ValueError(
                    f"vol_history schema version mismatch:\n"
                    f"  Expected: 1.0.0\n"
                    f"  Detected: {version}\n"
                    f"\n"
                    f"This equity_state.json file was written by a newer or incompatible\n"
                    f"version of the bot. Action required:\n"
                    f"  - If {version} is newer than 1.0.0, upgrade your bot code.\n"
                    f"  - If {version} is unrecognised, delete equity_state.json and\n"
                    f"    re-initialise (vol_history will rebuild from scratch).\n"
                    f"  - If the problem persists, contact the maintainer.\n"
                )
            data = raw.get("data", {})
        else:
            # Legacy format — treat as implicit v1.0.0
            data = raw

        # Restore per-symbol deques from the data dict.
        for sym, values in data.items():
            try:
                dq: deque = deque(maxlen=200)
                dq.extend(values[-200:])   # cap at maxlen to be safe
                self._vol_history[sym] = dq
            except Exception:
                continue   # silently skip malformed entries

    # ── END PERSIST-FIX + CHANGE 3 ──────────────────────────────────────────

    def sentiment_scale(self, s: float) -> float:
        """
        Piecewise-linear sentiment scale with neutral_band zero-return region.

        NEUTRAL-BAND-ZERO-FIX: Absolute value filter ensures that sentiment scores
        with |s| < neutral_band produce ZERO scale, blocking position entry entirely.
        This captures both barely-positive and barely-negative AI-neutral scores.

        Regions (with defaults no_trade_negative_threshold=-0.3, neutral_band=0.1):

          1. s < no_trade_negative_threshold             → return 0.0 (hard bearish block)
          2. abs(s) < neutral_band                       → return 0.0 (neutral block)
          3. no_trade_negative_threshold ≤ s < -neutral_band
                                                        → interpolate 0.0 → min_scale
          4. neutral_band ≤ s ≤ 1.0                      → interpolate min_scale → max_scale

        Region 2 is the critical fix: AI-neutral scores (e.g. s=0.05, s=-0.08) now
        produce zero scale instead of min_scale, ensuring no position is opened unless
        sentiment conviction exceeds neutral_band (default 0.1).

        Previous logic allowed s ∈ (0, neutral_band) to jump to min_scale, admitting
        barely-positive signals. The abs() check now blocks BOTH sides of the neutral
        zone symmetrically.
        """
        no_trade_neg = self.sentiment_cfg.no_trade_negative_threshold
        neutral_band = self.sentiment_cfg.neutral_band
        min_sc = self.sentiment_cfg.min_scale
        max_sc = self.sentiment_cfg.max_scale

        # Region 1: hard bearish block
        if s < no_trade_neg:
            return 0.0

        # Region 2: NEUTRAL-BAND-ZERO-FIX — block both barely-positive and barely-negative
        if abs(s) < neutral_band:
            return 0.0

        # Region 3: bearish but above hard floor → interpolate 0.0 → min_scale
        if s < 0.0:
            band_width = -neutral_band - no_trade_neg
            if band_width == 0.0:
                return 0.0
            return min_sc * (s - no_trade_neg) / band_width

        # Region 4: bullish conviction → interpolate min_scale → max_scale
        span = max_sc - min_sc
        denominator = 1.0 - neutral_band
        if denominator == 0.0:
            return max_sc
        frac = (s - neutral_band) / denominator
        return min(max_sc, min_sc + span * frac)

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
            # KELLY-MIN-FIX: use kelly_min_risk_pct floor instead of min_risk_per_trade_pct.
            # This allows Kelly-sized positions to take minimal risk (0.1% default)
            # without being constrained by the 0.5% fixed-fractional floor.
            risk_pct = min(
                self.limits.max_risk_per_trade_pct,
                max(self.limits.kelly_min_risk_pct, raw_risk_pct),
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
