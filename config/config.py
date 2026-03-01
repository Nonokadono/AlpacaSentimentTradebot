# CHANGES:
# FIX 7 — Added ema_crossover_norm_scale: float = 0.10 to TechnicalSignalConfig.
#   Previously _compute_simple_momentum_raw() divided raw EMA crossover by
#   momentum_norm_scale (0.05), causing saturation at ±1 for high-momentum
#   equities where the EMA spread exceeds 5%.  The new field widens the graded
#   range to ±10%.  Existing momentum_norm_scale (used by
#   _normalize_momentum_trend()) is unchanged.  New field carries a default so
#   backward compatibility with all existing configs is maintained.
#
# All prior changes (Fix L3, Change 2, Change 4) are preserved unchanged.

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict

ENV_MODE = os.getenv("APCA_API_ENV", "PAPER").upper()
if ENV_MODE not in ("PAPER", "LIVE"):
    raise ValueError(f"Invalid APCA_API_ENV={ENV_MODE}, expected PAPER or LIVE")

LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"


@dataclass
class RiskLimits:
    max_risk_per_trade_pct: float = 0.03     # 3% of equity
    min_risk_per_trade_pct: float = 0.005    # 0.5% of equity
    gross_exposure_cap_pct: float = 0.90     # 90% of equity
    daily_loss_limit_pct: float = 0.04       # 4% of start-of-day equity
    max_drawdown_pct: float = 0.09           # 9% from high watermark
    max_open_positions: int = 15
    # Task 4: Half-Kelly position sizing is now the VALIDATED DEFAULT.
    # Set to False ONLY for emergency rollback to fixed-fractional sizing.
    # Override via environment config or load_config() — do not change this line
    # without re-running the Kelly calibration tests.
    enable_kelly_sizing: bool = True
    # Improvement A: percentile of the rolling vol history (maxlen=200) used as
    # the denominator when normalising volatility inside _kelly_fraction.
    # 0.50 = median; raise toward 0.75 to be less sensitive to vol spikes.
    kelly_vol_norm_percentile: float = 0.50
    # Improvement C: weight applied to the clamped sentiment score in the
    # log-odds blend for Kelly p.  Increasing this makes sentiment more
    # influential on position size.
    kelly_sentiment_weight: float = 0.08


@dataclass
class SentimentConfig:
    neutral_band: float = 0.1
    min_scale: float = 0.2
    max_scale: float = 1.3
    no_trade_negative_threshold: float = -0.4
    # --- Sentiment-exit thresholds ---
    # Hard exit: raw_discrete == -2 -> always close, no delta check (unchanged).
    #
    # Soft exit: fires when delta > soft_exit_delta_threshold AND
    #            confidence > exit_confidence_min.
    #            Catches partial deterioration (e.g. +0.7 -> 0.0, delta=0.70).
    soft_exit_delta_threshold: float = 0.6
    #
    # Strong exit: fires when delta > strong_exit_delta_threshold AND
    #              confidence > strong_exit_confidence_min.
    #              Lower confidence bar because a large delta is self-confirming.
    strong_exit_delta_threshold: float = 1.0
    strong_exit_confidence_min: float = 0.4
    #
    # Minimum model confidence for the soft exit tier.
    exit_confidence_min: float = 0.5
    #
    # Improvement E: exponent applied to confidence before multiplying by base.
    # gamma=1.0 -> linear (original behaviour); gamma=2.0 -> convex weighting
    # that rewards high-confidence scores more and down-weights low-confidence
    # ones.  Clamped to [1.0, 4.0] at runtime.
    confidence_gamma: float = 2.0
    #
    # Legacy alias kept for any downstream code that reads this field directly.
    # Points at the strong threshold so behaviour is unchanged if old code path
    # is ever re-activated.  Do not remove.
    @property
    def exit_sentiment_delta_threshold(self) -> float:
        return self.strong_exit_delta_threshold


@dataclass
class TechnicalSignalConfig:
    weight_momentum_trend: float = 0.5
    weight_mean_reversion: float = 0.3
    weight_price_action: float = 0.2
    long_threshold: float = 0.2
    short_threshold: float = -0.2
    momentum_norm_scale: float = 0.05
    ma_distance_norm_scale: float = 0.05
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    breakout_lookback_bars: int = 20
    breakout_strength: float = 1.0
    base_stop_vol_mult: float = 1.5
    base_tp_vol_mult: float = 3.0
    max_tp_scale_from_signal: float = 1.3
    min_tp_scale_from_signal: float = 0.7
    # Improvement B: fraction of long_th / short_th that the cached sentiment
    # score is allowed to shift the entry threshold asymmetrically.
    # 0.25 means sentiment can tighten or widen each threshold by up to 25%.
    sentiment_th_scale: float = 0.25
    # Fix L3: explicit field so it is visible to introspection/serialisation
    # and can be overridden via config.  Previously accessed via getattr fallback.
    conflict_dampener_penalty: float = 0.6
    # FIX 7: normalisation scale for EMA crossover signal.
    # Wider than momentum_norm_scale (0.05) to prevent ±1 saturation for
    # high-momentum equities (NVDA, TSLA) where EMA spread can exceed 5%.
    # 0.10 allows a graded signal up to ±10% EMA divergence.
    ema_crossover_norm_scale: float = 0.10


@dataclass
class ExecutionConfig:
    enable_trailing_stop: bool = True
    trailing_stop_percent: float = 5.0   # 5% trailing distance
    enable_take_profit: bool = True
    exit_time_in_force: str = "day"
    entry_time_in_force: str = "day"
    post_entry_fill_timeout_sec: int = 15


@dataclass
class InstrumentMeta:
    symbol: str
    exchange: str
    lot_size: float
    fractional: bool
    shortable: bool
    marginable: bool
    trading_hours: str
    sector: str = "UNKNOWN"


@dataclass
class PortfolioConfig:
    enable_portfolio_veto: bool = False
    max_candidates_per_loop: int = 50
    # Fix 4: maximum positions allowed per sector in the selection loop.
    # Prevents the portfolio from being overweight in a single sector
    # (e.g. all 10 TECH symbols selected when only 3 are desired).
    max_positions_per_sector: int = 3


@dataclass
class BotConfig:
    env_mode: str
    live_trading_enabled: bool
    risk_limits: RiskLimits
    sentiment: SentimentConfig
    technical: TechnicalSignalConfig
    execution: ExecutionConfig
    instruments: Dict[str, InstrumentMeta]
    portfolio: PortfolioConfig


@dataclass
class AIConfig:
    api_url: str
    api_key: str


def _load_instrument_whitelist(path: Path) -> Dict[str, InstrumentMeta]:
    if not path.exists():
        raise FileNotFoundError(f"Instrument whitelist not found at {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    instruments: Dict[str, InstrumentMeta] = {}
    for sym, meta in data.items():
        instruments[sym] = InstrumentMeta(
            symbol=sym,
            exchange=meta.get("exchange", "NYSE"),
            lot_size=float(meta.get("lot_size", 1)),
            fractional=bool(meta.get("fractional", False)),
            shortable=bool(meta.get("shortable", False)),
            marginable=bool(meta.get("marginable", False)),
            trading_hours=meta.get("trading_hours", "09:30-16:00"),
            sector=meta.get("sector", "UNKNOWN"),
        )
    return instruments


def load_config() -> BotConfig:
    base = Path(__file__).resolve().parents[1]
    wl_path = base / "config" / "instrument_whitelist.yaml"
    instruments = _load_instrument_whitelist(wl_path)

    risk = RiskLimits()
    sentiment = SentimentConfig()
    technical = TechnicalSignalConfig()
    execution = ExecutionConfig()
    portfolio = PortfolioConfig()

    cfg = BotConfig(
        env_mode=ENV_MODE,
        live_trading_enabled=LIVE_TRADING_ENABLED,
        risk_limits=risk,
        sentiment=sentiment,
        technical=technical,
        execution=execution,
        instruments=instruments,
        portfolio=portfolio,
    )
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(asdict(cfg))
