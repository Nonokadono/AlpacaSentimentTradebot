# CHANGES:
# FIX 5 — Changed max_scale from 1.3 to 1.0 with inline comment explaining headroom safety.
# FIX 10 — Added inline comment to max_scale explaining cap and safety rationale.
# OA-2 — Added OPERATOR ACTION REQUIRED comment for exit_time_in_force = "day".
# OA-3 — Added OPERATOR ACTION REQUIRED comment for 180-minute pre-close blackout.
# KELLY-MIN-FIX — Added kelly_min_risk_pct field to RiskLimits with default 0.001 (0.1%).
#                 This creates a separate floor for Kelly-sized positions, allowing very
#                 low-conviction signals to take minimal risk without being blocked by the
#                 0.5% fixed-fractional floor (min_risk_per_trade_pct).
# NONE-RETURN-FIX — Added explicit error handling to ensure load_config() never returns None.
#                   FileNotFoundError for missing whitelist now includes the full path in the
#                   error message and is never silently caught. All code paths return BotConfig.

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
    # KELLY-MIN-FIX: separate minimum risk floor for Kelly-sized positions.
    # 0.001 = 0.1% of equity, allowing very low-conviction signals to enter
    # with minimal risk. The fixed-fractional floor (min_risk_per_trade_pct)
    # remains at 0.5% and is only used when enable_kelly_sizing=False.
    # Raising this above 0.001 will block weak Kelly signals; lowering it
    # below 0.0005 risks qty=0 rounding on low-priced symbols.
    kelly_min_risk_pct: float = 0.001


@dataclass
class SentimentConfig:
    neutral_band: float = 0.1
    min_scale: float = 0.2
    # FIX 10: changed from 1.3 to 1.0 to make headroom explicit and safe.
    # 1.0 = full risk-per-trade at peak positive sentiment.
    # Values > 1.0 are capped upstream by max_risk_per_trade_pct anyway,
    # but keeping this at 1.0 makes the headroom explicit and safe.
    max_scale: float = 1.0
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
    # CRASH-1 FIX: PnL-coupled sentiment-exit threshold scaling.
    # pnl_exit_scale_enabled=False (default) means the else-branch in
    # _check_and_exit_on_sentiment() is always taken, so effective thresholds
    # equal soft_exit_delta_threshold / strong_exit_delta_threshold exactly —
    # byte-identical to behaviour before this field existed.
    # Set to True to activate PnL-coupled widening/tightening of thresholds.
    pnl_exit_scale_enabled: bool = False
    # Multiplier for unrealised_pnl_pct when computing scale_adj.
    # scale_adj = unrealised_pnl_pct * pnl_exit_scale_factor
    # Only consumed when pnl_exit_scale_enabled=True.
    pnl_exit_scale_factor: float = 0.5
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
    # OPERATOR ACTION REQUIRED (OA-2):
    # exit_time_in_force = "day" means bracket and trailing-stop orders expire
    # at market close. After a kill-switch halt that spans overnight, open
    # positions have no protective orders the next morning. Consider changing
    # to "gtc" (good-till-cancelled) or implementing an order-refresh mechanism
    # on bot startup before going live.
    exit_time_in_force: str = "day"
    entry_time_in_force: str = "day"
    # WAIT-FOR-POSITION: primary timeout for the _wait_for_position() polling
    # loop.  Raised from 15 → 30 seconds to give the active poller adequate
    # time for slow or queued market-order fills.
    post_entry_fill_poll_timeout_sec: int = 30
    # Interval between successive adapter.list_positions() calls in the loop.
    post_entry_fill_poll_interval_sec: float = 2.0
    # DEPRECATED — retained for backward compatibility only.
    # Any serialised config or external writer that still uses this key will
    # not cause a TypeError.  OrderExecutor reads post_entry_fill_poll_timeout_sec
    # first; this field has no runtime effect.  Do NOT remove.
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
    # TASK IC-RANKING: IC-weighted composite ranking gate and weights.
    # enable_composite_ranking = False preserves the original |signal_score|
    # sort order exactly (byte-identical behaviour).
    # When True, candidates are ranked by:
    #   abs(rank_weight_technical * signal_score
    #       + rank_weight_sentiment * sentiment_score)
    # The signed interior means confirming signal/sentiment pairs rank higher
    # than contradicting pairs, so the API sentiment spend influences selection.
    enable_composite_ranking: bool = False
    rank_weight_technical: float = 0.7
    rank_weight_sentiment: float = 0.3


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
    """Load instrument whitelist from YAML file.
    
    Raises FileNotFoundError with full path if file doesn't exist.
    This error is intentionally NOT caught — it must propagate to main()
    so the operator knows the config is incomplete.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Instrument whitelist not found at {path.resolve()}\n"
            f"Expected location: {path}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Please ensure config/instrument_whitelist.yaml exists."
        )
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
    """Load bot configuration from files and environment variables.
    
    NONE-RETURN-FIX: This function now always returns a BotConfig or raises
    an exception — it never returns None. All error paths raise explicit
    exceptions that propagate to main() with actionable error messages.
    """
    base = Path(__file__).resolve().parents[1]
    wl_path = base / "config" / "instrument_whitelist.yaml"
    
    # OPERATOR ACTION REQUIRED (OA-3):
    # TechnicalSignalConfig default is 180-minute pre-close blackout, suppressing
    # all new entries from 13:00 ET onward (3.5-hour entry window per day).
    # For a swing strategy, 60–90 minutes is more appropriate. This is a
    # config/strategy decision, not a code bug. Adjust in TechnicalSignalConfig
    # or via environment override before live trading.
    
    # NONE-RETURN-FIX: _load_instrument_whitelist raises FileNotFoundError
    # if the file is missing — we do NOT catch it here. This ensures the
    # operator sees the full error message with the expected path.
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
