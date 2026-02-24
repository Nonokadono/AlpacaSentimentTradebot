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
    max_risk_per_trade_pct: float = 0.01     # 1% of equity
    min_risk_per_trade_pct: float = 0.005    # 0.5% of equity
    gross_exposure_cap_pct: float = 0.90     # 90% of equity
    daily_loss_limit_pct: float = 0.04       # 4% of start-of-day equity
    max_drawdown_pct: float = 0.09           # 9% from high watermark
    max_open_positions: int = 15


@dataclass
class SentimentConfig:
    neutral_band: float = 0.1
    min_scale: float = 0.2
    max_scale: float = 1.3
    no_trade_negative_threshold: float = -0.4


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


@dataclass
class BotConfig:
    env_mode: str
    live_trading_enabled: bool
    risk_limits: RiskLimits
    sentiment: SentimentConfig
    technical: TechnicalSignalConfig
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
    portfolio = PortfolioConfig()

    cfg = BotConfig(
        env_mode=ENV_MODE,
        live_trading_enabled=LIVE_TRADING_ENABLED,
        risk_limits=risk,
        sentiment=sentiment,
        technical=technical,
        instruments=instruments,
        portfolio=portfolio,
    )
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(asdict(cfg))






