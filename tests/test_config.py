"""
Tests for config/config.py — configuration loading, whitelist parsing, defaults.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import pytest
import yaml
from unittest.mock import patch
from dataclasses import asdict

from config.config import (
    RiskLimits, SentimentConfig, TechnicalSignalConfig, ExecutionConfig,
    PortfolioConfig, InstrumentMeta, BotConfig, _load_instrument_whitelist,
)


# ═══════════════════════════════════════════════════════════════════════════
#  RISK LIMITS DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskLimitsDefaults:

    def test_max_risk_per_trade(self):
        rl = RiskLimits()
        assert rl.max_risk_per_trade_pct == 0.03

    def test_min_risk_per_trade(self):
        rl = RiskLimits()
        assert rl.min_risk_per_trade_pct == 0.005

    def test_gross_exposure_cap(self):
        rl = RiskLimits()
        assert rl.gross_exposure_cap_pct == 0.90

    def test_daily_loss_limit(self):
        rl = RiskLimits()
        assert rl.daily_loss_limit_pct == 0.04

    def test_max_drawdown(self):
        rl = RiskLimits()
        assert rl.max_drawdown_pct == 0.09

    def test_max_open_positions(self):
        rl = RiskLimits()
        assert rl.max_open_positions == 15

    def test_kelly_enabled_by_default(self):
        rl = RiskLimits()
        assert rl.enable_kelly_sizing is True

    def test_kelly_min_risk_pct(self):
        rl = RiskLimits()
        assert rl.kelly_min_risk_pct == 0.001


# ═══════════════════════════════════════════════════════════════════════════
#  SENTIMENT CONFIG DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSentimentConfigDefaults:

    def test_neutral_band(self):
        sc = SentimentConfig()
        assert sc.neutral_band == 0.1

    def test_min_scale(self):
        sc = SentimentConfig()
        assert sc.min_scale == 0.2

    def test_max_scale(self):
        sc = SentimentConfig()
        assert sc.max_scale == 1.0

    def test_no_trade_threshold(self):
        sc = SentimentConfig()
        assert sc.no_trade_negative_threshold == -0.4

    def test_exit_thresholds(self):
        sc = SentimentConfig()
        assert sc.soft_exit_delta_threshold == 0.6
        assert sc.strong_exit_delta_threshold == 1.0

    def test_exit_confidence_mins(self):
        sc = SentimentConfig()
        assert sc.exit_confidence_min == 0.5
        assert sc.strong_exit_confidence_min == 0.4

    def test_pnl_exit_scale_disabled_by_default(self):
        sc = SentimentConfig()
        assert sc.pnl_exit_scale_enabled is False

    def test_legacy_alias(self):
        sc = SentimentConfig()
        assert sc.exit_sentiment_delta_threshold == sc.strong_exit_delta_threshold


# ═══════════════════════════════════════════════════════════════════════════
#  TECHNICAL SIGNAL CONFIG DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

class TestTechnicalSignalConfigDefaults:

    def test_weights_sum_to_one(self):
        tc = TechnicalSignalConfig()
        total = tc.weight_momentum_trend + tc.weight_mean_reversion + tc.weight_price_action
        assert total == pytest.approx(1.0)

    def test_momentum_blend_weights_sum_to_one(self):
        tc = TechnicalSignalConfig()
        total = tc.weight_ema_mom + tc.weight_macd_mom + tc.weight_trend_mom
        assert total == pytest.approx(1.0)

    def test_mr_blend_weights_sum_to_one(self):
        tc = TechnicalSignalConfig()
        total = tc.weight_rsi_mr + tc.weight_bb_mr + tc.weight_sma_dist_mr
        assert total == pytest.approx(1.0)

    def test_thresholds_symmetric(self):
        tc = TechnicalSignalConfig()
        assert tc.long_threshold == -tc.short_threshold

    def test_macd_defaults(self):
        tc = TechnicalSignalConfig()
        assert tc.macd_fast == 12
        assert tc.macd_slow == 26
        assert tc.macd_signal == 9

    def test_bollinger_defaults(self):
        tc = TechnicalSignalConfig()
        assert tc.bb_period == 20
        assert tc.bb_std_dev == 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  EXECUTION CONFIG DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutionConfigDefaults:

    def test_exit_tif_is_gtc(self):
        ec = ExecutionConfig()
        assert ec.exit_time_in_force == "gtc"

    def test_entry_tif_is_day(self):
        ec = ExecutionConfig()
        assert ec.entry_time_in_force == "day"

    def test_poll_timeout(self):
        ec = ExecutionConfig()
        assert ec.post_entry_fill_poll_timeout_sec == 30


# ═══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO CONFIG DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPortfolioConfigDefaults:

    def test_veto_disabled(self):
        pc = PortfolioConfig()
        assert pc.enable_portfolio_veto is False

    def test_max_positions_per_sector(self):
        pc = PortfolioConfig()
        assert pc.max_positions_per_sector == 3

    def test_composite_ranking_disabled(self):
        pc = PortfolioConfig()
        assert pc.enable_composite_ranking is False

    def test_ranking_weights(self):
        pc = PortfolioConfig()
        assert pc.rank_weight_technical + pc.rank_weight_sentiment == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  INSTRUMENT WHITELIST LOADING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestInstrumentWhitelist:

    def test_load_valid_whitelist(self, tmp_path):
        wl = {
            "AAPL": {
                "exchange": "NASDAQ",
                "lot_size": 1,
                "fractional": True,
                "shortable": True,
                "marginable": True,
                "trading_hours": "09:30-16:00",
                "sector": "TECH",
            },
            "SPY": {
                "exchange": "NYSE",
                "lot_size": 1,
                "fractional": True,
                "shortable": True,
                "marginable": True,
                "trading_hours": "09:30-16:00",
                "sector": "ETF_INDEX",
            },
        }
        wl_path = tmp_path / "whitelist.yaml"
        wl_path.write_text(yaml.dump(wl))

        instruments = _load_instrument_whitelist(wl_path)
        assert "AAPL" in instruments
        assert "SPY" in instruments
        assert instruments["AAPL"].sector == "TECH"
        assert instruments["SPY"].exchange == "NYSE"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_instrument_whitelist(tmp_path / "nonexistent.yaml")

    def test_defaults_for_missing_fields(self, tmp_path):
        wl = {"TEST": {}}
        wl_path = tmp_path / "whitelist.yaml"
        wl_path.write_text(yaml.dump(wl))

        instruments = _load_instrument_whitelist(wl_path)
        meta = instruments["TEST"]
        assert meta.exchange == "NYSE"
        assert meta.lot_size == 1.0
        assert meta.sector == "UNKNOWN"
        assert meta.fractional is False

    def test_empty_whitelist(self, tmp_path):
        wl_path = tmp_path / "whitelist.yaml"
        wl_path.write_text("")
        instruments = _load_instrument_whitelist(wl_path)
        assert instruments == {}


# ═══════════════════════════════════════════════════════════════════════════
#  BOT CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBotConfig:

    def test_serializable(self):
        """BotConfig should be JSON-serializable via asdict."""
        instruments = {
            "AAPL": InstrumentMeta(
                symbol="AAPL", exchange="NASDAQ", lot_size=1.0,
                fractional=True, shortable=True, marginable=True,
                trading_hours="09:30-16:00", sector="TECH",
            ),
        }
        cfg = BotConfig(
            env_mode="PAPER",
            live_trading_enabled=False,
            risk_limits=RiskLimits(),
            sentiment=SentimentConfig(),
            technical=TechnicalSignalConfig(),
            execution=ExecutionConfig(),
            instruments=instruments,
            portfolio=PortfolioConfig(),
        )
        d = asdict(cfg)
        assert d["env_mode"] == "PAPER"
        assert "AAPL" in d["instruments"]
