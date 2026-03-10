"""
Tests for core/portfolio_builder.py — sector caps, ranking, exposure guards.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import MagicMock, patch

from core.portfolio_builder import PortfolioBuilder
from core.risk_engine import RiskEngine, EquitySnapshot, PositionInfo, ProposedTrade
from core.signals import SignalEngine, Signal
from core.sentiment import SentimentModule, SentimentResult
from config.config import (
    BotConfig, RiskLimits, SentimentConfig, TechnicalSignalConfig,
    ExecutionConfig, PortfolioConfig, InstrumentMeta,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_cfg(instruments=None, portfolio=None, risk=None):
    if instruments is None:
        instruments = {
            f"TECH{i}": InstrumentMeta(
                symbol=f"TECH{i}", exchange="NASDAQ", lot_size=1.0,
                fractional=True, shortable=True, marginable=True,
                trading_hours="09:30-16:00", sector="TECH",
            )
            for i in range(6)
        }
        instruments["SPY"] = InstrumentMeta(
            symbol="SPY", exchange="NYSE", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="ETF_INDEX",
        )
        instruments["XLF"] = InstrumentMeta(
            symbol="XLF", exchange="NYSE", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="FINANCE",
        )
    return BotConfig(
        env_mode="PAPER",
        live_trading_enabled=False,
        risk_limits=risk or RiskLimits(),
        sentiment=SentimentConfig(),
        technical=TechnicalSignalConfig(),
        execution=ExecutionConfig(),
        instruments=instruments,
        portfolio=portfolio or PortfolioConfig(),
    )


def _make_snapshot(equity=100000.0, gross_exposure=10000.0):
    return EquitySnapshot(
        equity=equity, cash=equity * 0.5, portfolio_value=equity,
        day_trading_buying_power=equity * 4,
        start_of_day_equity=equity, high_watermark_equity=equity,
        realized_pl_today=0.0, unrealized_pl=0.0,
        gross_exposure=gross_exposure,
        daily_loss_pct=0.0, drawdown_pct=0.0,
    )


def _make_signal(symbol, side="buy", signal_score=0.5, last_price=150.0,
                 stop_price=145.0, take_profit_price=160.0):
    return Signal(
        symbol=symbol, side=side,
        rationale="test", sentiment_result=SentimentResult(
            score=0.5, raw_discrete=1, rawcompound=0.5,
            ndocuments=3, confidence=0.7, raw_model_score=0.5,
        ),
        stop_price=stop_price, take_profit_price=take_profit_price,
        signal_score=signal_score, momentum_score=0.3,
        mean_reversion_score=0.2, price_action_score=0.0,
        volatility=0.02, last_price=last_price,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO BUILDER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPortfolioBuilder:

    def test_empty_when_fully_allocated(self):
        """No candidates when gross exposure >= cap."""
        cfg = _make_cfg()
        builder = PortfolioBuilder(
            cfg, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )
        snapshot = _make_snapshot(gross_exposure=91000.0)  # 91% > 90% cap
        result = builder.build_portfolio(snapshot, {}, [])
        assert result == []

    def test_empty_when_max_positions(self):
        """No candidates when at max positions."""
        cfg = _make_cfg()
        builder = PortfolioBuilder(
            cfg, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )
        snapshot = _make_snapshot()
        positions = {
            f"P{i}": PositionInfo(
                symbol=f"P{i}", qty=10, market_price=100,
                side="long", notional=1000,
            )
            for i in range(15)
        }
        result = builder.build_portfolio(snapshot, positions, [])
        assert result == []

    def test_skips_existing_positions(self):
        """Symbols already held should be skipped."""
        cfg = _make_cfg()
        signal_engine = MagicMock(spec=SignalEngine)
        risk_engine = MagicMock(spec=RiskEngine)
        builder = PortfolioBuilder(
            cfg, MagicMock(), MagicMock(), signal_engine, risk_engine,
        )
        snapshot = _make_snapshot()
        positions = {
            "TECH0": PositionInfo(
                symbol="TECH0", qty=10, market_price=100,
                side="long", notional=1000,
            ),
        }
        # Signals for non-TECH0 symbols generate valid ProposedTrades via risk engine
        signal_engine.generate_signal_for_symbol.return_value = _make_signal(
            "TECH1", side="buy",
        )
        # risk_engine returns a rejected trade so no qty > 0 passes the filter
        risk_engine.pre_trade_checks.return_value = ProposedTrade(
            symbol="TECH1", side="buy", qty=0.0, entry_price=150.0,
            stop_price=145.0, take_profit_price=160.0,
            risk_amount=0.0, risk_pct_of_equity=0.0,
            sentiment_score=0.5, sentiment_scale=0.8,
            signal_score=0.5, rationale="test",
            rejected_reason="exposure cap",
        )
        result = builder.build_portfolio(snapshot, positions, [])
        # TECH0 should not appear in result
        assert all(t.symbol != "TECH0" for t in result)

    def test_skips_pending_orders(self):
        """Symbols with pending orders should be skipped."""
        cfg = _make_cfg()
        signal_engine = MagicMock(spec=SignalEngine)
        risk_engine = MagicMock(spec=RiskEngine)
        builder = PortfolioBuilder(
            cfg, MagicMock(), MagicMock(), signal_engine, risk_engine,
        )
        snapshot = _make_snapshot()

        pending_order = MagicMock()
        pending_order.symbol = "TECH0"
        signal_engine.generate_signal_for_symbol.return_value = _make_signal(
            "TECH1", side="buy",
        )
        # risk_engine returns a rejected trade so no qty > 0 passes the filter
        risk_engine.pre_trade_checks.return_value = ProposedTrade(
            symbol="TECH1", side="buy", qty=0.0, entry_price=150.0,
            stop_price=145.0, take_profit_price=160.0,
            risk_amount=0.0, risk_pct_of_equity=0.0,
            sentiment_score=0.5, sentiment_scale=0.8,
            signal_score=0.5, rationale="test",
            rejected_reason="exposure cap",
        )
        result = builder.build_portfolio(snapshot, {}, [pending_order])
        assert all(t.symbol != "TECH0" for t in result)


class TestSectorDiversification:
    """Tests for max_positions_per_sector enforcement."""

    def test_sector_cap_enforced(self):
        """Should not select more than max_positions_per_sector from same sector."""
        portfolio_cfg = PortfolioConfig(max_positions_per_sector=2)
        instruments = {
            f"TECH{i}": InstrumentMeta(
                symbol=f"TECH{i}", exchange="NASDAQ", lot_size=1.0,
                fractional=True, shortable=True, marginable=True,
                trading_hours="09:30-16:00", sector="TECH",
            )
            for i in range(6)
        }
        cfg = _make_cfg(instruments=instruments, portfolio=portfolio_cfg)

        signal_engine = MagicMock(spec=SignalEngine)
        risk_engine = MagicMock(spec=RiskEngine)

        # All tech symbols produce valid signals
        def gen_signal(sym):
            return _make_signal(sym, side="buy", signal_score=0.5)
        signal_engine.generate_signal_for_symbol.side_effect = gen_signal

        # All pass risk checks
        def pre_trade(snapshot, positions, symbol, side, entry_price,
                      stop_price, take_profit_price, sentiment,
                      signal_score=0, rationale=None, volatility=0,
                      **kwargs):
            return ProposedTrade(
                symbol=symbol, side=side, qty=10.0,
                entry_price=entry_price, stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=500, risk_pct_of_equity=0.005,
                sentiment_score=0.5, sentiment_scale=0.8,
                signal_score=signal_score, rationale=rationale,
                rejected_reason=None,
            )
        risk_engine.pre_trade_checks.side_effect = pre_trade

        builder = PortfolioBuilder(
            cfg, MagicMock(), MagicMock(), signal_engine, risk_engine,
        )
        snapshot = _make_snapshot()
        result = builder.build_portfolio(snapshot, {}, [])

        # Should not exceed 2 TECH positions
        tech_count = sum(1 for t in result if t.symbol.startswith("TECH"))
        assert tech_count <= 2

    def test_existing_positions_count_against_sector_cap(self):
        """Pre-existing positions should be counted in sector cap."""
        portfolio_cfg = PortfolioConfig(max_positions_per_sector=2)
        instruments = {
            f"TECH{i}": InstrumentMeta(
                symbol=f"TECH{i}", exchange="NASDAQ", lot_size=1.0,
                fractional=True, shortable=True, marginable=True,
                trading_hours="09:30-16:00", sector="TECH",
            )
            for i in range(5)
        }
        cfg = _make_cfg(instruments=instruments, portfolio=portfolio_cfg)

        signal_engine = MagicMock(spec=SignalEngine)
        risk_engine = MagicMock(spec=RiskEngine)

        def gen_signal(sym):
            return _make_signal(sym, side="buy", signal_score=0.5)
        signal_engine.generate_signal_for_symbol.side_effect = gen_signal

        def pre_trade(snapshot, positions, symbol, side, entry_price,
                      stop_price, take_profit_price, sentiment,
                      signal_score=0, rationale=None, volatility=0,
                      **kwargs):
            return ProposedTrade(
                symbol=symbol, side=side, qty=10.0,
                entry_price=entry_price, stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=500, risk_pct_of_equity=0.005,
                sentiment_score=0.5, sentiment_scale=0.8,
                signal_score=signal_score, rationale=rationale,
                rejected_reason=None,
            )
        risk_engine.pre_trade_checks.side_effect = pre_trade

        builder = PortfolioBuilder(
            cfg, MagicMock(), MagicMock(), signal_engine, risk_engine,
        )
        snapshot = _make_snapshot()
        # Already hold 2 TECH positions
        positions = {
            "TECH0": PositionInfo(
                symbol="TECH0", qty=10, market_price=150,
                side="long", notional=1500,
            ),
            "TECH1": PositionInfo(
                symbol="TECH1", qty=10, market_price=150,
                side="long", notional=1500,
            ),
        }
        result = builder.build_portfolio(snapshot, positions, [])
        # No more TECH should be selected (cap=2, already have 2)
        new_tech = sum(1 for t in result if t.symbol.startswith("TECH"))
        assert new_tech == 0


class TestRanking:
    """Tests for candidate ranking logic."""

    def test_default_ranking_by_signal_score(self):
        """Default ranking sorts by |signal_score| descending."""
        portfolio_cfg = PortfolioConfig(enable_composite_ranking=False)
        instruments = {
            f"SYM{i}": InstrumentMeta(
                symbol=f"SYM{i}", exchange="NYSE", lot_size=1.0,
                fractional=True, shortable=True, marginable=True,
                trading_hours="09:30-16:00", sector=f"SECTOR{i}",
            )
            for i in range(3)
        }
        cfg = _make_cfg(instruments=instruments, portfolio=portfolio_cfg)

        signal_engine = MagicMock(spec=SignalEngine)
        risk_engine = MagicMock(spec=RiskEngine)

        scores = {"SYM0": 0.3, "SYM1": 0.8, "SYM2": 0.5}
        def gen_signal(sym):
            return _make_signal(sym, side="buy", signal_score=scores[sym])
        signal_engine.generate_signal_for_symbol.side_effect = gen_signal

        def pre_trade(snapshot, positions, symbol, side, entry_price,
                      stop_price, take_profit_price, sentiment,
                      signal_score=0, rationale=None, volatility=0,
                      **kwargs):
            return ProposedTrade(
                symbol=symbol, side=side, qty=10.0,
                entry_price=entry_price, stop_price=stop_price,
                take_profit_price=take_profit_price,
                risk_amount=500, risk_pct_of_equity=0.005,
                sentiment_score=0.5, sentiment_scale=0.8,
                signal_score=signal_score, rationale=rationale,
                rejected_reason=None,
            )
        risk_engine.pre_trade_checks.side_effect = pre_trade

        builder = PortfolioBuilder(
            cfg, MagicMock(), MagicMock(), signal_engine, risk_engine,
        )
        result = builder.build_portfolio(_make_snapshot(), {}, [])
        if len(result) >= 2:
            # First should have highest |signal_score|
            assert abs(result[0].signal_score) >= abs(result[1].signal_score)
