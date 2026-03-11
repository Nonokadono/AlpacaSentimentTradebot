"""
Shared fixtures and mock helpers for the trading bot test suite.
"""
import sys
import os
import math
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from collections import deque

import pytest

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Mock Bar object (simulates broker bar) ──────────────────────────────────

@dataclass
class MockBar:
    """Simulates a broker bar with OHLCV fields."""
    o: float   # open
    h: float   # high
    l: float   # low
    c: float   # close
    v: float = 10000.0  # volume
    t: str = ""  # timestamp


def make_bars(closes: List[float], spread: float = 0.5, volume: float = 10000.0) -> List[MockBar]:
    """Generate mock bars from a list of close prices.

    Creates realistic OHLCV data where:
    - open = close ± small offset
    - high = max(open, close) + spread
    - low  = min(open, close) - spread
    """
    bars = []
    for i, c in enumerate(closes):
        o = c + (0.1 if i % 2 == 0 else -0.1)
        h = max(o, c) + spread
        l_val = min(o, c) - spread
        bars.append(MockBar(o=o, h=h, l=l_val, c=c, v=volume))
    return bars


def make_trending_bars(start: float, end: float, count: int, spread: float = 0.5) -> List[MockBar]:
    """Generate bars with a linear trend from start to end price."""
    step = (end - start) / max(count - 1, 1)
    closes = [start + step * i for i in range(count)]
    return make_bars(closes, spread=spread)


def make_flat_bars(price: float, count: int, noise: float = 0.05) -> List[MockBar]:
    """Generate bars around a flat price with small noise."""
    import random
    random.seed(42)
    closes = [price + random.uniform(-noise, noise) for _ in range(count)]
    return make_bars(closes)


# ── Mock broker objects ─────────────────────────────────────────────────────

@dataclass
class MockAccount:
    equity: str = "100000.00"
    cash: str = "50000.00"
    portfolio_value: str = "100000.00"
    buying_power: str = "200000.00"
    day_trading_buying_power: str = "400000.00"
    realized_pl: str = "0.00"
    unrealized_pl: str = "0.00"


@dataclass
class MockPosition:
    symbol: str
    qty: str
    current_price: str
    side: str
    avg_entry_price: str = "0.0"


@dataclass
class MockOrder:
    id: str
    symbol: str
    side: str = "buy"
    qty: str = "10"
    type: str = "market"
    status: str = "open"
    stop_price: Optional[str] = None
    limit_price: Optional[str] = None


@dataclass
class MockClock:
    is_open: bool = True
    next_close: str = "2026-03-08T21:00:00+00:00"
    next_open: str = "2026-03-09T14:30:00+00:00"


# ── Config fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def risk_limits():
    from config.config import RiskLimits
    return RiskLimits()


@pytest.fixture
def sentiment_config():
    from config.config import SentimentConfig
    return SentimentConfig()


@pytest.fixture
def technical_config():
    from config.config import TechnicalSignalConfig
    return TechnicalSignalConfig()


@pytest.fixture
def execution_config():
    from config.config import ExecutionConfig
    return ExecutionConfig()


@pytest.fixture
def portfolio_config():
    from config.config import PortfolioConfig
    return PortfolioConfig()


@pytest.fixture
def instrument_meta():
    from config.config import InstrumentMeta
    return {
        "AAPL": InstrumentMeta(
            symbol="AAPL", exchange="NASDAQ", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="TECH",
        ),
        "MSFT": InstrumentMeta(
            symbol="MSFT", exchange="NASDAQ", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="TECH",
        ),
        "GOOGL": InstrumentMeta(
            symbol="GOOGL", exchange="NASDAQ", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="TECH",
        ),
        "SPY": InstrumentMeta(
            symbol="SPY", exchange="NYSE", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="ETF_INDEX",
        ),
        "XLF": InstrumentMeta(
            symbol="XLF", exchange="NYSE", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="FINANCE",
        ),
        "BAC": InstrumentMeta(
            symbol="BAC", exchange="NYSE", lot_size=1.0,
            fractional=True, shortable=True, marginable=True,
            trading_hours="09:30-16:00", sector="FINANCE",
        ),
    }


# ── Snapshot fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def healthy_snapshot():
    from core.risk_engine import EquitySnapshot
    return EquitySnapshot(
        equity=100000.0,
        cash=50000.0,
        portfolio_value=100000.0,
        day_trading_buying_power=400000.0,
        start_of_day_equity=100000.0,
        high_watermark_equity=100000.0,
        realized_pl_today=0.0,
        unrealized_pl=0.0,
        gross_exposure=20000.0,
        daily_loss_pct=0.0,
        drawdown_pct=0.0,
    )


@pytest.fixture
def stressed_snapshot():
    from core.risk_engine import EquitySnapshot
    return EquitySnapshot(
        equity=92000.0,
        cash=20000.0,
        portfolio_value=92000.0,
        day_trading_buying_power=80000.0,
        start_of_day_equity=96000.0,
        high_watermark_equity=100000.0,
        realized_pl_today=-4000.0,
        unrealized_pl=-4000.0,
        gross_exposure=72000.0,
        daily_loss_pct=-0.0417,  # >4% daily loss
        drawdown_pct=-0.08,
    )


# ── Sentiment fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def bullish_sentiment():
    from core.sentiment import SentimentResult
    return SentimentResult(
        score=0.6, raw_discrete=1, rawcompound=0.6,
        ndocuments=5, explanation="Bullish", confidence=0.8,
        raw_model_score=0.6,
    )


@pytest.fixture
def bearish_sentiment():
    from core.sentiment import SentimentResult
    return SentimentResult(
        score=-0.5, raw_discrete=-1, rawcompound=-0.5,
        ndocuments=3, explanation="Bearish", confidence=0.7,
        raw_model_score=-0.5,
    )


@pytest.fixture
def neutral_sentiment():
    from core.sentiment import SentimentResult
    return SentimentResult(
        score=0.0, raw_discrete=0, rawcompound=0.0,
        ndocuments=2, explanation="Neutral", confidence=0.5,
        raw_model_score=0.0,
    )


@pytest.fixture
def chaos_sentiment():
    from core.sentiment import SentimentResult
    return SentimentResult(
        score=-1.0, raw_discrete=-2, rawcompound=-1.0,
        ndocuments=4, explanation="Chaos", confidence=0.9,
        raw_model_score=-2.0,
    )


# ── Temp directory fixture ──────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provides a temporary directory and patches paths for state files."""
    return tmp_path
