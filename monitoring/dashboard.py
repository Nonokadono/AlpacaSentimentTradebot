# REDESIGN: Modern frameless fintech-style PyQt5 dashboard.
#
# Architecture preserved:
#   - multiprocessing.Process GUI (PyQt5 requires QApplication in main thread of process)
#   - Pickle-based IPC via dashboard_state.pkl (atomic writes)
#   - build_position_rows() / build_price_rows() helpers called from main.py
#
# UI changes:
#   - Frameless window with custom title bar (drag, resize, min/max/close)
#   - Metric cards row (equity, unrealized P&L, realized today, daily %, drawdown, exposure)
#   - Enhanced positions table (side, qty, notional columns)
#   - Modern dark color palette with global QSS stylesheet
#   - Custom status bar with state badge

from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("tradebot")

# ── dev_mode flag ──────────────────────────────────────────────────────────────
dev_mode: bool = False

# ── PyQt5 availability check ───────────────────────────────────────────────────
try:
    from PyQt5.QtCore import QPoint, QSize, QTimer, Qt
    from PyQt5.QtGui import QColor, QCursor, QFont, QPainter, QPainterPath, QPen
    from PyQt5.QtWidgets import (
        QApplication,
        QGraphicsDropShadowEffect,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPushButton,
        QSizePolicy,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    _PYQT5_AVAILABLE = True
except ImportError:
    _PYQT5_AVAILABLE = False

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "bg_primary":          "#0B0E14",
    "bg_secondary":        "#111827",
    "bg_tertiary":         "#1A1F2E",
    "bg_elevated":         "#1F2937",
    "text_primary":        "#E5E7EB",
    "text_secondary":      "#9CA3AF",
    "text_muted":          "#4B5563",
    "accent":              "#3B82F6",
    "accent_hover":        "#60A5FA",
    "success":             "#10B981",
    "success_bg":          "#064E3B",
    "danger":              "#EF4444",
    "danger_bg":           "#7F1D1D",
    "warning":             "#F59E0B",
    "warning_bg":          "#78350F",
    "info":                "#06B6D4",
    "border":              "#1F2937",
    "border_subtle":       "#374151",
    "divider":             "#1F2937",
    "titlebar_bg":         "#0B0E14",
    "titlebar_btn_hover":  "#1F2937",
    "titlebar_close_hover": "#DC2626",
}

# ── Shared state dataclasses ───────────────────────────────────────────────────


@dataclass
class PositionRow:
    """Snapshot of a single open position for the UI positions table."""
    symbol: str
    current_price: float
    entry_price: float
    pct_change: float       # e.g. 0.032 == +3.2%
    pnl: float              # absolute dollar P&L
    take_profit: float
    stop_loss: float
    side: str = "long"      # "long" or "short"
    qty: float = 0.0
    notional: float = 0.0


@dataclass
class PriceRow:
    """Snapshot of a whitelisted symbol's live quote for the price tab."""
    symbol: str
    last_price: float
    bid: float
    ask: float
    daily_pct_change: float  # e.g. -0.012 == -1.2%


@dataclass
class DashboardState:
    """State container written by main.py, read by the GUI.

    Pickled to dashboard_state.pkl on every main loop cycle.
    The GUI polls this file and reconstructs the object.
    """
    cash: float = 0.0
    buying_power: float = 0.0
    cycle_count: int = 0
    last_run_ts: float = 0.0
    bot_state: str = "IDLE"
    market_open: bool = False
    positions: List[PositionRow] = field(default_factory=list)
    prices: List[PriceRow] = field(default_factory=list)
    # Extended equity/risk fields for metric cards
    equity: float = 0.0
    portfolio_value: float = 0.0
    unrealized_pl: float = 0.0
    realized_pl_today: float = 0.0
    daily_loss_pct: float = 0.0
    drawdown_pct: float = 0.0
    gross_exposure: float = 0.0
    high_watermark_equity: float = 0.0
    num_positions: int = 0
    max_positions: int = 0

    def __setstate__(self, state: dict) -> None:
        """Backward-compatible unpickle: fill missing fields with defaults."""
        defaults = DashboardState()
        for f in fields(DashboardState):
            if f.name not in state:
                state[f.name] = getattr(defaults, f.name)
        self.__dict__.update(state)

    def update(
        self,
        *,
        cash: Optional[float] = None,
        buying_power: Optional[float] = None,
        cycle_count: Optional[int] = None,
        bot_state: Optional[str] = None,
        market_open: Optional[bool] = None,
        positions: Optional[List[PositionRow]] = None,
        prices: Optional[List[PriceRow]] = None,
        equity: Optional[float] = None,
        portfolio_value: Optional[float] = None,
        unrealized_pl: Optional[float] = None,
        realized_pl_today: Optional[float] = None,
        daily_loss_pct: Optional[float] = None,
        drawdown_pct: Optional[float] = None,
        gross_exposure: Optional[float] = None,
        high_watermark_equity: Optional[float] = None,
        num_positions: Optional[int] = None,
        max_positions: Optional[int] = None,
    ) -> None:
        """Update one or more fields. Called by main.py before persist_state()."""
        if cash is not None:
            self.cash = cash
        if buying_power is not None:
            self.buying_power = buying_power
        if cycle_count is not None:
            self.cycle_count = cycle_count
            self.last_run_ts = time.time()
        if bot_state is not None:
            self.bot_state = bot_state
        if market_open is not None:
            self.market_open = market_open
        if positions is not None:
            self.positions = positions
        if prices is not None:
            self.prices = prices
        if equity is not None:
            self.equity = equity
        if portfolio_value is not None:
            self.portfolio_value = portfolio_value
        if unrealized_pl is not None:
            self.unrealized_pl = unrealized_pl
        if realized_pl_today is not None:
            self.realized_pl_today = realized_pl_today
        if daily_loss_pct is not None:
            self.daily_loss_pct = daily_loss_pct
        if drawdown_pct is not None:
            self.drawdown_pct = drawdown_pct
        if gross_exposure is not None:
            self.gross_exposure = gross_exposure
        if high_watermark_equity is not None:
            self.high_watermark_equity = high_watermark_equity
        if num_positions is not None:
            self.num_positions = num_positions
        if max_positions is not None:
            self.max_positions = max_positions


# ── Singleton state shared between main loop and GUI ───────────────────────────
dashboard_state: DashboardState = DashboardState()

# ── Pickle-based IPC file path ─────────────────────────────────────────────────
STATE_FILE = Path("dashboard_state.pkl")


def persist_state(state: DashboardState) -> None:
    """Pickle dashboard_state to disk for cross-process IPC.

    Uses atomic tempfile + os.replace pattern so a crash during
    write cannot corrupt the dashboard state file.
    """
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=STATE_FILE.parent,
            prefix=".dashboard_state_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            os.unlink(tmp_path)
            raise
        os.replace(tmp_path, STATE_FILE)
    except Exception as e:
        logger.warning(f"persist_state error: {e}")


def load_state() -> DashboardState:
    """Load dashboard_state from disk. Returns default state on error."""
    if not STATE_FILE.exists():
        return DashboardState()
    try:
        with STATE_FILE.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"load_state error: {e}")
        return DashboardState()


# ── Global QSS stylesheet ─────────────────────────────────────────────────────

_FONT_UI = "'Segoe UI', 'SF Pro Display', 'Helvetica Neue', sans-serif"
_FONT_MONO = "'Cascadia Code', 'Consolas', 'Courier New', monospace"

GLOBAL_QSS = f"""
/* === Scrollbars === */
QScrollBar:vertical {{
    background: {COLORS['bg_primary']};
    width: 8px;
    border: none;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border_subtle']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['text_muted']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
QScrollBar:horizontal {{
    background: {COLORS['bg_primary']};
    height: 8px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['border_subtle']};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS['text_muted']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* === Tab Widget === */
QTabWidget::pane {{
    background: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border']};
    border-top: none;
    border-radius: 0 0 8px 8px;
}}
QTabBar::tab {{
    background: {COLORS['bg_elevated']};
    color: {COLORS['text_muted']};
    padding: 8px 24px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-family: {_FONT_UI};
    font-size: 13px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {COLORS['bg_secondary']};
    color: {COLORS['accent']};
    border-bottom: 2px solid {COLORS['accent']};
}}
QTabBar::tab:hover:!selected {{
    background: {COLORS['bg_tertiary']};
    color: {COLORS['text_secondary']};
}}

/* === Tables === */
QTableWidget {{
    background-color: {COLORS['bg_secondary']};
    alternate-background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_primary']};
    gridline-color: {COLORS['divider']};
    border: none;
    font-family: {_FONT_MONO};
    font-size: 13px;
    selection-background-color: {COLORS['bg_tertiary']};
    selection-color: {COLORS['text_primary']};
    outline: none;
}}
QHeaderView::section {{
    background-color: {COLORS['bg_elevated']};
    color: {COLORS['text_secondary']};
    padding: 8px 10px;
    border: none;
    border-bottom: 1px solid {COLORS['border']};
    border-right: 1px solid {COLORS['border']};
    font-family: {_FONT_UI};
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}}
QTableWidget::item {{
    padding: 6px 10px;
    border-bottom: 1px solid {COLORS['divider']};
}}

/* === Tooltip === */
QToolTip {{
    background: {COLORS['bg_elevated']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_subtle']};
    padding: 4px 8px;
    font-family: {_FONT_UI};
    font-size: 12px;
}}
"""

# ── PyQt5 GUI implementation ───────────────────────────────────────────────────

if _PYQT5_AVAILABLE:

    # ── Custom widgets ─────────────────────────────────────────────────────

    class TitleBarButton(QPushButton):
        """Flat title-bar button (minimize / maximize / close)."""

        def __init__(self, text: str, is_close: bool = False, parent: QWidget = None) -> None:
            super().__init__(text, parent)
            self._is_close = is_close
            self.setFixedSize(46, 36)
            self.setCursor(QCursor(Qt.PointingHandCursor))
            base = (
                f"QPushButton {{ background: transparent; border: none; "
                f"color: {COLORS['text_secondary']}; font-size: 16px; "
                f"font-family: {_FONT_UI}; }}"
            )
            if is_close:
                hover = (
                    f"QPushButton:hover {{ background: {COLORS['titlebar_close_hover']}; "
                    f"color: #FFFFFF; }}"
                )
            else:
                hover = (
                    f"QPushButton:hover {{ background: {COLORS['titlebar_btn_hover']}; "
                    f"color: {COLORS['text_primary']}; }}"
                )
            self.setStyleSheet(base + hover)

    class TitleBar(QWidget):
        """Custom frameless title bar with drag-to-move and window controls."""

        def __init__(self, parent: QWidget) -> None:
            super().__init__(parent)
            self._parent = parent
            self._drag_pos: Optional[QPoint] = None
            self.setFixedHeight(40)
            self.setStyleSheet(
                f"background: {COLORS['titlebar_bg']}; "
                f"border-bottom: 1px solid {COLORS['border']};"
            )

            layout = QHBoxLayout(self)
            layout.setContentsMargins(16, 0, 0, 0)
            layout.setSpacing(0)

            # App title
            title = QLabel("Alpaca TradeBot")
            title.setStyleSheet(
                f"color: {COLORS['text_secondary']}; "
                f"font-family: {_FONT_UI}; font-size: 13px; font-weight: 600; "
                f"background: transparent; border: none;"
            )
            layout.addWidget(title)
            layout.addStretch()

            # Window control buttons
            self._btn_min = TitleBarButton("\u2014")  # em-dash
            self._btn_max = TitleBarButton("\u25a1")  # white square
            self._btn_close = TitleBarButton("\u2715", is_close=True)  # multiplication X
            self._btn_min.clicked.connect(self._on_minimize)
            self._btn_max.clicked.connect(self._on_maximize)
            self._btn_close.clicked.connect(self._on_close)
            layout.addWidget(self._btn_min)
            layout.addWidget(self._btn_max)
            layout.addWidget(self._btn_close)

        def _on_minimize(self) -> None:
            self._parent.showMinimized()

        def _on_maximize(self) -> None:
            if self._parent.isMaximized():
                self._parent.showNormal()
                self._btn_max.setText("\u25a1")
            else:
                self._parent.showMaximized()
                self._btn_max.setText("\u25a3")  # filled square

        def _on_close(self) -> None:
            QApplication.quit()
            os._exit(0)

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.LeftButton:
                self._drag_pos = event.globalPos() - self._parent.frameGeometry().topLeft()
                event.accept()

        def mouseDoubleClickEvent(self, event) -> None:
            if event.button() == Qt.LeftButton:
                self._on_maximize()

        def mouseMoveEvent(self, event) -> None:
            if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
                if self._parent.isMaximized():
                    self._parent.showNormal()
                    self._btn_max.setText("\u25a1")
                    # Recalculate drag pos after un-maximize
                    self._drag_pos = QPoint(
                        self._parent.width() // 2,
                        self.height() // 2,
                    )
                self._parent.move(event.globalPos() - self._drag_pos)
                event.accept()

        def mouseReleaseEvent(self, event) -> None:
            self._drag_pos = None

    class MetricCard(QWidget):
        """Rounded card showing a single KPI metric."""

        def __init__(self, title: str, parent: QWidget = None) -> None:
            super().__init__(parent)
            self.setMinimumWidth(140)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.setFixedHeight(80)
            self.setStyleSheet(
                f"MetricCard {{ background: {COLORS['bg_secondary']}; "
                f"border: 1px solid {COLORS['border']}; border-radius: 8px; }}"
            )

            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 10, 14, 10)
            layout.setSpacing(2)

            self._title_label = QLabel(title.upper())
            self._title_label.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-family: {_FONT_UI}; "
                f"font-size: 10px; font-weight: 600; letter-spacing: 1px; "
                f"background: transparent; border: none;"
            )
            layout.addWidget(self._title_label)

            self._value_label = QLabel("\u2014")  # em-dash placeholder
            self._value_label.setStyleSheet(
                f"color: {COLORS['text_primary']}; font-family: {_FONT_MONO}; "
                f"font-size: 20px; font-weight: 700; "
                f"background: transparent; border: none;"
            )
            layout.addWidget(self._value_label)

        def set_value(self, text: str, color: Optional[str] = None) -> None:
            self._value_label.setText(text)
            c = color or COLORS['text_primary']
            self._value_label.setStyleSheet(
                f"color: {c}; font-family: {_FONT_MONO}; "
                f"font-size: 20px; font-weight: 700; "
                f"background: transparent; border: none;"
            )

    class StatusBadge(QLabel):
        """Self-coloring bot state badge."""

        _STATE_STYLES = {
            "IDLE": (COLORS['text_muted'], COLORS['bg_elevated']),
            "SCANNING": ("#FEF08A", COLORS['warning_bg']),
            "EXECUTING": ("#86EFAC", COLORS['success_bg']),
            "HALTED": ("#FCA5A5", COLORS['danger_bg']),
        }

        def __init__(self, parent: QWidget = None) -> None:
            super().__init__("IDLE", parent)
            self.set_state("IDLE")

        def set_state(self, state: str) -> None:
            s = state.upper()
            fg, bg = self._STATE_STYLES.get(s, self._STATE_STYLES["IDLE"])
            self.setText(s)
            self.setStyleSheet(
                f"background: {bg}; color: {fg}; "
                f"padding: 3px 12px; border-radius: 4px; "
                f"font-family: {_FONT_UI}; font-size: 11px; font-weight: 700; "
                f"letter-spacing: 1px;"
            )

    # ── Main window ────────────────────────────────────────────────────────

    class TradeBotMainWindow(QWidget):
        """Modern frameless PyQt5 dashboard for the Alpaca trading bot."""

        _EDGE_SIZE = 6  # pixels from edge for resize grab

        def __init__(self, refresh_seconds: float = 5.0) -> None:
            super().__init__()
            self._refresh_ms = int(refresh_seconds * 1000)
            self._pulse = False

            # Frameless + translucent for rounded corners
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setWindowTitle("Alpaca TradeBot Dashboard")
            self.setMinimumSize(960, 600)
            self.resize(1300, 760)

            # Resize tracking
            self._resize_edge: Optional[str] = None
            self._resize_start_pos: Optional[QPoint] = None
            self._resize_start_geo = None
            self.setMouseTracking(True)

            # Outer layout (margin for drop shadow)
            outer_layout = QVBoxLayout(self)
            outer_layout.setContentsMargins(10, 10, 10, 10)
            outer_layout.setSpacing(0)

            # Main content container (painted background)
            self._container = QWidget()
            self._container.setObjectName("MainContainer")
            self._container.setStyleSheet(
                f"QWidget#MainContainer {{ "
                f"background: {COLORS['bg_primary']}; "
                f"border: 1px solid {COLORS['border']}; "
                f"border-radius: 10px; }}"
            )
            outer_layout.addWidget(self._container)

            # Drop shadow
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(24)
            shadow.setOffset(0, 6)
            shadow.setColor(QColor(0, 0, 0, 100))
            self._container.setGraphicsEffect(shadow)

            # Container layout
            container_layout = QVBoxLayout(self._container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)

            # Title bar
            self._title_bar = TitleBar(self)
            container_layout.addWidget(self._title_bar)

            # Content area
            content = QWidget()
            content.setStyleSheet("background: transparent; border: none;")
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(16, 12, 16, 0)
            content_layout.setSpacing(10)
            container_layout.addWidget(content, stretch=1)

            # -- Metric cards row --
            self._build_metric_cards(content_layout)

            # -- Account info bar --
            self._build_account_bar(content_layout)

            # -- Tabs (positions / prices) --
            self._build_tabs(content_layout)

            # -- Status bar --
            self._build_status_bar(container_layout)

            # Refresh timer
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._refresh)
            self._timer.start(self._refresh_ms)
            self._refresh()

        # ── Build helpers ──────────────────────────────────────────────────

        def _build_metric_cards(self, parent_layout: QVBoxLayout) -> None:
            row = QWidget()
            row.setStyleSheet("background: transparent; border: none;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(10)

            self._card_equity = MetricCard("Equity")
            self._card_unrealized = MetricCard("Unrealized P&L")
            self._card_realized = MetricCard("Realized Today")
            self._card_daily = MetricCard("Daily Change")
            self._card_drawdown = MetricCard("Drawdown")
            self._card_exposure = MetricCard("Gross Exposure")

            for card in (
                self._card_equity,
                self._card_unrealized,
                self._card_realized,
                self._card_daily,
                self._card_drawdown,
                self._card_exposure,
            ):
                row_layout.addWidget(card)

            parent_layout.addWidget(row)

        def _build_account_bar(self, parent_layout: QVBoxLayout) -> None:
            bar = QWidget()
            bar.setStyleSheet(
                f"background: {COLORS['bg_secondary']}; "
                f"border: 1px solid {COLORS['border']}; border-radius: 6px;"
            )
            bar.setFixedHeight(36)
            bar_layout = QHBoxLayout(bar)
            bar_layout.setContentsMargins(14, 0, 14, 0)
            bar_layout.setSpacing(24)

            lbl_style = (
                f"color: {COLORS['text_secondary']}; font-family: {_FONT_UI}; "
                f"font-size: 12px; background: transparent; border: none;"
            )
            val_style = (
                f"color: {COLORS['info']}; font-family: {_FONT_MONO}; "
                f"font-size: 12px; font-weight: 600; background: transparent; border: none;"
            )

            self._acct_cash = QLabel("$0.00")
            self._acct_cash.setStyleSheet(val_style)
            self._acct_bp = QLabel("$0.00")
            self._acct_bp.setStyleSheet(val_style)
            self._acct_pos = QLabel("0 / 0")
            self._acct_pos.setStyleSheet(val_style)

            cash_lbl = QLabel("Cash")
            cash_lbl.setStyleSheet(lbl_style)
            bp_lbl = QLabel("Buying Power")
            bp_lbl.setStyleSheet(lbl_style)
            pos_lbl = QLabel("Positions")
            pos_lbl.setStyleSheet(lbl_style)

            for lbl, val in ((cash_lbl, self._acct_cash), (bp_lbl, self._acct_bp), (pos_lbl, self._acct_pos)):
                bar_layout.addWidget(lbl)
                bar_layout.addWidget(val)

            bar_layout.addStretch()
            parent_layout.addWidget(bar)

        def _build_tabs(self, parent_layout: QVBoxLayout) -> None:
            self._tabs = QTabWidget()
            self._tabs.setStyleSheet(
                "background: transparent; border: none;"
            )

            self._positions_table = self._create_table(
                ["Symbol", "Side", "Qty", "Entry", "Current", "% Chg", "P&L", "Notional", "TP", "SL"]
            )
            self._prices_table = self._create_table(
                ["Symbol", "Last Price", "Bid", "Ask", "Daily % Change"]
            )
            self._tabs.addTab(self._positions_table, "Positions")
            self._tabs.addTab(self._prices_table, "Prices")

            parent_layout.addWidget(self._tabs, stretch=1)

        def _create_table(self, headers: List[str]) -> QTableWidget:
            table = QTableWidget()
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setShowGrid(False)
            table.verticalHeader().setVisible(False)
            table.setFocusPolicy(Qt.NoFocus)
            # Stretch columns proportionally
            header = table.horizontalHeader()
            header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for i in range(len(headers)):
                header.setSectionResizeMode(i, QHeaderView.Stretch)
            return table

        def _build_status_bar(self, parent_layout: QVBoxLayout) -> None:
            bar = QWidget()
            bar.setFixedHeight(32)
            bar.setStyleSheet(
                f"background: {COLORS['bg_secondary']}; "
                f"border-top: 1px solid {COLORS['border']}; "
                f"border-bottom-left-radius: 10px; "
                f"border-bottom-right-radius: 10px;"
            )
            bar_layout = QHBoxLayout(bar)
            bar_layout.setContentsMargins(16, 0, 16, 0)
            bar_layout.setSpacing(16)

            lbl_style = (
                f"color: {COLORS['text_muted']}; font-family: {_FONT_UI}; "
                f"font-size: 11px; background: transparent; border: none;"
            )

            self._live_dot = QLabel("\u25cf")  # filled circle
            self._live_dot.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 12px; "
                f"background: transparent; border: none;"
            )
            self._market_label = QLabel("Market Closed")
            self._market_label.setStyleSheet(lbl_style)
            self._cycle_label = QLabel("Cycle #0")
            self._cycle_label.setStyleSheet(lbl_style)
            self._lastrun_label = QLabel("Last Run: \u2014")
            self._lastrun_label.setStyleSheet(lbl_style)
            self._state_badge = StatusBadge()

            bar_layout.addWidget(self._live_dot)
            bar_layout.addWidget(self._market_label)
            bar_layout.addWidget(self._cycle_label)
            bar_layout.addWidget(self._lastrun_label)
            bar_layout.addStretch()
            bar_layout.addWidget(self._state_badge)

            parent_layout.addWidget(bar)

        # ── Refresh logic ──────────────────────────────────────────────────

        def _refresh(self) -> None:
            state = load_state()

            # Metric cards
            self._card_equity.set_value(f"${state.equity:,.2f}")
            pl_color = COLORS['success'] if state.unrealized_pl >= 0 else COLORS['danger']
            self._card_unrealized.set_value(f"${state.unrealized_pl:+,.2f}", pl_color)
            rpl_color = COLORS['success'] if state.realized_pl_today >= 0 else COLORS['danger']
            self._card_realized.set_value(f"${state.realized_pl_today:+,.2f}", rpl_color)
            dl_color = COLORS['success'] if state.daily_loss_pct >= 0 else COLORS['danger']
            self._card_daily.set_value(f"{state.daily_loss_pct * 100:+.3f}%", dl_color)
            dd_color = COLORS['danger'] if state.drawdown_pct < 0 else COLORS['text_primary']
            self._card_drawdown.set_value(f"{state.drawdown_pct * 100:.3f}%", dd_color)
            self._card_exposure.set_value(f"${state.gross_exposure:,.2f}")

            # Account bar
            self._acct_cash.setText(f"${state.cash:,.2f}")
            self._acct_bp.setText(f"${state.buying_power:,.2f}")
            self._acct_pos.setText(f"{state.num_positions} / {state.max_positions}")

            # Status bar
            self._pulse = not self._pulse
            if state.market_open:
                dot_char = "\u25cf" if self._pulse else "\u25cb"
                self._live_dot.setText(dot_char)
                self._live_dot.setStyleSheet(
                    f"color: {COLORS['success']}; font-size: 12px; "
                    f"background: transparent; border: none;"
                )
                self._market_label.setText("Market Open")
                self._market_label.setStyleSheet(
                    f"color: {COLORS['success']}; font-family: {_FONT_UI}; "
                    f"font-size: 11px; font-weight: 600; "
                    f"background: transparent; border: none;"
                )
            else:
                self._live_dot.setText("\u25cb")  # hollow circle
                self._live_dot.setStyleSheet(
                    f"color: {COLORS['text_muted']}; font-size: 12px; "
                    f"background: transparent; border: none;"
                )
                self._market_label.setText("Market Closed")
                self._market_label.setStyleSheet(
                    f"color: {COLORS['text_muted']}; font-family: {_FONT_UI}; "
                    f"font-size: 11px; background: transparent; border: none;"
                )

            self._cycle_label.setText(f"Cycle #{state.cycle_count:,}")
            if state.last_run_ts > 0:
                elapsed = time.time() - state.last_run_ts
                self._lastrun_label.setText(f"Last Run: {elapsed:.1f}s ago")

            self._state_badge.set_state(state.bot_state)

            # Tables
            self._update_positions_table(state.positions)
            self._update_prices_table(state.prices)

        # ── Table renderers ────────────────────────────────────────────────

        def _update_positions_table(self, positions: List[PositionRow]) -> None:
            table = self._positions_table
            table.setRowCount(0)

            if not positions:
                table.setRowCount(1)
                item = QTableWidgetItem("\u2014  No open positions")
                item.setForeground(QColor(COLORS['text_muted']))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(0, 0, item)
                for col in range(1, table.columnCount()):
                    table.setItem(0, col, QTableWidgetItem(""))
                return

            mono = QFont("Consolas", 11)
            mono_bold = QFont("Consolas", 11, QFont.Bold)

            for row_data in positions:
                row = table.rowCount()
                table.insertRow(row)

                # Symbol
                sym_item = QTableWidgetItem(row_data.symbol)
                sym_item.setFont(mono_bold)
                sym_item.setForeground(QColor(COLORS['text_primary']))
                table.setItem(row, 0, sym_item)

                # Side
                side_upper = row_data.side.upper()
                side_item = QTableWidgetItem(side_upper)
                side_item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                side_color = COLORS['success'] if side_upper == "LONG" else COLORS['danger']
                side_item.setForeground(QColor(side_color))
                table.setItem(row, 1, side_item)

                # Qty
                qty_item = QTableWidgetItem(f"{abs(row_data.qty):,.0f}")
                qty_item.setFont(mono)
                qty_item.setForeground(QColor(COLORS['text_secondary']))
                table.setItem(row, 2, qty_item)

                # Entry Price
                entry_item = QTableWidgetItem(f"${row_data.entry_price:,.2f}")
                entry_item.setFont(mono)
                entry_item.setForeground(QColor(COLORS['text_secondary']))
                table.setItem(row, 3, entry_item)

                # Current Price
                cur_item = QTableWidgetItem(f"${row_data.current_price:,.2f}")
                cur_item.setFont(mono)
                cur_item.setForeground(QColor(COLORS['info']))
                table.setItem(row, 4, cur_item)

                # % Change
                pct_str = f"{row_data.pct_change * 100:+.2f}%"
                pct_item = QTableWidgetItem(pct_str)
                pct_item.setFont(mono)
                pct_color = COLORS['success'] if row_data.pct_change >= 0 else COLORS['danger']
                pct_item.setForeground(QColor(pct_color))
                table.setItem(row, 5, pct_item)

                # P&L
                pnl_str = f"${row_data.pnl:+,.2f}"
                pnl_item = QTableWidgetItem(pnl_str)
                pnl_item.setFont(mono_bold)
                pnl_color = COLORS['success'] if row_data.pnl >= 0 else COLORS['danger']
                pnl_item.setForeground(QColor(pnl_color))
                table.setItem(row, 6, pnl_item)

                # Notional
                not_item = QTableWidgetItem(f"${row_data.notional:,.2f}")
                not_item.setFont(mono)
                not_item.setForeground(QColor(COLORS['text_secondary']))
                table.setItem(row, 7, not_item)

                # Take Profit
                tp_text = f"${row_data.take_profit:,.2f}" if row_data.take_profit > 0 else "\u2014"
                tp_item = QTableWidgetItem(tp_text)
                tp_item.setFont(mono)
                tp_item.setForeground(QColor(COLORS['warning']))
                table.setItem(row, 8, tp_item)

                # Stop Loss
                sl_text = f"${row_data.stop_loss:,.2f}" if row_data.stop_loss > 0 else "\u2014"
                sl_item = QTableWidgetItem(sl_text)
                sl_item.setFont(mono)
                sl_item.setForeground(QColor(COLORS['warning']))
                table.setItem(row, 9, sl_item)

        def _update_prices_table(self, prices: List[PriceRow]) -> None:
            table = self._prices_table
            table.setRowCount(0)

            if not prices:
                table.setRowCount(1)
                item = QTableWidgetItem("\u2014  No price data")
                item.setForeground(QColor(COLORS['text_muted']))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(0, 0, item)
                for col in range(1, table.columnCount()):
                    table.setItem(0, col, QTableWidgetItem(""))
                return

            mono = QFont("Consolas", 11)
            mono_bold = QFont("Consolas", 11, QFont.Bold)

            for row_data in prices:
                row = table.rowCount()
                table.insertRow(row)

                # Symbol
                sym_item = QTableWidgetItem(row_data.symbol)
                sym_item.setFont(mono_bold)
                sym_item.setForeground(QColor(COLORS['text_primary']))
                table.setItem(row, 0, sym_item)

                # Last Price
                last_item = QTableWidgetItem(f"${row_data.last_price:,.2f}")
                last_item.setFont(mono)
                last_item.setForeground(QColor(COLORS['info']))
                table.setItem(row, 1, last_item)

                # Bid
                bid_item = QTableWidgetItem(f"${row_data.bid:,.2f}")
                bid_item.setFont(mono)
                bid_item.setForeground(QColor(COLORS['text_secondary']))
                table.setItem(row, 2, bid_item)

                # Ask
                ask_item = QTableWidgetItem(f"${row_data.ask:,.2f}")
                ask_item.setFont(mono)
                ask_item.setForeground(QColor(COLORS['text_secondary']))
                table.setItem(row, 3, ask_item)

                # Daily % Change
                daily_str = f"{row_data.daily_pct_change * 100:+.2f}%"
                daily_item = QTableWidgetItem(daily_str)
                daily_item.setFont(mono)
                daily_color = COLORS['success'] if row_data.daily_pct_change >= 0 else COLORS['danger']
                daily_item.setForeground(QColor(daily_color))
                table.setItem(row, 4, daily_item)

        # ── Edge resize handling ───────────────────────────────────────────

        def _edge_at(self, pos: QPoint) -> Optional[str]:
            """Return resize edge string or None if not on an edge."""
            r = self.rect()
            e = self._EDGE_SIZE
            x, y = pos.x(), pos.y()
            on_left = x <= e
            on_right = x >= r.width() - e
            on_top = y <= e
            on_bottom = y >= r.height() - e
            if on_top and on_left:
                return "tl"
            if on_top and on_right:
                return "tr"
            if on_bottom and on_left:
                return "bl"
            if on_bottom and on_right:
                return "br"
            if on_left:
                return "l"
            if on_right:
                return "r"
            if on_top:
                return "t"
            if on_bottom:
                return "b"
            return None

        def _cursor_for_edge(self, edge: Optional[str]):
            cursors = {
                "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
                "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
                "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
            }
            return cursors.get(edge, Qt.ArrowCursor)

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.LeftButton:
                edge = self._edge_at(event.pos())
                if edge:
                    self._resize_edge = edge
                    self._resize_start_pos = event.globalPos()
                    self._resize_start_geo = self.geometry()
                    event.accept()
                    return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event) -> None:
            if self._resize_edge and self._resize_start_pos:
                delta = event.globalPos() - self._resize_start_pos
                geo = self._resize_start_geo
                minw, minh = self.minimumWidth(), self.minimumHeight()
                new_geo = geo.__class__(geo)

                if "r" in self._resize_edge:
                    new_geo.setWidth(max(minw, geo.width() + delta.x()))
                if "b" in self._resize_edge:
                    new_geo.setHeight(max(minh, geo.height() + delta.y()))
                if "l" in self._resize_edge:
                    new_w = max(minw, geo.width() - delta.x())
                    new_geo.setLeft(geo.right() - new_w)
                if "t" in self._resize_edge:
                    new_h = max(minh, geo.height() - delta.y())
                    new_geo.setTop(geo.bottom() - new_h)

                self.setGeometry(new_geo)
                event.accept()
                return

            edge = self._edge_at(event.pos())
            self.setCursor(self._cursor_for_edge(edge))
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event) -> None:
            self._resize_edge = None
            self._resize_start_pos = None
            self._resize_start_geo = None
            super().mouseReleaseEvent(event)


# ── Public helpers called by main.py ───────────────────────────────────────────

def build_position_rows(
    positions: dict,          # Dict[str, PositionInfo] from PositionManager
    open_orders: list,        # list of Alpaca order objects (kept for backward compat)
) -> List[PositionRow]:
    """Convert a PositionManager positions dict into UI PositionRow objects.

    TP and SL are read directly from PositionInfo.take_profit_price and
    PositionInfo.stop_price, which PositionManager already populates from
    the broker's open-order list.  This avoids a redundant order re-parse
    and ensures the dashboard stays in sync with the console log.
    """
    rows: List[PositionRow] = []

    for sym, pos in positions.items():
        entry_price = float(getattr(pos, "avg_entry_price", 0.0) or 0.0)
        current_price = float(pos.market_price)
        if entry_price > 0:
            pct_change = (current_price - entry_price) / entry_price
        else:
            pct_change = 0.0
        qty = float(pos.qty)
        pnl = (current_price - entry_price) * qty if entry_price > 0 else 0.0
        rows.append(
            PositionRow(
                symbol=sym,
                current_price=current_price,
                entry_price=entry_price,
                pct_change=pct_change,
                pnl=pnl,
                take_profit=float(pos.take_profit_price) if pos.take_profit_price is not None else 0.0,
                stop_loss=float(pos.stop_price) if pos.stop_price is not None else 0.0,
                side=str(pos.side),
                qty=qty,
                notional=float(pos.notional),
            )
        )
    return rows


def build_price_rows(
    instruments: dict,   # Dict[str, InstrumentMeta]
    adapter,             # AlpacaAdapter instance
) -> List[PriceRow]:
    """Fetch live quotes for all whitelisted symbols and return PriceRow list."""
    rows: List[PriceRow] = []
    for sym in instruments:
        try:
            last_price = adapter.get_last_quote(sym)
        except Exception:
            last_price = 0.0

        bid = last_price
        ask = last_price
        try:
            q = adapter.rest.get_latest_quote(sym)
            bid = float(getattr(q, "bp", last_price) or last_price)
            ask = float(getattr(q, "ap", last_price) or last_price)
        except Exception:
            pass

        daily_pct = 0.0
        try:
            bars = adapter.get_recent_bars(sym, timeframe="1Day", lookback_bars=2)
            if len(bars) >= 2:
                prev_close = float(bars[-2].c)
                if prev_close > 0:
                    daily_pct = (last_price - prev_close) / prev_close
        except Exception:
            pass

        rows.append(
            PriceRow(
                symbol=sym,
                last_price=last_price,
                bid=bid,
                ask=ask,
                daily_pct_change=daily_pct,
            )
        )
    return rows


# ── GUI process singleton ──────────────────────────────────────────────────────
_gui_process: Optional[multiprocessing.Process] = None


def _run_gui_process(refresh_seconds: float) -> None:
    """QApplication event loop entry point for the GUI process."""
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)
    window = TradeBotMainWindow(refresh_seconds=refresh_seconds)
    window.show()
    sys.exit(app.exec_())


def launch_dashboard(refresh_seconds: float = 5.0) -> None:
    """Launch the PyQt5 desktop GUI in a separate OS process.

    Checks three conditions before spawning:
      1. dev_mode == False
      2. PyQt5 is installed (_PYQT5_AVAILABLE == True)
      3. Not already running (singleton guard)
    """
    global _gui_process

    if dev_mode:
        logger.info("Dashboard launch skipped: dev_mode=True (headless operation)")
        return

    if not _PYQT5_AVAILABLE:
        logger.warning(
            "Dashboard launch skipped: PyQt5 not installed. "
            "Install it with: pip install PyQt5"
        )
        return

    if _gui_process is not None and _gui_process.is_alive():
        logger.warning("Dashboard already running, skipping duplicate launch")
        return

    _gui_process = multiprocessing.Process(
        target=_run_gui_process,
        args=(refresh_seconds,),
        daemon=True,
        name="pyqt5-dashboard",
    )
    _gui_process.start()
    logger.info("PyQt5 dashboard launched in separate desktop window.")
