# CHANGES:
# FIX 1: Replace threading with multiprocessing to comply with PyQt5's
#   requirement that QApplication must be created in the main thread of a process.
#   The GUI now spawns as a separate OS process via multiprocessing.Process.
#
# FIX 2: Remove invalid QTabWidget.setShortcut() calls. QTabWidget doesn't have
#   a setShortcut method. Keyboard shortcuts (Ctrl+P, Ctrl+O) removed for now
#   (can be re-added later using QShortcut if needed, but not blocking launch).
#
# FIX 3: _run_gui() now creates a fresh QApplication(sys.argv) unconditionally
#   since it runs in a separate process with its own main thread.

from __future__ import annotations

import logging
import multiprocessing
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("tradebot")

# ── dev_mode flag ──────────────────────────────────────────────────────────────
# Set to True to run headless (raw logging, no GUI).
# Set to False (default) to launch the full PyQt5 desktop window.
dev_mode: bool = False

# ── PyQt5 availability check ───────────────────────────────────────────────────
try:
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QPalette
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    _PYQT5_AVAILABLE = True
except ImportError:
    _PYQT5_AVAILABLE = False

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
    last_run_ts: float = 0.0          # time.time() of last cycle start
    bot_state: str = "IDLE"           # "SCANNING" | "EXECUTING" | "IDLE" | "HALTED"
    market_open: bool = False
    positions: List[PositionRow] = field(default_factory=list)
    prices: List[PriceRow] = field(default_factory=list)

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


# ── Singleton state shared between main loop and GUI ───────────────────────────
dashboard_state: DashboardState = DashboardState()

# ── Pickle-based IPC file path ─────────────────────────────────────────────────
STATE_FILE = Path("dashboard_state.pkl")


def persist_state(state: DashboardState) -> None:
    """Pickle dashboard_state to disk for cross-process IPC."""
    try:
        with STATE_FILE.open("wb") as f:
            pickle.dump(state, f)
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


# ── PyQt5 GUI implementation ───────────────────────────────────────────────────

if _PYQT5_AVAILABLE:

    class TradeBotMainWindow(QMainWindow):
        """PyQt5 main window for the Alpaca trading bot dashboard."""

        def __init__(self, refresh_seconds: float = 5.0) -> None:
            super().__init__()
            self._refresh_seconds = int(refresh_seconds * 1000)  # ms
            self._pulse = False
            self.setWindowTitle("Alpaca TradeBot Dashboard")
            self.setGeometry(100, 100, 1200, 700)

            # Apply dark theme
            self._apply_dark_theme()

            # Build UI
            self._build_menu()
            self._build_central_widget()
            self._build_status_bar()

            # Start refresh timer
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._refresh)
            self._timer.start(self._refresh_seconds)

            # Initial load
            self._refresh()

        def _apply_dark_theme(self) -> None:
            """Apply custom dark palette to the application."""
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor("#0d0d0d"))
            palette.setColor(QPalette.WindowText, QColor("#e0e0e0"))
            palette.setColor(QPalette.Base, QColor("#111827"))
            palette.setColor(QPalette.AlternateBase, QColor("#1f2937"))
            palette.setColor(QPalette.Text, QColor("#e0e0e0"))
            palette.setColor(QPalette.Button, QColor("#1f2937"))
            palette.setColor(QPalette.ButtonText, QColor("#9ca3af"))
            palette.setColor(QPalette.Highlight, QColor("#22d3ee"))
            palette.setColor(QPalette.HighlightedText, QColor("#0d0d0d"))
            QApplication.instance().setPalette(palette)

        def _build_menu(self) -> None:
            """Build the top menu bar."""
            menubar = self.menuBar()
            file_menu = menubar.addMenu("File")

            settings_action = file_menu.addAction("Settings")
            settings_action.triggered.connect(self._show_settings_todo)

            console_action = file_menu.addAction("Console")
            console_action.triggered.connect(self._show_console_todo)

            file_menu.addSeparator()

            exit_action = file_menu.addAction("Exit")
            exit_action.triggered.connect(self._exit_app)

        def _build_central_widget(self) -> None:
            """Build the main content area."""
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)

            # Top panel: account info
            account_panel = QWidget()
            account_layout = QHBoxLayout(account_panel)
            account_layout.setContentsMargins(8, 8, 8, 8)
            self._cash_label = QLabel("Cash: $0.00")
            self._cash_label.setFont(QFont("Courier New", 11, QFont.Bold))
            self._cash_label.setStyleSheet("color: #22d3ee;")
            self._bp_label = QLabel("Buying Power: $0.00")
            self._bp_label.setFont(QFont("Courier New", 11, QFont.Bold))
            self._bp_label.setStyleSheet("color: #22d3ee;")
            account_layout.addWidget(self._cash_label)
            account_layout.addWidget(self._bp_label)
            account_layout.addStretch()
            account_panel.setStyleSheet(
                "background-color: #111827; border: 1px solid #1f2937; border-radius: 4px;"
            )
            layout.addWidget(account_panel)

            # Tabbed content: Positions and Prices
            self._tabs = QTabWidget()
            self._tabs.setStyleSheet(
                "QTabWidget::pane { border: 1px solid #1f2937; background: #111827; }"
                "QTabBar::tab { background: #1f2937; color: #9ca3af; padding: 6px 12px; }"
                "QTabBar::tab:selected { background: #111827; color: #22d3ee; }"
            )

            # Positions tab
            self._positions_table = self._create_table(
                ["Symbol", "Current Price", "Entry Price", "% Change", "P&L ($)", "Take Profit", "Stop Loss"]
            )
            self._tabs.addTab(self._positions_table, "Positions")

            # Prices tab
            self._prices_table = self._create_table(
                ["Symbol", "Last Price", "Bid", "Ask", "Daily % Change"]
            )
            self._tabs.addTab(self._prices_table, "Prices")

            # FIX 2: Removed invalid QTabWidget.setShortcut() calls.
            # Keyboard shortcuts can be re-added later using QShortcut if needed.

            layout.addWidget(self._tabs)

        def _create_table(self, headers: List[str]) -> QTableWidget:
            """Create a styled QTableWidget with the given headers."""
            table = QTableWidget()
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.horizontalHeader().setStretchLastSection(True)
            table.setStyleSheet(
                "QTableWidget { background-color: #111827; color: #e0e0e0; gridline-color: #1f2937; }"
                "QHeaderView::section { background-color: #1f2937; color: #9ca3af; padding: 4px; border: none; }"
            )
            return table

        def _build_status_bar(self) -> None:
            """Build the bottom status bar."""
            status = self.statusBar()
            status.setStyleSheet("background-color: #111827; color: #6b7280;")

            # Live indicator
            self._live_dot = QLabel("●")
            self._live_dot.setStyleSheet("color: #4b5563; font-size: 14px;")
            status.addWidget(self._live_dot)

            # Cycle count
            self._cycle_label = QLabel("Bot Cycle: #0")
            status.addWidget(self._cycle_label)

            # Last run
            self._lastrun_label = QLabel("Last Run: —")
            status.addWidget(self._lastrun_label)

            # Bot state badge
            self._state_badge = QLabel("IDLE")
            self._state_badge.setStyleSheet(
                "background-color: #1f2937; color: #6b7280; padding: 2px 8px; border-radius: 3px;"
            )
            status.addPermanentWidget(self._state_badge)

        def _refresh(self) -> None:
            """Load state from pickle file and update all widgets."""
            state = load_state()

            # Update account info
            self._cash_label.setText(f"Cash: ${state.cash:,.2f}")
            self._bp_label.setText(f"Buying Power: ${state.buying_power:,.2f}")

            # Update status bar
            self._pulse = not self._pulse
            if state.market_open:
                self._live_dot.setText("●" if self._pulse else "○")
                self._live_dot.setStyleSheet("color: #22c55e; font-size: 14px;")
            else:
                self._live_dot.setText("○")
                self._live_dot.setStyleSheet("color: #4b5563; font-size: 14px;")

            self._cycle_label.setText(f"Bot Cycle: #{state.cycle_count:,}")

            if state.last_run_ts > 0:
                elapsed = time.time() - state.last_run_ts
                self._lastrun_label.setText(f"Last Run: {elapsed:.1f}s ago")

            state_upper = state.bot_state.upper()
            self._state_badge.setText(state_upper)
            if state_upper == "SCANNING":
                self._state_badge.setStyleSheet(
                    "background-color: #854d0e; color: #fef08a; padding: 2px 8px; border-radius: 3px;"
                )
            elif state_upper == "EXECUTING":
                self._state_badge.setStyleSheet(
                    "background-color: #14532d; color: #86efac; padding: 2px 8px; border-radius: 3px;"
                )
            elif state_upper == "HALTED":
                self._state_badge.setStyleSheet(
                    "background-color: #7f1d1d; color: #fca5a5; padding: 2px 8px; border-radius: 3px;"
                )
            else:
                self._state_badge.setStyleSheet(
                    "background-color: #1f2937; color: #6b7280; padding: 2px 8px; border-radius: 3px;"
                )

            # Update positions table
            self._update_positions_table(state.positions)

            # Update prices table
            self._update_prices_table(state.prices)

        def _update_positions_table(self, positions: List[PositionRow]) -> None:
            """Refresh the positions table with current data."""
            table = self._positions_table
            table.setRowCount(0)

            if not positions:
                table.setRowCount(1)
                item = QTableWidgetItem("— No open positions —")
                item.setForeground(QColor("#4b5563"))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(0, 0, item)
                for col in range(1, table.columnCount()):
                    table.setItem(0, col, QTableWidgetItem(""))
                return

            mono_font = QFont("Courier New", 10)
            for row_data in positions:
                row = table.rowCount()
                table.insertRow(row)

                # Symbol
                sym_item = QTableWidgetItem(row_data.symbol)
                sym_item.setFont(QFont("Courier New", 10, QFont.Bold))
                table.setItem(row, 0, sym_item)

                # Current Price
                cur_item = QTableWidgetItem(f"${row_data.current_price:,.2f}")
                cur_item.setFont(mono_font)
                cur_item.setForeground(QColor("#22d3ee"))
                table.setItem(row, 1, cur_item)

                # Entry Price
                entry_item = QTableWidgetItem(f"${row_data.entry_price:,.2f}")
                entry_item.setFont(mono_font)
                table.setItem(row, 2, entry_item)

                # % Change
                pct_str = f"{row_data.pct_change * 100:+.2f}%"
                pct_item = QTableWidgetItem(pct_str)
                pct_item.setFont(mono_font)
                pct_item.setForeground(QColor("#22c55e" if row_data.pct_change >= 0 else "#ef4444"))
                table.setItem(row, 3, pct_item)

                # P&L
                pnl_str = f"${row_data.pnl:+,.2f}"
                pnl_item = QTableWidgetItem(pnl_str)
                pnl_item.setFont(mono_font)
                pnl_item.setForeground(QColor("#22c55e" if row_data.pnl >= 0 else "#ef4444"))
                table.setItem(row, 4, pnl_item)

                # Take Profit
                tp_item = QTableWidgetItem(f"${row_data.take_profit:,.2f}")
                tp_item.setFont(mono_font)
                tp_item.setForeground(QColor("#f59e0b"))
                table.setItem(row, 5, tp_item)

                # Stop Loss
                sl_item = QTableWidgetItem(f"${row_data.stop_loss:,.2f}")
                sl_item.setFont(mono_font)
                sl_item.setForeground(QColor("#f59e0b"))
                table.setItem(row, 6, sl_item)

        def _update_prices_table(self, prices: List[PriceRow]) -> None:
            """Refresh the prices table with current data."""
            table = self._prices_table
            table.setRowCount(0)

            if not prices:
                table.setRowCount(1)
                item = QTableWidgetItem("— No price data —")
                item.setForeground(QColor("#4b5563"))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(0, 0, item)
                for col in range(1, table.columnCount()):
                    table.setItem(0, col, QTableWidgetItem(""))
                return

            mono_font = QFont("Courier New", 10)
            for row_data in prices:
                row = table.rowCount()
                table.insertRow(row)

                # Symbol
                sym_item = QTableWidgetItem(row_data.symbol)
                sym_item.setFont(QFont("Courier New", 10, QFont.Bold))
                table.setItem(row, 0, sym_item)

                # Last Price
                last_item = QTableWidgetItem(f"${row_data.last_price:,.2f}")
                last_item.setFont(mono_font)
                last_item.setForeground(QColor("#22d3ee"))
                table.setItem(row, 1, last_item)

                # Bid
                bid_item = QTableWidgetItem(f"${row_data.bid:,.2f}")
                bid_item.setFont(mono_font)
                table.setItem(row, 2, bid_item)

                # Ask
                ask_item = QTableWidgetItem(f"${row_data.ask:,.2f}")
                ask_item.setFont(mono_font)
                table.setItem(row, 3, ask_item)

                # Daily % Change
                daily_str = f"{row_data.daily_pct_change * 100:+.2f}%"
                daily_item = QTableWidgetItem(daily_str)
                daily_item.setFont(mono_font)
                daily_item.setForeground(QColor("#22c55e" if row_data.daily_pct_change >= 0 else "#ef4444"))
                table.setItem(row, 4, daily_item)

        def _show_settings_todo(self) -> None:
            """Show TODO message for Settings menu item."""
            QMessageBox.information(self, "TODO", "Settings panel not yet implemented.")

        def _show_console_todo(self) -> None:
            """Show TODO message for Console menu item."""
            QMessageBox.information(self, "TODO", "Console overlay not yet implemented.")

        def _exit_app(self) -> None:
            """Exit the application (both GUI and main process)."""
            QApplication.quit()
            import os
            os._exit(0)


# ── Public helpers called by main.py ───────────────────────────────────────────

def build_position_rows(
    positions: dict,          # Dict[str, PositionInfo] from PositionManager
    open_orders: list,        # list of Alpaca order objects (used for TP/SL lookup)
) -> List[PositionRow]:
    """Convert a PositionManager positions dict into UI PositionRow objects.

    TP and SL are extracted by matching the symbol against the open_orders list.
    If no matching bracket leg is found the values default to 0.0 — the UI
    renders them as $0.00 rather than crashing.
    """
    rows: List[PositionRow] = []
    # Build a quick lookup: symbol -> (tp_price, sl_price) from open orders
    tp_map: Dict[str, float] = {}
    sl_map: Dict[str, float] = {}
    for o in open_orders:
        sym = getattr(o, "symbol", None)
        if sym is None:
            continue
        order_type = str(getattr(o, "type", "")).lower()
        if order_type == "limit":
            lp = getattr(o, "limit_price", None)
            if lp is not None:
                tp_map[sym] = float(lp)
        if order_type in ("stop", "trailing_stop", "stop_limit"):
            sp = getattr(o, "stop_price", None)
            if sp is not None:
                sl_map[sym] = float(sp)

    for sym, pos in positions.items():
        # Extract entry price from avg_entry_price if available, otherwise 0.0
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
                take_profit=tp_map.get(sym, 0.0),
                stop_loss=sl_map.get(sym, 0.0),
            )
        )
    return rows


def build_price_rows(
    instruments: dict,   # Dict[str, InstrumentMeta]
    adapter,             # AlpacaAdapter instance
) -> List[PriceRow]:
    """Fetch live quotes for all whitelisted symbols and return PriceRow list.

    Uses adapter.get_last_quote() for the last trade price.  Bid/ask are
    fetched via adapter.rest.get_latest_quote() when available; on failure
    they fall back to last_price so the table is never empty.

    Daily % change is computed as (last - prev_close) / prev_close using the
    first bar of the current session fetched via get_recent_bars().  Falls back
    to 0.0 on any error so the UI degrades gracefully.
    """
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
# FIX 1: Replace threading.Thread with multiprocessing.Process so the GUI
# runs in its own process with its own main thread (PyQt5 requirement).
_gui_process: Optional[multiprocessing.Process] = None


def _run_gui_process(refresh_seconds: float) -> None:
    """QApplication event loop entry point for the GUI process.
    
    FIX 3: Creates a fresh QApplication since this runs in a separate process.
    """
    app = QApplication(sys.argv)
    window = TradeBotMainWindow(refresh_seconds=refresh_seconds)
    window.show()
    sys.exit(app.exec_())


def launch_dashboard(refresh_seconds: float = 5.0) -> None:
    """Launch the PyQt5 desktop GUI in a separate OS process.

    Checks three conditions before spawning:
      1. dev_mode == False
      2. PyQt5 is installed (_PYQT5_AVAILABLE == True)
      3. Not already running (singleton guard)

    When any condition fails, logs a message and returns silently.
    The bot never crashes when PyQt5 is missing.
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

    # FIX 1: Spawn a separate process instead of a thread.
    _gui_process = multiprocessing.Process(
        target=_run_gui_process,
        args=(refresh_seconds,),
        daemon=True,
        name="pyqt5-dashboard",
    )
    _gui_process.start()
    logger.info("PyQt5 dashboard launched in separate desktop window.")
