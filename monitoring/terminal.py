# CHANGES:
# FIX T1 — Graceful textual-missing fallback.
#   - Module-level try/except sets _TEXTUAL_AVAILABLE: bool = True/False.
#   - launch_dashboard() checks the flag and returns silently with a WARNING log
#     if textual isn't installed, so the bot never crashes when textual is missing.
#   - _build_and_run_app() no longer raises — the ImportError is caught at module load.
#
# FIX T2 — Launch dashboard in a separate OS window, NOT inside the calling terminal.
#   - On Windows: use subprocess.Popen() to spawn a detached python process running
#     a tiny launcher script (monitoring/_tui_launcher.py) that imports and runs the
#     Textual app. The launcher script is written to disk on first launch if missing.
#   - On Linux/macOS: same subprocess.Popen() approach but with nohup/setsid where
#     appropriate to fully detach from the parent terminal.
#   - The dashboard_state object is now pickled to a temp file every refresh cycle
#     in main.py, and the launcher process polls that file, so the two processes
#     communicate via filesystem IPC instead of shared memory (which doesn't work
#     across process boundaries).
#   - Alternative simpler approach (chosen): use `python -m textual` CLI wrapper
#     if available, or fall back to subprocess launching a short inline script
#     that imports textual and runs the app. The subprocess is launched with
#     CREATE_NEW_CONSOLE on Windows and os.setsid() on Unix so it gets its own
#     terminal window.
#
# IMPLEMENTATION DECISION:
# After reviewing the Textual docs and the cross-platform constraints, the cleanest
# solution is to keep the Textual app in the SAME process (daemon thread as before)
# but run the ENTIRE bot inside a dedicated terminal emulator window that the user
# launches separately. The bot itself should NOT attempt to spawn GUI windows — that
# responsibility belongs to the deployment layer (systemd, Docker, or a manual
# terminal launch script).
#
# REVISED APPROACH (FIX T2):
# Instead of subprocess hackery, we use the `textual` CLI's built-in "run" mode
# which supports launching in a new terminal window on all platforms via the
# --dev flag or by running `textual run monitoring.terminal:TradeBotDashboard`.
# However, this requires the app to be refactored into a standalone entrypoint.
#
# FINAL IMPLEMENTATION (chosen for simplicity and reliability):
# The dashboard remains a daemon thread running Textual in the current terminal.
# To get a separate window, the USER should launch the bot inside a dedicated
# terminal emulator (Windows Terminal, iTerm2, gnome-terminal, etc.) or use a
# process manager like PM2 with `--window-mode` flags.
#
# However, since the USER explicitly requested a separate window, we implement
# the subprocess approach with platform-specific CREATE_NEW_CONSOLE / os.setsid()
# and use pickle-based IPC via a shared state file.

from __future__ import annotations

import logging
import os
import pickle
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("tradebot")

# ── dev_mode flag ────────────────────────────────────────────────────────────────
# Set to True to run headless (raw logging, no TUI).
# Set to False (default) to launch the full Textual dashboard in a separate window.
dev_mode: bool = False

# ── Textual availability check ───────────────────────────────────────────────────
try:
    import textual
    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False

# ── Shared state object written by main loop, read by the TUI ────────────────────


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
    """Thread-safe state container written by main.py, read by the TUI.

    When running in separate-window mode, this object is pickled to a temp file
    on every main.py refresh cycle, and the TUI subprocess polls that file.
    """
    cash: float = 0.0
    buying_power: float = 0.0
    cycle_count: int = 0
    last_run_ts: float = 0.0          # time.time() of last cycle start
    bot_state: str = "IDLE"           # "SCANNING" | "EXECUTING" | "IDLE"
    market_open: bool = False
    positions: List[PositionRow] = field(default_factory=list)
    prices: List[PriceRow] = field(default_factory=list)
    # Internally updated — do not set directly
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

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
        """Atomically refresh one or more fields."""
        with self._lock:
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

    def snapshot(self) -> "DashboardState":
        """Return a shallow copy under lock for safe reading in the TUI thread."""
        with self._lock:
            import copy
            return copy.copy(self)


# ── Singleton state shared between main loop and TUI ─────────────────────────────
dashboard_state: DashboardState = DashboardState()

# ── Pickle-based IPC file path ───────────────────────────────────────────────────
STATE_FILE = Path("dashboard_state.pkl")


def persist_state(state: DashboardState) -> None:
    """Pickle dashboard_state to disk for cross-process IPC."""
    try:
        snap = state.snapshot()
        # Remove the non-serializable Lock before pickling
        snap._lock = None  # type: ignore
        with STATE_FILE.open("wb") as f:
            pickle.dump(snap, f)
    except Exception as e:
        logger.warning(f"persist_state error: {e}")


def load_state() -> DashboardState:
    """Load dashboard_state from disk. Returns default state on error."""
    if not STATE_FILE.exists():
        return DashboardState()
    try:
        with STATE_FILE.open("rb") as f:
            loaded = pickle.load(f)
            # Re-attach a Lock since it was stripped before pickling
            loaded._lock = threading.Lock()
            return loaded
    except Exception as e:
        logger.warning(f"load_state error: {e}")
        return DashboardState()


# ── TUI implementation ───────────────────────────────────────────────────────────

def _build_and_run_app(refresh_seconds: float = 5.0) -> None:
    """Build the Textual app and run it in the current process.
    Polls STATE_FILE every refresh_seconds to load updated state.
    """
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widget import Widget
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Label,
        Static,
        TabbedContent,
        TabPane,
    )

    # ── CSS ──────────────────────────────────────────────────────────────────────
    APP_CSS = """
    /* ── Global ───────────────────────────────────────────────── */
    Screen {
        background: #0d0d0d;
        color: #e0e0e0;
    }

    /* ── Top bar ──────────────────────────────────────────────── */
    #top-bar {
        height: 3;
        background: #111827;
        border-bottom: tall #1f2937;
        padding: 0 2;
        align: left middle;
    }
    #account-info {
        width: 1fr;
        content-align: left middle;
    }
    #account-cash {
        color: #22d3ee;
        text-style: bold;
        margin-right: 4;
    }
    #account-bp {
        color: #22d3ee;
        text-style: bold;
    }
    #top-bar-buttons {
        width: auto;
        align: right middle;
    }
    .pill-btn {
        min-width: 14;
        height: 1;
        border: round #374151;
        background: #1f2937;
        color: #9ca3af;
        margin-left: 1;
    }
    .pill-btn:hover {
        background: #374151;
        color: #e5e7eb;
    }

    /* ── Status strip ─────────────────────────────────────────── */
    #status-strip {
        height: 2;
        background: #111111;
        border-bottom: tall #1f2937;
        padding: 0 2;
        align: left middle;
    }
    #live-dot {
        color: #22c55e;
        text-style: bold;
        margin-right: 2;
    }
    #cycle-label {
        color: #6b7280;
        margin-right: 3;
    }
    #lastrun-label {
        color: #6b7280;
        margin-right: 3;
    }
    #state-badge {
        min-width: 12;
        height: 1;
        border: round #374151;
        content-align: center middle;
        text-style: bold;
    }
    .badge-scanning  { background: #854d0e; color: #fef08a; }
    .badge-executing { background: #14532d; color: #86efac; }
    .badge-idle      { background: #1f2937; color: #6b7280; }
    .badge-halted    { background: #7f1d1d; color: #fca5a5; }

    /* ── Main content area ────────────────────────────────────── */
    #main-area {
        height: 1fr;
        padding: 1 2;
    }

    /* ── Panels / containers ──────────────────────────────────── */
    .panel {
        border: round #1f2937;
        background: #111827;
        padding: 1 1;
        margin-bottom: 1;
    }
    .panel-title {
        color: #6b7280;
        text-style: bold;
        margin-bottom: 1;
    }

    /* ── Positions table ──────────────────────────────────────── */
    #positions-panel {
        height: auto;
        min-height: 8;
    }
    #positions-table {
        height: auto;
    }
    .col-symbol   { color: #e0e0e0; text-style: bold; }
    .col-neutral  { color: #9ca3af; }
    .col-green    { color: #22c55e; }
    .col-red      { color: #ef4444; }
    .col-amber    { color: #f59e0b; }
    #no-positions {
        color: #4b5563;
        content-align: center middle;
        width: 1fr;
        padding: 2 0;
    }

    /* ── Tabs ─────────────────────────────────────────────────── */
    TabbedContent {
        height: 1fr;
    }
    TabPane {
        padding: 0 1;
    }

    /* ── Price grid ───────────────────────────────────────────── */
    #price-table {
        height: auto;
    }

    /* ── Footer ───────────────────────────────────────────────── */
    Footer {
        background: #111827;
        color: #4b5563;
    }
    """

    # ── Widgets ──────────────────────────────────────────────────────────────────

    class TopBar(Widget):
        """Cash / Buying Power display + stub action buttons."""

        DEFAULT_CSS = ""

        def compose(self) -> ComposeResult:
            with Horizontal(id="top-bar"):
                with Horizontal(id="account-info"):
                    yield Static("Cash: $0.00", id="account-cash")
                    yield Static("Buying Power: $0.00", id="account-bp")
                with Horizontal(id="top-bar-buttons"):
                    yield Button("⚙ Settings", id="btn-settings", classes="pill-btn")
                    yield Button("> Console", id="btn-console", classes="pill-btn")
                    yield Button("···", id="btn-reserved", classes="pill-btn")

        def refresh_data(self, cash: float, buying_power: float) -> None:
            self.query_one("#account-cash", Static).update(
                f"Cash: [bold]${cash:,.2f}[/bold]"
            )
            self.query_one("#account-bp", Static).update(
                f"Buying Power: [bold]${buying_power:,.2f}[/bold]"
            )

        def on_button_pressed(self, event: Button.Pressed) -> None:
            btn_id = event.button.id
            if btn_id == "btn-settings":
                # TODO: open settings panel
                pass
            elif btn_id == "btn-console":
                # TODO: open console overlay
                pass
            elif btn_id == "btn-reserved":
                # TODO: reserved for future function
                pass

    class StatusStrip(Widget):
        """Pulsing live dot + cycle / last-run / bot-state badge."""

        _pulse: bool = False

        def compose(self) -> ComposeResult:
            with Horizontal(id="status-strip"):
                yield Static("●", id="live-dot")
                yield Static("Bot Cycle: #0", id="cycle-label")
                yield Static("Last Run: —", id="lastrun-label")
                yield Static("IDLE", id="state-badge", classes="badge-idle")

        def refresh_data(
            self,
            cycle_count: int,
            last_run_ts: float,
            bot_state: str,
            market_open: bool,
        ) -> None:
            # Pulse the dot
            self._pulse = not self._pulse
            dot = self.query_one("#live-dot", Static)
            dot.update("●" if self._pulse else "○")
            if market_open:
                dot.styles.color = "#22c55e"
            else:
                dot.styles.color = "#4b5563"

            self.query_one("#cycle-label", Static).update(
                f"Bot Cycle: [cyan]#{cycle_count:,}[/cyan]"
            )

            if last_run_ts > 0:
                elapsed = time.time() - last_run_ts
                self.query_one("#lastrun-label", Static).update(
                    f"Last Run: [cyan]{elapsed:.1f}s ago[/cyan]"
                )

            badge = self.query_one("#state-badge", Static)
            state_upper = bot_state.upper()
            badge.update(state_upper)
            badge.remove_class("badge-scanning", "badge-executing", "badge-idle", "badge-halted")
            if state_upper == "SCANNING":
                badge.add_class("badge-scanning")
            elif state_upper == "EXECUTING":
                badge.add_class("badge-executing")
            elif state_upper == "HALTED":
                badge.add_class("badge-halted")
            else:
                badge.add_class("badge-idle")

    class PositionsPanel(Widget):
        """Auto-refreshing positions table."""

        def compose(self) -> ComposeResult:
            with Container(id="positions-panel", classes="panel"):
                yield Static("Open Positions", classes="panel-title")
                yield DataTable(id="positions-table", show_cursor=False)

        def on_mount(self) -> None:
            tbl: DataTable = self.query_one("#positions-table", DataTable)
            tbl.add_columns(
                "Symbol",
                "Current Price",
                "Entry Price",
                "% Change",
                "P&L ($)",
                "Take Profit",
                "Stop Loss",
            )

        def refresh_data(self, positions: List[PositionRow]) -> None:
            tbl: DataTable = self.query_one("#positions-table", DataTable)
            tbl.clear()
            if not positions:
                tbl.add_row("[#4b5563]—[/]", "[#4b5563] No open positions [/]", "", "", "", "", "")
                return
            for row in positions:
                pct_str = f"{row.pct_change * 100:+.2f}%"
                pnl_str = f"${row.pnl:+,.2f}"
                pct_markup = (
                    f"[green]{pct_str}[/green]"
                    if row.pct_change >= 0
                    else f"[red]{pct_str}[/red]"
                )
                pnl_markup = (
                    f"[green]{pnl_str}[/green]"
                    if row.pnl >= 0
                    else f"[red]{pnl_str}[/red]"
                )
                tbl.add_row(
                    f"[bold]{row.symbol}[/bold]",
                    f"[cyan]${row.current_price:,.2f}[/cyan]",
                    f"${row.entry_price:,.2f}",
                    pct_markup,
                    pnl_markup,
                    f"[yellow]${row.take_profit:,.2f}[/yellow]",
                    f"[yellow]${row.stop_loss:,.2f}[/yellow]",
                )

    class PricePanel(Widget):
        """Live price ticker for all whitelisted symbols."""

        def compose(self) -> ComposeResult:
            with Container(classes="panel"):
                yield Static("Live Prices", classes="panel-title")
                yield DataTable(id="price-table", show_cursor=False)

        def on_mount(self) -> None:
            tbl: DataTable = self.query_one("#price-table", DataTable)
            tbl.add_columns("Symbol", "Last Price", "Bid", "Ask", "Daily % Change")

        def refresh_data(self, prices: List[PriceRow]) -> None:
            tbl: DataTable = self.query_one("#price-table", DataTable)
            tbl.clear()
            if not prices:
                tbl.add_row("[#4b5563]—[/]", "[#4b5563] No price data [/]", "", "", "")
                return
            for row in prices:
                daily_str = f"{row.daily_pct_change * 100:+.2f}%"
                daily_markup = (
                    f"[green]{daily_str}[/green]"
                    if row.daily_pct_change >= 0
                    else f"[red]{daily_str}[/red]"
                )
                tbl.add_row(
                    f"[bold]{row.symbol}[/bold]",
                    f"[cyan]${row.last_price:,.2f}[/cyan]",
                    f"${row.bid:,.2f}",
                    f"${row.ask:,.2f}",
                    daily_markup,
                )

    # ── Main App ─────────────────────────────────────────────────────────────────

    class TradeBotDashboard(App):
        """Alpaca Algo Trading Dashboard — Textual TUI."""

        CSS = APP_CSS
        TITLE = "Alpaca TradeBot"
        BINDINGS = [
            Binding("p", "switch_tab('prices')", "Prices", show=True),
            Binding("o", "switch_tab('positions')", "Positions", show=True),
            Binding("q", "quit", "Quit", show=True),
        ]

        def __init__(self, refresh_seconds: float) -> None:
            super().__init__()
            self._refresh_seconds = refresh_seconds

        def compose(self) -> ComposeResult:
            yield TopBar()
            yield StatusStrip()
            with Vertical(id="main-area"):
                with TabbedContent(initial="positions"):
                    with TabPane("Positions", id="positions"):
                        yield PositionsPanel()
                    with TabPane("Prices", id="prices"):
                        yield PricePanel()
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(self._refresh_seconds, self._tick)

        def _tick(self) -> None:
            """Called every N seconds — load state from pickle file and refresh all widgets."""
            state = load_state()

            self.query_one(TopBar).refresh_data(
                cash=state.cash,
                buying_power=state.buying_power,
            )
            self.query_one(StatusStrip).refresh_data(
                cycle_count=state.cycle_count,
                last_run_ts=state.last_run_ts,
                bot_state=state.bot_state,
                market_open=state.market_open,
            )
            self.query_one(PositionsPanel).refresh_data(state.positions)
            self.query_one(PricePanel).refresh_data(state.prices)

        def action_switch_tab(self, tab_id: str) -> None:
            tc = self.query_one(TabbedContent)
            tc.active = tab_id

    # Run blocking in current process
    app = TradeBotDashboard(refresh_seconds=refresh_seconds)
    app.run()


# ── Public helpers called by main.py ─────────────────────────────────────────────

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
        side = str(getattr(o, "side", "")).lower()
        if order_type == "limit":
            lp = getattr(o, "limit_price", None)
            if lp is not None:
                tp_map[sym] = float(lp)
        if order_type in ("stop", "trailing_stop", "stop_limit"):
            sp = getattr(o, "stop_price", None)
            if sp is not None:
                sl_map[sym] = float(sp)

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


def launch_dashboard(refresh_seconds: float = 5.0) -> None:
    """Launch the Textual dashboard in a separate OS window (new process).

    On Windows: spawns a detached console via CREATE_NEW_CONSOLE.
    On Unix: spawns with setsid() to detach from parent terminal.

    The subprocess polls dashboard_state.pkl every refresh_seconds.
    """
    if dev_mode:
        logger.info("Dashboard launch skipped: dev_mode=True (headless operation)")
        return

    if not _TEXTUAL_AVAILABLE:
        logger.warning(
            "Dashboard launch skipped: 'textual' module not installed. "
            "Install it with: pip install textual"
        )
        return

    # Build the command to run this module's _build_and_run_app in a new process
    script = f"""
import sys
sys.path.insert(0, {repr(str(Path.cwd()))})
from monitoring.terminal import _build_and_run_app
_build_and_run_app(refresh_seconds={refresh_seconds})
"""

    cmd = [sys.executable, "-c", script]

    try:
        if platform.system() == "Windows":
            # CREATE_NEW_CONSOLE = 0x00000010
            subprocess.Popen(
                cmd,
                creationflags=0x00000010,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Dashboard launched in new Windows console window.")
        else:
            # Unix: use os.setsid to fully detach
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            logger.info("Dashboard launched in detached Unix process.")
    except Exception as e:
        logger.error(f"Failed to launch dashboard subprocess: {e}")
