# CHANGES:
# - No functional changes from baseline.
# - Full file reproduced verbatim to restore all ANSI colour formatting,
#   separator helpers, italicize_technical(), sentiment_score_fragment(),
#   nextlinecolor cycling, and all log_* functions that were present in the
#   original 400-line version.

import logging
import threading
from datetime import datetime
from typing import List, Optional

from core.risk_engine import EquitySnapshot, ProposedTrade, PositionInfo
from core.sentiment import SentimentResult
from config.config import ENV_MODE

logger = logging.getLogger("tradebot")

# ── ANSI colour constants ──────────────────────────────────────────────────────

RESET         = "\033[0m"
BOLD          = "\033[1m"
ITALIC        = "\033[3m"
DEEPBLUE      = "\033[38;5;27m"
BRIGHTPURPLE  = "\033[38;5;135m"
SIGNALGREEN   = "\033[38;5;46m"
SIGNALRED     = "\033[38;5;196m"
MARKETOPEN    = "\033[38;5;82m"
MARKETCLOSED  = "\033[38;5;214m"

# Colour palette cycled per log block so adjacent blocks are visually distinct.
_LINE_COLOURS = [
    "\033[38;5;33m",   # steel blue
    "\033[38;5;39m",   # sky blue
    "\033[38;5;44m",   # cyan-ish
    "\033[38;5;75m",   # light blue
    "\033[38;5;111m",  # periwinkle
]
_colour_index = 0
_colour_lock  = threading.Lock()


def nextlinecolor() -> str:
    """Return the next colour from the cycling palette (thread-safe)."""
    global _colour_index
    with _colour_lock:
        c = _LINE_COLOURS[_colour_index % len(_LINE_COLOURS)]
        _colour_index += 1
    return c


# ── Separator helpers ─────────────────────────────────────────────────────────

def _thick(width: int = 80) -> str:
    return "═" * width

def _thin(width: int = 80) -> str:
    return "─" * width

def thick(width: int = 80) -> str:
    return _thick(width)

def thin(width: int = 80) -> str:
    return _thin(width)

def separatorline(width: int = 80) -> str:
    return "-" * width


# ── Text helpers ──────────────────────────────────────────────────────────────

def italicize_technical(text: str) -> str:
    """Wrap text in ANSI italic escape codes."""
    return f"{ITALIC}{text}{RESET}"


def sentiment_score_fragment(score: float, lc: str) -> str:
    """
    Return a colour-coded sentiment score string.
      score >  0.5  → green
      score < -0.5  → red
      otherwise     → line colour (lc)
    """
    if score > 0.5:
        sc_color = SIGNALGREEN
    elif score < -0.5:
        sc_color = SIGNALRED
    else:
        sc_color = lc
    return f"{sc_color}{score:.3f}{lc}"


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(env_mode: str = "PAPER") -> None:
    """Configure root logger. Call once at startup."""
    import sys
    level = logging.DEBUG if env_mode == "PAPER" else logging.INFO
    fmt     = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(handler)
    root.setLevel(level)


def log_environment_switch(env_mode: str, user: str = "") -> None:
    tag = f" (triggered by: {user})" if user else ""
    lc  = nextlinecolor()
    logger.info(f"{lc}{_thick()}{RESET}")
    logger.info(f"{lc}ENVIRONMENT: {env_mode}{tag}{RESET}")
    logger.info(f"{lc}{_thick()}{RESET}")


# ── Instrument report (called by SignalEngine for every symbol) ───────────────

def log_instrument_report(
    symbol: str,
    signal_score: float,
    sentiment: SentimentResult,
    momentum_score: float,
    mean_reversion_score: float,
    price_action_score: float,
    env_mode: str = "PAPER",
) -> None:
    """
    Unified per-symbol evaluation block emitted by SignalEngine for every
    symbol it processes — both trade (buy/sell) and skip paths.
    Renders:
      - Signal decomposition: momentum, mean-reversion, price-action, composite
      - Sentiment block: score, discrete, confidence, ndocuments, explanation
    Replaces the two deprecated shims log_signal_score and log_sentiment_for_symbol
    that were previously called separately from signals.py.
    """
    lc  = nextlinecolor()
    W   = 80
    thn = f"{lc}{_thin(W)}{RESET}"

    # Composite signal colour
    if signal_score > 0:
        sig_color = SIGNALGREEN
    elif signal_score < 0:
        sig_color = SIGNALRED
    else:
        sig_color = lc

    # Sentiment score colour
    sc = sentiment.score
    if sc > 0.5:
        sc_color = SIGNALGREEN
    elif sc < -0.5:
        sc_color = SIGNALRED
    else:
        sc_color = lc

    symbol_str = f"{BRIGHTPURPLE}{symbol}{lc}"
    sig_str    = f"{sig_color}{signal_score:.3f}{lc}"
    sc_str     = f"{sc_color}{sc:.3f}{lc}"

    expl_raw = (sentiment.explanation or "").replace("\n", " ").strip()
    expl_fmt = italicize_technical(expl_raw)

    logger.info(thn)
    logger.info(
        f"{lc} {env_mode} {DEEPBLUE}SIGNAL{lc} "
        f"{symbol_str}  composite={sig_str}{RESET}"
    )
    logger.info(
        f"{lc}   momentum={momentum_score:.3f}  "
        f"meanrev={mean_reversion_score:.3f}  "
        f"priceaction={price_action_score:.3f}{RESET}"
    )
    logger.info(
        f"{lc}   sentiment={sc_str}  "
        f"discrete={sentiment.raw_discrete}  "
        f"conf={sentiment.confidence:.2f}  "
        f"docs={sentiment.ndocuments}{RESET}"
    )
    if expl_fmt:
        logger.info(f"{lc}   {expl_fmt}{RESET}")
    logger.info(thn)


# ── Proposed trade log ────────────────────────────────────────────────────────

def log_proposed_trade(trade: ProposedTrade, env_mode: str = "PAPER") -> None:
    """
    Emit a richly formatted block for a ProposedTrade — accepted or rejected.
    Unexpected trade.side emits logger.warning instead of silently producing
    an empty action tag.
    """
    lc         = nextlinecolor()
    line_color = lc
    sep        = f"{line_color}{separatorline()}{RESET}"

    sentiment_part = sentiment_score_fragment(trade.sentiment_score, line_color)

    if trade.rejected_reason is None and trade.qty > 0:
        if trade.side == "buy":
            action_tag = f" {SIGNALGREEN}BUY{line_color}"
        elif trade.side == "sell":
            action_tag = f" {SIGNALRED}SELL{line_color}"
        else:
            logger.warning(
                f"log_proposed_trade: unexpected side={trade.side} "
                f"for {trade.symbol}; expected 'buy' or 'sell'."
            )
            action_tag = f" {trade.side.upper()}"
    else:
        action_tag = ""

    notional    = trade.qty * trade.entry_price
    symbol_part = f"{BRIGHTPURPLE}{trade.symbol}{line_color}"
    sig_part    = f"{BRIGHTPURPLE}{trade.signal_score:.3f}{line_color}"

    msg = (
        f"{datetime.utcnow().isoformat()}Z  {env_mode}  ProposedTrade  "
        f"symbol={symbol_part}  {trade.side}  "
        f"qty={trade.qty:.4f}  notional={notional:.2f}  "
        f"entry={trade.entry_price:.4f}  stop={trade.stop_price:.4f}  tp={trade.take_profit_price:.4f}  "
        f"riskamt={trade.risk_amount:.2f}  riskpct={trade.risk_pct_of_equity:.4f}  "
        f"sentiment={sentiment_part}  scale={trade.sentiment_scale:.3f}  "
        f"rejected={trade.rejected_reason}  "
        f"{action_tag}{sig_part}"
    )
    logger.info(f"{line_color}{msg}{RESET}")
    logger.info(sep)


# ── Legacy shim: log_sentiment_for_symbol ────────────────────────────────────

def log_sentiment_for_symbol(
    symbol: str,
    sentiment: SentimentResult,
    env_mode: str,
) -> None:
    """
    Legacy shim retained because main.py imports this function by name.
    Delegates to log_instrument_report with zeroed technical scores so any
    remaining callers continue to work without error.
    """
    log_instrument_report(
        symbol=symbol,
        signal_score=0.0,
        sentiment=sentiment,
        momentum_score=0.0,
        mean_reversion_score=0.0,
        price_action_score=0.0,
        env_mode=env_mode,
    )


# ── Portfolio overview ────────────────────────────────────────────────────────

def log_portfolio_overview(trades: List[ProposedTrade], env_mode: str = "PAPER") -> None:
    lc  = nextlinecolor()
    sep = f"{lc}{separatorline()}{RESET}"

    logger.info(sep)

    if not trades:
        header = f"{datetime.utcnow().isoformat()}Z  {env_mode}  PORTFOLIO OVERVIEW"
        logger.info(f"{DEEPBLUE}{header}{RESET}")
        logger.info(f"{DEEPBLUE}No new trades selected in this cycle.{RESET}")
        logger.info(sep)
        return

    header = f"{datetime.utcnow().isoformat()}Z  {env_mode}  PORTFOLIO OVERVIEW"
    logger.info(f"{lc}{header}{RESET}")

    for t in trades:
        notional    = t.qty * t.entry_price
        symbol_part = f"{BRIGHTPURPLE}{t.symbol}{lc}"
        sig_str     = f"{BRIGHTPURPLE}{t.signal_score:.3f}{lc}"
        msg = (
            f"  symbol={symbol_part}  side={t.side}  "
            f"qty={t.qty:.4f}  notional={notional:.2f}  "
            f"signalscore={sig_str}  rejected={t.rejected_reason}"
        )
        logger.info(f"{lc}{msg}{RESET}")

    logger.info(sep)


# ── Equity snapshot ───────────────────────────────────────────────────────────

def log_equity_snapshot(
    snapshot: EquitySnapshot,
    market_open: bool = False,
) -> None:
    lc = nextlinecolor()

    market_tag = (
        f"{MARKETOPEN}MARKET OPEN{lc}"
        if market_open
        else f"{MARKETCLOSED}MARKET CLOSED{lc}"
    )
    daily_color = SIGNALRED  if snapshot.daily_loss_pct < 0 else SIGNALGREEN
    dd_color    = SIGNALRED  if snapshot.drawdown_pct   < 0 else SIGNALGREEN

    msg = (
        f"{datetime.utcnow().isoformat()}Z  EquitySnapshot  "
        f"equity={snapshot.equity:.2f}  cash={snapshot.cash:.2f}  "
        f"portfoliovalue={snapshot.portfolio_value:.2f}  "
        f"grossexp={snapshot.gross_exposure:.2f}  "
        f"dailyloss={daily_color}{snapshot.daily_loss_pct:.3%}{lc}  "
        f"drawdown={dd_color}{snapshot.drawdown_pct:.3%}{lc}  "
        f"{market_tag}"
    )
    logger.info(f"{lc}{msg}{RESET}")
    logger.info(f"{lc}{separatorline()}{RESET}")


# ── Kill switch ───────────────────────────────────────────────────────────────

def log_kill_switch_state(ks_state) -> None:
    """
    Log the result of each kill-switch evaluation.
    ks_state is a KillSwitchState dataclass from monitoring/kill_switch.py.
    """
    if ks_state.halted:
        logger.warning(
            f"{SIGNALRED}KILL SWITCH ACTIVE{RESET}  {ks_state.reason}  "
            f"daily_pnl={ks_state.daily_loss_pct:+.2%}  "
            f"drawdown={ks_state.drawdown_pct:+.2%}"
        )
    else:
        logger.debug(
            f"Kill switch OK  "
            f"daily_pnl={ks_state.daily_loss_pct:+.2%}  "
            f"drawdown={ks_state.drawdown_pct:+.2%}"
        )


# ── Sentiment position check (per open position, per rescore cycle) ───────────

def log_sentiment_position_check(
    position: PositionInfo,
    entry_compound: float,
    current_sentiment: SentimentResult,
    delta: float,
    delta_threshold: float,
    confidence_min: float,
    closing: bool,
    close_reason: str,
    env_mode: str = "PAPER",
    stop_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
) -> None:
    """
    Emit a detailed per-position sentiment-check block each rescore cycle.
    Chaos exit (raw_discrete == -2) also marks triggered_tag and delta_color
    so the delta row is visually consistent with the Verdict.
    """
    lc  = nextlinecolor()
    W   = 80
    thk = f"{lc}{_thick(W)}{RESET}"
    thn = f"{lc}{_thin(W)}{RESET}"

    rd = current_sentiment.raw_discrete

    symbol_str  = f"{BRIGHTPURPLE}{position.symbol}{lc}"
    side_str    = (
        f"{SIGNALGREEN}{position.side.upper()}{lc}"
        if position.side == "long"
        else f"{SIGNALRED}{position.side.upper()}{lc}"
    )
    notional_val = abs(position.qty * position.market_price)
    header_body  = (
        f" {DEEPBLUE}SENTIMENT CHECK{lc}  "
        f"{symbol_str}  {side_str}  "
        f"qty={position.qty:.4f}  notional={notional_val:.2f}"
    )

    # Sentiment score colour
    sc       = current_sentiment.score
    sc_color = SIGNALGREEN if sc > 0.5 else SIGNALRED if sc < -0.5 else lc
    sc_str   = f"{sc_color}{sc:.3f}{lc}"

    # Entry compound colour
    ec_color = SIGNALGREEN if entry_compound > 0 else SIGNALRED
    ec_str   = f"{ec_color}{entry_compound:.3f}{lc}"

    sl_str = f"{stop_price:.2f}"       if stop_price       is not None else "N/A"
    tp_str = f"{take_profit_price:.2f}" if take_profit_price is not None else "N/A"

    meta_row = (
        f" Current Sentiment={sc_str}  "
        f"Opening Compound={ec_str}  "
        f"Stop Loss={sl_str}  Take Profit={tp_str}"
    )

    # Delta colour — red if adverse (closing), green if stable
    delta_color = SIGNALRED if closing else SIGNALGREEN
    chaos_exit  = (rd == -2)

    if chaos_exit:
        triggered_tag = f"  {SIGNALRED}[CHAOS raw_discrete=-2]{lc}"
        delta_color   = SIGNALRED
    elif closing:
        triggered_tag = f"  {SIGNALRED}[THRESHOLD BREACHED]{lc}"
    else:
        triggered_tag = ""

    delta_str = (
        f" Δ={delta_color}{delta:+.3f}{lc}  "
        f"threshold={delta_threshold:.3f}  "
        f"conf={current_sentiment.confidence:.2f}  "
        f"conf_min={confidence_min:.2f}  "
        f"discrete={rd}"
        f"{triggered_tag}"
    )

    expl_raw = (current_sentiment.explanation or "").replace("\n", " ").strip()
    expl_fmt = italicize_technical(expl_raw)

    if closing:
        if chaos_exit:
            verdict_detail = f"{SIGNALRED} CLOSING  raw_discrete=-2  CHAOS absolute exit{lc}"
        else:
            verdict_detail = (
                f"{SIGNALRED} CLOSING  sentiment shift "
                f"Δ={delta:.3f} > threshold {delta_threshold:.3f}{lc}"
            )
    else:
        verdict_detail = f"{SIGNALGREEN} HOLDING  no exit condition met{lc}"

    logger.info(thk)
    logger.info(f"{lc}{header_body}{RESET}")
    logger.info(thk)
    logger.info(f"{lc}{meta_row}{RESET}")
    logger.info(thn)
    logger.info(f"")
    logger.info(f"{lc} Entry compound    {ec_str}{RESET}")
    logger.info(f"")
    logger.info(f"")
    logger.info(f"{lc} {delta_str}{RESET}")
    logger.info(f"")
    logger.info(f"{lc} Explanation  {expl_fmt}{RESET}")
    logger.info(f"")
    logger.info(f"{lc} docs={current_sentiment.ndocuments}{RESET}")
    logger.info(thn)
    logger.info(f"")
    logger.info(f"{lc} Verdict  {verdict_detail}{RESET}")
    logger.info(f"")
    logger.info(thk)


# ── Sentiment close decision (called by OrderExecutor) ───────────────────────

def log_sentiment_close_decision(
    symbol: str,
    side: str,
    qty: float,
    sentiment_score: float,
    confidence: float,
    explanation: str,
    env_mode: str,
    reason: str,
) -> None:
    lc        = nextlinecolor()
    score_frag = sentiment_score_fragment(sentiment_score, lc)

    expl_raw = (explanation or "").replace("\n", " ").strip()
    expl_fmt = italicize_technical(expl_raw)

    force_msg   = f"{SIGNALRED}FORCE-CLOSED DUE TO BAD SENTIMENT{lc}"
    symbol_part = f"{BRIGHTPURPLE}{symbol}{lc}"

    msg = (
        f"{datetime.utcnow().isoformat()}Z  {env_mode}  SentimentExit  "
        f"symbol={symbol_part}  side={side}  qty={qty:.4f}  "
        f"{score_frag}  conf={confidence:.2f}  "
        f"reason_for_exit={reason}  "
        f"{force_msg}  "
        f"{expl_fmt}"
    )
    logger.warning(f"{lc}{msg}{RESET}")
    logger.info(f"{lc}{separatorline()}{RESET}")
