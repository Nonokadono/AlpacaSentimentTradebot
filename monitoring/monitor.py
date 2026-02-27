# CHANGES:
#   - log_sentiment_position_check(): fixed broken logger.info line at the
#     "docs=" row. On-disk line was:
#         logger.info(f"docs={current_sentiment.ndocuments}){RESET}")
#     which is missing {lc}, missing the "  " indent, and has a stray literal ")"
#     inside the string. Fixed to:
#         logger.info(f"{lc}  docs={current_sentiment.ndocuments}{RESET}")

import logging
import threading
from datetime import datetime
from typing import List, Optional

from core.risk_engine import EquitySnapshot, ProposedTrade, PositionInfo
from core.sentiment import SentimentResult
from monitoring.kill_switch import KillSwitchState

logger = logging.getLogger("tradebot")

# ── ANSI colour constants ─────────────────────────────────────────────────────
RESET        = "\033[0m"
LINEBLUE     = "\033[38;2;138;185;241m"
LINEYELLOW   = "\033[38;2;248;222;126m"
SEPCOLOR     = "\033[38;2;222;211;151m"
SIGNALRED    = "\033[38;2;220;53;69m"
SIGNALGREEN  = "\033[38;2;40;167;69m"
ITALICON     = "\033[3m"
ITALICOFF    = "\033[23m"
BRIGHTPURPLE = "\033[38;2;191;64;191m"
DEEPBLUE     = "\033[38;2;0;13;222m"

MARKETOPEN   = SIGNALGREEN
MARKETCLOSED = SIGNALRED

# ── Thread-safe line-colour toggle ────────────────────────────────────────────
_linetoggle      = 0
_linetoggle_lock = threading.Lock()


def next_line_color() -> str:
    global _linetoggle
    with _linetoggle_lock:
        _linetoggle = 0 if _linetoggle == 1 else 1
        return LINEBLUE if _linetoggle == 0 else LINEYELLOW


def separator_line() -> str:
    return f"{SEPCOLOR}" + "-" * 80 + f"{RESET}"


def _thick(w: int = 80) -> str:
    """Border glyphs only — no colour escapes; callers wrap in lc / RESET."""
    return "═" * w


def _thin(w: int = 80) -> str:
    """Border glyphs only — no colour escapes; callers wrap in lc / RESET."""
    return "─" * w


def sentiment_score_fragment(score: float, line_color: str) -> str:
    base = "sentiment="
    if score <= -0.5:
        return f"{base}{SIGNALRED}{score:.3f}{line_color}"
    if score >= 0.5:
        return f"{base}{SIGNALGREEN}{score:.3f}{line_color}"
    return f"{base}{score:.3f}"


def italicize_technical(text: str) -> str:
    if not text:
        return text
    terms = [
        "RSI", "MACD", "moving average", "moving averages", "EMA", "SMA",
        "Bollinger", "support", "resistance", "candles", "candle", "bar", "bars",
    ]
    out = text
    for term in terms:
        out = out.replace(term, f"{ITALICON}{term}{ITALICOFF}")
    return out


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_logging(env_mode: str = "") -> None:
    """
    Initialise root logger at INFO level (no-op if already initialised).
    Calls log_startup_banner() so callers need no separate banner call.
    Pass cfg.env_mode for a labelled banner; omit for an unlabelled one.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log_startup_banner(env_mode)


def log_startup_banner(env_mode: str) -> None:
    logger.info(separator_line())
    logger.info(f"{DEEPBLUE}   TRADE BOT STARTING UP   {RESET}")
    if env_mode:
        logger.info(f"{DEEPBLUE}   Environment: {env_mode}{RESET}")
    logger.info(separator_line())


def log_environment_switch(env_mode: str, user: str) -> None:
    line_color = next_line_color()
    msg = f"{datetime.utcnow().isoformat()}Z ENV switch event: mode={env_mode}, user={user}"
    logger.warning(f"{line_color}{msg}{RESET}")


# ── Equity snapshot ───────────────────────────────────────────────────────────

def log_equity_snapshot(snapshot: EquitySnapshot, market_open: bool = False) -> None:
    line_color = next_line_color()
    market_badge = (
        f"{MARKETOPEN}[MARKET OPEN]{line_color}"
        if market_open
        else f"{MARKETCLOSED}[MARKET CLOSED]{line_color}"
    )
    msg = (
        f"{datetime.utcnow().isoformat()}Z EquitySnapshot: "
        f"{market_badge} "
        f"equity={snapshot.equity:.2f}, "
        f"cash={snapshot.cash:.2f}, "
        f"gross_exposure={snapshot.gross_exposure:.2f}, "
        f"daily_loss_pct={snapshot.daily_loss_pct:.3f}, "
        f"drawdown_pct={snapshot.drawdown_pct:.3f}"
    )
    logger.info(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())


# ── Kill-switch ───────────────────────────────────────────────────────────────

def log_kill_switch_state(state: KillSwitchState) -> None:
    """
    BUG #3 fix: OK-branch was logger.debug() — invisible at INFO level.
    Now logger.info() so the heartbeat is visible every loop.
    """
    line_color = next_line_color()
    if state.halted:
        msg = f"{datetime.utcnow().isoformat()}Z KILL-SWITCH ACTIVATED: {state.reason}"
        logger.error(f"{line_color}{msg}{RESET}")
    else:
        msg = f"{datetime.utcnow().isoformat()}Z Kill-switch OK"
        logger.info(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())


# ── Instrument Report ─────────────────────────────────────────────────────────

def log_instrument_report(
    symbol: str,
    signal_score: float,
    sentiment: SentimentResult,
    momentum_score: float,
    mean_reversion_score: float,
    price_action_score: float,
    env_mode: str,
) -> None:
    """
    Emit a structured Instrument Report block for a single symbol scan.

    ════════════════════════════════════════════════════════════════════════════
     [PAPER] Instrument Report  │  AAPL  |  value=+0.350
    Current Sentiment=+0.900
    reason=Earnings beat, raised guidance; analysts bullish short-term.
    ────────────────────────────────────────────────────────────────────────────
    Indicator Listing
    compound=+0.350   momentum=+0.412   mean_reversion=-0.050   price_action=+0.000
    ════════════════════════════════════════════════════════════════════════════
    """
    W  = 80
    lc = next_line_color()

    thick = _thick(W)
    thin  = _thin(W)

    # ── Header row ────────────────────────────────────────────────────────────
    env_badge   = f"[{env_mode}] " if env_mode else ""
    symbol_str  = f"{BRIGHTPURPLE}{symbol}{lc}"
    val_color   = SIGNALGREEN if signal_score >= 0 else SIGNALRED
    val_str     = f"{val_color}{signal_score:+.3f}{lc}"
    header_body = (
        f" {DEEPBLUE}{env_badge}Instrument Report{lc}  │  "
        f"{symbol_str}  |  value={val_str}"
    )

    # ── Current Sentiment row ─────────────────────────────────────────────────
    sc       = sentiment.score
    sc_color = SIGNALGREEN if sc >= 0.5 else (SIGNALRED if sc <= -0.5 else lc)
    sc_str   = f"{sc_color}{sc:+.3f}{lc}"
    sent_row = f"Current Sentiment={sc_str}"

    # ── Reason row ────────────────────────────────────────────────────────────
    expl_raw   = (sentiment.explanation or "").replace("\n", " ").strip()
    expl_fmt   = italicize_technical(expl_raw)
    reason_row = f"reason={expl_fmt}"

    # ── Indicator colours ─────────────────────────────────────────────────────
    # compound = signal_score (weighted technical composite).
    # This is signal_score, NOT sentiment.rawcompound (which is the AI score).
    cp       = signal_score
    cp_color = SIGNALGREEN if cp >= 0 else SIGNALRED
    cp_str   = f"{cp_color}{cp:+.3f}{lc}"

    mom_color = SIGNALGREEN if momentum_score >= 0 else SIGNALRED
    mom_str   = f"{mom_color}{momentum_score:+.3f}{lc}"

    mr_color  = SIGNALGREEN if mean_reversion_score >= 0 else SIGNALRED
    mr_str    = f"{mr_color}{mean_reversion_score:+.3f}{lc}"

    pa_color  = SIGNALGREEN if price_action_score >= 0 else SIGNALRED
    pa_str    = f"{pa_color}{price_action_score:+.3f}{lc}"

    # ── Indicator Listing — all four on one line ───────────────────────────────
    ind_row = (
        f"compound={cp_str}   "
        f"momentum={mom_str}   "
        f"mean_reversion={mr_str}   "
        f"price_action={pa_str}"
    )

    # ── Formula row — italicised factor breakdown ──────────────────────────────
    formula_row = (
        f"  Compound Score={cp_str}   "
        f"{ITALICON}momentum={mom_color}{momentum_score:+.3f}{lc}{ITALICOFF}   "
        f"{ITALICON}mean_reversion={mr_color}{mean_reversion_score:+.3f}{lc}{ITALICOFF}   "
        f"{ITALICON}price_action={pa_color}{price_action_score:+.3f}{lc}{ITALICOFF}"
    )

    logger.info(f"{lc}{thick}{RESET}")
    logger.info(f"{lc}{header_body}{RESET}")
    logger.info(f"{lc}{sent_row}{RESET}")
    logger.info(f"{lc}{reason_row}{RESET}")
    logger.info(f"{lc}{thin}{RESET}")
    logger.info(f"{lc}Indicator Listing{RESET}")
    logger.info(f"{lc}{ind_row}{RESET}")
    logger.info(f"{lc}{thin}{RESET}")
    logger.info(f"{lc}{formula_row}{RESET}")
    logger.info(f"{lc}{thick}{RESET}")


# ── Deprecated shims ──────────────────────────────────────────────────────────
# These exist only so any OTHER file that still imports them doesn't crash.
# core/signals.py no longer calls them. Do NOT add logger.warning() here —
# the warnings were the user-visible noise we're eliminating.

def log_sentiment_for_symbol(symbol: str, sentiment: SentimentResult, env_mode: str) -> None:
    """Deprecated shim — preserved for import compatibility only."""
    log_instrument_report(
        symbol=symbol,
        signal_score=0.0,
        sentiment=sentiment,
        momentum_score=0.0,
        mean_reversion_score=0.0,
        price_action_score=0.0,
        env_mode=env_mode,
    )


def log_signal_score(
    symbol: str,
    signal_score: float,
    momentum_score: float,
    mean_reversion_score: float,
    price_action_score: float,
    env_mode: str,
) -> None:
    """Deprecated shim — preserved for import compatibility only."""
    from core.sentiment import SentimentResult  # local import avoids circular at module load
    log_instrument_report(
        symbol=symbol,
        signal_score=signal_score,
        sentiment=SentimentResult(
            score=0.0, raw_discrete=0, rawcompound=0.0,
            ndocuments=0, explanation=None, confidence=0.0,
        ),
        momentum_score=momentum_score,
        mean_reversion_score=mean_reversion_score,
        price_action_score=price_action_score,
        env_mode=env_mode,
    )


# ── Proposed trade ────────────────────────────────────────────────────────────

def log_proposed_trade(trade: ProposedTrade, env_mode: str) -> None:
    """
    BUG #5 fix: unexpected trade.side now emits logger.warning() instead of
    silently producing an empty action_tag.
    """
    line_color     = next_line_color()
    sentiment_part = sentiment_score_fragment(trade.sentiment_score, line_color)
    action_tag     = ""
    if trade.rejected_reason is None and trade.qty > 0:
        if trade.side == "buy":
            action_tag = f" {SIGNALGREEN}BUY{line_color}"
        elif trade.side == "sell":
            action_tag = f" {SIGNALRED}SELL{line_color}"
        else:
            logger.warning(
                f"log_proposed_trade: unexpected side='{trade.side}' for "
                f"{trade.symbol} — expected 'buy' or 'sell'."
            )
    notional    = trade.qty * trade.entry_price
    symbol_part = f"{BRIGHTPURPLE}{trade.symbol}{line_color}"
    sig_part    = f"{BRIGHTPURPLE}{trade.signal_score:.3f}{line_color}"
    msg = (
        f"{datetime.utcnow().isoformat()}Z {env_mode} ProposedTrade "
        f"symbol={symbol_part} {trade.side} "
        f"qty={trade.qty:.4f} notional={notional:.2f} "
        f"entry={trade.entry_price:.4f} stop={trade.stop_price:.4f} tp={trade.take_profit_price:.4f} "
        f"risk_amt={trade.risk_amount:.2f} risk_pct={trade.risk_pct_of_equity:.4f} "
        f"{sentiment_part} scale={trade.sentiment_scale:.3f} "
        f"rejected={trade.rejected_reason} "
        f"{action_tag} "
        f"{sig_part}"
    )
    logger.info(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())


# ── Sentiment close decision ──────────────────────────────────────────────────

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
    line_color  = next_line_color()
    score_frag  = sentiment_score_fragment(sentiment_score, line_color)
    expl_raw    = (explanation or "").replace("\n", " ").strip()
    expl_fmt    = italicize_technical(expl_raw)
    force_msg   = f"{SIGNALRED}FORCE-CLOSED DUE TO BAD SENTIMENT{line_color}"
    symbol_part = f"{BRIGHTPURPLE}{symbol}{line_color}"
    msg = (
        f"{datetime.utcnow().isoformat()}Z {env_mode} SentimentExit "
        f"symbol={symbol_part} side={side} qty={qty:.4f} "
        f"{score_frag} conf={confidence:.2f} "
        f"reason_for_exit={reason} "
        f"{force_msg} "
        f"{expl_fmt}"
    )
    logger.warning(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())


# ── Portfolio overview ────────────────────────────────────────────────────────

def log_portfolio_overview(trades: List[ProposedTrade], env_mode: str) -> None:
    logger.info(separator_line())
    if not trades:
        header = f"{datetime.utcnow().isoformat()}Z {env_mode} PORTFOLIO OVERVIEW"
        logger.info(f"{DEEPBLUE}{header}{RESET}")
        logger.info(f"{DEEPBLUE}No new trades selected in this cycle.{RESET}")
        logger.info(separator_line())
        return

    line_color = next_line_color()
    header     = f"{datetime.utcnow().isoformat()}Z {env_mode} PORTFOLIO OVERVIEW"
    logger.info(f"{line_color}{header}{RESET}")
    for t in trades:
        notional    = t.qty * t.entry_price
        symbol_part = f"{BRIGHTPURPLE}{t.symbol}{line_color}"
        sig_str     = f"{BRIGHTPURPLE}{t.signal_score:.3f}{line_color}"
        msg = (
            f"  symbol={symbol_part} side={t.side} "
            f"qty={t.qty:.4f} notional={notional:.2f} "
            f"signal_score={sig_str} "
            f"rejected={t.rejected_reason}"
        )
        logger.info(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())


# ── Sentiment position check ──────────────────────────────────────────────────

def log_sentiment_position_check(
    position: PositionInfo,
    entry_compound: float,
    current_sentiment: SentimentResult,
    delta: float,
    delta_threshold: float,
    confidence_min: float,
    closing: bool,
    close_reason: str,
    env_mode: str,
    stop_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
) -> None:
    """
    BUG #2 fix: triggered_tag and delta_color are also set when raw_discrete == -2
    (chaos exit), regardless of numeric delta, so the delta row is always visually
    consistent with the Verdict.

    BUG #4 fix: thick/thin carry no embedded RESET — callers wrap in lc/RESET.

    Current compound now uses current_sentiment.score (not .rawcompound).
    rawcompound == score always (set identically in _call_ai); using .score is
    semantically correct and consistent with how delta is computed in main.py.
    """
    W  = 80
    lc = next_line_color()

    thick = _thick(W)
    thin  = _thin(W)

    rd = current_sentiment.raw_discrete

    # ── Header row ────────────────────────────────────────────────────────────
    symbol_str   = f"{BRIGHTPURPLE}{position.symbol}{lc}"
    side_str     = (
        f"{SIGNALGREEN}{position.side.upper()}{lc}"
        if position.side == "long"
        else f"{SIGNALRED}{position.side.upper()}{lc}"
    )
    notional_val = abs(position.qty * position.market_price)
    header_body  = (
        f" {DEEPBLUE}SENTIMENT CHECK{lc}  │  "
        f"{symbol_str}  │  {side_str}  │  "
        f"qty={position.qty:.4f}  │  notional={notional_val:.2f}"
    )

    # ── Metadata row ──────────────────────────────────────────────────────────
    sc       = current_sentiment.score
    sc_color = SIGNALGREEN if sc >= 0.5 else (SIGNALRED if sc <= -0.5 else lc)
    sc_str   = f"{sc_color}{sc:+.3f}{lc}"

    ec_color = SIGNALGREEN if entry_compound >= 0 else SIGNALRED
    ec_str   = f"{ec_color}{entry_compound:+.3f}{lc}"

    sl_str = f"{stop_price:.2f}"        if stop_price        is not None else "N/A"
    tp_str = f"{take_profit_price:.2f}" if take_profit_price is not None else "N/A"

    meta_row = (
        f"Current Sentiment={sc_str}   |   "
        f"Opening Compound={ec_str}   |   "
        f"Stop Loss={sl_str}   |   "
        f"Take Profit={tp_str}"
    )

    # ── Entry compound ────────────────────────────────────────────────────────
    ec_str2 = f"{ec_color}{entry_compound:+.3f}{lc}"

    # ── Delta — BUG #2 fix: chaos exit also marks delta as triggered ──────────
    chaos_exit    = (rd == -2)
    delta_trigger = (delta >= delta_threshold)
    show_trigger  = delta_trigger or chaos_exit

    delta_color   = SIGNALRED if show_trigger else lc
    triggered_tag = (
        f"  {SIGNALRED}← TRIGGERED{lc}"
        if show_trigger
        else ""
    )
    delta_str = (
        f"{delta_color}{delta:+.3f}{lc}   "
        f"[threshold ≥ {delta_threshold:.3f}]{triggered_tag}"
    )

    # ── Explanation ───────────────────────────────────────────────────────────
    expl_raw = (current_sentiment.explanation or "").replace("\n", " ").strip()
    expl_fmt = italicize_technical(expl_raw)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if closing:
        if chaos_exit:
            verdict_detail = (
                f"{SIGNALRED}⛔ CLOSING — raw_discrete = -2 (CHAOS / absolute exit){lc}"
            )
        else:
            verdict_detail = (
                f"{SIGNALRED}⛔ CLOSING — sentiment shift "
                f"Δ={delta:+.3f} ≥ threshold {delta_threshold:.3f}{lc}"
            )
    else:
        verdict_detail = f"{SIGNALGREEN}✔  HOLDING — no exit condition met{lc}"

    logger.info(f"{lc}{thick}{RESET}")
    logger.info(f"{lc}{header_body}{RESET}")
    logger.info(f"{lc}{thick}{RESET}")
    logger.info(f"{lc}{meta_row}{RESET}")
    logger.info(f"{lc}{thin}{RESET}")
    logger.info("")
    logger.info(f"{lc}  Entry compound   :  {ec_str2}{RESET}")
    logger.info("")
    logger.info("")
    logger.info(f"{lc}  Δ compound       :  {delta_str}{RESET}")
    logger.info("")
    logger.info(f"{lc}  Explanation      :  {expl_fmt}{RESET}")
    logger.info("")
    logger.info(f"{lc}{thin}{RESET}")
    logger.info("")
    logger.info(f"{lc}  Verdict          :  {verdict_detail}{RESET}")
    logger.info("")
    logger.info(f"{lc}{thick}{RESET}")
