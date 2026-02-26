# monitoring/monitor.py
import logging
from datetime import datetime
from typing import List

from core.risk_engine import EquitySnapshot, ProposedTrade
from core.sentiment import SentimentResult
from monitoring.kill_switch import KillSwitchState

logger = logging.getLogger("tradebot")

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

# Market status colours
MARKETOPEN   = "\033[38;2;40;167;69m"   # green  - same hue as SIGNALGREEN
MARKETCLOSED = "\033[38;2;220;53;69m"   # red    - same hue as SIGNALRED

_linetoggle = 0

def next_line_color() -> str:
    global _linetoggle
    _linetoggle = 0 if _linetoggle == 1 else 1
    return LINEBLUE if _linetoggle == 0 else LINEYELLOW


def separator_line() -> str:
    return f"{SEPCOLOR}" + "-" * 80 + f"{RESET}"


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


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_environment_switch(env_mode: str, user: str) -> None:
    line_color = next_line_color()
    msg = f"{datetime.utcnow().isoformat()}Z ENV switch event: mode={env_mode}, user={user}"
    logger.warning(f"{line_color}{msg}{RESET}")


def log_equity_snapshot(snapshot: EquitySnapshot, market_open: bool = False) -> None:
    """
    Logs the equity snapshot with a coloured MARKET OPEN / MARKET CLOSED badge.
    """
    line_color = next_line_color()

    if market_open:
        market_badge = f"{MARKETOPEN}[MARKET OPEN]{line_color}"
    else:
        market_badge = f"{MARKETCLOSED}[MARKET CLOSED]{line_color}"

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


def log_kill_switch_state(state: KillSwitchState) -> None:
    line_color = next_line_color()
    if state.halted:
        msg = f"{datetime.utcnow().isoformat()}Z KILL-SWITCH ACTIVATED: {state.reason}"
        logger.error(f"{line_color}{msg}{RESET}")
    else:
        msg = f"{datetime.utcnow().isoformat()}Z Kill-switch OK"
        logger.debug(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())


def log_sentiment_for_symbol(symbol: str, sentiment: SentimentResult, env_mode: str) -> None:
    line_color = next_line_color()
    score_frag = sentiment_score_fragment(sentiment.score, line_color)
    expl_raw = (sentiment.explanation or "").replace("\n", " ").strip()
    expl_fmt = italicize_technical(expl_raw)
    symbol_part = f"{BRIGHTPURPLE}{symbol}{line_color}"
    msg = (
        f"{datetime.utcnow().isoformat()}Z {env_mode} Sentiment "
        f"symbol={symbol_part} "
        f"{score_frag} "
        f"conf={sentiment.confidence:.2f} "
        f"docs={sentiment.ndocuments} "
        f"reason={expl_fmt}"
    )
    logger.info(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())


def log_signal_score(
    symbol: str,
    signal_score: float,
    momentum_score: float,
    mean_reversion_score: float,
    price_action_score: float,
    env_mode: str,
) -> None:
    line_color = next_line_color()
    symbol_part = f"{BRIGHTPURPLE}{symbol}{line_color}"
    composite_str = f"{BRIGHTPURPLE}{signal_score:.3f}{line_color}"
    msg = (
        f"{datetime.utcnow().isoformat()}Z {env_mode} Signal "
        f"symbol={symbol_part} "
        f"composite={composite_str} "
        f"mom={momentum_score:.3f} "
        f"mr={mean_reversion_score:.3f} "
        f"pa={price_action_score:.3f}"
    )
    logger.info(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())


def log_proposed_trade(trade: ProposedTrade, env_mode: str) -> None:
    line_color = next_line_color()
    sentiment_part = sentiment_score_fragment(trade.sentiment_score, line_color)
    action_tag = ""
    if trade.rejected_reason is None and trade.qty > 0:
        if trade.side == "buy":
            action_tag = f" {SIGNALGREEN}BUY{line_color}"
        elif trade.side == "sell":
            action_tag = f" {SIGNALRED}SELL{line_color}"
    notional = trade.qty * trade.entry_price
    symbol_part = f"{BRIGHTPURPLE}{trade.symbol}{line_color}"
    sig_part = f"{BRIGHTPURPLE}{trade.signal_score:.3f}{line_color}"
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
    line_color = next_line_color()
    score_frag = sentiment_score_fragment(sentiment_score, line_color)
    expl_raw = (explanation or "").replace("\n", " ").strip()
    expl_fmt = italicize_technical(expl_raw)
    force_msg = f"{SIGNALRED}FORCE-CLOSED DUE TO BAD SENTIMENT{line_color}"
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


def log_portfolio_overview(trades: List[ProposedTrade], env_mode: str) -> None:
    logger.info(separator_line())
    if not trades:
        header = f"{datetime.utcnow().isoformat()}Z {env_mode} PORTFOLIO OVERVIEW"
        logger.info(f"{DEEPBLUE}{header}{RESET}")
        logger.info(f"{DEEPBLUE}No new trades selected in this cycle.{RESET}")
        logger.info(separator_line())
        return

    line_color = next_line_color()
    header = f"{datetime.utcnow().isoformat()}Z {env_mode} PORTFOLIO OVERVIEW"
    logger.info(f"{line_color}{header}{RESET}")
    for t in trades:
        notional = t.qty * t.entry_price
        symbol_part = f"{BRIGHTPURPLE}{t.symbol}{line_color}"
        sig_str = f"{BRIGHTPURPLE}{t.signal_score:.3f}{line_color}"
        msg = (
            f"  symbol={symbol_part} side={t.side} "
            f"qty={t.qty:.4f} notional={notional:.2f} "
            f"signal_score={sig_str} "
            f"rejected={t.rejected_reason}"
        )
        logger.info(f"{line_color}{msg}{RESET}")
    logger.info(separator_line())



















