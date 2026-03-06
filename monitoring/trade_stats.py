# CHANGES:
# - Added monitoring/trade_stats.py, a standalone trade analytics module for
#   recording closed trades, tracking counterfactual post-stop outcomes, and
#   rendering human-readable summaries plus an HTML report.
# - Supports debugging stop-loss / take-profit settings by showing whether a
#   stopped trade would have become profitable without the stop and by how much.
# - Uses only the Python standard library so it can run in the current repo
#   without adding dependencies.

from __future__ import annotations

import argparse
import json
import statistics
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

TRADE_STATS_PATH = Path("data/trade_stats.jsonl")
TRADE_STATS_HTML_PATH = Path("data/trade_stats_report.html")


@dataclass
class TradeStat:
    trade_id: str
    symbol: str
    side: str
    qty: float
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    stop_price: float
    take_profit_price: float
    exit_reason: str
    bars_held: int
    realized_pnl: float
    realized_return_pct: float
    max_favorable_price: float
    max_adverse_price: float
    max_favorable_pnl: float
    max_adverse_pnl: float
    hit_stop_loss: bool
    hit_take_profit: bool
    no_stop_last_price: float
    no_stop_best_price: float
    no_stop_worst_price: float
    no_stop_best_pnl: float
    no_stop_worst_pnl: float
    no_stop_would_be_profitable: bool
    no_stop_profit_delta_vs_realized: float
    notes: str


@dataclass
class SummaryStats:
    total_trades: int
    winners: int
    losers: int
    hit_rate: float
    avg_pnl: float
    avg_return_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_stop_distance_pct: float
    stop_exits: int
    stop_exits_that_would_have_been_profitable: int
    stop_recovery_rate: float
    avg_counterfactual_best_pnl: float


class TradeStatsTracker:
    def __init__(self, path: Path = TRADE_STATS_PATH) -> None:
        self.path = path

    def record_trade(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        entry_ts: Optional[str] = None,
        exit_ts: Optional[str] = None,
        stop_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        exit_reason: str = "unknown",
        bars_held: int = 0,
        max_favorable_price: Optional[float] = None,
        max_adverse_price: Optional[float] = None,
        hit_stop_loss: bool = False,
        hit_take_profit: bool = False,
        notes: str = "",
    ) -> TradeStat:
        side_clean = side.lower().strip()
        if side_clean not in {"long", "short", "buy", "sell"}:
            raise ValueError(f"Unsupported side: {side}")
        normalized_side = "long" if side_clean in {"long", "buy"} else "short"

        entry_ts = entry_ts or _utc_now_iso()
        exit_ts = exit_ts or _utc_now_iso()
        stop_price = float(stop_price or 0.0)
        take_profit_price = float(take_profit_price or 0.0)
        qty = float(qty)
        entry_price = float(entry_price)
        exit_price = float(exit_price)
        max_favorable_price = float(max_favorable_price if max_favorable_price is not None else exit_price)
        max_adverse_price = float(max_adverse_price if max_adverse_price is not None else exit_price)

        realized_pnl = _pnl(normalized_side, qty, entry_price, exit_price)
        realized_return_pct = _return_pct(normalized_side, entry_price, exit_price)
        max_favorable_pnl = _pnl(normalized_side, qty, entry_price, max_favorable_price)
        max_adverse_pnl = _pnl(normalized_side, qty, entry_price, max_adverse_price)

        trade = TradeStat(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            side=normalized_side,
            qty=qty,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            exit_reason=exit_reason,
            bars_held=bars_held,
            realized_pnl=realized_pnl,
            realized_return_pct=realized_return_pct,
            max_favorable_price=max_favorable_price,
            max_adverse_price=max_adverse_price,
            max_favorable_pnl=max_favorable_pnl,
            max_adverse_pnl=max_adverse_pnl,
            hit_stop_loss=bool(hit_stop_loss),
            hit_take_profit=bool(hit_take_profit),
            no_stop_last_price=exit_price,
            no_stop_best_price=exit_price,
            no_stop_worst_price=exit_price,
            no_stop_best_pnl=realized_pnl,
            no_stop_worst_pnl=max_adverse_pnl,
            no_stop_would_be_profitable=realized_pnl > 0.0,
            no_stop_profit_delta_vs_realized=0.0,
            notes=notes,
        )
        self._append(trade)
        return trade

    def update_counterfactual_path(
        self,
        trade_id: str,
        observed_prices: Sequence[float],
    ) -> Optional[TradeStat]:
        trades = self.load_trades()
        updated: Optional[TradeStat] = None

        for idx, trade in enumerate(trades):
            if trade.trade_id != trade_id:
                continue
            if not observed_prices:
                updated = trade
                break

            best_price = _best_price_for_side(trade.side, observed_prices)
            worst_price = _worst_price_for_side(trade.side, observed_prices)
            last_price = float(observed_prices[-1])
            best_pnl = _pnl(trade.side, trade.qty, trade.entry_price, best_price)
            worst_pnl = _pnl(trade.side, trade.qty, trade.entry_price, worst_price)
            would_be_profitable = best_pnl > 0.0

            updated = TradeStat(
                **{
                    **asdict(trade),
                    "no_stop_last_price": last_price,
                    "no_stop_best_price": best_price,
                    "no_stop_worst_price": worst_price,
                    "no_stop_best_pnl": best_pnl,
                    "no_stop_worst_pnl": worst_pnl,
                    "no_stop_would_be_profitable": would_be_profitable,
                    "no_stop_profit_delta_vs_realized": best_pnl - trade.realized_pnl,
                }
            )
            trades[idx] = updated
            break

        if updated is not None:
            self._rewrite(trades)
        return updated

    def load_trades(self) -> List[TradeStat]:
        if not self.path.exists():
            return []

        known_fields = {f.name for f in fields(TradeStat)}
        trades: List[TradeStat] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                filtered = {k: v for k, v in raw.items() if k in known_fields}
                trades.append(TradeStat(**filtered))
        return trades

    def summary(self) -> SummaryStats:
        trades = self.load_trades()
        return _summarize(trades)

    def render_text_report(self, limit: int = 15) -> str:
        trades = self.load_trades()
        summary = _summarize(trades)
        lines = [
            "Trade Stats Report",
            "==================",
            f"Trades: {summary.total_trades}",
            f"Hit rate: {summary.hit_rate * 100:.1f}%",
            f"Avg P&L: {summary.avg_pnl:+.2f}",
            f"Avg return: {summary.avg_return_pct * 100:+.2f}%",
            f"Avg win / loss: {summary.avg_win:+.2f} / {summary.avg_loss:+.2f}",
            f"Profit factor: {summary.profit_factor:.2f}",
            f"Stop recovery rate: {summary.stop_recovery_rate * 100:.1f}%",
            "",
            "Recent trades:",
        ]
        for trade in trades[-limit:]:
            lines.append(
                f"- {trade.symbol} {trade.side} pnl={trade.realized_pnl:+.2f} "
                f"exit={trade.exit_reason} stop={trade.stop_price:.2f} "
                f"no_stop_best={trade.no_stop_best_pnl:+.2f} "
                f"recover={trade.no_stop_would_be_profitable}"
            )
        return "\n".join(lines)

    def render_html_report(self, output_path: Path = TRADE_STATS_HTML_PATH) -> Path:
        trades = self.load_trades()
        summary = _summarize(trades)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(summary, trades)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _append(self, trade: TradeStat) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trade), ensure_ascii=False) + "\n")

    def _rewrite(self, trades: Iterable[TradeStat]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for trade in trades:
                handle.write(json.dumps(asdict(trade), ensure_ascii=False) + "\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pnl(side: str, qty: float, entry_price: float, exit_price: float) -> float:
    if side == "long":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty



def _return_pct(side: str, entry_price: float, exit_price: float) -> float:
    if entry_price <= 0.0:
        return 0.0
    if side == "long":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price



def _best_price_for_side(side: str, prices: Sequence[float]) -> float:
    return float(max(prices) if side == "long" else min(prices))



def _worst_price_for_side(side: str, prices: Sequence[float]) -> float:
    return float(min(prices) if side == "long" else max(prices))



def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0



def _summarize(trades: Sequence[TradeStat]) -> SummaryStats:
    winners = [t for t in trades if t.realized_pnl > 0.0]
    losers = [t for t in trades if t.realized_pnl <= 0.0]
    stop_exits = [t for t in trades if t.hit_stop_loss or t.exit_reason.lower() == "stop_loss"]
    recovered = [t for t in stop_exits if t.no_stop_would_be_profitable]
    gross_profit = sum(t.realized_pnl for t in winners)
    gross_loss = abs(sum(t.realized_pnl for t in losers))

    stop_distances = []
    for trade in trades:
        if trade.entry_price > 0.0 and trade.stop_price > 0.0:
            stop_distances.append(abs(trade.entry_price - trade.stop_price) / trade.entry_price)

    return SummaryStats(
        total_trades=len(trades),
        winners=len(winners),
        losers=len(losers),
        hit_rate=(len(winners) / len(trades)) if trades else 0.0,
        avg_pnl=_mean([t.realized_pnl for t in trades]),
        avg_return_pct=_mean([t.realized_return_pct for t in trades]),
        avg_win=_mean([t.realized_pnl for t in winners]),
        avg_loss=_mean([t.realized_pnl for t in losers]),
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0.0 else 0.0,
        avg_stop_distance_pct=_mean(stop_distances),
        stop_exits=len(stop_exits),
        stop_exits_that_would_have_been_profitable=len(recovered),
        stop_recovery_rate=(len(recovered) / len(stop_exits)) if stop_exits else 0.0,
        avg_counterfactual_best_pnl=_mean([t.no_stop_best_pnl for t in stop_exits]),
    )



def _metric_card(title: str, value: str) -> str:
    return (
        '<div class="card">'
        f'<div class="card-title">{escape(title)}</div>'
        f'<div class="card-value">{escape(value)}</div>'
        '</div>'
    )



def _bar(label: str, value: float, good: bool) -> str:
    pct = max(0.0, min(100.0, value * 100.0))
    color = "#16a34a" if good else "#dc2626"
    return (
        '<div class="bar-row">'
        f'<div class="bar-label">{escape(label)}</div>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div>'
        '</div>'
        f'<div class="bar-value">{pct:.1f}%</div>'
        '</div>'
    )



def _format_money(value: float) -> str:
    return f"{value:+,.2f}"



def _build_html(summary: SummaryStats, trades: Sequence[TradeStat]) -> str:
    stopped = [t for t in trades if t.hit_stop_loss or t.exit_reason.lower() == "stop_loss"]
    recent_rows = []
    for trade in reversed(trades[-30:]):
        recent_rows.append(
            "<tr>"
            f"<td>{escape(trade.symbol)}</td>"
            f"<td>{escape(trade.side)}</td>"
            f"<td>{escape(trade.exit_reason)}</td>"
            f"<td>{trade.stop_price:.2f}</td>"
            f"<td>{trade.take_profit_price:.2f}</td>"
            f"<td>{_format_money(trade.realized_pnl)}</td>"
            f"<td>{_format_money(trade.no_stop_best_pnl)}</td>"
            f"<td>{'YES' if trade.no_stop_would_be_profitable else 'NO'}</td>"
            "</tr>"
        )

    stopped_rows = []
    for trade in reversed(stopped[-30:]):
        stopped_rows.append(
            "<tr>"
            f"<td>{escape(trade.symbol)}</td>"
            f"<td>{trade.entry_price:.2f}</td>"
            f"<td>{trade.stop_price:.2f}</td>"
            f"<td>{trade.exit_price:.2f}</td>"
            f"<td>{trade.no_stop_best_price:.2f}</td>"
            f"<td>{_format_money(trade.realized_pnl)}</td>"
            f"<td>{_format_money(trade.no_stop_best_pnl)}</td>"
            f"<td>{_format_money(trade.no_stop_profit_delta_vs_realized)}</td>"
            f"<td>{'YES' if trade.no_stop_would_be_profitable else 'NO'}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Trade Stats Report</title>
<style>
body {{ font-family: Arial, sans-serif; background:#0f172a; color:#e2e8f0; margin:0; padding:24px; }}
h1 {{ margin:0 0 18px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:24px; }}
.card {{ background:#111827; border:1px solid #1f2937; border-radius:10px; padding:14px; }}
.card-title {{ color:#94a3b8; font-size:13px; margin-bottom:6px; }}
.card-value {{ font-size:24px; font-weight:700; }}
.section {{ background:#111827; border:1px solid #1f2937; border-radius:10px; padding:16px; margin-bottom:20px; }}
.bar-row {{ display:grid; grid-template-columns:180px 1fr 70px; gap:10px; align-items:center; margin:10px 0; }}
.bar-track {{ background:#1f2937; border-radius:999px; overflow:hidden; height:12px; }}
.bar-fill {{ height:12px; border-radius:999px; }}
table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:14px; }}
th, td {{ border-bottom:1px solid #1f2937; padding:8px; text-align:left; }}
th {{ color:#93c5fd; }}
.good {{ color:#22c55e; }}
.bad {{ color:#ef4444; }}
.muted {{ color:#94a3b8; }}
</style>
</head>
<body>
<h1>Trade Stats Report</h1>
<div class=\"grid\">
{_metric_card('Total trades', str(summary.total_trades))}
{_metric_card('Hit rate', f'{summary.hit_rate * 100:.1f}%')}
{_metric_card('Average P&L', _format_money(summary.avg_pnl))}
{_metric_card('Average return', f'{summary.avg_return_pct * 100:+.2f}%')}
{_metric_card('Profit factor', f'{summary.profit_factor:.2f}')}
{_metric_card('Avg stop distance', f'{summary.avg_stop_distance_pct * 100:.2f}%')}
{_metric_card('Stop exits', str(summary.stop_exits))}
{_metric_card('Stop recovery rate', f'{summary.stop_recovery_rate * 100:.1f}%')}
</div>
<div class=\"section\">
<h2>Quick debug view</h2>
<p class=\"muted\">Use stop recovery rate and no-stop best P&amp;L to see whether your stops are too tight.</p>
{_bar('Hit rate', summary.hit_rate, True)}
{_bar('Stop recovery rate', summary.stop_recovery_rate, True)}
</div>
<div class=\"section\">
<h2>Recent trades</h2>
<table>
<thead>
<tr><th>Symbol</th><th>Side</th><th>Exit reason</th><th>Stop</th><th>Take profit</th><th>Realized P&amp;L</th><th>No-stop best P&amp;L</th><th>Would recover?</th></tr>
</thead>
<tbody>
{''.join(recent_rows) or '<tr><td colspan="8" class="muted">No trades recorded yet.</td></tr>'}
</tbody>
</table>
</div>
<div class=\"section\">
<h2>Stopped trades that might deserve wider stops</h2>
<table>
<thead>
<tr><th>Symbol</th><th>Entry</th><th>Stop</th><th>Exit</th><th>Best later price</th><th>Realized P&amp;L</th><th>No-stop best P&amp;L</th><th>Delta</th><th>Would recover?</th></tr>
</thead>
<tbody>
{''.join(stopped_rows) or '<tr><td colspan="9" class="muted">No stop-loss exits recorded yet.</td></tr>'}
</tbody>
</table>
</div>
</body>
</html>
"""



def main() -> int:
    parser = argparse.ArgumentParser(description="Trade stats recorder and report generator")
    parser.add_argument("--html", action="store_true", help="Render HTML report")
    parser.add_argument("--text", action="store_true", help="Print text summary")
    args = parser.parse_args()

    tracker = TradeStatsTracker()

    if args.html:
        path = tracker.render_html_report()
        print(path)
        return 0
    if args.text or (not args.html and not args.text):
        print(tracker.render_text_report())
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
