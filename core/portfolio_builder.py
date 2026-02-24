# core/portfolio_builder.py

from typing import Dict, List

from adapters.alpaca_adapter import AlpacaAdapter
from core.signals import SignalEngine, Signal
from core.sentiment import SentimentModule
from core.risk_engine import RiskEngine, EquitySnapshot, PositionInfo, ProposedTrade
from config.config import BotConfig
from .portfolio_veto import PortfolioVeto


class PortfolioBuilder:
    """
    Portfolio-level builder:
    - Compute signal_score and side for each instrument.
    - Run pre_trade_checks -> ProposedTrade.
    - Rank by |signal_score| and select until portfolio limits are hit.
    - Optional Sonar veto.
    """

    def __init__(
        self,
        cfg: BotConfig,
        adapter: AlpacaAdapter,
        sentiment: SentimentModule,
        signal_engine: SignalEngine,
        risk_engine: RiskEngine,
    ) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.sentiment = sentiment
        self.signal_engine = signal_engine
        self.risk_engine = risk_engine
        self.veto = PortfolioVeto()

    def _build_candidate_for_symbol(
        self,
        symbol: str,
        snapshot: EquitySnapshot,
        positions: Dict[str, PositionInfo],
    ) -> ProposedTrade:
        sig: Signal = self.signal_engine.generate_signal_for_symbol(symbol)

        if sig.side == "skip":
            return ProposedTrade(
                symbol=sig.symbol,
                side="buy",
                qty=0.0,
                entry_price=0.0,
                stop_price=0.0,
                take_profit_price=0.0,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                sentiment_score=sig.sentiment_result.score,
                sentiment_scale=0.0,
                signal_score=sig.signal_score,
                rationale=sig.rationale,
                rejected_reason="Signal neutral/skip",
            )

        entry_price = self.adapter.get_last_quote(symbol)

        proposed = self.risk_engine.pre_trade_checks(
            snapshot=snapshot,
            positions=positions,
            symbol=sig.symbol,
            side=sig.side,
            entry_price=entry_price,
            stop_price=sig.stop_price,
            take_profit_price=sig.take_profit_price,
            sentiment=sig.sentiment_result,
            signal_score=sig.signal_score,
            rationale=sig.rationale,
        )
        return proposed

    def build_portfolio(
        self,
        snapshot: EquitySnapshot,
        positions: Dict[str, PositionInfo],
    ) -> List[ProposedTrade]:
        symbols = list(self.cfg.instruments.keys())
        max_candidates = self.cfg.portfolio.max_candidates_per_loop
        if max_candidates > 0:
            symbols = symbols[:max_candidates]

        candidates: List[ProposedTrade] = []
        for sym in symbols:
            candidate = self._build_candidate_for_symbol(sym, snapshot, positions)
            candidates.append(candidate)

        feasible: List[ProposedTrade] = [
            t for t in candidates if t.qty > 0 and t.rejected_reason is None
        ]
        if not feasible:
            return []

        feasible.sort(key=lambda t: abs(t.signal_score), reverse=True)

        selected: List[ProposedTrade] = []
        current_gross = snapshot.gross_exposure
        current_open_positions = len(positions)

        for t in feasible:
            notional = t.qty * t.entry_price
            projected_gross = current_gross + abs(notional)

            if projected_gross > snapshot.equity * self.cfg.risk_limits.gross_exposure_cap_pct:
                continue

            new_symbol = t.symbol not in positions
            projected_positions = current_open_positions + (1 if new_symbol else 0)
            if projected_positions > self.cfg.risk_limits.max_open_positions:
                continue

            selected.append(t)
            current_gross = projected_gross
            if new_symbol:
                current_open_positions = projected_positions

        if not selected:
            return []

        if self.cfg.portfolio.enable_portfolio_veto:
            selected = self.veto.apply_veto(selected)

        return selected

