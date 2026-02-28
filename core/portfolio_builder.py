# CHANGES:
# Fix C4 — Enforce max_positions_per_sector in build_portfolio() selection loop.
#   sector_counts: Dict[str, int] is initialised to {} before the for t in
#   feasible loop.  Before appending t to selected, the symbol's sector is
#   looked up from self.cfg.instruments (defaulting to "UNKNOWN" if missing).
#   If sector_counts.get(sector, 0) >= self.cfg.portfolio.max_positions_per_sector,
#   the candidate is skipped via continue.  Otherwise sector_counts[sector] is
#   incremented.  No variable renames.
# Fix H4 — Added volatility=sig.volatility as a named keyword argument to the
#   self.risk_engine.pre_trade_checks() call in _build_candidate_for_symbol().
#   Previously this was omitted, so the Kelly path always received volatility=0.0
#   (the default), making the adaptive vol_norm history useless for new trades.

from typing import Dict, List, Any

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
        pending_symbols: set,
    ) -> ProposedTrade:

        # --- DUPLICATE-ORDER & NO EQUITY GUARD ---
        # If we already hold a position in this symbol, skip immediately (no AI API calls).
        if symbol in positions:
            return ProposedTrade(
                symbol=symbol,
                side="buy",
                qty=0.0,
                entry_price=0.0,
                stop_price=0.0,
                take_profit_price=0.0,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                sentiment_score=0.0,
                sentiment_scale=0.0,
                signal_score=0.0,
                rationale="Position already open for this symbol",
                rejected_reason="Position already open; skipping to prevent duplicate order and save API calls",
            )

        # If an order is already pending, skip immediately (no AI API calls).
        if symbol in pending_symbols:
            return ProposedTrade(
                symbol=symbol,
                side="buy",
                qty=0.0,
                entry_price=0.0,
                stop_price=0.0,
                take_profit_price=0.0,
                risk_amount=0.0,
                risk_pct_of_equity=0.0,
                sentiment_score=0.0,
                sentiment_scale=0.0,
                signal_score=0.0,
                rationale="Pending order exists for this symbol",
                rejected_reason="Pending order exists; skipping to prevent duplicate order and save API calls",
            )

        # Signal Engine is queried AFTER validation, saving an AI API call if conditions are unmet.
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
            volatility=sig.volatility,  # Fix H4: was previously omitted
        )
        return proposed

    def build_portfolio(
        self,
        snapshot: EquitySnapshot,
        positions: Dict[str, PositionInfo],
        open_orders: List[Any],
    ) -> List[ProposedTrade]:

        # Defense-in-depth: if fully allocated globally, skip all portfolio generation.
        exposure_cap_notional = snapshot.equity * self.cfg.risk_limits.gross_exposure_cap_pct
        if snapshot.gross_exposure >= exposure_cap_notional or len(positions) >= self.cfg.risk_limits.max_open_positions:
            return []

        pending_symbols = {getattr(o, "symbol", "") for o in open_orders if hasattr(o, "symbol")}

        symbols = list(self.cfg.instruments.keys())
        max_candidates = self.cfg.portfolio.max_candidates_per_loop
        if max_candidates > 0:
            symbols = symbols[:max_candidates]

        candidates: List[ProposedTrade] = []
        for sym in symbols:
            candidate = self._build_candidate_for_symbol(sym, snapshot, positions, pending_symbols)
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
        # Fix C4: per-sector position counter initialised before the loop.
        sector_counts: Dict[str, int] = {}

        for t in feasible:
            # Secondary guard (defensive): never select a symbol already in positions.
            if t.symbol in positions:
                continue

            notional = t.qty * t.entry_price
            projected_gross = current_gross + abs(notional)

            if projected_gross > snapshot.equity * self.cfg.risk_limits.gross_exposure_cap_pct:
                continue

            new_symbol = t.symbol not in positions
            projected_positions = current_open_positions + (1 if new_symbol else 0)
            if projected_positions > self.cfg.risk_limits.max_open_positions:
                continue

            # Fix C4: enforce max_positions_per_sector.
            meta = self.cfg.instruments.get(t.symbol)
            sector = meta.sector if meta is not None else "UNKNOWN"
            if sector_counts.get(sector, 0) >= self.cfg.portfolio.max_positions_per_sector:
                continue

            selected.append(t)
            current_gross = projected_gross
            if new_symbol:
                current_open_positions = projected_positions
            # Fix C4: increment sector counter after selection.
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        if not selected:
            return []

        if self.cfg.portfolio.enable_portfolio_veto:
            selected = self.veto.apply_veto(selected)

        return selected
