# CHANGES:
# FIX 1 — sector_counts pre-seeded from existing `positions` before the feasible
#   candidate selection loop begins.  Iterates over every symbol in `positions`,
#   looks up its InstrumentMeta, and increments sector_counts so that already-open
#   TECH (or any other sector) positions are visible to the per-sector cap check.
#   Without this, a bot restart with 3 open TECH positions would admit a 4th because
#   the counter started empty.
#
# FIX 2 — UNKNOWN-sector symbols (meta is None) are now exempt from the sector cap
#   check entirely.  Previously all missing-meta symbols shared one "UNKNOWN" bucket
#   and competed for the same cap slots, incorrectly blocking unrelated symbols.
#   Guard changed to:  if meta is not None and sector_counts.get(...) >= cap: continue
#   Sector counter is only incremented when meta is not None.
#
# All prior changes (Fix C4, Fix H4) are preserved unchanged.

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

        # FIX 1: Pre-seed sector_counts with sectors of already-open positions so
        # that existing holdings count against the per-sector cap from the start.
        # Without this, a bot restart with N open TECH positions would not
        # recognise them and admit further TECH candidates up to the cap.
        for sym in positions:
            meta = self.cfg.instruments.get(sym)
            if meta is not None:
                sector = meta.sector
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            # FIX 2: If meta is None we do NOT create an "UNKNOWN" entry here —
            # missing-meta symbols are exempt from the cap entirely (see selection
            # loop below), so pre-seeding "UNKNOWN" would be misleading.

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

            # Fix C4 / FIX 2: enforce max_positions_per_sector.
            # FIX 2: symbols whose InstrumentMeta is missing (meta is None) are
            # treated as uncapped — they do NOT share an "UNKNOWN" bucket that
            # would incorrectly block unrelated missing-meta symbols.
            meta = self.cfg.instruments.get(t.symbol)
            if meta is not None:
                sector = meta.sector
                if sector_counts.get(sector, 0) >= self.cfg.portfolio.max_positions_per_sector:
                    continue

            selected.append(t)
            current_gross = projected_gross
            if new_symbol:
                current_open_positions = projected_positions
            # Fix C4 / FIX 2: only increment when meta is not None.
            if meta is not None:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        if not selected:
            return []

        if self.cfg.portfolio.enable_portfolio_veto:
            selected = self.veto.apply_veto(selected)

        return selected
