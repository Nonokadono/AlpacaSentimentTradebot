# CHANGES:
# MONITOR-FIX — PositionManager now queries open orders (`list_orders(status="open")`) alongside
#               positions to retrieve the active trailing-stop, stop, and take-profit limits.
#               These are mapped to `stop_price` and `take_profit_price` on PositionInfo.
# CRASH-2 FIX — Populate avg_entry_price in the PositionInfo constructor.
# CONFIDENCE-STORE — opening_compound now represents a tuple (compound, confidence) rather than
#                    a scalar. PositionInfo.opening_compound remains a scalar for backward
#                    compatibility, but the registry (Dict[str, tuple]) stores both values.
#                    The confidence value is reserved for future delta-exit threshold scaling.
#                    get_positions() unpacks the tuple and stores only the compound in the
#                    dataclass; the confidence is preserved in the main registry.

import logging
from typing import Dict, Optional

from adapters.ibkr_adapter import IbkrAdapter
from core.risk_engine import PositionInfo

logger = logging.getLogger("tradebot")


class PositionManager:
    """
    Maps raw IBKR REST positions into typed PositionInfo objects.

    Responsibilities:
      - Fetch all open positions from the broker.
      - Convert raw IBKR position objects into PositionInfo dataclasses.
      - Patch opening_compound from the in-memory registry (_opening_compounds)
        so downstream sentiment-exit logic has a valid entry-time baseline.
      - Populate avg_entry_price from the IBKR position object so that
        PnL-coupled sentiment-exit threshold scaling has a valid denominator.
      - MONITOR-FIX: Fetch open orders to bind current TP/SL prices to positions.
      - CONFIDENCE-STORE: Unpack (compound, confidence) tuples from the registry.
    """

    def __init__(self, adapter: IbkrAdapter) -> None:
        self.adapter = adapter

    def get_positions(
        self,
        opening_compounds: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, PositionInfo]:
        """
        Fetch and map all open IBKR positions.

        Parameters
        ----------
        opening_compounds : dict, optional
            Registry of {symbol: (entry_sentiment_score, confidence)} maintained
            by main. When provided, each PositionInfo.opening_compound is populated
            with the compound score (the confidence is stored separately and not
            passed to the dataclass).

        Returns
        -------
        Dict[str, PositionInfo]
            Keyed by symbol string.
        """
        raw_positions = self.adapter.list_positions()
        
        # MONITOR-FIX: Fetch open orders to map stop and limit prices to positions
        open_orders = self.adapter.list_orders(status="open")
        orders_by_symbol = {}
        for order in open_orders:
            sym = getattr(order, "symbol", None)
            if sym:
                if sym not in orders_by_symbol:
                    orders_by_symbol[sym] = []
                orders_by_symbol[sym].append(order)

        positions: Dict[str, PositionInfo] = {}

        for pos in raw_positions:
            try:
                symbol: str = str(pos.symbol)
                qty: float = float(pos.qty)
                market_price: float = float(pos.current_price)
                side: str = str(pos.side)  # "long" or "short"
                notional: float = abs(qty * market_price)

                # CONFIDENCE-STORE: unpack tuple; fallback to (0.0, 0.0) if missing
                opening_compound: float = 0.0
                if opening_compounds is not None:
                    opening_data = opening_compounds.get(symbol, (0.0, 0.0))
                    if isinstance(opening_data, tuple):
                        opening_compound = float(opening_data[0])
                    else:
                        # Legacy scalar — treat as compound only
                        opening_compound = float(opening_data)

                # CRASH-2 FIX: read avg_entry_price from the raw IBKR position
                # object. getattr with default 0.0 is safe for mock/test objects
                # that omit this attribute; the guard in order_executor.py
                # (`if pos.avg_entry_price > 0.0`) treats 0.0 as "not available"
                # and skips PnL scaling, preserving legacy behaviour exactly.
                avg_entry_price: float = float(getattr(pos, "avg_entry_price", 0.0))

                # MONITOR-FIX: Look up active exit legs for this position
                current_stop: Optional[float] = None
                current_tp: Optional[float] = None
                for o in orders_by_symbol.get(symbol, []):
                    o_type = getattr(o, "type", "")
                    if o_type in ("stop", "stop_limit", "trailing_stop"):
                        sp = getattr(o, "stop_price", None)
                        if sp is not None:
                            current_stop = float(sp)
                    elif o_type == "limit":
                        lp = getattr(o, "limit_price", None)
                        if lp is not None:
                            current_tp = float(lp)

                positions[symbol] = PositionInfo(
                    symbol=symbol,
                    qty=qty,
                    market_price=market_price,
                    side=side,
                    notional=notional,
                    opening_compound=opening_compound,
                    avg_entry_price=avg_entry_price,
                    stop_price=current_stop,
                    take_profit_price=current_tp,
                )
            except Exception as e:
                logger.warning(f"PositionManager: failed to parse position {getattr(pos, 'symbol', '?')}: {e}")

        return positions
