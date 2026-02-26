from typing import Dict, Optional
from adapters.alpaca_adapter import AlpacaAdapter
from core.risk_engine import PositionInfo


class PositionManager:
    def __init__(self, adapter: AlpacaAdapter):
        self.adapter = adapter

    def get_positions(
        self,
        opening_compounds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, PositionInfo]:
        """
        Returns the current open positions as a symbol-keyed dict.

        opening_compounds: optional mapping of symbol -> entry-time rawcompound.
        When supplied, each PositionInfo.opening_compound is populated from it
        so the sentiment-exit loop in main.py can compare entry vs. current
        compound even after the SentimentModule cache has been refreshed.
        """
        alpaca_positions = self.adapter.list_positions()
        compounds = opening_compounds or {}
        out: Dict[str, PositionInfo] = {}
        for p in alpaca_positions:
            qty = float(p.qty)
            price = float(p.current_price)
            notional = qty * price
            side = "long" if qty > 0 else "short"
            out[p.symbol] = PositionInfo(
                symbol=p.symbol,
                qty=qty,
                market_price=price,
                side=side,
                notional=notional,
                opening_compound=compounds.get(p.symbol, 0.0),
            )
        return out
