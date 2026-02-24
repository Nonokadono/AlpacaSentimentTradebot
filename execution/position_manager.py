from typing import Dict
from adapters.alpaca_adapter import AlpacaAdapter
from core.risk_engine import PositionInfo


class PositionManager:
    def __init__(self, adapter: AlpacaAdapter):
        self.adapter = adapter

    def get_positions(self) -> Dict[str, PositionInfo]:
        alpaca_positions = self.adapter.list_positions()
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
            )
        return out
