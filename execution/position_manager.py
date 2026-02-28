# CHANGES:
# - No functional changes from baseline.
# - get_positions() accepts an optional opening_compounds dict and patches
#   each PositionInfo.opening_compound from it so the sentiment-exit delta
#   check in main._check_and_exit_on_sentiment() has a valid entry baseline.

import logging
from typing import Dict, Optional

from adapters.alpaca_adapter import AlpacaAdapter
from core.risk_engine import PositionInfo

logger = logging.getLogger("tradebot")


class PositionManager:
    """
    Maps raw Alpaca REST positions into typed PositionInfo objects.

    Responsibilities:
      - Fetch all open positions from the broker.
      - Convert raw Alpaca position objects into PositionInfo dataclasses.
      - Patch opening_compound from the in-memory registry (_opening_compounds)
        so downstream sentiment-exit logic has a valid entry-time baseline.
    """

    def __init__(self, adapter: AlpacaAdapter) -> None:
        self.adapter = adapter

    def get_positions(
        self,
        opening_compounds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, PositionInfo]:
        """
        Fetch and map all open Alpaca positions.

        Parameters
        ----------
        opening_compounds : dict, optional
            Registry of {symbol: entry_sentiment_score} maintained by main.
            When provided, each PositionInfo.opening_compound is populated
            so that _check_and_exit_on_sentiment() can compute a meaningful delta.

        Returns
        -------
        Dict[str, PositionInfo]
            Keyed by symbol string.
        """
        raw_positions = self.adapter.list_positions()
        positions: Dict[str, PositionInfo] = {}

        for pos in raw_positions:
            try:
                symbol: str = str(pos.symbol)
                qty: float = float(pos.qty)
                market_price: float = float(pos.current_price)
                side: str = str(pos.side)  # "long" or "short"
                notional: float = abs(qty * market_price)

                opening_compound: float = 0.0
                if opening_compounds is not None:
                    opening_compound = float(opening_compounds.get(symbol, 0.0))

                positions[symbol] = PositionInfo(
                    symbol=symbol,
                    qty=qty,
                    market_price=market_price,
                    side=side,
                    notional=notional,
                    opening_compound=opening_compound,
                )
            except Exception as e:
                logger.warning(f"PositionManager: failed to parse position {getattr(pos, 'symbol', '?')}: {e}")

        return positions
