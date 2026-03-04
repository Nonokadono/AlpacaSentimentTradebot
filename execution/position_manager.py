# CHANGES:
# CRASH-2 FIX — Populate avg_entry_price in the PositionInfo constructor.
#
#   order_executor._check_and_exit_on_sentiment() reads pos.avg_entry_price to
#   compute unrealised P&L for PnL-coupled threshold scaling.  The field now
#   exists in PositionInfo (added in core/risk_engine.py), but it must also be
#   populated from the raw Alpaca position object here.
#
#   Added inside the per-position try block:
#     avg_entry_price: float = float(getattr(pos, "avg_entry_price", 0.0))
#
#   getattr with default 0.0 means:
#     - If the Alpaca SDK returns the field (as it does for real positions),
#       the true entry price is captured and PnL scaling works correctly.
#     - If the field is absent (e.g. mock/test objects), the value is 0.0,
#       which causes the `if pos.avg_entry_price > 0.0` guard in
#       order_executor.py to skip scaling — identical to legacy behaviour.
#
#   avg_entry_price is passed as a keyword argument to the PositionInfo
#   constructor so the field ordering in the dataclass is respected.
#
# All prior changes (opening_compound patching from opening_compounds dict)
# are preserved unchanged.

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
      - Populate avg_entry_price from the Alpaca position object so that
        PnL-coupled sentiment-exit threshold scaling has a valid denominator.
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

                # CRASH-2 FIX: read avg_entry_price from the raw Alpaca position
                # object.  getattr with default 0.0 is safe for mock/test objects
                # that omit this attribute; the guard in order_executor.py
                # (`if pos.avg_entry_price > 0.0`) treats 0.0 as "not available"
                # and skips PnL scaling, preserving legacy behaviour exactly.
                avg_entry_price: float = float(getattr(pos, "avg_entry_price", 0.0))

                positions[symbol] = PositionInfo(
                    symbol=symbol,
                    qty=qty,
                    market_price=market_price,
                    side=side,
                    notional=notional,
                    opening_compound=opening_compound,
                    avg_entry_price=avg_entry_price,
                )
            except Exception as e:
                logger.warning(f"PositionManager: failed to parse position {getattr(pos, 'symbol', '?')}: {e}")

        return positions
