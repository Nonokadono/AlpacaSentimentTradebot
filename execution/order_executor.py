# execution/order_executor.py

import logging

from alpaca_trade_api.rest import APIError
from adapters.alpaca_adapter import AlpacaAdapter
from core.risk_engine import ProposedTrade, PositionInfo
from core.sentiment import SentimentResult
from monitoring.monitor import (
    log_proposed_trade,
    log_sentiment_close_decision,
)

logger = logging.getLogger("tradebot")


class OrderExecutor:
    def __init__(self, adapter: AlpacaAdapter, env_mode: str, live_trading_enabled: bool) -> None:
        self.adapter = adapter
        self.env_mode = env_mode
        self.live_trading_enabled = live_trading_enabled

    def _can_place_orders(self) -> bool:
        """
        Returns True if it is allowed to place real orders in the current environment.
        PAPER: always True.
        LIVE: only if live_trading_enabled is True.
        """
        if self.env_mode == "LIVE" and not self.live_trading_enabled:
            return False
        return True

    def execute_proposed_trade(self, trade: ProposedTrade):
        # Log the proposed trade with environment mode
        log_proposed_trade(trade, self.env_mode)

        # Skip if rejected or zero qty
        if trade.rejected_reason is not None or trade.qty <= 0:
            return None

        # LIVE dry‑run protection
        if not self._can_place_orders():
            return None

        side = trade.side
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        # Round TP and SL to 2 decimals to avoid sub‑penny issues
        tp_price = round(trade.take_profit_price, 2)
        sl_price = round(trade.stop_price, 2)

        try:
            order = self.adapter.submit_bracket_order(
                symbol=trade.symbol,
                qty=trade.qty,
                side=side,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
                time_in_force="day",
            )
            return order
        except APIError as e:
            logger.error(f"Order placement failed for {trade.symbol}: {e}")
            return None

    def close_position_due_to_sentiment(
        self,
        position: PositionInfo,
        sentiment: SentimentResult,
        reason: str,
    ):
        """
        Close an existing position immediately because sentiment has flipped strongly against it.

        For a long position, we submit a sell market order of full qty.
        For a short position, we submit a buy market order of full qty.
        """
        log_sentiment_close_decision(
            symbol=position.symbol,
            side=position.side,
            qty=position.qty,
            sentiment_score=sentiment.score,
            confidence=sentiment.confidence,
            explanation=sentiment.explanation,
            env_mode=self.env_mode,
            reason=reason,
        )

        if not self._can_place_orders():
            return None

        if position.qty <= 0:
            return None

        if position.side == "long":
            side = "sell"
        elif position.side == "short":
            side = "buy"
        else:
            raise ValueError(f"Unexpected position side {position.side}")

        try:
            order = self.adapter.rest.submit_order(
                symbol=position.symbol,
                side=side,
                type="market",
                qty=abs(position.qty),
                time_in_force="day",
            )
            return order
        except APIError as e:
            logger.error(f"Close position failed for {position.symbol}: {e}")
            return None





