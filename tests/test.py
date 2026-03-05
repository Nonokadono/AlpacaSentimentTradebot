import logging
from dataclasses import dataclass

# 1. Setup minimal logging to see the output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tradebot")

# 2. Mock the PositionInfo and SentimentResult objects 
@dataclass
class PositionInfo:
    symbol: str
    side: str
    qty: float

@dataclass
class SentimentResult:
    score: float
    confidence: float
    explanation: str
    raw_discrete: int

# 3. Mock the function signature exactly as it appears in monitoring/monitor.py
def log_sentiment_close_decision(
    symbol: str,
    side: str,
    qty: float,
    sentiment_score: float,
    confidence: float,
    explanation: str,
    env_mode: str,
    reason: str,
) -> None:
    # Simplified print block simulating your custom log formatting
    logger.warning(
        f"SentimentExit: symbol={symbol} | side={side} | qty={qty:.4f} | "
        f"score={sentiment_score:.3f} | conf={confidence:.2f} | "
        f"reason={reason} | explanation='{explanation}'"
    )

if __name__ == "__main__":
    # Simulate a position and a chaos sentiment result
    mock_position = PositionInfo(symbol="AAPL", side="long", qty=10.0)
    mock_sentiment = SentimentResult(
        score=-0.85, 
        confidence=0.92, 
        explanation="Negative earnings report and unexpected CEO departure.", 
        raw_discrete=-2
    )
    
    print("--- Executing the CORRECTED call ---")
    try:
        # The corrected arguments passed to the function
        log_sentiment_close_decision(
            symbol=mock_position.symbol,
            side=mock_position.side,
            qty=mock_position.qty,
            sentiment_score=mock_sentiment.score,
            confidence=mock_sentiment.confidence,
            explanation=mock_sentiment.explanation,
            env_mode="LIVE",
            reason="hard_exit_chaos",
        )
        print("\nSuccess! The arguments map correctly without throwing a TypeError.")
    except Exception as e:
        print(f"\nFailed! Exception raised: {type(e).__name__}: {e}")
