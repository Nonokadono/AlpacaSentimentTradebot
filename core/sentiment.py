# core/sentiment.py
from dataclasses import dataclass
from typing import List, Optional
from ai_client import NewsReasoner



@dataclass
class SentimentResult:
    """
    score: continuous sentiment score in [-1, 1] for risk sizing.
           -1 strongly negative, +1 strongly positive.
           For discrete -2 (utterly undesirable / unstable), score is fixed at -1
           and should trigger no-trade / forced exit logic at the risk engine level.
    raw_discrete: the raw discrete value from the model in {-2, -1, 0, 1}
    rawcompound: legacy field; kept for compatibility, here = score
    ndocuments: number of news items used
    explanation: optional short explanation
    confidence: model-reported confidence in [0, 1]
    """
    score: float
    raw_discrete: int
    rawcompound: float
    ndocuments: int
    explanation: Optional[str] = None
    confidence: float = 0.0


class SentimentModule:
    """
    Sentiment engine backed by Perplexity Sonar via NewsReasoner.

    It expects a list of newsitems-style dicts and uses Sonar to produce a discrete
    sentiment in {-2, -1, 0, 1} with confidence in [0, 1]. That is then mapped into
    a continuous score in [-1, 1] for the risk engine.
    """

    def __init__(self) -> None:
        self.reasoner = NewsReasoner()

    def _map_discrete_to_score(self, sdisc: int, confidence: float) -> float:
        """
        Map discrete sentiment {-2, -1, 0, 1} plus confidence into a continuous
        score in [-1, 1].

        Rules:
            -2 : treat as 'do not trade / extremely bad / unstable' => score = -1.0
                 (risk engine should then enforce zero size or forced close).
            -1 : clearly negative      => base -1, scaled by confidence
            0  : neutral or mixed      => base 0, scaled by confidence (≈ 0)
            1  : clearly positive      => base +1, scaled by confidence
        """
        confidence = max(0.0, min(1.0, confidence))

        if sdisc == -2:
            # Hard floor at -1 to signal "utterly undesirable / unstable".
            return -1.0

        if sdisc == -1:
            base = -1.0
        elif sdisc == 0:
            base = 0.0
        elif sdisc == 1:
            base = 1.0
        else:
            base = 0.0

        return max(-1.0, min(1.0, base * confidence))

    def scorenewsitems(self, symbol: str, newsitems: List[dict]) -> SentimentResult:
        """
        newsitems: list of dicts typically from Alpaca's news API,
                   each with at least 'headline' and/or 'summary'.
        """
        res = self.reasoner.scorenews(symbol, newsitems)

        sdisc = int(res.get("sentiment", 0))
        if sdisc not in (-2, -1, 0, 1):
            sdisc = 0

        try:
            confidence = float(res.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        explanation = res.get("explanation", "")

        score = self._map_discrete_to_score(sdisc, confidence)

        return SentimentResult(
            score=score,
            raw_discrete=sdisc,
            rawcompound=score,
            ndocuments=len(newsitems),
            explanation=explanation,
            confidence=confidence,
        )


