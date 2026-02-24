# core/portfolio_veto.py
import json
import os
from typing import Dict, List

import requests

from core.risk_engine import ProposedTrade


class PortfolioVeto:
    """
    Optional portfolio-level veto layer using Perplexity Sonar.

    Given a list of ProposedTrade candidates (already risk-checked),
    prepares a compact prompt and expects a JSON mapping symbol -> 0/1,
    where 0 means veto (do not trade), 1 means allow.

    Controlled by config.portfolio.enable_portfolio_veto.
    """

    def __init__(self) -> None:
        self.apiurl = os.getenv("AI_API_URL", "https://api.perplexity.ai/chat/completions")
        self.apikey = os.getenv("AI_API_KEY")
        if not self.apikey:
            # Fail soft: without API key, veto layer is effectively a no-op
            self.disabled = True
        else:
            self.disabled = False

    def _build_prompt(self, trades: List[ProposedTrade]) -> str:
        lines = []
        for t in trades:
            notional = t.qty * t.entry_price
            line = {
                "symbol": t.symbol,
                "side": t.side,
                "notional": round(notional, 2),
                "signal_score": round(t.signal_score, 3),
                "sentiment_score": round(t.sentiment_score, 3),
                "reason": (t.rationale or "")[:200],
            }
            lines.append(line)

        return (
            "You are a risk-aware portfolio reviewer.\n"
            "You will receive a list of proposed short-term trades for US equities.\n"
            "Each trade includes symbol, side, notional size, a technical signal_score in [-1,1],\n"
            "and a sentiment_score in [-1,1].\n\n"
            "Your task: return a single JSON object mapping each symbol to 0 or 1.\n"
            "1 = trade is acceptable, 0 = veto due to extreme risk, obvious conflict, or\n"
            "major concern (e.g. very unstable situation, illogical rationale).\n"
            "Be conservative but do NOT overfit; use 0 only for clearly problematic trades.\n\n"
            "Input trades:\n"
            + json.dumps(lines, indent=2)
        )

    def apply_veto(self, trades: List[ProposedTrade]) -> List[ProposedTrade]:
        if self.disabled or not trades:
            return trades

        prompt = self._build_prompt(trades)
        headers = {
            "Authorization": f"Bearer {self.apikey}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise portfolio veto engine. "
                        "You only output strict JSON with {symbol: 0 or 1} entries."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        }

        try:
            resp = requests.post(self.apiurl, headers=headers, json=payload, timeout=30)
            if not resp.ok:
                print("PortfolioVeto API error", resp.status_code, resp.text)
                return trades

            data = resp.json()
            rawcontent = data["choices"][0]["message"]["content"].strip()
            try:
                result = json.loads(rawcontent)
            except json.JSONDecodeError:
                start = rawcontent.find("{")
                end = rawcontent.rfind("}")
                if start == -1 or end == -1:
                    return trades
                result = json.loads(rawcontent[start : end + 1])

            allowed: List[ProposedTrade] = []
            for t in trades:
                flag = result.get(t.symbol, 1)
                try:
                    flag_int = int(flag)
                except (TypeError, ValueError):
                    flag_int = 1
                if flag_int == 1:
                    allowed.append(t)
            return allowed
        except Exception as e:
            print("PortfolioVeto exception", e)
            return trades
