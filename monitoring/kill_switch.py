# CHANGES:
# - No functional changes from baseline.
# - KillSwitchState dataclass and KillSwitch.check() are fully typed.
# - Integrates with EquitySnapshot fields daily_loss_pct and drawdown_pct
#   already computed by get_equity_snapshot_from_account() in main.py.

import logging
from dataclasses import dataclass

from core.risk_engine import EquitySnapshot
from config.config import RiskLimits

logger = logging.getLogger("tradebot")


@dataclass
class KillSwitchState:
    """
    Result of a kill-switch evaluation.

    halted : bool
        True if trading should be suspended this loop iteration.
    reason : str
        Human-readable reason for the halt, or empty string if not halted.
    daily_loss_pct : float
        Observed daily P&L percentage (negative = loss).
    drawdown_pct : float
        Observed drawdown from high-watermark (negative = drawdown).
    """
    halted: bool
    reason: str
    daily_loss_pct: float
    drawdown_pct: float


class KillSwitch:
    """
    Hard-halt mechanism that suspends new trade entry when either:
      - The daily P&L loss exceeds daily_loss_limit_pct of start-of-day equity, OR
      - The portfolio drawdown from high-watermark exceeds max_drawdown_pct.

    Does NOT close existing positions — that is the responsibility of the
    sentiment-exit layer and the broker's bracket/stop orders.

    Thresholds are sourced directly from RiskLimits to stay in sync with
    RiskEngine.pre_trade_checks() guard logic.
    """

    def __init__(self, risk_limits: RiskLimits) -> None:
        self.limits = risk_limits

    def check(self, snapshot: EquitySnapshot) -> KillSwitchState:
        """
        Evaluate kill-switch conditions against the current equity snapshot.

        Parameters
        ----------
        snapshot : EquitySnapshot
            Current account state as computed by get_equity_snapshot_from_account().

        Returns
        -------
        KillSwitchState
        """
        daily_loss_pct = snapshot.daily_loss_pct
        drawdown_pct = snapshot.drawdown_pct

        # Daily loss breach
        if daily_loss_pct <= -self.limits.daily_loss_limit_pct:
            reason = (
                f"KILL SWITCH: daily loss {daily_loss_pct:.2%} breached limit "
                f"-{self.limits.daily_loss_limit_pct:.2%}. "
                f"Halting new entries for this cycle."
            )
            logger.warning(reason)
            return KillSwitchState(
                halted=True,
                reason=reason,
                daily_loss_pct=daily_loss_pct,
                drawdown_pct=drawdown_pct,
            )

        # Drawdown breach
        if drawdown_pct <= -self.limits.max_drawdown_pct:
            reason = (
                f"KILL SWITCH: drawdown {drawdown_pct:.2%} breached limit "
                f"-{self.limits.max_drawdown_pct:.2%}. "
                f"Halting new entries for this cycle."
            )
            logger.warning(reason)
            return KillSwitchState(
                halted=True,
                reason=reason,
                daily_loss_pct=daily_loss_pct,
                drawdown_pct=drawdown_pct,
            )

        return KillSwitchState(
            halted=False,
            reason="",
            daily_loss_pct=daily_loss_pct,
            drawdown_pct=drawdown_pct,
        )
