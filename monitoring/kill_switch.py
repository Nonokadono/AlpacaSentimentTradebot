from dataclasses import dataclass
from config.config import RiskLimits
from core.risk_engine import EquitySnapshot


@dataclass
class KillSwitchState:
    halted: bool
    reason: str


class KillSwitch:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def check(self, snapshot: EquitySnapshot) -> KillSwitchState:
        if snapshot.daily_loss_pct <= -self.limits.daily_loss_limit_pct:
            return KillSwitchState(True, "Daily loss limit hit")
        if snapshot.drawdown_pct <= -self.limits.max_drawdown_pct:
            return KillSwitchState(True, "Max drawdown limit hit")
        return KillSwitchState(False, "")
