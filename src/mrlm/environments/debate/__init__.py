"""
Multi-agent debate environment.

Environments for training agents to engage in structured debates.
"""

from mrlm.environments.debate.debate_env import (
    DebateEnvironment,
    DebateTopic,
    DebatePosition,
)
from mrlm.environments.debate.judge import DebateJudge, RuleBasedJudge

__all__ = [
    "DebateEnvironment",
    "DebateTopic",
    "DebatePosition",
    "DebateJudge",
    "RuleBasedJudge",
]
