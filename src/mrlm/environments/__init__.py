"""Built-in task environments for MRLM."""

from mrlm.environments.code import CodeExecutionEnvironment, PythonExecutor
from mrlm.environments.math import MathReasoningEnvironment
from mrlm.environments.debate import DebateEnvironment, DebateTopic, RuleBasedJudge
from mrlm.environments.tools import (
    ToolUseEnvironment,
    ToolRegistry,
    CalculatorTool,
    WebSearchTool,
    PythonREPLTool,
)

__all__ = [
    "CodeExecutionEnvironment",
    "MathReasoningEnvironment",
    "PythonExecutor",
    "DebateEnvironment",
    "DebateTopic",
    "RuleBasedJudge",
    "ToolUseEnvironment",
    "ToolRegistry",
    "CalculatorTool",
    "WebSearchTool",
    "PythonREPLTool",
]
