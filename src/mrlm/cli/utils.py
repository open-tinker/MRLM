"""
Utility functions for CLI commands.
"""

from typing import List
from mrlm.core.base import BaseEnvironment
from mrlm.core.types import EnvironmentMode
from mrlm.config.training_config import EnvironmentConfig


def create_environment_by_name(name: str) -> BaseEnvironment:
    """
    Create an environment by name.

    Args:
        name: Environment name (code, math, debate, tool)

    Returns:
        Environment instance
    """
    if name == "code":
        from mrlm.environments.code import CodeExecutionEnvironment, CodeProblemGenerator

        generator = CodeProblemGenerator()
        return CodeExecutionEnvironment(
            problem_generator=generator,
            mode=EnvironmentMode.CLIENT,
        )

    elif name == "math":
        from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator

        generator = MathProblemGenerator()
        return MathReasoningEnvironment(
            problem_generator=generator,
            mode=EnvironmentMode.CLIENT,
        )

    elif name == "debate":
        from mrlm.environments.debate import DebateEnvironment, RuleBasedJudge

        judge = RuleBasedJudge()
        return DebateEnvironment(
            mode=EnvironmentMode.CLIENT,
            judge=judge,
        )

    elif name == "tool":
        from mrlm.environments.tools import ToolUseEnvironment
        from mrlm.environments.tools.builtin_tools import create_default_tool_registry

        registry = create_default_tool_registry()
        return ToolUseEnvironment(
            tool_registry=registry,
            mode=EnvironmentMode.CLIENT,
        )

    else:
        raise ValueError(f"Unknown environment: {name}")


def create_environments(env_configs: List[EnvironmentConfig]) -> List[BaseEnvironment]:
    """
    Create multiple environments from configs.

    Args:
        env_configs: List of environment configurations

    Returns:
        List of environment instances
    """
    environments = []

    for config in env_configs:
        mode = EnvironmentMode.SERVER if config.mode == "server" else EnvironmentMode.CLIENT

        if config.env_type == "code":
            from mrlm.environments.code import CodeExecutionEnvironment, CodeProblemGenerator

            generator = CodeProblemGenerator()
            env = CodeExecutionEnvironment(
                problem_generator=generator,
                mode=mode,
                max_turns=config.max_turns if hasattr(config, "max_turns") else 3,
            )
            environments.append(env)

        elif config.env_type == "math":
            from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator

            difficulty_range = getattr(config, "difficulty_range", [1, 3])
            generator = MathProblemGenerator(difficulty_range=tuple(difficulty_range))
            env = MathReasoningEnvironment(
                problem_generator=generator,
                mode=mode,
                max_turns=config.max_turns if hasattr(config, "max_turns") else 5,
            )
            environments.append(env)

        elif config.env_type == "debate":
            from mrlm.environments.debate import DebateEnvironment, RuleBasedJudge

            judge = RuleBasedJudge()
            env = DebateEnvironment(
                mode=mode,
                judge=judge,
                max_turns=config.max_turns if hasattr(config, "max_turns") else 6,
            )
            environments.append(env)

        elif config.env_type == "tool":
            from mrlm.environments.tools import ToolUseEnvironment
            from mrlm.environments.tools.builtin_tools import create_default_tool_registry

            registry = create_default_tool_registry()
            env = ToolUseEnvironment(
                tool_registry=registry,
                mode=mode,
                max_turns=config.max_turns if hasattr(config, "max_turns") else 5,
            )
            environments.append(env)

        else:
            raise ValueError(f"Unknown environment type: {config.env_type}")

    return environments
