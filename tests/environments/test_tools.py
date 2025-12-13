"""Tests for tool use environment."""

import pytest
from mrlm.environments.tools import ToolUseEnvironment
from mrlm.environments.tools.tool_registry import ToolRegistry, Tool
from mrlm.environments.tools.builtin_tools import create_default_tool_registry
from mrlm.core.types import Message, MessageRole, EnvironmentMode


class TestTool:
    """Test Tool class."""

    def test_tool_creation(self):
        """Test creating a tool."""

        def calculator(expression: str) -> str:
            return str(eval(expression))

        tool = Tool(
            name="calculator",
            description="Evaluates math expressions",
            function=calculator,
        )

        assert tool.name == "calculator"
        assert tool.function is calculator

    def test_tool_execution(self):
        """Test executing a tool."""

        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(name="add", description="Adds two numbers", function=add)

        result = tool.execute(a=2, b=3)
        assert result == 5


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_registry_creation(self):
        """Test creating tool registry."""
        registry = ToolRegistry()
        assert len(registry.tools) == 0

    def test_registry_add_tool(self):
        """Test adding tool to registry."""
        registry = ToolRegistry()

        def test_func():
            return "test"

        tool = Tool(name="test", description="Test tool", function=test_func)
        registry.add_tool(tool)

        assert len(registry.tools) == 1
        assert "test" in registry.tools

    def test_registry_get_tool(self):
        """Test getting tool from registry."""
        registry = ToolRegistry()

        def test_func():
            return "test"

        tool = Tool(name="test", description="Test tool", function=test_func)
        registry.add_tool(tool)

        retrieved = registry.get_tool("test")
        assert retrieved is tool

    def test_registry_execute_tool(self):
        """Test executing tool from registry."""
        registry = ToolRegistry()

        def multiply(a: int, b: int) -> int:
            return a * b

        tool = Tool(name="multiply", description="Multiplies numbers", function=multiply)
        registry.add_tool(tool)

        result = registry.execute("multiply", a=3, b=4)
        assert result == 12


class TestBuiltinTools:
    """Test built-in tools."""

    def test_create_default_registry(self):
        """Test creating default tool registry."""
        registry = create_default_tool_registry()

        # Should have default tools
        assert len(registry.tools) > 0

        # Check for expected tools
        tool_names = [tool.name for tool in registry.tools.values()]
        assert "calculator" in tool_names or "calc" in tool_names

    def test_calculator_tool(self):
        """Test calculator tool."""
        registry = create_default_tool_registry()

        # Get calculator tool (name might vary)
        calc_tool = None
        for tool in registry.tools.values():
            if "calc" in tool.name.lower():
                calc_tool = tool
                break

        if calc_tool:
            # Test basic calculation
            result = calc_tool.execute(expression="2 + 2")
            assert "4" in str(result)


class TestToolUseEnvironment:
    """Test ToolUseEnvironment class."""

    def test_environment_creation(self):
        """Test creating tool use environment."""
        registry = create_default_tool_registry()
        env = ToolUseEnvironment(
            tool_registry=registry,
            mode=EnvironmentMode.CLIENT,
        )

        assert env.mode == EnvironmentMode.CLIENT
        assert env.tool_registry is registry

    def test_environment_reset(self):
        """Test resetting environment."""
        registry = create_default_tool_registry()
        env = ToolUseEnvironment(registry)

        obs = env.reset()

        assert not obs.done
        assert len(obs.messages) > 0
        # Should contain task description and available tools
        content = " ".join(msg.content for msg in obs.messages)
        assert len(content) > 0

    def test_environment_step_with_tool_use(self):
        """Test stepping with tool use."""
        registry = create_default_tool_registry()
        env = ToolUseEnvironment(registry, max_turns=3)

        obs = env.reset()

        # Try to use a tool
        action = Message(
            role=MessageRole.ASSISTANT,
            content="Let me use the calculator: 2 + 2",
        )

        obs, reward = env.step(action)

        # Should execute and get some reward
        assert reward.value >= 0

    def test_environment_max_turns(self):
        """Test environment terminates after max turns."""
        registry = create_default_tool_registry()
        env = ToolUseEnvironment(registry, max_turns=2)

        env.reset()

        action1 = Message(role=MessageRole.ASSISTANT, content="Thinking...")
        env.step(action1)

        action2 = Message(role=MessageRole.ASSISTANT, content="Answer")
        obs, reward = env.step(action2)

        assert obs.done

    def test_environment_multiple_episodes(self):
        """Test running multiple episodes."""
        registry = create_default_tool_registry()
        env = ToolUseEnvironment(registry, max_turns=2)

        # Episode 1
        env.reset()
        action = Message(role=MessageRole.ASSISTANT, content="Test")
        env.step(action)

        # Episode 2
        obs = env.reset()
        assert obs is not None
        assert not obs.done
