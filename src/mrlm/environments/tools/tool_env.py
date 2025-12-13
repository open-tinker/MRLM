"""
Tool use environment for training agents to use external tools.

This environment teaches agents to:
1. Recognize when a tool is needed
2. Select the appropriate tool
3. Format tool calls correctly
4. Interpret tool results
5. Use results to answer questions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import json
import re
import random

from mrlm.core.base import SimulatedEnvironment
from mrlm.core.types import Message, MessageRole, Observation, Reward, EnvironmentMode


@dataclass
class Tool:
    """
    A tool that can be used by the agent.

    Attributes:
        name: Tool name
        description: What the tool does
        parameters: Parameter schema (JSON schema format)
        function: The actual tool function
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    def call(self, **kwargs) -> str:
        """
        Call the tool with parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool result as string
        """
        try:
            result = self.function(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def to_schema(self) -> Dict[str, Any]:
        """
        Get tool schema in OpenAI function calling format.

        Returns:
            Tool schema dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Tool]:
        """Get list of all tools."""
        return list(self.tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.to_schema() for tool in self.tools.values()]


@dataclass
class ToolUseTask:
    """
    A task that requires tool use.

    Attributes:
        question: The question to answer
        required_tools: Tools needed to solve this task
        ground_truth: Expected answer
        hints: Optional hints about which tools to use
    """

    question: str
    required_tools: List[str]
    ground_truth: str
    hints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolUseEnvironment(SimulatedEnvironment):
    """
    Environment for training agents to use tools.

    The agent receives a question that requires using one or more tools
    to answer. It must:
    1. Decide which tool(s) to use
    2. Format the tool call correctly
    3. Interpret the result
    4. Provide a final answer
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tasks: Optional[List[ToolUseTask]] = None,
        max_turns: int = 5,
        mode: EnvironmentMode = EnvironmentMode.CLIENT,
    ):
        """
        Initialize tool use environment.

        Args:
            tool_registry: Registry of available tools
            tasks: List of tasks (generates default if None)
            max_turns: Maximum number of turns
            mode: Environment mode
        """
        super().__init__(mode=mode)

        self.tool_registry = tool_registry
        self.tasks = tasks or self._generate_default_tasks()
        self.max_turns = max_turns

        # Current task state
        self.current_task: Optional[ToolUseTask] = None
        self.turn_count = 0
        self.tools_used: List[str] = []
        self.tool_results: List[Dict[str, Any]] = []

    def _generate_default_tasks(self) -> List[ToolUseTask]:
        """Generate default tasks based on available tools."""
        # This will be populated based on registered tools
        return []

    def reset(self) -> Observation:
        """
        Reset environment with a new task.

        Returns:
            Initial observation
        """
        # Select random task
        self.current_task = random.choice(self.tasks)
        self.turn_count = 0
        self.tools_used = []
        self.tool_results = []

        self._state = {
            "task": self.current_task,
            "turn_count": 0,
            "tools_used": [],
        }

        # Create system message with tool descriptions
        tool_schemas = self.tool_registry.get_schemas()
        tools_desc = json.dumps(tool_schemas, indent=2)

        system_message = Message(
            role=MessageRole.SYSTEM,
            content=f"""You are an AI assistant that can use tools to answer questions.

Available tools:
{tools_desc}

To use a tool, respond with:
<tool_call>
{{"name": "tool_name", "parameters": {{"param1": "value1", ...}}}}
</tool_call>

After using tools, provide your final answer with:
<answer>Your final answer here</answer>

Question: {self.current_task.question}""",
        )

        return Observation(
            messages=[system_message],
            state=self._state.copy(),
            done=False,
        )

    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """
        Process agent's action (tool call or final answer).

        Args:
            action: Agent's response

        Returns:
            Tuple of (observation, reward)
        """
        content = action.content
        self.turn_count += 1
        self._state["turn_count"] = self.turn_count

        # Check for tool call
        tool_call = self._parse_tool_call(content)
        if tool_call:
            return self._handle_tool_call(tool_call)

        # Check for final answer
        answer = self._parse_answer(content)
        if answer:
            return self._handle_final_answer(answer)

        # No tool call or answer found
        if self.turn_count >= self.max_turns:
            # Max turns reached without answer
            reward = Reward(
                value=0.0,
                components={"timeout": 0.0},
                info={"reason": "max_turns_reached"},
            )
            obs = Observation(
                messages=[
                    Message(
                        role=MessageRole.SYSTEM,
                        content="Maximum turns reached. Task failed.",
                    )
                ],
                state=self._state.copy(),
                done=True,
            )
            return obs, reward

        # Invalid format - ask for clarification
        feedback = Message(
            role=MessageRole.SYSTEM,
            content="Invalid format. Please use <tool_call>...</tool_call> to call a tool or <answer>...</answer> to provide your final answer.",
        )

        reward = Reward(value=-0.1, components={"format_error": -0.1})
        obs = Observation(messages=[feedback], state=self._state.copy(), done=False)

        return obs, reward

    def _parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool call from response.

        Args:
            content: Response content

        Returns:
            Tool call dict or None
        """
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
        if match:
            try:
                tool_call = json.loads(match.group(1))
                return tool_call
            except json.JSONDecodeError:
                return None
        return None

    def _parse_answer(self, content: str) -> Optional[str]:
        """
        Parse final answer from response.

        Args:
            content: Response content

        Returns:
            Answer string or None
        """
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _handle_tool_call(
        self, tool_call: Dict[str, Any]
    ) -> Tuple[Observation, Reward]:
        """
        Handle a tool call.

        Args:
            tool_call: Tool call dictionary

        Returns:
            Tuple of (observation, reward)
        """
        tool_name = tool_call.get("name")
        parameters = tool_call.get("parameters", {})

        # Get tool
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            # Unknown tool
            feedback = Message(
                role=MessageRole.SYSTEM,
                content=f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(self.tool_registry.tools.keys())}",
            )
            reward = Reward(value=-0.2, components={"invalid_tool": -0.2})
            obs = Observation(messages=[feedback], state=self._state.copy(), done=False)
            return obs, reward

        # Call tool
        result = tool.call(**parameters)

        # Record tool use
        self.tools_used.append(tool_name)
        self.tool_results.append(
            {"tool": tool_name, "parameters": parameters, "result": result}
        )
        self._state["tools_used"] = self.tools_used

        # Check if correct tool was used
        is_correct_tool = tool_name in self.current_task.required_tools
        tool_reward = 0.3 if is_correct_tool else -0.1

        # Provide feedback
        feedback = Message(
            role=MessageRole.SYSTEM,
            content=f"Tool result: {result}\n\nYou can use another tool or provide your final answer.",
        )

        reward = Reward(
            value=tool_reward,
            components={"tool_use": tool_reward},
            info={"tool_used": tool_name, "correct_tool": is_correct_tool},
        )

        # Check if max turns reached
        done = self.turn_count >= self.max_turns
        obs = Observation(messages=[feedback], state=self._state.copy(), done=done)

        return obs, reward

    def _handle_final_answer(self, answer: str) -> Tuple[Observation, Reward]:
        """
        Handle final answer.

        Args:
            answer: Final answer string

        Returns:
            Tuple of (observation, reward)
        """
        # Evaluate answer
        is_correct = self._check_answer(answer, self.current_task.ground_truth)

        # Compute reward components
        components = {}

        # Correctness (main component)
        correctness = 1.0 if is_correct else 0.0
        components["correctness"] = correctness

        # Tool usage (bonus for using required tools)
        required_tools = set(self.current_task.required_tools)
        used_tools = set(self.tools_used)
        tool_coverage = len(required_tools & used_tools) / max(len(required_tools), 1)
        components["tool_coverage"] = tool_coverage * 0.3

        # Efficiency (bonus for fewer turns)
        efficiency = 1.0 - (self.turn_count / self.max_turns)
        components["efficiency"] = efficiency * 0.2 if is_correct else 0.0

        # Total reward
        total_reward = correctness * 0.5 + components["tool_coverage"] + components["efficiency"]

        reward = Reward(
            value=total_reward,
            components=components,
            info={
                "is_correct": is_correct,
                "answer": answer,
                "ground_truth": self.current_task.ground_truth,
                "tools_used": self.tools_used,
                "required_tools": self.current_task.required_tools,
            },
        )

        # Create feedback message
        if is_correct:
            feedback_text = f"✓ Correct! Final reward: {total_reward:.2f}"
        else:
            feedback_text = f"✗ Incorrect. Expected: {self.current_task.ground_truth}. Final reward: {total_reward:.2f}"

        feedback = Message(role=MessageRole.SYSTEM, content=feedback_text)

        obs = Observation(
            messages=[feedback],
            state=self._state.copy(),
            done=True,
        )

        return obs, reward

    def _check_answer(self, answer: str, ground_truth: str) -> bool:
        """
        Check if answer is correct.

        Args:
            answer: Agent's answer
            ground_truth: Correct answer

        Returns:
            True if correct
        """
        # Normalize strings
        answer = answer.lower().strip()
        ground_truth = ground_truth.lower().strip()

        # Exact match
        if answer == ground_truth:
            return True

        # Try to extract numbers and compare
        try:
            answer_num = float(re.sub(r"[^\d.-]", "", answer))
            truth_num = float(re.sub(r"[^\d.-]", "", ground_truth))
            return abs(answer_num - truth_num) < 0.01
        except (ValueError, AttributeError):
            pass

        # Fuzzy match (contains ground truth)
        return ground_truth in answer

    def close(self):
        """Clean up resources."""
        pass
