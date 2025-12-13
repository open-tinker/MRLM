"""
Built-in tools for the tool use environment.

Provides commonly used tools like calculator, web search, etc.
"""

import math
import operator
import re
from typing import Any, Dict
from pathlib import Path

from mrlm.environments.tools.tool_env import Tool


class CalculatorTool(Tool):
    """
    Calculator tool for mathematical operations.

    Supports basic arithmetic and mathematical functions.
    """

    def __init__(self):
        def calculate(expression: str) -> str:
            """
            Evaluate a mathematical expression.

            Args:
                expression: Mathematical expression (e.g., "2 + 2 * 3")

            Returns:
                Result of the calculation
            """
            # Safe evaluation using limited scope
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "exp": math.exp,
                "pi": math.pi,
                "e": math.e,
            }

            try:
                # Remove any potentially dangerous characters
                safe_expr = re.sub(r"[^0-9+\-*/().,\s]", "", expression)
                result = eval(safe_expr, {"__builtins__": {}}, allowed_names)
                return f"{result}"
            except Exception as e:
                return f"Error evaluating expression: {e}"

        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Supports basic arithmetic (+, -, *, /) and functions (sqrt, sin, cos, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')",
                    }
                },
                "required": ["expression"],
            },
            function=calculate,
        )


class PythonREPLTool(Tool):
    """
    Python REPL tool for executing Python code.

    WARNING: This executes arbitrary code. Use with caution.
    """

    def __init__(self, timeout: int = 5):
        def execute_python(code: str) -> str:
            """
            Execute Python code and return the result.

            Args:
                code: Python code to execute

            Returns:
                Output of the code execution
            """
            import io
            import contextlib

            # Capture stdout
            output = io.StringIO()

            try:
                with contextlib.redirect_stdout(output):
                    # Execute code
                    # Note: This is potentially unsafe - in production, use proper sandboxing
                    exec_globals = {"__builtins__": __builtins__}
                    exec(code, exec_globals)

                result = output.getvalue()
                return result if result else "Code executed successfully (no output)"

            except Exception as e:
                return f"Error: {e}"

        super().__init__(
            name="python_repl",
            description="Execute Python code and return the output. Useful for computations, data processing, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
            function=execute_python,
        )


class WebSearchTool(Tool):
    """
    Web search tool (simulated).

    In a real implementation, this would call a search API.
    For training, we use a simulated version with predefined knowledge.
    """

    def __init__(self, knowledge_base: Dict[str, str] = None):
        self.knowledge_base = knowledge_base or self._get_default_knowledge()

        def search(query: str) -> str:
            """
            Search for information.

            Args:
                query: Search query

            Returns:
                Search results
            """
            query_lower = query.lower()

            # Simple keyword matching
            results = []
            for key, value in self.knowledge_base.items():
                if any(word in key.lower() for word in query_lower.split()):
                    results.append(f"{key}: {value}")

            if results:
                return "\n".join(results[:3])  # Top 3 results
            else:
                return "No results found for this query."

        super().__init__(
            name="web_search",
            description="Search for information on the web. Returns relevant facts and data.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
            function=search,
        )

    def _get_default_knowledge(self) -> Dict[str, str]:
        """Get default knowledge base for simulated search."""
        return {
            "Python programming language": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
            "Machine learning": "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
            "Eiffel Tower height": "The Eiffel Tower is 330 meters (1,083 feet) tall, including antennas.",
            "Speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "Mount Everest height": "Mount Everest is 8,849 meters (29,032 feet) tall, the highest mountain on Earth.",
            "World population": "The world population is approximately 8 billion people as of 2024.",
            "First moon landing": "The first moon landing was on July 20, 1969, by Apollo 11 astronauts Neil Armstrong and Buzz Aldrin.",
            "Photosynthesis": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into oxygen and glucose.",
            "DNA structure": "DNA has a double helix structure, discovered by Watson and Crick in 1953.",
            "Human genome": "The human genome contains approximately 3 billion base pairs and about 20,000-25,000 genes.",
        }


class FileSystemTool(Tool):
    """
    File system tool for reading/writing files.

    Restricted to a specific directory for safety.
    """

    def __init__(self, base_dir: str = "/tmp/mrlm_workspace"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        def file_operation(operation: str, path: str, content: str = None) -> str:
            """
            Perform file operations.

            Args:
                operation: Operation type ('read', 'write', 'list')
                path: File path (relative to workspace)
                content: Content to write (for 'write' operation)

            Returns:
                Operation result
            """
            try:
                # Ensure path is within base_dir
                full_path = (self.base_dir / path).resolve()
                if not str(full_path).startswith(str(self.base_dir)):
                    return "Error: Access denied - path outside workspace"

                if operation == "read":
                    if full_path.exists():
                        return full_path.read_text()
                    else:
                        return f"Error: File {path} not found"

                elif operation == "write":
                    if content is None:
                        return "Error: No content provided for write operation"
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content)
                    return f"Successfully wrote to {path}"

                elif operation == "list":
                    if full_path.is_dir():
                        files = [f.name for f in full_path.iterdir()]
                        return "\n".join(files) if files else "Directory is empty"
                    else:
                        return f"Error: {path} is not a directory"

                else:
                    return f"Error: Unknown operation '{operation}'"

            except Exception as e:
                return f"Error: {e}"

        super().__init__(
            name="file_system",
            description="Read, write, and list files in the workspace directory.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "list"],
                        "description": "Operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path (relative to workspace)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (required for 'write' operation)",
                    },
                },
                "required": ["operation", "path"],
            },
            function=file_operation,
        )


def create_default_tool_registry() -> "ToolRegistry":
    """
    Create a tool registry with default built-in tools.

    Returns:
        ToolRegistry with calculator, python_repl, web_search, and file_system
    """
    from mrlm.environments.tools.tool_env import ToolRegistry

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(PythonREPLTool())
    registry.register(WebSearchTool())
    registry.register(FileSystemTool())

    return registry
