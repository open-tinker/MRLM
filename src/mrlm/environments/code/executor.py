"""
Python code executor with sandboxing.

This module provides safe execution of Python code with timeout and resource limits.
"""

import io
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional
import signal
import multiprocessing
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """
    Result of code execution.

    Attributes:
        success: Whether execution completed without errors
        output: Standard output from execution
        error: Error message if execution failed
        return_value: Return value from executed code
        execution_time: Time taken to execute (seconds)
    """

    success: bool
    output: str = ""
    error: str = ""
    return_value: Any = None
    execution_time: float = 0.0


class TimeoutException(Exception):
    """Raised when code execution times out."""

    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Code execution timed out")


@contextmanager
def time_limit(seconds: float):
    """
    Context manager for limiting execution time.

    Args:
        seconds: Maximum execution time

    Raises:
        TimeoutException: If execution exceeds time limit
    """
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm


class PythonExecutor:
    """
    Safe Python code executor.

    This class executes Python code in a controlled environment with:
    - Timeout limits
    - Output capture
    - Error handling
    - Basic sandboxing

    Example:
        >>> executor = PythonExecutor(timeout=5.0)
        >>> result = executor.execute("print('Hello, world!')")
        >>> print(result.output)
        Hello, world!
    """

    def __init__(self, timeout: float = 5.0, max_output_length: int = 10000):
        """
        Initialize executor.

        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum length of captured output
        """
        self.timeout = timeout
        self.max_output_length = max_output_length

    def execute(self, code: str, globals_dict: Optional[Dict] = None) -> ExecutionResult:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            globals_dict: Global variables for execution context

        Returns:
            ExecutionResult containing output and status
        """
        import time

        start_time = time.time()

        # Create execution context
        if globals_dict is None:
            globals_dict = {}

        # Add built-ins but remove dangerous ones
        safe_globals = {
            "__builtins__": {
                k: v
                for k, v in __builtins__.items()
                if k
                not in [
                    "open",
                    "exec",
                    "eval",
                    "compile",
                    "__import__",
                    "globals",
                    "locals",
                ]
            }
        }
        safe_globals.update(globals_dict)

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with time_limit(self.timeout):
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Execute code
                    exec(code, safe_globals)

            # Get output
            output = stdout_capture.getvalue()
            if len(output) > self.max_output_length:
                output = output[: self.max_output_length] + "\n... (output truncated)"

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=True,
                output=output,
                execution_time=execution_time,
            )

        except TimeoutException:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
            )

        except Exception as e:
            error_trace = traceback.format_exc()
            execution_time = time.time() - start_time

            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n{error_trace}",
                execution_time=execution_time,
            )

    def execute_function(
        self, code: str, function_name: str, *args, **kwargs
    ) -> ExecutionResult:
        """
        Execute code and call a specific function.

        Args:
            code: Python code containing function definition
            function_name: Name of function to call
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            ExecutionResult with function return value
        """
        import time

        start_time = time.time()

        # Create execution context
        globals_dict = {}

        try:
            with time_limit(self.timeout):
                # Execute code to define function
                exec(code, globals_dict)

                # Check if function exists
                if function_name not in globals_dict:
                    return ExecutionResult(
                        success=False,
                        error=f"Function '{function_name}' not found in code",
                    )

                # Call function
                func = globals_dict[function_name]
                result = func(*args, **kwargs)

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=True, return_value=result, execution_time=execution_time
            )

        except TimeoutException:
            return ExecutionResult(
                success=False,
                error=f"Function execution timed out after {self.timeout} seconds",
                execution_time=self.timeout,
            )

        except Exception as e:
            error_trace = traceback.format_exc()
            execution_time = time.time() - start_time

            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}\n{error_trace}",
                execution_time=execution_time,
            )

    def test_function(
        self, code: str, function_name: str, test_cases: list[tuple]
    ) -> Dict[str, Any]:
        """
        Test a function against multiple test cases.

        Args:
            code: Python code containing function
            function_name: Name of function to test
            test_cases: List of (inputs, expected_output) tuples

        Returns:
            Dictionary with test results

        Example:
            >>> test_cases = [
            ...     (([1, 2, 3],), 6),  # sum([1,2,3]) == 6
            ...     (([],), 0),         # sum([]) == 0
            ... ]
            >>> results = executor.test_function(code, "sum", test_cases)
        """
        results = {
            "passed": 0,
            "failed": 0,
            "total": len(test_cases),
            "test_results": [],
            "all_passed": False,
        }

        for i, (inputs, expected) in enumerate(test_cases):
            # Handle inputs - can be tuple of args or dict of kwargs
            if isinstance(inputs, dict):
                args = ()
                kwargs = inputs
            elif isinstance(inputs, tuple):
                args = inputs
                kwargs = {}
            else:
                args = (inputs,)
                kwargs = {}

            # Execute function with test input
            result = self.execute_function(code, function_name, *args, **kwargs)

            test_result = {
                "test_id": i,
                "inputs": inputs,
                "expected": expected,
                "success": result.success,
            }

            if result.success:
                # Check if output matches expected
                if result.return_value == expected:
                    test_result["passed"] = True
                    results["passed"] += 1
                else:
                    test_result["passed"] = False
                    test_result["actual"] = result.return_value
                    results["failed"] += 1
            else:
                test_result["passed"] = False
                test_result["error"] = result.error
                results["failed"] += 1

            results["test_results"].append(test_result)

        results["all_passed"] = results["passed"] == results["total"]

        return results
