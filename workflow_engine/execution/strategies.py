# workflow_engine/execution/strategies.py
from abc import ABC, abstractmethod


# The "Strategy" Interface
class CodeExecutionStrategy(ABC):
    @abstractmethod
    def execute_code(self, code_string: str, inputs: str) -> str:
        pass


# A concrete "Strategy" for Python code
class PythonExecutionStrategy(CodeExecutionStrategy):
    def execute_code(self, code_string: str, inputs: str) -> str:
        # In a real app, use a secure sandbox.
        exec_globals: dict[str, str] = {"inputs": inputs, "output": ""}
        exec(code_string, exec_globals)
        return exec_globals.get("output", "")
