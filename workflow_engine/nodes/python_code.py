# workflow_engine/nodes/python_code.py
from .base import SimpleNode
from ..execution.strategies import CodeExecutionStrategy, PythonExecutionStrategy


class PythonCodeNode(SimpleNode):
    def __init__(
        self,
        name: str,
        description: str,
        code_string: str,
        strategy: CodeExecutionStrategy | None = None,
    ):
        super().__init__(name, description)
        self.code_string = code_string
        self.strategy = strategy or PythonExecutionStrategy()

    def execute(self, inputs: str):
        print(f"Executing Python Code Node: {self.name}")
        self.output = self.strategy.execute_code(self.code_string, inputs)
