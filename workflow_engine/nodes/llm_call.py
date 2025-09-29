# workflow_engine/nodes/llm_call.py
from .base import SimpleNode
from ..integrations.llm import ActionImplementation


class LLMCallNode(SimpleNode):
    def __init__(
        self,
        name: str,
        description: str,
        implementation: ActionImplementation,
        model_name: str,
    ):
        super().__init__(name, description)
        self.implementation = implementation
        self.model_name = model_name

    def execute(self, inputs: str):
        print(f"Executing LLM Call Node: {self.name}")
        self.output = self.implementation.perform(self.model_name, inputs)
