# workflow_engine/nodes/base.py
from abc import ABC, abstractmethod


class WorkflowNode(ABC):
    """The Component Interface for the Composite pattern."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.output: str = ""

    @abstractmethod
    def execute(self, inputs: str):
        pass


class SimpleNode(WorkflowNode):
    """The Leaf in the Composite pattern."""

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
