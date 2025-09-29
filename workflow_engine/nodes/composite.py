# workflow_engine/nodes/composite.py
from .base import WorkflowNode


class CompositeNode(WorkflowNode):
    """The Composite in the Composite pattern."""

    def __init__(
        self, name: str, description: str, children: list[WorkflowNode] | None = None
    ):
        super().__init__(name, description)
        self.children: list[WorkflowNode] = children or []

    def add_child(self, node: WorkflowNode):
        self.children.append(node)

    def execute(self, inputs: str):
        current_input: str = inputs
        for child in self.children:
            child.execute(current_input)
            current_input = child.output
        self.output = current_input
