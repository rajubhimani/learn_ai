from abc import ABC, abstractmethod


# The Composite-Command "Component" Interface
class WorkflowNode(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.output: str = ""  # To store the output after execution

    @abstractmethod
    def execute(self, inputs: str):
        """
        The core method to be implemented by all nodes.
        It takes inputs and produces an output.
        """
        pass


# The "Leaf" in the Composite pattern
# These are the simple, individual nodes (e.g., HTTP request, LLM call)
class SimpleNode(WorkflowNode):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)


# The "Composite" in the Composite pattern
# This allows us to group multiple nodes into a single, reusable unit.
class CompositeNode(WorkflowNode):
    def __init__(
        self, name: str, description: str, children: list[WorkflowNode] | None = None
    ):
        super().__init__(name, description)
        self.children: list[WorkflowNode] = children or []

    def add_child(self, node: WorkflowNode):
        self.children.append(node)

    def execute(self, inputs: str):
        # The Composite's execute() method delegates to its children.
        current_input: str = inputs
        for child in self.children:
            child.execute(current_input)
            current_input: str = (
                child.output
            )  # Output of one node becomes input of the next
        self.output = current_input


# The "Invoker" for our Command pattern
class WorkflowRunner:
    def run(self, workflow: WorkflowNode, initial_inputs: str):
        print(f"--- Running Workflow: {workflow.name} ---")
        workflow.execute(initial_inputs)
        print(f"--- Workflow Completed. Final Output: {workflow.output} ---")


# The "Strategy" Interface
class CodeExecutionStrategy(ABC):
    @abstractmethod
    def execute_code(self, code_string: str, inputs: str) -> str:
        pass


# A concrete "Strategy" for Python code
class PythonExecutionStrategy(CodeExecutionStrategy):
    def execute_code(self, code_string: str, inputs: str) -> str:
        # We'll use exec() for simplicity, but in a real app,
        # you'd need a secure sandbox to prevent malicious code execution.
        exec_globals: dict[str, str] = {"inputs": inputs, "output": ""}
        exec(code_string, exec_globals)
        return exec_globals.get("output", "")


# The "Context" that uses the Strategy
# This is our custom node for Python code.
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
        # Use a default strategy if none is provided
        self.strategy = strategy or PythonExecutionStrategy()

    def execute(self, inputs: str):
        print(f"Executing Python Code Node: {self.name}")
        self.output = self.strategy.execute_code(self.code_string, inputs)


# Example of a regular node that calls an LLM (simplified)
class LLMCallNode(SimpleNode):
    def __init__(self, name: str, description: str, model_name: str):
        super().__init__(name, description)
        self.model_name = model_name

    def execute(self, inputs: str):
        print(f"Calling LLM Model '{self.model_name}' with input: '{inputs}'")
        # In a real app, this would use a library like langchain/langgraph
        # For simplicity, we'll just simulate the output.
        self.output = (
            f"LLM_RESPONSE: The input '{inputs}' is a great example of a simple phrase."
        )


# A simple workflow to showcase a single node
single_node_workflow = LLMCallNode(
    name="Single LLM Call",
    description="A simple workflow that calls an LLM.",
    model_name="GPT-4",
)

# A composite workflow that combines an LLM call with custom Python logic
custom_python_logic = """
# This is the code provided by the user in the UI
# The 'inputs' variable is automatically available
# We will manipulate it to prepare a final output.
processed_text = inputs.upper() + " (PROCESSED by Python)"
output = processed_text
"""

composite_workflow = CompositeNode(
    name="Hybrid LLM and Python Workflow",
    description="Combines an LLM call with custom Python code.",
)

# Add the LLM node as a child
composite_workflow.add_child(
    LLMCallNode(
        name="Step 1: LLM Call",
        description="Get the raw LLM response.",
        model_name="GPT-4",
    )
)

# Add the custom Python node as a child
composite_workflow.add_child(
    PythonCodeNode(
        name="Step 2: Python Code",
        description="Process the LLM's response.",
        code_string=custom_python_logic,
    )
)

# Run the workflows
runner = WorkflowRunner()

# Run the simple workflow
runner.run(single_node_workflow, initial_inputs="Hello, world.")

print("\n")

# Run the composite workflow
runner.run(composite_workflow, initial_inputs="Hello, world.")
