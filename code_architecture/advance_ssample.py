# The "Implementation" hierarchy
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


class ActionImplementation(ABC):
    @abstractmethod
    def perform(self, model_name: str, inputs: str) -> str:
        pass


class OpenaiLLMAction(ActionImplementation):
    def perform(self, model_name: str, inputs: str) -> str:
        # This would contain the actual code to call the OpenAI API
        print(f"Calling OpenAI API for {model_name}...")
        return f"OpenAI_RESPONSE: Processed '{inputs}'."


class AnthropicLLMAction(ActionImplementation):
    def perform(self, model_name: str, inputs: str) -> str:
        # This would contain the actual code to call the Anthropic API
        print(f"Calling Anthropic API for {model_name}...")
        return f"ANTHROPIC_RESPONSE: Processed '{inputs}'."


# The "Abstraction" hierarchy (our existing Node classes)
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


# Create an LLM node with the OpenAI implementation
openai_node = LLMCallNode(
    name="OpenAI Node",
    description="Calls the OpenAI API",
    implementation=OpenaiLLMAction(),
    model_name="gpt-4",
)

# Create an LLM node with the Anthropic implementation
anthropic_node = LLMCallNode(
    name="Anthropic Node",
    description="Calls the Anthropic API",
    implementation=AnthropicLLMAction(),
    model_name="claude-2",
)


# The "State" Interface
class WorkflowState(ABC):
    @abstractmethod
    def handle(self, runner: "WorkflowRunner"):
        pass


# Our updated "Context" (the WorkflowRunner)
class WorkflowRunner:
    def __init__(self):
        self.workflow: SimpleNode | CompositeNode | None = None
        self.inputs: str | None = None
        self.current_state: WorkflowState | None = None
        self.output: str | None = None

    def transition_to(self, state: WorkflowState):
        self.current_state = state

    def run(self, workflow: SimpleNode | CompositeNode, initial_inputs: str):
        self.workflow = workflow
        self.inputs = initial_inputs
        self.transition_to(PendingState())
        # Kick off the state machine
        self.current_state.handle(self)


class PendingState(WorkflowState):
    def handle(self, runner: WorkflowRunner):
        print("Workflow is Pending...")
        # Transition to RunningState and start execution
        runner.transition_to(RunningState())
        runner.current_state.handle(runner)


class RunningState(WorkflowState):
    def handle(self, runner: WorkflowRunner):
        print("Workflow is Running...")
        try:
            # Execute the workflow's command
            runner.workflow.execute(runner.inputs)
            # On success, transition to CompletedState
            runner.transition_to(CompletedState())
            runner.current_state.handle(runner)
        except Exception as e:
            print(f"Workflow failed with error: {e}")
            # On failure, transition to FailedState
            runner.transition_to(FailedState())


class CompletedState(WorkflowState):
    def handle(self, runner: WorkflowRunner):
        print("Workflow Completed Successfully.")
        print(f"Final Output: {runner.workflow.output}\n")


class FailedState(WorkflowState):
    def handle(self, runner: WorkflowRunner):
        print("Workflow Failed.\n")


# Run the workflows
runner = WorkflowRunner()
# You can run both interchangeably
runner.run(openai_node, initial_inputs="What is the weather?")
# exit(0)
runner.run(anthropic_node, initial_inputs="What is the weather?")


# A simple workflow to showcase a single node
single_node_workflow = LLMCallNode(
    name="Single LLM Call",
    description="A simple workflow that calls an LLM.",
    implementation=OpenaiLLMAction(),
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
        implementation=OpenaiLLMAction(),
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
# Run the composite workflow
runner.run(composite_workflow, initial_inputs="Hello, world.")
