# main.py
from workflow_engine.nodes.composite import CompositeNode
from workflow_engine.nodes.llm_call import LLMCallNode
from workflow_engine.nodes.python_code import PythonCodeNode
from workflow_engine.integrations.llm import OpenaiLLMAction, AnthropicLLMAction
from workflow_engine.execution.runner import WorkflowRunner


# A simple workflow to showcase a single node
single_node_workflow = LLMCallNode(
    name="Single LLM Call",
    description="A simple workflow that calls an LLM.",
    implementation=OpenaiLLMAction(),
    model_name="gpt-4",
)

# A composite workflow
custom_python_logic = """
processed_text = inputs.upper() + " (PROCESSED by Python)"
output = processed_text
"""

composite_workflow = CompositeNode(
    name="Hybrid LLM and Python Workflow",
    description="Combines an LLM call with custom Python code.",
)

composite_workflow.add_child(
    LLMCallNode(
        name="Step 1: LLM Call",
        description="Get the raw LLM response.",
        implementation=AnthropicLLMAction(),
        model_name="claude-3-opus",
    )
)

composite_workflow.add_child(
    PythonCodeNode(
        name="Step 2: Python Code",
        description="Process the LLM's response.",
        code_string=custom_python_logic,
    )
)

# Run the workflows
runner = WorkflowRunner()

print("--- Running Simple Workflow ---")
runner.run(single_node_workflow, initial_inputs="Hello, world.")

print("--- Running Composite Workflow ---")
runner.run(composite_workflow, initial_inputs="Hello, world.")
