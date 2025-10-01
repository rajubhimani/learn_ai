from workflow_engine.nodes.llm_call import LLMCallNode
from workflow_engine.integrations.llm import PerplexityLLMAction
from workflow_engine.execution.runner import WorkflowRunner


def test_perplexity():
    # Example input and model output
    # Run the workflows
    # A simple workflow to showcase a single node
    single_node_workflow = LLMCallNode(
        name="Single LLM Call",
        description="A simple workflow that calls an LLM.",
        implementation=PerplexityLLMAction(),
        model_name="gpt-4",
    )
    runner = WorkflowRunner()

    print("--- Running Simple Workflow ---")
    runner.run(
        single_node_workflow, initial_inputs="Tell me something interesting about AI."
    )


if __name__ == "__main__":
    test_perplexity()
