# workflow_engine/execution/runner.py
from abc import ABC, abstractmethod
from ..nodes.base import WorkflowNode


# The "State" Interface
class WorkflowState(ABC):
    @abstractmethod
    def handle(self, runner: "WorkflowRunner"):
        pass


# The "Context" that uses the State
class WorkflowRunner:
    def __init__(self):
        self.workflow: WorkflowNode | None = None
        self.inputs: str | None = None
        self.current_state: WorkflowState | None = None
        self.output: str | None = None

    def transition_to(self, state: WorkflowState):
        self.current_state = state

    def run(self, workflow: WorkflowNode, initial_inputs: str):
        self.workflow = workflow
        self.inputs = initial_inputs
        self.transition_to(PendingState())
        self.current_state.handle(self)


class PendingState(WorkflowState):
    def handle(self, runner: WorkflowRunner):
        print("Workflow is Pending...")
        runner.transition_to(RunningState())
        runner.current_state.handle(runner)


class RunningState(WorkflowState):
    def handle(self, runner: WorkflowRunner):
        print("Workflow is Running...")
        try:
            runner.workflow.execute(runner.inputs)
            runner.transition_to(CompletedState())
            runner.current_state.handle(runner)
        except Exception as e:
            print(f"Workflow failed with error: {e}")
            runner.transition_to(FailedState())
            runner.current_state.handle(runner)


class CompletedState(WorkflowState):
    def handle(self, runner: WorkflowRunner):
        print("Workflow Completed Successfully.")
        print(f"Final Output: {runner.workflow.output}\n")


class FailedState(WorkflowState):
    def handle(self, runner: WorkflowRunner):
        print("Workflow Failed.\n")
