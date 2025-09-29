# workflow_engine/integrations/llm.py
from abc import ABC, abstractmethod


# The "Implementation" Interface for the Bridge Pattern
class ActionImplementation(ABC):
    @abstractmethod
    def perform(self, model_name: str, inputs: str) -> str:
        pass


class OpenaiLLMAction(ActionImplementation):
    def perform(self, model_name: str, inputs: str) -> str:
        print(f"Calling OpenAI API for {model_name}...")
        return f"OpenAI_RESPONSE: Processed '{inputs}'."


class AnthropicLLMAction(ActionImplementation):
    def perform(self, model_name: str, inputs: str) -> str:
        print(f"Calling Anthropic API for {model_name}...")
        return f"ANTHROPIC_RESPONSE: Processed '{inputs}'."
