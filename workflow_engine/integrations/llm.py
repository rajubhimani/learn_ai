# workflow_engine/integrations/llm.py
from abc import ABC, abstractmethod
from pydantic import SecretStr
from langchain_core.prompts import ChatPromptTemplate
from langchain_perplexity import ChatPerplexity
from workflow_engine.config import settings


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


class PerplexityLLMAction(ActionImplementation):
    def perform(self, model_name: str, inputs: str) -> str:
        print(f"Calling Perplexity API for {model_name} using LangGraph...")
        llm = ChatPerplexity(
            temperature=0.7,
            model="sonar",
            timeout=60,
            api_key=SecretStr(settings.PPLX_API_KEY),
        )
        template = ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot."),
                ("human", "{user_input}"),
            ]
        )
        response = llm.invoke(template.format_messages(user_input=inputs))
        return f"PERPLEXITY_RESPONSE: {response.text()}"
