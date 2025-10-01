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


from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.memory import BaseMemory
from langchain_core.chains import LLMChain


class LangChainLLMNode(SimpleNode):
    """
    A unified node for calling any LangChain-compatible LLM.
    This replaces the Bridge pattern (LLMCallNode and ActionImplementations).
    Works with the latest LangChain API (v0.1+).
    """

    def __init__(self, name: str, description: str, llm: BaseLanguageModel):
        super().__init__(name, description)
        self.llm = llm
        # Modern PromptTemplate (langchain_core.prompts)
        self.prompt_template = PromptTemplate.from_template(
            """
            The following is a friendly conversation between a human and an AI.
            The AI is talkative and provides lots of specific details from its context.

            Current conversation:
            {history}
            Human: {input}
            AI:
            """
        )

    def execute(self, inputs: str, memory: BaseMemory | None = None) -> None:
        """
        Executes the LLM call using up-to-date LangChain API components.
        If memory is provided, uses LLMChain to manage prompt and history.
        Otherwise, calls the model directly.
        """
        print(
            f"Executing LangChain LLM Node: {self.name} with model {type(self.llm).__name__}"
        )

        if memory is not None:
            # Modern LLMChain invocation
            chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt_template,
                memory=memory,
                verbose=False,  # Optional, for debugging
            )
            response = chain.invoke({"input": inputs})
            self.output = response["text"]
        else:
            # Direct invoke pattern for a single-shot prompt
            response = self.llm.invoke(inputs)
            # Latest BaseLanguageModel returns an AIMessage; use content property
            self.output = response.content

        print(f"Output: {self.output}")
