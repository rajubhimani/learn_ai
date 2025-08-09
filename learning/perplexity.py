from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

# load environment variables
load_dotenv()

llm = ChatPerplexity(
    model="sonar",
    temperature=0.7,
    timeout=30,
    max_retries=3,
)

system = "You are a helpful assistant."
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chain = prompt | llm

response = chain.invoke(
    {
        "input": """Current Temperature in Ahmedabad? What is the weather like?
                         Also mention when was the last update?"""
    }
)
print(response.content)
print("\n\nSearch results:")
search_results = response.additional_kwargs.get("search_results", [])
print(search_results)
