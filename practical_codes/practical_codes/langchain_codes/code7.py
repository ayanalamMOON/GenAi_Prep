from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Initialize the LLM with Ollama
llm = OllamaLLM(model="qwen3:4b", temperature=0.9)

prompt = PromptTemplate(
    input_variables=["country"],
    template="what your opinion on this country {country}?",
)

# Create a chain
chain = prompt | llm

# Invoke the chain
result = chain.invoke({"country": "India"})
print(result)