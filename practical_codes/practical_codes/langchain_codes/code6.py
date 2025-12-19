from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Initialize the LLM with Ollama
llm = OllamaLLM(model="qwen3:4b")

prompt = PromptTemplate(
    input_variables=["input"],
    template="What is the capital of {input}?",
)

# Create a chain
chain = prompt | llm

# Invoke the chain
result = chain.invoke({"input": "China"})
print(result)