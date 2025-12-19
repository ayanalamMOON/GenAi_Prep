from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="qwen3:4b", temperature=0.7)

prompt = PromptTemplate(
    template="what is the capital of {input}"
)

# Using pipe operator instead of deprecated LLMChain
pipeline = prompt | llm

result = pipeline.invoke({"input": "china"})

print(result)