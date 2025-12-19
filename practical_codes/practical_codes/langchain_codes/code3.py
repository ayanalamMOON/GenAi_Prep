from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:4b", temperature=0.7)

result = llm.invoke("What is the capital of France?")

print(result)