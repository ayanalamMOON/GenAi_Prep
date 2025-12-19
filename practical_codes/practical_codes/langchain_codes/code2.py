from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="qwen3:4b",
    temperature=0.7,
    max_tokens=100,
)

response = llm.invoke("Hello, how are you?")

print(response)