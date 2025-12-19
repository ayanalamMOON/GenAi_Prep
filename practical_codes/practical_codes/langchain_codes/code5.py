from langchain_ollama import OllamaLLM

llm1 = OllamaLLM(model="qwen3:4b", temperature=0.5)
llm2 = OllamaLLM(model="qwen3:4b", temperature=0.7)  # Change model name to ones installed for better experience
llm3 = OllamaLLM(model="qwen3:4b", temperature=0.7)  # Change model name to ones installed for better experience

result1 = llm1.invoke("Opinion on Pakistan?")
result2 = llm2.invoke("Opinion on Pakistan?")
result3 = llm3.invoke("Opinion on Pakistan?")

print("OpenAI result: ", result1)
print("Google result: ", result2)
print("Anthropic result: ", result3)