from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:4b", temperature=0.7)

while True:
    input_text = input("Enter your query: ")
    result = llm.invoke(input_text)
    print("AI:", result)
    print("--------------------------------------------------")