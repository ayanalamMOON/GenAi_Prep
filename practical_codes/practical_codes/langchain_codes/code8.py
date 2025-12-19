from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_ollama import OllamaLLM

# Initialize the LLM with Ollama
llm = OllamaLLM(model="qwen3:4b", temperature=0.9)

examples = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."},
    {"question": "What is the capital of Italy?", "answer": "The capital of Italy is Rome."},
]

examples_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=examples_prompt,
    input_variables=["question"],
    suffix="Answer:",
    prefix="You are a helpful assistant. Answer the following question.",
)

# Create a chain
chain = few_shot_prompt | llm

# Invoke the chain
result = chain.invoke({"question": "What is the capital of Spain?"})
print(result)