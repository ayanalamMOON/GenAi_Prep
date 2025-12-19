from transformers import pipeline

# Initialize the pipeline with Qwen model
pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

# Build the few-shot prompt manually
examples = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."},
    {"question": "What is the capital of Italy?", "answer": "The capital of Italy is Rome."},
]

prompt = "You are a helpful assistant. Answer the following question.\n"
for ex in examples:
    prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n"
prompt += "Question: What is the capital of Spain?\nAnswer:"

# Generate text
result = pipe(prompt, max_new_tokens=100, temperature=0.9, top_p=0.9)
print(result[0]['generated_text'])