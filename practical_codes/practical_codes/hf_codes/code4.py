from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

result = pipe("What is the capital of France?", max_new_tokens=50, temperature=0.7, num_return_sequences=1)

print(result[0]['generated_text'])