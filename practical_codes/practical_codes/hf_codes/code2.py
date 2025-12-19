from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

response = pipe("Hello, how are you?", max_new_tokens=50, temperature=0.7, num_return_sequences=1)

print(response[0]['generated_text'])