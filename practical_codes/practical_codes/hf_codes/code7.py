from transformers import pipeline

# Initialize the pipeline with Qwen model
pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

# Format the prompt
prompt = "what your opinion on this country India?"

# Generate text
result = pipe(prompt, max_new_tokens=100, temperature=0.9, top_p=0.9)
print(result[0]['generated_text'])