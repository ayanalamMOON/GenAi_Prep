from transformers import pipeline

# Initialize the pipeline with Qwen model
pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

# Format the prompt
prompt = "What is the capital of China?"

# Generate text
result = pipe(prompt, max_new_tokens=50)
print(result[0]['generated_text'])