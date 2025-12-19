from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

prompt_template = "what is the capital of {input}"

input_value = "china"
prompt = prompt_template.format(input=input_value)

result = pipe(prompt, max_new_tokens=50, temperature=0.7, num_return_sequences=1)

print(result[0]['generated_text'])