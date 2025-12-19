from transformers import pipeline

pipe1 = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
pipe2 = pipeline("text-generation", model="Qwen/Qwen3-0.6B")  # Change model name to ones installed for better experience
pipe3 = pipeline("text-generation", model="Qwen/Qwen3-0.6B")  # Change model name to ones installed for better experience

result1 = pipe1("Opinion on Pakistan?", max_new_tokens=50, temperature=0.5, num_return_sequences=1)
result2 = pipe2("Opinion on Pakistan?", max_new_tokens=50, temperature=0.7, num_return_sequences=1)
result3 = pipe3("Opinion on Pakistan?", max_new_tokens=50, temperature=0.7, num_return_sequences=1)

print("OpenAI result: ", result1[0]['generated_text'])
print("Google result: ", result2[0]['generated_text'])
print("Anthropic result: ", result3[0]['generated_text'])