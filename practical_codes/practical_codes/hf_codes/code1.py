from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

while True:
    input_text = input("Enter your query: ")
    result = pipe(input_text, max_length=100, temperature=0.7, num_return_sequences=1)
    print("AI:", result[0]['generated_text'])
    print("--------------------------------------------------")