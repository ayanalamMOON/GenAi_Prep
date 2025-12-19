from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")

chat_history = []

while True:
    inp = input("Enter the query: ")
    # Keep only last 2 exchanges (4 messages: 2 user + 2 AI)
    if len(chat_history) > 4:
        chat_history = chat_history[-4:]
    history_str = "\n".join(chat_history)
    prompt = f"""

   "summary in 50 words technical

    Conversation history:
    {history_str}
    User: {inp}
    AI:
    """
    result = pipe(prompt, max_new_tokens=100, temperature=0.7, num_return_sequences=1)
    response = result[0]['generated_text'].split("AI:")[-1].strip() if "AI:" in result[0]['generated_text'] else result[0]['generated_text']
    print(response)
    chat_history.append(f"User: {inp}")
    chat_history.append(f"AI: {response}")
    print(chat_history)