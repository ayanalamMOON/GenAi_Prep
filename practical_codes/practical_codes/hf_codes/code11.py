from transformers import pipeline
import streamlit as st
st.title("MY FIRST CHATBOT")

generator = pipeline("text-generation", model="gpt2")

usertext = st.text_input("USER:")
if usertext:
    prompt = f"You are a helpful assistant. {usertext}"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    st.write(result[0]['generated_text'])