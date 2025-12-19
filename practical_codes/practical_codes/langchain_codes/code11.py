# Component -1
from langchain_ollama import OllamaLLM
import streamlit as st
st.title("MY FIRST CHATBOT")
my_llm = OllamaLLM(model="qwen3:4b", temperature=0.7)

# Component -2  (ChatPromptTemplate Version)
from langchain_core.prompts import ChatPromptTemplate

my_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# Using pipe operator instead of deprecated LLMChain
my_pipeline = my_prompt | my_llm
usertext=st.text_input("USER:")
result = my_pipeline.invoke({"input": usertext})
st.write(result["text"])