# Component -1
from langchain.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
st.title("MY FIRST CHATBOT")
my_llm = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)

# Component -2  (ChatPromptTemplate Version)
from langchain.prompts import ChatPromptTemplate

my_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

from langchain.chains import LLMChain

my_pipeline = LLMChain(
    llm=my_llm,
    prompt=my_prompt
)
usertext=st.text_input("USER:")
result = my_pipeline.invoke({"input": usertext})
st.write(result["text"])
