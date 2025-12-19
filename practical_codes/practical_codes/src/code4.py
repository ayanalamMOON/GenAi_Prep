from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv
import os
load_dotenv()

llm=ChatGoogleGenAI(temperature=0.7, model="gemini-1.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))


result=llm.invoke("What is the capital of France?")
print(result)