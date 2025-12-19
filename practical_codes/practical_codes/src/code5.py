from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenAI

from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
import os


llm1=ChatOpenAI(model="gpt-3.5-turbo",temperature=0.5,openai_api_key=os.getenv("OPENAI_API_KEY"))
llm2=ChatGoogleGenAI(temperature=0.7, model="gemini-1.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
llm3=ChatAnthropic(temperature=0.7, model="claude-2", anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))

result1=llm1.invoke("Opinion on Pakistan?")
result2=llm2.invoke("Opinion on Pakistan?")
result3=llm3.invoke("Opinion on Pakistan?")

print("OpenAI result: ",result1)
print("Google result: ",result2)
print("Anthropic result: ",result3)
