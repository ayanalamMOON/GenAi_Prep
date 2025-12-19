from langchain.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()

llm = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)
while True:
    input_text=input("Enter your query: ")
    result = llm.invoke(input_text)
    print("AI:",result.content)
    print("--------------------------------------------------")