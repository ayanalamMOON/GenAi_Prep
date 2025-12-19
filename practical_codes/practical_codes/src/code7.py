from langchain.prompts import PromptTemplate

from langchain.chat_models import AzureChatOpenAI

from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
load_dotenv()
# Fetch from .env
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
llm=AzureChatOpenAI(
    deployment_name=deployment_name,
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=api_base,  
    model_name="gpt-4",       
    temperature=0.9,
    top_p=0.9,
    max_tokens=100,
# Maximum number of tokens in the response
)
promt=PromptTemplate(
    input_variables=["country"],
    template="what your opinion on this country {country}?",
)
chain=LLMChain(llm=llm,prompt=promt)
result=chain.invoke({"country":"India"})
print(result['text'])



