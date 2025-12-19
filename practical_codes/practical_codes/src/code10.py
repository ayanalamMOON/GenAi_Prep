#component 1 LLM
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


from langchain.chains import LLMChain

#compenent 2 PROMPT
from langchain.prompts import PromptTemplate
prompt=PromptTemplate(
    template="what is the capital of {input}"
)

pipeline=LLMChain(
    llm=llm,
    prompt=prompt
)


result=pipeline.invoke(input="china")

print(result['text'])