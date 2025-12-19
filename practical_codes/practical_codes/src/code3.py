#ChatAnthropic - cluade

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
load_dotenv()
llm=ChatAnthropic(temperature=0.7, model="claude-2", anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
result=llm.invoke("What is the capital of France?")
print(result)
