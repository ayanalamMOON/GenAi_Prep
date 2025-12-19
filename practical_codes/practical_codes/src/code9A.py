from langchain.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

llm_object = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.7,
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=2
)

prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="""

   "summary in 50 words technical

    Conversation history:
    {chat_history}
    User: {user_input}
    AI:
    """
)

chain = LLMChain(
    llm=llm_object,
    prompt=prompt,
    memory=memory
)

while True:
    inp = input("Enter the query: ")
    result = chain.invoke({"user_input": inp})
    print(result['text'])
    print(memory.buffer)
