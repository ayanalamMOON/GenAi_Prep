from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory

llm = OllamaLLM(model="qwen3:4b", temperature=0.7)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
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

# Using pipe operator instead of deprecated LLMChain
chain = prompt | llm

while True:
    inp = input("Enter the query: ")
    result = chain.invoke({"user_input": inp, "chat_history": memory.buffer})
    print(result)
    memory.save_context({"input": inp}, {"output": result})