from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationSummaryMemory

llm_object = OllamaLLM(model="qwen3:4b", temperature=0.7)

memory = ConversationSummaryMemory(
    memory_key="chat_history",
    return_messages=True,
    llm=llm_object
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
chain = prompt | llm_object

while True:
    inp = input("Enter the query: ")
    result = chain.invoke({"user_input": inp, "chat_history": memory.buffer})
    print(result)
    memory.save_context({"input": inp}, {"output": result})