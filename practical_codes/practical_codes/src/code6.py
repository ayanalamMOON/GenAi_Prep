from langchain.prompts import PromptTemplate

prompt=PromptTemplate(
    input_variables=["input"],
    template="What is the capital of {input}?",
)

print(prompt.format(input="China"))