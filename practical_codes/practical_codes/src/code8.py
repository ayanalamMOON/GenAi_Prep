from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
import os
load_dotenv()

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
examples = [
{"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."},
{"question": "What is the capital of Italy?", "answer": "The capital of Italy is Rome."},
]

examples_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}",

)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    examples_prompt=examples_prompt,
    input_variables=["question"],
    suffix="Answer:",
    prefix="You are a helpful assistant. Answer the following question.",
)

formatted_prompt = few_shot_prompt.format(question="What is the capital of Spain?")
print(formatted_prompt)

llmchain=LLMChain(
    llm=llm,
    prompt=few_shot_prompt,
)
result=llmchain.invoke({"question":"What is the capital of Spain?"})
print(formatted_prompt)
print(result['text'])



