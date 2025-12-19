import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Fetch from .env or directly assign as needed
api_key = ""
api_base = "https://chatgpt-key.openai.azure.com/"
deployment_name = "gpt-4o-2"
api_version = "2024-12-01-preview"  # Ensure this is a valid API version

print(f"API Key: {api_key}")

# Initialize the model
llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    api_version=api_version,
    api_key=api_key,
    azure_endpoint=api_base,  
    model_name="gpt-4o",       
    temperature=0.7,
    max_tokens=100,  # Maximum number of tokens in the response
)


# Use the predict method instead of chat
response = llm.predict("Hello, how are you?")

# Print the response
print(response)
