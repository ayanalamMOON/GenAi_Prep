
import os
from langchain_openai import AzureOpenAIEmbeddings

AZURE_OPENAI_API_KEY = ""  #Put your own API key
AZURE_OPENAI_ENDPOINT = "https://chatgpt-key.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-2"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-large" # Replace with your embedding deployment name
AZURE_OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-large" # Replace with your embedding model name


embeddings = AzureOpenAIEmbeddings(
    deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

text = "I love machine learning"


embedding = embeddings.embed_query(text)

print(f"Embedding vector length: {len(embedding)}")
print(embedding)