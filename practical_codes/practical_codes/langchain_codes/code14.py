from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

text = "I love machine learning"

embedding = embedding_model.embed_query(text)

print(f"Embedding vector length: {len(embedding)}")
print(embedding)