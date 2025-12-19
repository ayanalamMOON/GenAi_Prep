from langchain_community.document_loaders import TextLoader


path='ipl.txt'

def load_doc(path):
  if path.endswith('.txt'):
    loader=TextLoader(path)
  documents=loader.load()
  return documents

documents=load_doc(path)
print(documents)


# #Now we do chunking as already told model have limit and answer also become accurate 


from langchain_community.embeddings import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


from langchain_experimental.text_splitter import SemanticChunker


chunker = SemanticChunker(embeddings)
chunks = chunker.create_documents([documents])

for c in chunks:
    print(c.page_content)