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

# #1. CharacterTextSplitter

from langchain_text_splitters import CharacterTextSplitter


chunking=CharacterTextSplitter(chunk_size=70,chunk_overlap=39)


docs=chunking.split_documents(documents)

print(len(docs))


for i in range(len(docs)):
    print(docs[i])
    print("------------------------------------------")