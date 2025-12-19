# In a RAG pipeline you must:

# 1. Collect documents (PDFs, text files, DOCX, website pages, database rows, etc.)


# 2. Load them into your program


# 3. Split them into chunks


# 4. Convert them into embeddings


# 5. Store them in a vector database



# Here we focus on Step 2 — Loading documents.


# ---

# Most Common Ways to Load Documents for RAG

# Here are the most frequently used loaders in Python (LangChain):


# ---

# Load PDFs

# Using LangChain’s PyPDFLoader

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print(documents)




# Load DOCX / Word files

from langchain.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("sample.docx")
documents = loader.load()



# Load Text Files (.txt)

from langchain.document_loaders import TextLoader

loader = TextLoader("notes.txt")
documents = loader.load()


# Load Multiple Documents in a Folder

from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "data/", 
    glob="*/.pdf",  
    show_progress=True
)

documents = loader.load()


Load Website Pages (HTML)

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("http://shooliniuniversity.com/")
											
documents = loader.load()


# Load Markdown Files (.md)

from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("readme.md")
documents = loader.load()


# Load CSV Files

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader("data.csv")
documents = loader.load()

