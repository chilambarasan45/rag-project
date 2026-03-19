import os
print("Current working directory:", os.getcwd())

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings




# Load PDF
loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

print(f"Total pages loaded: {len(documents)}")

for i, doc in enumerate(documents):
    print(f"\nPage {i+1} length:", len(doc.page_content))
    print(doc.page_content[:200])

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("\nTotal chunks created:", len(chunks))
print("\nFirst chunk:\n", chunks[0].page_content)

# Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

print("\nVector DB created successfully")
