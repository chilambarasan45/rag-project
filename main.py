import os
print("Current working directory:", os.getcwd())

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from transformers import pipeline


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
query = input("\nEnter your question: ")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

docs = retriever.invoke(query)

print("\nRetrieved Chunks:\n")

for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}:")
    print(doc.page_content)

generator = pipeline("text-generation", model="google/flan-t5-base")


context = "\n\n".join([doc.page_content for doc in docs])

prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{query}
"""

response = generator(prompt, max_length=200)

print("\nFinal Answer:\n")
print(response[0]['generated_text'])
