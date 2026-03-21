import os
print("Current working directory:", os.getcwd())

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
# Get query
# Get query
query = input("\nEnter your question: ")

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke(query)

print("\nRetrieved Chunks:\n")
for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}:")
    print(doc.page_content)

# Build context AFTER retrieval
context = "\n\n".join([doc.page_content for doc in docs])

# Final prompt
prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{query}
"""

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs, max_new_tokens=150)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nFinal Answer:\n")
print(answer)
