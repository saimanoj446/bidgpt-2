import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Paths
documents_path = "docs"
persist_directory = "chroma"

# Load and split documents
all_docs = []
for filename in os.listdir(documents_path):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(documents_path, filename))
        all_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(all_docs)

# Create vector store using lighter embedding model
embedding = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
vectordb = Chroma.from_documents(split_docs, embedding=embedding, persist_directory=persist_directory)

print("✅ Chroma DB created and persisted in:", persist_directory)
