import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.db import store_db
from src.constants import DOCS_PATH, CHUNK_SIZE_500, CHUNK_OVERLAP_100

# Upload Function: Handles document loading
def load_documents():
    documents = []

    for filename in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, filename)

        # Support PDF and TXT files for ingestion; skip unsupported formats
        # PDFs are loaded with PyPDFLoader, while text files are loaded with TextLoader (with fallback encoding)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load()

        elif filename.endswith(".txt"):
            try:
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
            except Exception:
                loader = TextLoader(path, encoding="latin-1")
                docs = loader.load()
        else:
            continue

        for doc in docs:
            doc.metadata["source"] = filename

        documents.extend(docs)

    return documents

# Ingest Function: Main function to  split and store uploaded documents in the vector database
def ingest(progress_callback=None):
    documents = load_documents()

    # Progress bar for UI update
    if progress_callback:
        progress_callback(0.2, "Loaded documents")

    # Initialize text splitter
    # uses overlap to maintain context across chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_500,
        chunk_overlap=CHUNK_OVERLAP_100
    )

    # Split the documents using the text splitter
    docs = text_splitter.split_documents(documents)

    if progress_callback:
        progress_callback(0.5, "Split into chunks")
    
    # Store the chunks in the vector database 
    store_db(docs)
    if progress_callback:
        progress_callback(1.0, "Ingestion complete")

    return f"Ingested {len(docs)} chunks."


if __name__ == "__main__":
    print(ingest())