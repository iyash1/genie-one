import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.db import store_db
from src.constants import DOCS_PATH, CHUNK_SIZE_500, CHUNK_OVERLAP_100

def load_documents():
    documents = []

    for file in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            try:
                loader = TextLoader(path, encoding="utf-8")
                documents.extend(loader.load())
            except Exception:
                # fallback encoding
                loader = TextLoader(path, encoding="latin-1")
                documents.extend(loader.load())

    return documents


def ingest(progress_callback=None):
    documents = load_documents()

    if progress_callback:
        progress_callback(0.2, "Loaded documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_500,
        chunk_overlap=CHUNK_OVERLAP_100
    )

    docs = text_splitter.split_documents(documents)
    if progress_callback:
        progress_callback(0.5, "Split into chunks")
    
    store_db(docs)
    if progress_callback:
        progress_callback(1.0, "Ingestion complete")

    return f"Ingested {len(docs)} chunks."


if __name__ == "__main__":
    print(ingest())