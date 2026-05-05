import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.db import store_db
from src.constants import DEFAULT_EMBEDDING_MODEL, DB_PATH, DOCS_PATH, CHUNK_SIZE_500, CHUNK_OVERLAP_100

def load_documents():
    documents = []

    for file in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())

    return documents


def ingest():
    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_500,
        chunk_overlap=CHUNK_OVERLAP_100
    )

    docs = text_splitter.split_documents(documents)
    store_db(docs)

    return f"Ingested {len(docs)} chunks."


if __name__ == "__main__":
    print(ingest())