import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.db import load_db, store_db
from src.constants import DOCS_PATH, CHUNK_SIZE_800, CHUNK_OVERLAP_150

# ---------- TEXT CLEANING ----------
def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove excessive whitespace, line breaks, weird spacing
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())

    return text


# ---------- LOAD DOCUMENTS ----------
def load_documents(file_list=None):
    documents = []

    files = file_list if file_list else os.listdir(DOCS_PATH)

    for filename in files:
        path = os.path.join(DOCS_PATH, filename)

        if not os.path.exists(path):
            continue

        print(f"\nLoading file: {filename}")

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
            # CLEAN TEXT (CRITICAL FIX)
            doc.page_content = clean_text(doc.page_content)

            # ✅ Attach metadata
            doc.metadata["source"] = filename

        # 🔍 Debug PDF quality
        if filename.endswith(".pdf") and docs:
            print("\n--- PDF SAMPLE (first 300 chars) ---")
            print(docs[0].page_content[:300])
            print("-----------------------------------")

        documents.extend(docs)

    return documents


# ---------- INGEST ----------
def ingest_documents(documents, progress_callback=None):
    if progress_callback:
        progress_callback(0.2, "Loaded documents")

    # ✅ Better chunking for PDFs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_800,       
        chunk_overlap=CHUNK_OVERLAP_150  
    )

    docs = text_splitter.split_documents(documents)

    print(f"\nTotal chunks before filtering: {len(docs)}")

    docs = filter_existing_docs(docs)

    print(f"Total chunks after filtering: {len(docs)}")

    if progress_callback:
        progress_callback(0.5, "Split into chunks")

    if not docs:
        return "No new documents to ingest."

    store_db(docs)

    if progress_callback:
        progress_callback(1.0, "Ingestion complete")

    return f"Ingested {len(docs)} chunks."


# ---------- FILTER EXISTING ----------
def filter_existing_docs(docs):
    try:
        db = load_db()
        data = db.get()

        existing_sources = set(
            meta["source"]
            for meta in data["metadatas"]
            if meta and "source" in meta
        )

        # Only skip if file already exists AND user is re-uploading same file
        filtered = [
            doc for doc in docs
            if doc.metadata["source"] not in existing_sources
        ]

        if not filtered:
            print("All documents already ingested. Skipping.")

        return filtered

    except Exception:
        # If DB not initialized yet → ingest everything
        return docs


if __name__ == "__main__":
    docs = load_documents()
    print(ingest_documents(docs))