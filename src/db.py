import os
from langchain_community.vectorstores import Chroma
from src.embedding import embedding
from src.constants import DB_PATH, DEFAULT_EMBEDDING_MODEL  # Vector database for embeddings

# ========== DATABASE LOADER ==========
def load_db(db_path=DB_PATH, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Load and initialize the Chroma vector database.
    Returns a Chroma instance connected to the persisted database.
    """
    print(f"Loading database from {db_path}...")
    print(f"Using {embedding_model} for embeddings...")
    return Chroma(
        persist_directory=db_path,
        embedding_function=embedding(embedding_model) # Uses MiniLM-L6-v2: lightweight but effective for semantic similarity
    )

# ========== DATABASE STORER ==========
def store_db(docs, embedding_model=DEFAULT_EMBEDDING_MODEL, db_path=DB_PATH):
    if docs is None or len(docs) == 0:
        print("No documents to store in the database.")
        return
    
    print(f"Storing {len(docs)} documents in the database...")
    print(f"Using {embedding_model} for embeddings...")

    embed = embedding(embedding_model)

    # Check if DB already exists
    if os.path.exists(db_path):
        print("Loading existing database...")
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embed
        )

        db.add_documents(docs)
    else:
        print("Creating new database...")
        db = Chroma.from_documents(
            docs,
            embed,
            persist_directory=db_path
        )

    db.persist()
    print(f"Database stored at {db_path}")

# ========= DATABASE INSPECTOR (FOR DEBUGGING) ==========
def inspect_db(embedding_model=DEFAULT_EMBEDDING_MODEL, db_path=DB_PATH):
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding(embedding_model)
    )

    data = db.get()

    print(f"Total chunks: {len(data['documents'])}")

    for i in range(min(5, len(data['documents']))):
        print("\n--- Chunk ---")
        print("Text:", data['documents'][i][:200])
        print("Metadata:", data['metadatas'][i])