from langchain_community.embeddings import HuggingFaceEmbeddings  # Sentence embedding model

# ========== EMBEDDING MODEL ==========
# Initialize HuggingFace embeddings for semantic search
def embedding(EMBEDDING_MODEL):
    if EMBEDDING_MODEL is None:
        print("\n\nNo embedding model specified!\n\n")
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    return embedding