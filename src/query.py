# Import required libraries for RAG (Retrieval-Augmented Generation) pipeline
from langchain_community.llms import Ollama
from src.db import load_db
from src.constants import LLAMA_MODEL, OLLAMA_LOCAL_URL

# Load the Chroma vector database from persistent storage
# The database contains pre-indexed documents converted to embeddings for fast retrieval
db = load_db()

# Initialize the Ollama LLM client
# Connects to a local Ollama instance running the llama3 model on port 11434
llm = Ollama(model=LLAMA_MODEL, base_url=OLLAMA_LOCAL_URL)

# Main query loop: continuously prompt user for questions until 'exit' command
while True:
    # Prompt user to enter a question or type 'exit' to quit the program
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break

    docs = db.max_marginal_relevance_search(query, k=5, fetch_k=10)

    print("\nRetrieved sources:")
    for d in docs:
        print(d.metadata.get("source"))

    context = "\n\n".join([
        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
        for doc in docs
    ])

    prompt = f"""
    You are answering using multiple documents.

    Use ALL relevant information from the context.
    Combine information from multiple sources when needed.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    sources = set(d.metadata.get("source") for d in docs)

    print("\nAnswer:", response)
    print("\nSources:")
    for s in sources:
        print("-", s)