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
    
    # Check if user wants to exit the loop
    if query == "exit":
        break

    # Retrieve the top 3 most relevant documents from vector DB using semantic similarity
    # The query is converted to embeddings and matched against stored document embeddings
    docs = db.similarity_search(query, k=3)

    # Combine the content of retrieved documents into a single context string
    # Separated by double newlines for readability in the prompt
    context = "\n\n".join([doc.page_content for doc in docs])

    # Construct the prompt for the LLM with the retrieved context and user query
    # This ensures the model answers based only on provided context (RAG approach)
    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    # Send the prompt to the Ollama LLM and get the response
    response = llm.invoke(prompt)

    # Display the LLM's answer to the user
    print("\nAnswer:", response)