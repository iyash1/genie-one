# ========== IMPORTS ==========
# Gradio: Framework for building web interfaces for ML models
import gradio as gr
# OS and file operations: Path management and file utilities
import os
import shutil

# LangChain Community: Tools for LLM integration and RAG (Retrieval-Augmented Generation)
from langchain_community.llms import Ollama  # Local LLM via Ollama


# Custom ingestion module: Handles document processing and vector store population
from src.ingest import ingest
from src.constants import DOCS_PATH, LLAMA_MODEL, OLLAMA_LOCAL_URL
from src.db import load_db  # Function to load the Chroma vector database

# Ensure docs directory exists
os.makedirs(DOCS_PATH, exist_ok=True)

# ========== LLM INITIALIZATION ==========
# Initialize local Ollama LLM model
# Connects to Ollama service running on Docker container
llm = Ollama(
    model=LLAMA_MODEL,
    base_url=OLLAMA_LOCAL_URL
)

# ========== CHAT FUNCTION ========
def chat_fn(message, history):
    """
    Core RAG (Retrieval-Augmented Generation) chat function.
    Retrieves relevant documents from vector store and generates context-aware responses.
    
    Args:
        message (str): User's query
        history (list): Chat history of previous messages and responses
    
    Returns:
        tuple: Updated (history, history) for Gradio chatbot display
    """
    print(f"RECEIVED MESSAGE: {message}")
    # Load vector database connection
    db = load_db()

    # Retrieve top 3 most similar documents to the user's query
    docs = db.similarity_search(message, k=3)

    # Combine retrieved documents into a single context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Construct prompt with instructions and context for the LLM
    prompt = f"""
    Answer based only on context.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question: {message}
    """

    # Get response from local LLM
    response = llm.invoke(prompt)

    # Append user message and AI response to chat history
    history.append((message, response))
    return history, history


# ========== FILE UPLOAD FUNCTION ========
def upload_and_ingest(files, progress=gr.Progress()):
    """
    Handle file upload and trigger ingestion into vector database.
    Copies uploaded files to docs directory and processes them into embeddings.
    
    Args:
        files (list): List of file paths uploaded by user
    
    Returns:
        str: Success message with uploaded filenames and ingestion result
    """
    print("BEGINNING UPLOAD AND INGESTION PROCESS...")

    # Track successfully uploaded files
    saved_files = []

    # Process each uploaded file
    for file in files:
        # Extract filename from file path
        filename = os.path.basename(file)

        # Define destination path in docs directory
        dest_path = os.path.join(DOCS_PATH, filename)

        # Copy file to docs directory
        shutil.copy(file, dest_path)

        # Record filename
        saved_files.append(filename)

    # Progress-aware ingestion
    def update_progress(p, msg):
        progress(p, desc=msg)
    
    # Trigger ingestion process to embed documents into vector store
    result = ingest(progress_callback=update_progress)

    return f"Uploaded: {saved_files}\n\n{result}"

# ========== VIEW FUNCTIONS ========
# Functions to view ingested documents files for debugging and transparency in the UI
# Files can be uploaded but not ingested if they are in an unsupported format, so this helps users understand what data is available for retrieval
def view_ingested_docs():
    print("FETCHING INGESTED DOCUMENTS...")
    db = load_db()
    data = db.get()

    sources = set()

    for meta in data["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])

    return "\n".join(sorted(sources))

# View uploaded files in the docs directory
def view_uploaded_files():
    print("FETCHING UPLOADED FILES...")
    files = os.listdir(DOCS_PATH)
    return "\n".join([f"{i+1}. {file}" for i, file in enumerate(files)])

# -------- UI --------
with gr.Blocks() as app:
    gr.Markdown("# Local RAG App")

    with gr.Tabs():

        # TAB 1 — CHAT
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Ask something...")
            clear = gr.Button("Clear")

            msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot])
            clear.click(lambda: [], None, chatbot)

        # TAB 2 — UPLOAD
        with gr.Tab("Upload & Ingest"):
            file_upload = gr.File(file_count="multiple")
            upload_btn = gr.Button("Upload & Ingest")
            output = gr.Textbox()

            upload_btn.click(
                upload_and_ingest,
                inputs=file_upload,
                outputs=output
            )

        # TAB 3 — VIEW INGESTED DOCS
        with gr.Tab("View Ingested Files"):
            btn = gr.Button("Show Ingested Data")
            out = gr.Textbox(lines=20)

            btn.click(view_ingested_docs, None, out)

        # TAB 4 — VIEW UPLOADED FILES
        with gr.Tab("View Files Uploaded"):
            btn = gr.Button("Show Uploaded Files")
            out = gr.Textbox(lines=20)

            btn.click(view_uploaded_files, None, out)

# Launch the Gradio app on all network interfaces (important for Docker) and share it publicly
if __name__ == "__main__":
    app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True   # important for Docker environment
)