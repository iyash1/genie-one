# ========== IMPORTS ==========
import gradio as gr
import os
import shutil

from langchain_community.llms import Ollama

from src.ingest import ingest_documents, load_documents
from src.constants import DOCS_PATH, LLAMA_MODEL, OLLAMA_LOCAL_URL
from src.db import load_db

# Ensure docs directory exists
os.makedirs(DOCS_PATH, exist_ok=True)

# ========== LLM INITIALIZATION ==========
llm = Ollama(
    model=LLAMA_MODEL,
    base_url=OLLAMA_LOCAL_URL
)

# ========== CHAT FUNCTION ==========
def chat_fn(message, history):
    print(f"\n=== RECEIVED MESSAGE: {message} ===")

    db = load_db()

    # ✅ Use MMR for better diversity across documents
    docs = db.max_marginal_relevance_search(
        message,
        k=8,
        fetch_k=25
    )

    # ✅ Remove weak chunks
    docs = [d for d in docs if len(d.page_content) > 100]

    # ✅ Prioritize richer chunks
    docs = sorted(docs, key=lambda x: len(x.page_content), reverse=True)

    print("\n--- RETRIEVED DOCUMENTS ---")
    for d in docs:
        print("SOURCE:", d.metadata.get("source"))
        print("TEXT:", d.page_content[:150])
        print("-------------------------")

    # ✅ Structured context with source separation
    context = "\n\n".join([
        f"""
        DOCUMENT: {doc.metadata.get('source')}
        CONTENT:
        {doc.page_content}
        """
        for doc in docs
    ])

    # ✅ Strong prompt
    prompt = f"""
        You are answering a question using multiple documents.

        Carefully read ALL the context below.

        - Combine information from multiple chunks if needed
        - If partial information exists, provide the most complete answer possible
        - Only say "I don't know" if absolutely nothing relevant is found

        Context:
        {context}

        Question: {message}
    """

    print("\n--- CONTEXT PREVIEW ---")
    print(context[:1000])
    print("------------------------")
    
    response = llm.invoke(prompt)

    # ✅ Show sources
    sources = sorted(set(d.metadata.get("source") for d in docs))

    response_with_sources = f"""{response}
    Sources:
    {chr(10).join(f"- {s}" for s in sources)}
    """

    history.append((message, response_with_sources))
    return history, history


# ========== FILE UPLOAD FUNCTION ==========
def upload_and_ingest(files, progress=gr.Progress()):
    saved_files = []

    for file in files:
        filepath = str(file)
        filename = os.path.basename(filepath)
        dest_path = os.path.join(DOCS_PATH, filename)

        shutil.copy(filepath, dest_path)
        saved_files.append(filename)

    # ✅ Only ingest new files
    documents = load_documents(saved_files)

    def update_progress(p, msg):
        progress(p, desc=msg)

    result = ingest_documents(documents, update_progress)

    return f"Uploaded: {saved_files}\n\n{result}"


# ========== VIEW FUNCTIONS ==========

# Show files actually ingested (from DB metadata)
def view_ingested_docs():
    print("FETCHING INGESTED DOCUMENTS...")
    db = load_db()
    data = db.get()

    sources = set()

    for meta in data["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])

    return "\n".join(sorted(sources)) if sources else "No documents ingested."


# Show raw uploaded files
def view_uploaded_files():
    print("FETCHING UPLOADED FILES...")
    files = os.listdir(DOCS_PATH)

    if not files:
        return "No files uploaded."

    return "\n".join([f"{i+1}. {file}" for i, file in enumerate(files)])


# ========== UI ==========
with gr.Blocks() as app:
    gr.Markdown("# 🚀 Local RAG App")

    with gr.Tabs():

        # ---------- CHAT ----------
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(placeholder="Ask something...")
            clear = gr.Button("Clear")

            msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot])
            clear.click(lambda: [], None, chatbot)

        # ---------- UPLOAD ----------
        with gr.Tab("Upload & Ingest"):
            file_upload = gr.File(file_count="multiple")
            upload_btn = gr.Button("Upload & Ingest")
            output = gr.Textbox()

            upload_btn.click(
                upload_and_ingest,
                inputs=file_upload,
                outputs=output
            )

        # ---------- VIEW INGESTED ----------
        with gr.Tab("View Ingested Files"):
            btn_ingested = gr.Button("Show Ingested Files")
            out_ingested = gr.Textbox(lines=20)

            btn_ingested.click(view_ingested_docs, None, out_ingested)

        # ---------- VIEW UPLOADED ----------
        with gr.Tab("View Uploaded Files"):
            btn_uploaded = gr.Button("Show Uploaded Files")
            out_uploaded = gr.Textbox(lines=20)

            btn_uploaded.click(view_uploaded_files, None, out_uploaded)


# ========== APP ENTRY ==========
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True # Important for Docker implementation to allow external access
    )