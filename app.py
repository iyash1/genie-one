import gradio as gr
import os
import shutil

from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingest import ingest

DOCS_PATH = "docs"
DB_PATH = "db"

os.makedirs(DOCS_PATH, exist_ok=True)

# Load embedding + DB
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_db():
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )

# Load LLM
llm = Ollama(
    model="llama3",
    base_url="http://ollama:11434"
)

# -------- CHAT FUNCTION --------
def chat_fn(message, history):
    db = load_db()

    docs = db.similarity_search(message, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer based only on context.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question: {message}
    """

    response = llm.invoke(prompt)

    history.append((message, response))
    return history, history


# -------- FILE UPLOAD FUNCTION --------
def upload_and_ingest(files):
    print("DEBUG: NEW VERSION RUNNING")
    saved_files = []

    for file in files:
        # file is a path-like object
        filename = os.path.basename(file)
        dest_path = os.path.join(DOCS_PATH, filename)

        shutil.copy(file, dest_path)

        saved_files.append(filename)

    result = ingest()

    return f"Uploaded: {saved_files}\n\n{result}"


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


if __name__ == "__main__":
    app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True   # important for Docker env
)