# Genie One — Local RAG App 🧠📚

A simple local Retrieval-Augmented Generation (RAG) demo that:
- ingests PDFs into a Chroma vector DB
- creates embeddings with HuggingFace sentence-transformers
- answers queries via a local LLM (Ollama) from the stored context
- provides both CLI and Streamlit UI interfaces

> Note: The repository ignores the contents of `/docs` so you can add your own documents safely.

---

## Features ✨
- PDF ingestion → chunking → embedding → Chroma DB
- CLI query tool (`query.py`)
- Streamlit web UI (`app.py`)
- Minimal dependency set (see `requirements.txt`)

---

## Project structure

```
genie-one/
├─ README.md
├─ ingest.py            # PDF ingestion -> chunking -> embeddings
├─ query.py             # CLI query tool
├─ app.py               # Streamlit UI
├─ requirements.txt
├─ docs/                # Add your PDFs here (gitignored)
├─ db/                  # Chroma persist directory (generated)
├─ models/              # Optional local/downloaded model files
├─ scripts/             # helper scripts (e.g., maintenance, build)
├─ notebooks/           # optional exploration notebooks
├─ tests/               # unit / integration tests
├─ .gitignore
├─ env.example          # example env vars
└─ LICENSE
```
---

## Prerequisites ✅
- Miniconda (recommended)
- git
- Python 3.10+ (will be installed inside the conda env)
- Ollama (or another local LLM accessible via LangChain) — ensure your LLM is installed and running if you plan to use `query.py` / `app.py`

---

## Quick Setup (Miniconda) 🐍

1. Clone the repo
```bash
git clone <repo-url>
cd genie-one
```

2. Create and activate conda environment
```bash
conda create -n genie-one python=3.10 -y
conda activate genie-one
```

3. Install Python deps
```bash
pip install -r requirements.txt
```

4. (Optional) Install and start Ollama or other local LLM
- Follow your LLM provider's install/run instructions. The code expects an Ollama-compatible local service (model `"llama3"` by default). Adjust `query.py` / `app.py` if using another LLM.

---

## Using the Project

1. Add your documents
- Place PDFs you want to index into the `docs/` folder (this folder's contents are intentionally not tracked).

2. Ingest documents (build vector DB)
```bash
python ingest.py
```
- This will chunk documents and store embeddings into `db/` (Chroma persist directory).

3. Query via CLI
```bash
python query.py
```
- Type questions; enter `exit` to quit.

4. Run Streamlit UI
```bash
streamlit run app.py
```
- Open the provided localhost URL, type a question, and get answers.

---

## Configuration & Customization ⚙️
- Embeddings model: change `model_name` in `ingest.py`, `query.py`, and `app.py` (`sentence-transformers/all-MiniLM-L6-v2` by default).
- LLM: change `Ollama(model="llama3")` to the model you have available or replace with another LangChain-compatible LLM.
- Chroma persist dir: `db/` — safe to back up / reuse.

---

## Troubleshooting 🩺
- If ingestion is slow: try smaller chunk sizes or a lighter embedding model.
- If Ollama connection fails: ensure Ollama daemon/service is running and the model is available locally.
- If Streamlit shows stale results: stop app, re-run `ingest.py` and restart Streamlit.

---

## TODO (project ideas) 🧩
- Containerize (Docker / Docker Compose)
- Support multiple document types & directories
- Standardized UI with conversation history
- Add `README.md` (this file) and env constants
- Environment variable support for models and paths

(From repo TODOs)

---

Enjoy building on this local RAG starter! 🚀