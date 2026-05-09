# Genie One — Local RAG App 🧠📚

> This requires a huge amount of disk, approximately 12GB ~ 15GB

A simple local Retrieval-Augmented Generation (RAG) demo that:
- ingests PDFs (and TXT) into a Chroma vector DB
- creates embeddings with HuggingFace sentence-transformers
- answers queries via a local LLM (Ollama) from the stored context
- provides both a CLI and a Gradio web UI
- includes optional Docker Compose setup for local deployment

> Note: The repository ignores the contents of `/docs` so you can add your own documents safely.

---

## Features ✨
- PDF & TXT ingestion → chunking → embedding → Chroma DB
- CLI query tool (`query.py`)
- Gradio web UI (`app.py`)
- Docker + docker-compose support with an Ollama service
- Minimal dependency set (see `Dockerfile` / `requirements.txt` if present)
- Helper scripts (e.g., `scripts/reset.sh`) to wipe and reset local state

---

## Project structure

```
genie-one/
├─ README.md
├─ Dockerfile
├─ docker-compose.yml
├─ .gitignore
│
├─ Core Scripts
│  ├─ ingest.py            # PDF/TXT ingestion -> chunking -> embeddings
│  ├─ query.py             # CLI query tool
│  └─ app.py               # Gradio web UI
│
├─ Configuration & Data
│  ├─ docs/                # Add your PDFs/TXT here (gitignored)
│  ├─ db/                  # Chroma persist directory (generated, gitignored)
│  └─ storage/             # Optional: alternative data storage location
│
├─ Scripts & Utilities
│  ├─ scripts/
│  │  └─ reset.sh          # wipe db, stop containers, prune docker
│  └─ utils/               # Utility functions & helpers
│
├─ Testing & Documentation
│  ├─ tests/               # unit / integration tests
│  ├─ design/              # design docs & flowcharts
│  └─ examples/            # example usage & notebooks
│
└─ Dependencies
   └─ requirements.txt     # Python package dependencies (if present)
```

---

## Prerequisites ✅
- Miniconda (recommended) or system Python 3.10+
- git
- Docker & docker-compose (optional, recommended for easy local deployment)
- Ollama (if running without Docker) — ensure your LLM is installed and running if you plan to use `query.py` / `app.py` directly

---

## Quick Setup (local / Miniconda) 🐍

1. Clone the repo
```bash
git clone https://github.com/iyash1/genie-one.git
cd genie-one
```

2. (Optional) Create and activate conda environment
```bash
conda create -n genie-one python=3.10 -y
conda activate genie-one
```

3. Install Python deps
```bash
pip install -r requirements.txt
# or install packages listed in Dockerfile
```

4. (Optional) Install and start Ollama or ensure your LLM service is available
- The code expects an Ollama-compatible service (default model `"llama3"`). Adjust `query.py` / `app.py` if using another LLM.

---

## Run with Docker Compose (recommended)
This repo includes a docker-compose.yml that starts both the app and an Ollama container (it will attempt to ensure `llama3` is available).

1. Build & start
```bash
docker compose up --build
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama list
```

2. App will be available at http://localhost:7860 (Gradio)

3. To stop and wipe (see scripts/reset.sh for a helper)
```bash
docker compose down -v
```

---

## Using the Project (local Python)

1. Add your documents
- Place PDFs or .txt files into the `docs/` folder.

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

4. Run Gradio web UI
```bash
python app.py
```
- Open the provided localhost URL, ask questions, and ingest files via the Upload tab.

---

## Reset / Cleanup
A helper script exists at `scripts/reset.sh` that:
- removes the `db/` directory
- stops containers and removes volumes
- prunes Docker images and volumes

Use with caution — it deletes local DB and prunes Docker resources.

---

## Configuration & Customization ⚙️
- Embeddings model: change `model_name` in `ingest.py`, `query.py`, and `app.py` (`sentence-transformers/all-MiniLM-L6-v2` by default).
- LLM: change `Ollama(model="llama3")` and `base_url` if using another model/service.
- Chroma persist dir: `db/` — safe to back up / reuse.
- Docker Compose: `OLLAMA_BASE_URL` env var is set in `docker-compose.yml` and the Ollama service attempts to pull `llama3`.

---

## Troubleshooting 🩺
- If ingestion is slow: try smaller chunk sizes or a lighter embedding model.
- If Ollama connection fails: ensure Ollama (or the Docker Ollama service) is running and the model is available.
- If Gradio shows stale results: stop the app, re-run `ingest.py`, and restart the app.

---

Enjoy building on this local RAG starter! 🚀