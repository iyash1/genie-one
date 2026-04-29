FROM python:3.10-slim

# Avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
# COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pydantic==2.10.6

RUN pip install --no-cache-dir gradio-client==1.3.0 gradio==4.44.1 fastapi==0.110.0 starlette==0.36.3 jinja2==3.1.3

RUN pip install --no-cache-dir langchain==0.1.0 langchain-community==0.0.10 chromadb==0.4.24

RUN pip install --no-cache-dir sentence-transformers==2.2.2 pypdf==3.17.1 huggingface-hub==0.19.4

# Copy app files
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Run app
CMD ["python", "app.py"]