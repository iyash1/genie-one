import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("Local RAG App")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

llm = Ollama(model="llama3")

query = st.text_input("Ask a question")

if query:
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer based only on context.

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    st.write(response)