# Import PyPDFLoader to read PDF documents
from langchain_community.document_loaders import PyPDFLoader

# Import text splitter to break documents into manageable chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import HuggingFace embeddings to convert text to vector representations
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import Chroma vector database for storing and retrieving embeddings
from langchain_community.vectorstores import Chroma

# 1. Load document to learn from
# Initialize the PDF loader with the path to your PDF file
loader = PyPDFLoader("docs/The Ultimate Guide To Body Recomposition b - Unknown.pdf")
# Load the PDF and convert it into LangChain Document objects
documents = loader.load()

# 2. Split the document into chunks
# Create a text splitter with specified chunk size and overlap for context preservation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Each chunk will be max 500 characters
    chunk_overlap=100  # 100 characters of overlap between chunks for continuity
)

# Split all documents into smaller chunks
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
# Initialize the embedding model (a lightweight sentence transformer)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in vector DB
# Create a Chroma vector database from the document chunks using the embeddings
db = Chroma.from_documents(
    docs,  # The split document chunks
    embedding,  # The embedding model to convert text to vectors
    persist_directory="db"  # Directory where the vector database will be stored
)

# Persist the database to disk so it can be reused later
db.persist()

# Notify user that the ingestion process is complete
print("Ingestion complete.")