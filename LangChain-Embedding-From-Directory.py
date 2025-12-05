import os
from dotenv import load_dotenv

# LangChain imports (NEW 2024 format)
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   # <-- FIXED
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Get API keys and index name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # Specify which embedding model

# Load PDF files from a directory
directory_path = "./papers"
loader = DirectoryLoader(
    directory_path,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# print(documents)

# Split PDFs into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

split_documents = text_splitter.split_documents(documents)

print(len(split_documents))

# -------------------------
# Connect to Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

print(index)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# Add chunks to Pinecone
vectorstore.add_documents(split_documents)

print("PDF embeddings created and inserted into Pinecone successfully!")
