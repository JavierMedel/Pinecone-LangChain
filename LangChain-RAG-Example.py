import os
from dotenv import load_dotenv
import openai

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone

# Set your API keys for OpenAI and Pinecone
# Load environment variables
load_dotenv()

# Get API keys and index name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize OpenAI Embeddings using LangChain
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Specify which embedding model

# -------------------------
# Connect to Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Connect to the Pinecone index using LangChain's Pinecone wrapper
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# Define the retrieval mechanism
retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve top-1 relevant documents

# Initialize GPT-4 with OpenAI
llm = ChatOpenAI( model="gpt-5-nano", openai_api_key=OPENAI_API_KEY, temperature=0.7 )

# Define Prompt Template
prompt_template = PromptTemplate(
    template="""
    Use the following context to answer the question as accurately as possible:
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)

# Create LLM Chain
llm_chain = prompt_template | llm | StrOutputParser()

# Retrieve documents
query = "What are main idea behind SAM 3"
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])
    
# Run LLM chain with the retrieved context
answer = llm_chain.invoke({"context": context, "question": query})

# Output the Answer and Sources
print("Answer:", answer)