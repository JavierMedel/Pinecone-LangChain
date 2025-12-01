import os
import time
from pinecone import Pinecone, ServerlessSpec 

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
    
# Set your API keys for Pinecone
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# Create Index if not already created
print(PINECONE_INDEX)

if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX, 
        dimension=1536, # '1536' is the dimension for ada-002 embeddings
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
     
    while not pc.describe_index(PINECONE_INDEX).index.status['ready']:
        time.sleep(1)
    
    print("Pinecone Index provisioned")
else:
    print("Pinecone Index Already Provisioned")