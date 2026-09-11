# Importing modules

from retriever.retriever import HybridRetriever
from client.vector_handler import get_nvidia_key, get_pinecone_info
import config

key, index = get_pinecone_info(config.PINECONE_API_KEY)
n_key = get_nvidia_key()

Ran = HybridRetriever(key, index, n_key)



print(Ran.retrieve(query="What is attention?", top_k=10))