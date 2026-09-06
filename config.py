import os

from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME") or "sift"
NAMESPACE = "hrbird"
TEXT_KEY = "text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
PARSED_DIR = "data/parsed"
