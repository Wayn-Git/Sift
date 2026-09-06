import os

from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException, PineconeException

from langchain_pinecone import PineconeVectorStore

from embeddings.nvidia_embeddings import NVIDIAEmbeddings
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


def get_nvidia_key(nvidia_api_key: str | None = None) -> str:
    """Resolve NVIDIA API key: arg > env var."""
    key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError("NVIDIA_API_KEY missing")
    return key


def get_pinecone_info(pinecone_api_key: str | None, pinecone_index_name: str | None = None) -> str:
    """Resolve Pinecone API key: arg > env var."""
    key = pinecone_api_key or PINECONE_API_KEY
    if not key:
        raise ValueError("PINECONE_API_KEY missing")
    index = pinecone_index_name or PINECONE_INDEX_NAME
    return key, index


def get_vector_store(
    pinecone_api_key: str | None = None,
    index_name: str | None = None,
    nvidia_api_key: str | None = None,
) -> PineconeVectorStore:
    """Initialize and return a PineconeVectorStore with NVIDIA embeddings."""
    pc_key, index = get_pinecone_info(pinecone_api_key)
    nv_key = get_nvidia_key(nvidia_api_key)

    index_name = index_name or PINECONE_INDEX_NAME

    pc = Pinecone(api_key=pc_key)
    index = pc.Index(index_name)

    embeddings = NVIDIAEmbeddings(api_key=nv_key)

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )
