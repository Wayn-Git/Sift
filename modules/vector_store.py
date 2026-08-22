import os

from pinecone import Pinecone
from pinecone.exceptions import PineconeApiException, PineconeException

from langchain_pinecone import PineconeVectorStore

try:
    from embed import NVIDIAEmbeddings
except ModuleNotFoundError:
    from modules.embed import NVIDIAEmbeddings


class VectorStore:
    def __init__(
        self,
        pinecone_api_key: str | None = None,
        index_name: str | None = None,
        nvidia_api_key: str | None = None,
        # backward compat with old signature VectorStore(api_key, INDEX_NAME)
        api_key: str | None = None,
        INDEX_NAME: str | None = None,
    ):
        # resolve keys  explicit arg > env
        pinecone_api_key = pinecone_api_key or api_key or os.getenv("PINECONE_API_KEY")
        nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        index_name = index_name or INDEX_NAME or os.getenv("PINECONE_INDEX_NAME") or "sift"

        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY missing")
        if not nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY missing")

        self.PINECONE_API_KEY = pinecone_api_key
        self.INDEX_NAME = index_name

        pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index = pc.Index(self.INDEX_NAME)

        embeddings = NVIDIAEmbeddings(api_key=nvidia_api_key)

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=embeddings,
        )

    def add_to_store(self, chunks):
        try:
            self.vector_store.add_documents(chunks)
            return "Added Successfully"
        except PineconeException as e:
            print(f"Pinecone Error: {e}")
            raise

    def vector_search(self, query: str, k: int = 4):
        try:
            return self.vector_store.similarity_search(query, k=k)
        except PineconeException as e:
            print(f"Error occured while searching: {e}")
            raise
