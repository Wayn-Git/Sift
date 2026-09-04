from langchain_community.retrievers import PineconeHybridSearchRetriever

from embeddings.dense_embed import DenseEmbeddings
from embeddings.sparec_embed import SparseEmbeddings
from client.vector_handler import get_pinecone_info


class HybridRetriever:
    def __init__(
        self,
        pinecone_api_key: str | None = None,
        pinecone_index_name: str | None = None,
    ):
        key, index = get_pinecone_info(pinecone_api_key, pinecone_index_name)

        self.retriever = PineconeHybridSearchRetriever(
            embeddings=DenseEmbeddings(nvidia_api_key=None),
            sparse_encoder=SparseEmbeddings(pc_api_key=key).get_sparse_model(),
            key=key,
            index=index,
        )

    def retrieve(self, query: str, top_k: int = 5):
        return self.retriever.invoke(query, top_k=top_k)
