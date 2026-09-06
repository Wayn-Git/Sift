from langchain_community.retrievers import PineconeHybridSearchRetriever

from embeddings.dense_embed import DenseEmbeddings
from embeddings.sparec_embed import SparseEmbeddings, SparseEncoderAdapter
from client.vector_handler import get_pinecone_info
from config import NAMESPACE, TEXT_KEY


class HybridRetriever:
    def __init__(
        self,
        pinecone_api_key: str | None = None,
        pinecone_index_name: str | None = None,
        nvidia_api_key: str | None = None,
        namespace: str = NAMESPACE,
        top_k: int = 5,
    ):
        key, index = get_pinecone_info(pinecone_api_key, pinecone_index_name)

        self.retriever = PineconeHybridSearchRetriever(
            embeddings=DenseEmbeddings(nvidia_api_key=nvidia_api_key).embeddings,
            sparse_encoder=SparseEncoderAdapter(
                SparseEmbeddings(pc_api_key=key)
            ),
            index=index,
            namespace=namespace,
            top_k=top_k,
            text_key=TEXT_KEY,
        )

    def retrieve(self, query: str, top_k: int = 5):
        self.retriever.top_k = top_k
        return self.retriever.invoke(query)
