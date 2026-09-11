from langchain_community.retrievers import PineconeHybridSearchRetriever

from pinecone import Pinecone

from embeddings.dense_embed import DenseEmbeddings, DenseEncoderAdapter
from embeddings.sparec_embed import SparseEmbeddings, SparseEncoderAdapter
from client.vector_handler import get_pinecone_info
from config import NAMESPACE, TEXT_KEY, PINECONE_INDEX_NAME


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

        pc = Pinecone(api_key=key)

        index_object = pc.Index(PINECONE_INDEX_NAME)

        self.retriever = PineconeHybridSearchRetriever(
            embeddings=DenseEmbeddings(nvidia_api_key=nvidia_api_key).embeddings,
            sparse_encoder=SparseEncoderAdapter(
                SparseEmbeddings(pc_api_key=key)
            ),
            index=index_object,
            namespace=namespace,
            top_k=top_k,
            text_key=TEXT_KEY,
        )

        
    def retrieve(self, query: str, top_k: int = 5):
        # Use .invoke() and pass top_k via the search_kwargs configuration
        return self.retriever.invoke(
            query, 
            config={"search_kwargs": {"top_k": top_k}}
        )
