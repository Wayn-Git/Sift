import os

from pinecone import Pinecone


class SparseEmbeddings:

    def __init__(
        self,
        pc_api_key: str | None = None,
        model: str = "pinecone-sparse-english-v0",
    ):

        if pc_api_key is None:
            pc_api_key = os.environ.get("PINECONE_API_KEY")

        if pc_api_key is None:
            raise ValueError("Pinecone API key not provided")

        self.pc = Pinecone(api_key=pc_api_key)
        self.model = model

    def get_sparse_model(self):
        return self.model 

    def generate_embeddings(
        self,
        texts: list[str],
        input_type: str = "passage",
) -> list[dict]:

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every item in texts must be a string")

        response = self.pc.inference.embed(
            model=self.model,
            inputs=texts,
            parameters={
                "input_type": input_type,
                "truncate": "END",
            },
        )

        return [
            {
                "indices": item["sparse_indices"],
                "values": item["sparse_values"],
            }
            for item in response
        ]


class SparseEncoderAdapter:
    """Adapter exposing encode_queries/encode_documents for hybrid retriever."""
    def __init__(self, encoder: SparseEmbeddings):
        self.encoder = encoder

    def encode_queries(self, text: str) -> dict:
        return self.encoder.generate_embeddings([text], input_type="query")[0]

    def encode_documents(self, texts: list[str]) -> list[dict]:
        return self.encoder.generate_embeddings(texts, input_type="passage")