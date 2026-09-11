import os

from embeddings.nvidia_embeddings import NVIDIAEmbeddings


class DenseEmbeddings:

    def __init__(
        self,
        nvidia_api_key: str | None = None,
    ):

        if nvidia_api_key is None:
            nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

        if nvidia_api_key is None:
            raise ValueError("NVIDIA API key not provided")

        self.embeddings = NVIDIAEmbeddings(
            api_key=nvidia_api_key
        )

    def get_dense_model(self):
        return NVIDIAEmbeddings.get_model()

    def generate_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every item in texts must be a string")

        return self.embeddings.embed_documents(texts)

class DenseEncoderAdapter:
    """Adapter exposing encode_queries/encode_documents for hybrid retriever."""
    def __init__(self, encoder: DenseEmbeddings):
        self.encoder = encoder

    def encode_queries(self, text: str) -> dict:
        return self.encoder.generate_embeddings([text], input_type="query")[0]

    def encode_documents(self, texts: list[str]) -> list[dict]:
        return self.encoder.generate_embeddings(texts, input_type="passage")