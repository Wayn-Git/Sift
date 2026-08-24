from langchain_core.embeddings import Embeddings
from openai import OpenAI
import os

class NVIDIAEmbeddings(Embeddings):

    def __init__(self, api_key: str, model: str = "nvidia/llama-nemotron-embed-1b-v2"):
        api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY missing")
        
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float",
            extra_body={
                "input_type": "passage",
                "truncate": "NONE"
            }
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            encoding_format="float",
            extra_body={
                "input_type": "query",
                "truncate": "NONE"
            }
        )

        return response.data[0].embedding