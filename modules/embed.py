from langchain_core.embeddings import Embeddings
from openai import OpenAI

class NVIDIAEmbeddings(Embeddings):

    def __init__(self, client, model):
        self.client = client
        self.model = model

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