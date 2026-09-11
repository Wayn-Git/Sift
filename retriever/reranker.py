import requests

from client.vector_handler import get_nvidia_key


class Reranker:

    def __init__(
        self,
        nvidia_api_key: str | None = None,
        model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
    ):

        self.api_key = get_nvidia_key(nvidia_api_key)
        self.model = model
        self.url = url

    def get_rerank_model(self):
        return self.model

    def generate_rankings(
        self,
        query: str,
        docs: list,
        top_n: int = 5,
    ) -> list:

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not isinstance(docs, list):
            raise TypeError("docs must be a list of documents")

        if not docs:
            return []

        try:
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                json={
                    "model": self.model,
                    "query": {"text": query},
                    "passages": [{"text": d.page_content[:2000]} for d in docs],
                },
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Rerank Error: {e}")
            raise

        rankings = response.json()["rankings"]
        top = sorted(rankings, key=lambda r: r["logit"], reverse=True)[:top_n]

        return [docs[r["index"]] for r in top]
