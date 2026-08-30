from client import client_handler 


class ResultFusion:
    def __init__(self, nvidia_api_key: str, pinecone_api_key: str, index_name: str):
        self.n_key = client_handler.get_nvidia_key(nvidia_api_key)
        self.p_key = client_handler.get_pinecone_key(pinecone_api_key)
        self.llm = client_handler.get_llm(self.n_key)
        self.ps = client_handler.get_vector_store(
            self.p_key,
            self.n_key,
            index_name
        )

    def normalize_scores(self, scores):
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def hybrid_score(self, dense_score, sparse_score, alpha=0.7):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "The value of alpha must be between 0.0 and 1.0"
            )

        return (
            alpha * dense_score
            + (1 - alpha) * sparse_score
        )