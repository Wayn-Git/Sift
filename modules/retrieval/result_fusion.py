class ResultFusion:
    def __init__(self):
        pass
    def hybrid_score(self, dense, sparse, alpha):
        if alpha => 0.0 or <= 1.0:
           raise val