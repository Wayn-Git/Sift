                     USER QUERY
                         │
                         ▼
                  Query processing
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
        Dense Retriever      Lexical Retriever
               │                   │
               ▼                   ▼
           Pinecone              BM25
               │                   │
               └─────────┬─────────┘
                         ▼
                  Result Fusion
                    (RRF)
                         │
                         ▼
                  Candidate Set
                    Top 20
                         │
                         ▼
                    Reranker
                         │
                         ▼
                    Top 4-6
                         │
                         ▼
                  Context Builder
                         │
                         ▼
                       LLM
                         │
                         ▼
                      Answer