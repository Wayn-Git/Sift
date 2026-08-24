# Sift — Project Structure

```
Sift/
├── main.py                 # demo: rag_chain.invoke() + hallucination guard
├── requirements.txt
├── .env                    # NVIDIA_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
│
├── modules/                # domain-organized, names = what they do
│   ├── ingestion/
│   │   └── mineru_loader.ipynb    # ex document_loader.ipynb — MinerU PDF → data/parsed/*.md
│   ├── embeddings/
│   │   ├── nvidia_embeddings.py   # ex embed.py — NVIDIA llama-nemotron-embed-1b-v2 wrapper
│   │   ├── dense_index.ipynb      # ex embedding.ipynb — chunk 1000/150 → dense cosine
│   │   └── sparse_index.ipynb     # ex sparse_embedding.ipynb — SPLADE dotProduct
│   ├── retrieval/
│   │   ├── pinecone_store.py      # ex vector_store.py — Pinecone + NVIDIAEmbeddings
│   │   └── rag_chain.py           # ex retrieval_chain.py — retriever | prompt | LLM
│   ├── llm/
│   │   └── nvidia_llm.py          # ex language_model.py — ChatOpenAI @ NVIDIA
│   └── visual/
│       └── colqwen_visual.py      # ex colpali.py — ColQwen2.5 + PyMuPDF rendering
│
└── data/
    ├── paper/attention_is_all_you_need.pdf
    └── parsed/             # MinerU output — 11 pages
        ├── page_001.md ... page_011.md
        └── paper_images/page_001.png ... page_011.png
```

## Renames (no logic added)

| Old | New | Reason |
|-----|-----|--------|
| `embed.py` | `nvidia_embeddings.py` | explicit provider+purpose |
| `embedding.ipynb` | `dense_index.ipynb` | pairs with `sparse_index` |
| `sparse_embedding.ipynb` | `sparse_index.ipynb` | parallel naming |
| `vector_store.py` | `pinecone_store.py` | backend explicit |
| `retrieval_chain.py` | `rag_chain.py` | shorter, pattern name |
| `language_model.py` | `nvidia_llm.py` | provider explicit |
| `colpali.py` | `colqwen_visual.py` | model explicit |
| `document_loader.ipynb` | `mineru_loader.ipynb` | tool explicit |

## Pipeline

1. `ingestion/mineru_loader.ipynb` → `data/parsed/*.md`
2. `visual/colqwen_visual.py` → `paper_images/*.png` + ColQwen
3. `embeddings/dense_index.ipynb` → Pinecone `cosine`
4. `embeddings/sparse_index.ipynb` → Pinecone `dotProduct` (separate index)
5. `retrieval/pinecone_store.py:58` — `VectorStore().vector_search()`
6. `llm/nvidia_llm.py` — `get_llm()`
7. `retrieval/rag_chain.py:43` — `get_retrieval_chain(k=4)`
8. `main.py:6` — `from modules.retrieval.rag_chain import get_retrieval_chain`

## Notes

- All moves verbatim, imports updated with fallback chains.
- Notebook `../data` → `../../data` after nesting 2 deep.
- Sparse bug kept as-is per request — don't mix `cosine` + `dotProduct` in same index.
