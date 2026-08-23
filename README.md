# Sift

RAG for "Attention Is All You Need" — text chunks + Pinecone + NVIDIA embeddings.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add NVIDIA_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
```

## Pipeline
1. `modules/document_loader.ipynb` — MinerU splits PDF → `data/parsed/*.md` + `data/parsed/paper_images/*.png`
2. `modules/embedding.ipynb` — chunk (1000/150) → NVIDIA `llama-nemotron-embed-1b-v2` → Pinecone
3. `modules/vector_store.py` — `VectorStore().vector_search(query)`
4. `modules/language_model.py` — `get_llm()` (NVIDIA ChatOpenAI)
5. `modules/retrieval_chain.py` — `get_retrieval_chain(k=4)` (retriever + prompt + LLM)
6. `main.py` — demo

## Run
```bash
python main.py
# or
python -c "from modules.retrieval_chain import get_retrieval_chain; print(get_retrieval_chain().invoke('What is self-attention?'))"
```

## Visual (optional)
`modules/colpali.py` — ColQwen2.5 for image retrieval. Requires `colpali-engine` + 8GB model `vidore/colqwen2-v1.0`. Skip for text-only.
