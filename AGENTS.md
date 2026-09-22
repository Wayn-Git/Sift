# AGENTS.md

## What this is

Sift: hybrid RAG demo over "Attention Is All You Need". Dense + sparse retrieval via Pinecone, NVIDIA APIs for embeddings/LLM/rerank. LangChain-based, no web server, no tests, no lint/typecheck config.

Target architecture lives in `.doc/structure.md` (query → hybrid retrieve → RRF → rerank → context → LLM). Not fully built yet.

## Commands

```bash
source .venv/bin/activate   # always; deps live here, no global install
python main.py              # only entrypoint: hybrid retrieval smoke test
```

No test/lint/typecheck commands exist. After dependency changes: `.venv/bin/pip freeze > requirements.txt`.

## Environment

- `.env` (gitignored) or `.env.example`: `NVIDIA_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`. Loaded by `config.py` via `load_dotenv()` — import `config`, don't re-read `.env` yourself.
- All packages go in `.venv`. Never assume a package is installed; check `.venv/bin/pip list`.
- `requirements.txt` is a full `pip freeze` of `.venv`, not a curated direct-deps list.

## Architecture facts that filenames don't show

- Run everything from repo root; imports are absolute from root (`from embeddings... import`, `from client... import`). Imports missing the package prefix (`from llm_handler import ...`) break — a bug that has shipped before.
- Two embedding paths exist and must stay compatible with the Pinecone index: dense = NVIDIA `nemotron-3-embed-1b` (dim 2048), sparse = `pinecone-sparse-english-v0`. Index MUST use `metric="dotproduct"` for hybrid — see `storage/hybrid_store.ipynb`.
- Hybrid retrieval = `retriever/retriever.py` wrapping LangChain's `PineconeHybridSearchRetriever` (namespace `hrbird`, `text_key="text"`). Reranker (`retriever/reranker.py`, NVIDIA hosted API) exists but is NOT wired into any pipeline yet.
- `chain/prompts.py` and `chain/rag_chain.py` are intentionally empty (generation stage not built).
- Index upsert happens only from `storage/hybrid_store.ipynb` (notebook, not a script). Notebook deletes namespace `hrbird` before upsert — re-run safe. Vector IDs are still positional (`hrbird-{i}`).
- Chunking (`injestion/chunking.py`) reads `data/parsed/*.md`. Produce those files with `injestion/data_processor.ipynb` (MinerU PDF → markdown; source PDF in `data/raw/`).

## Known landmines

- `retriever/retriever.py`: `pc.Index(pinecone_index_name)` ignores the `index` value from `get_pinecone_info()` — passing only an API key (index arg `None`) crashes. Use the resolved `index` var when touching this file.
- `DenseEncoderAdapter` in `embeddings/dense_embed.py` is broken (calls `generate_embeddings(..., input_type=...)` but that method takes no `input_type`) and is dead code. Sparse adapter is fine. Don't wire the dense adapter in without fixing it.
- `storage/pinecone_store.py` writes dense-only vectors (no `sparse_values`) — wrong for the hybrid index. Only upsert via `storage/hybrid_store.ipynb`.
- README describes an old `modules/` layout that no longer exists. Trust the directory tree, not the README.

## Conventions

- Package dirs are lowercase. `sparec_embed.py` was renamed to `sparse_embed.py` — keep all imports on `embeddings.sparse_embed`.
- Empty `__init__.py` files make every subdir a package; keep them.
- Secrets never enter git: `.env` and `.venv` are gitignored. Do not paste key values into code, commits, or notebooks.
