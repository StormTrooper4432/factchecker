Project: RAG + LLM Nutrition Fact Checker

Key files and folders
- backend.py — Flask API that orchestrates claim evaluation, retrieval, and Gemini fact-checking (hybrid or LLM-only).
- foodrag/config.py — Global configuration (data paths, chunk sizes, model names).
- foodrag/factcheck.py — Builds/executes fact-check prompts (hybrid and LLM-only), parses Gemini responses, heuristic fallback.
- foodrag/retrieval.py — Term extraction, corpus filtering, chunking, similarity scoring, and Chroma querying/merging.
- foodrag/embeddings.py — Lightweight domain embedding function used for Chroma.
- foodrag/ingest.py — CLI to ingest text/XML corpora into Chroma with tagging and chunking.
- foodrag/tagging.py — Simple keyword-based tag generator for ingested documents.
- foodrag/text_utils.py — Sentence splitting, cleaning, and chunk helpers.
- foodrag/storage.py — Chroma client/collection management.
- foodrag/hardcoded_prompts.py — Predefined claims/responses for quick demos.
- chunk_crawl_data.py — Pre-chunks crawled text into `data/crawl_chunks` for faster retrieval.
- crawl_open_domains.py — Lightweight crawler for allowlisted health/nutrition domains; stores raw text + metadata.
- ingest_missing_chunks.py — Adds any missing chunk files from `data/crawl_chunks` into Chroma.
- pmc_pull_api.py — PMC E-utilities wrapper to search/download OA articles (nxml).
- direct_fact_check.py — CLI to run fact-check directly over local files (no Chroma).
- part4_fact_checking.py — Simpler pipeline example: retrieve-top-k then fact-check.
- factcheck_live_pmc.py — End-to-end live PMC pull + retrieval + Gemini verdict CLI.
- query_chroma.py — Queries existing Chroma collection and fact-checks.
- frontend/ — Vite/React UI; `src/App.jsx` handles claim form, progress states, and renders verdict/evidence.
- data/ — Local corpora (oa_api, crawl, crawl_chunks, etc.); required at runtime.
- chroma_db/ — Persistent Chroma vector store.

Notes
- Set GEMINI_API_KEY before running hybrid or LLM-only fact checks.
- Start backend with `python -B backend.py`; start frontend from `frontend` with `npm run dev`.
