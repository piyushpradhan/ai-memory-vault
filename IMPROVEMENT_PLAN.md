# ai-memory: close the accuracy gap with mem0 / supermemory / memclaw

## Context

Current retrieval is vector-only chroma with `all-MiniLM-L6-v2`, section-level chunks, top-3, floor 0.35. The benchmarks show where it falls short:

- `bench/custom_results.json`: **70% top-3** (14/20). Misses are vocabulary mismatch (query wording ≠ note wording) and near-duplicate notes splitting semantic mass (`ai-preferences` vs `communication-style-preference` vs `preferences-concise-communication`). Same note also occupies 2 of 3 result slots in several queries.
- LOCOMO smoke: 50% overall, **0% multi-hop**.
- Token efficiency is already good (mean 257 chars injected) — accuracy is the gap, not tokens.

What the SOTA products actually do, mapped to local equivalents (constraint from vault goals: local, offline, no API keys → no server-side LLM):

| Product | Technique | Local equivalent |
|---|---|---|
| memclaw / OpenClaw | BM25 + vector hybrid | inline BM25 + RRF fusion |
| supermemory | rerank, hybrid, decay | local cross-encoder rerank |
| mem0 | LLM extract + ADD/UPDATE/DELETE consolidation | push the LLM role to the calling agent (it *is* an LLM); server detects near-dupes |

All changes in `src/aimemory/` (513 lines today); ~150–200 new lines total.

## Changes

### 1. Hybrid retrieval: BM25 + RRF — `retriever.py`
Corpus is ~190 chunks, so no index needed: inline BM25-Okapi (~30 lines, stdlib only) over tokenized chunk text held in memory, rebuilt on `index()` and `add_memory()`. Merge BM25 and vector rankings with reciprocal rank fusion (k=60) — no weight tuning. Fixes the vocabulary-mismatch misses (exact terms like "Redux", slugs, project names).

### 2. Better embedder — `config.py`
`EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"` (MTEB retrieval ~52 vs MiniLM-L6's ~42, similar speed/size). Needs a ~10-line custom chroma embedding function to add the bge query prefix on queries only. One-time ~130MB download; reindex after.

### 3. Local cross-encoder rerank — `retriever.py`
Hybrid retrieval fetches top-20 candidates; `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` (sentence-transformers already installed) reranks to final top-n. This is the single biggest accuracy lever and the main multi-hop fix (relevant-but-differently-worded chunks survive into candidates, reranker sorts them out). Lazy-loaded at first query. Latency ~30–80ms on CPU vs 5ms today — fine for hooks. Relevance floor moves to sigmoid(reranker score); recalibrate on the 20 custom-bench questions so the "empty output on irrelevant queries" property survives.

### 4. One result per note — `retriever.py`
Group candidates by `source` before final top-n, keep the best chunk per note. Eval shows duplicate slugs wasting slots today.

### 5. mem0-style consolidation, agent-driven — `retriever.py` + `server.py`
- `add_memory` checks for near-duplicate notes (rerank score above threshold, different slug) and returns them in the `remember` response: `Saved. Overlaps: communication-style-preference (0.82) — read_note + merge + forget the loser.`
- New `forget(slug)` MCP tool + `delete_note()`: moves the file to `vault/.trash/` (soft delete, excluded from indexing), reindexes. Without delete, consolidation can't actually happen.
- Update `remember`/`forget` docstrings + the CLAUDE.md / AGENTS.md protocol blurb so calling agents know the merge workflow.

### 6. Token-efficiency tweaks (minor)
Truncate at sentence boundary instead of mid-character; `/context` enforces a total char budget (~1500) instead of per-item only.

**Skipped** (say so once, revisit only if benchmarks demand):
- Wikilink graph expansion — speculative; rerank+k=20 covers multi-hop cheaper.
- Recency decay — vault is curated; consolidation handles staleness.
- Server-side LLM extraction (mem0's extractor) — violates local/no-API-key goal; calling agents already extract.

## Files
- `src/aimemory/config.py` — embedder, candidate-k, recalibrated floor
- `src/aimemory/retriever.py` — BM25, RRF, rerank, per-note dedupe, near-dupe check, `delete_note`
- `src/aimemory/server.py` — `forget` tool, richer `remember` response, docstrings
- `~/.claude/CLAUDE.md` + vault `AGENTS.md` protocol — merge/forget workflow note

## Verification
1. `python bench/run_custom.py` before/after — target ≥85% top-3 (from 70%).
2. `python bench/run_locomo.py --convs 1 --max-questions 10 --k 5` smoke — expect multi-hop > 0%.
3. Floor sanity: 3–4 deliberately irrelevant queries must return empty (zero-token property).
4. `curl POST /index`, `GET /health`, `GET /context?q=...` against the running server; restart launchd service.
