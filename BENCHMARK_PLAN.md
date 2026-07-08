# Benchmark ai-memory vs supermemory / memclaw

## Context

We want a defensible answer to "is ai-memory better than supermemory / memclaw?"
You can't claim "better" without a number on a yardstick the competitors also
report. **LOCOMO** is that yardstick: supermemory, memclaw, and mem0 all publish
LOCOMO scores, and mem0 open-sourced a harness (`mem0ai/memory-benchmarks`), so we
reuse the methodology instead of inventing one.

Two facts from the codebase shape the design:
1. **No auto-extraction.** `remember()` expects distilled notes; competitors
   extract facts from raw conversation via an LLM on `add`. LOCOMO conversations
   are raw, so our add-phase needs an LLM extractor step — meaning the benchmark
   measures **{extractor + ai-memory retrieval}**, which is exactly how it's used
   in real life (Claude calls `remember`). State this in results.
2. **One ChromaDB collection, no namespaces.** LOCOMO has 10 independent
   conversations; retrieval must not leak across them. So we run **one
   conversation at a time against a throwaway vault** (clear between
   conversations), never the real vault at `vault/`.

Decisions (locked): benchmark = **LOCOMO**; competitor numbers = **published
baselines** (no self-hosting supermemory/memclaw); answerer = **deepseek-v4-pro**,
judge = **claude-sonnet** (`claude-sonnet-4-6`). Extractor also deepseek-v4-pro
(cheap). Note: a non-`gpt-4o-mini` judge means published numbers are a *rough
baseline*, not a strict same-config match — you accepted this.

## Approach

Reuse mem0's LOCOMO **dataset + exact judge prompt** (for methodology
comparability) but write our own compact runner, because we're swapping both the
answerer and the judge to DeepSeek/Claude — that touches most of mem0's
OpenAI-wired harness anyway, so forking it is more work than a ~200-line loop.

Isolation without editing the real vault: point ai-memory at a temp vault +
temp chroma dir. `config.py` paths are module constants (not env-overridable
today), so add a **2-line env override** (`os.getenv`) for `VAULT_DIR` /
`CHROMA_DIR` — the only production-code change. Then the runner imports the
`Retriever` in-process against temp dirs (no server, no port).

## Files

New, all under `bench/` (fewest files):
- `bench/run_locomo.py` — the runner: load dataset → per-conversation {clear temp
  vault, extract+`remember` each session, `index`, then per-question
  recall→answer→judge} → write `bench/results.json` + printed summary.
- `bench/judge_prompt.txt` — mem0's LOCOMO judge prompt, verbatim (comparability).
- `bench/README.md` — one-paragraph how-to-run + the honest caveats.

Reuse (do not rewrite):
- `src/aimemory/retriever.py` — `Retriever.query(text, n, category, min_relevance)`
  for recall; the add/index path for ingest. Instantiate with temp paths.
- `src/aimemory/config.py` — add env override for `VAULT_DIR`/`CHROMA_DIR` only.
- LOCOMO dataset JSON (`locomo10.json`) — download from mem0's repo / HF; vendor
  into `bench/data/` (gitignored). ~10 convs, ~1.5k QA across 5 categories.

## Steps

1. `config.py`: `VAULT_DIR = Path(os.getenv("AIMEM_VAULT_DIR", <default>))`,
   same for chroma. One line each. (Also lets the real server stay untouched.)
2. Download `locomo10.json` + copy mem0's judge prompt into `bench/`.
3. `bench/run_locomo.py`:
   - Deepseek + Anthropic clients from env keys (`DEEPSEEK_API_KEY`,
     `ANTHROPIC_API_KEY`). DeepSeek via its OpenAI-compatible base_url.
   - Per conversation: wipe temp vault/chroma → for each session, deepseek
     extracts atomic facts → `remember(title, content, category="facts",
     tags=[conv_id])` → reindex.
   - Per question: `recall(q, n=k)` (sweep k∈{3,5,10}; default n=3 is our real
     setting, but report a k-sweep so retrieval isn't the bottleneck) → deepseek
     answers from retrieved context → claude-sonnet judges answer vs gold
     (correct/incorrect) using the vendored prompt.
   - Aggregate: **LLM-judge accuracy overall + per category** (single-hop,
     multi-hop, temporal, open-domain, adversarial/abstention), plus efficiency
     axes: mean tokens injected per query, recall@k, p50/p95 recall latency,
     embedding $ cost (=$0, local).
4. Run, drop numbers into the comparison table below.

## What "better" means (multi-axis — don't overclaim accuracy)

Published LOCOMO baselines (LLM-judge accuracy, their configs):

| System | LOCOMO acc | Cost model | Local/Private |
|---|---|---|---|
| mem0 | ~66.9% overall | paid API | no |
| supermemory | ~83–92% (est.) | paid SaaS | no |
| memclaw (local, emanuilo) | local-first, MD + hybrid | free | yes |
| human | ~87.9% F1 | — | — |
| **ai-memory** | *(fill from run)* | **$0 (local embeds)** | **yes** |

Honest framing: ai-memory's peers on *architecture* are the local-first,
markdown-vault tools (emanuilo/memclaw), not the cloud SaaS. Likely finding:
accuracy competitive-but-not-SOTA, winning decisively on **cost ($0 vs paid),
privacy (fully local), latency (local HNSW), and token efficiency** (floor 0.35 +
1200-char truncation → tight context). The claim to make is the **Pareto
tradeoff**, not "highest accuracy." One knob if accuracy lags: relevance floor
0.35 may be too aggressive for LOCOMO multi-hop — the k-sweep + a floor-sweep will
show whether retrieval or the floor is the ceiling.

## Caveats to print in results (so the comparison isn't smuggled)

1. Judge ≠ leaderboard's `gpt-4o-mini` → published numbers are a rough baseline.
2. LOCOMO measures conversational QA memory; ai-memory's real use is
   dev-assistant/personal notes. A good LOCOMO score doesn't fully transfer.
   **Add a 20-line custom sanity eval** (`bench/run_custom.py`): ~20 hand-written
   questions over your *real* `vault/`, scored on "did the right note surface in
   top-3." Cheap, and it's the only number that measures *your* use case.
3. Add-phase uses an LLM extractor → benchmark = extractor + retrieval, not
   retrieval alone.

## Verification

- `python bench/run_locomo.py --convs 1 --k 5` on one conversation first —
  confirms extract→remember→index→recall→answer→judge end-to-end before the full
  ~1.5k-question run.
- Assert temp vault path ≠ real `vault/` (guard so a bug can't touch real notes).
- Sanity: `recall` on a fact just written returns it above floor; judge marks an
  obviously-correct answer correct and an obviously-wrong one wrong (2-case
  self-check in `__main__`).
- Full run writes `bench/results.json`; eyeball per-category accuracy + the
  efficiency axes against the table.

## Skipped (add when needed)

- Self-hosting supermemory/memclaw for strict same-config numbers — skip; add only
  if published-baseline gap is <5% and you need a definitive win.
- LongMemEval — skip; add if you want a second, personal-assistant-flavored
  yardstick after LOCOMO lands.
- Matching `gpt-4o-mini` judge — skip; add a single gpt-4o-mini re-judge on a
  100-question subset only if a reviewer disputes the Claude-judge numbers.

## Sources

- mem0 memory-benchmarks — https://github.com/mem0ai/memory-benchmarks
- AI Memory Benchmarks 2026 — https://mem0.ai/blog/ai-memory-benchmarks-in-2026
- LoCoMo Refined — https://github.com/mem-eval-suite/LoCoMo_refined
- LongMemEval — https://github.com/xiaowu0162/longmemeval
- emanuilo/memclaw — https://github.com/emanuilo/memclaw
- caura-ai/memclaw — https://github.com/caura-ai/caura-memclaw
