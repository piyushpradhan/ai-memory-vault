# ai-memory benchmarks

LOCOMO-10 runner comparing ai-memory against published mem0/supermemory baselines, plus a custom vault sanity eval.

## Quick start (no API keys needed)

Uses your existing **OpenCode Go** subscription for extract + answer. Judge auto-picks the best available option.

```bash
# 1. Self-check
python bench/run_locomo.py --self-check

# 2. Smoke test — one conversation, five questions
python bench/run_locomo.py --convs 1 --max-questions 5 --k 5

# 3. Full LOCOMO run (~1.5k questions; slow — each Q spawns opencode subprocesses)
python bench/run_locomo.py

# 4. Custom vault eval (no LLMs at all; measures your real use case)
python bench/run_custom.py
```

## Backends (auto-detected by default)

| Role | Default | How |
|---|---|---|
| Extract + answer | `opencode` | `opencode run -m opencode-go/deepseek-v4-pro` |
| Judge | auto | `claude -p` if logged in, else opencode (same DeepSeek) |

### Judge options

**Option A — Claude $20 sub via Claude Code** (preferred, matches benchmark plan):

```bash
claude /login          # one-time
python bench/run_locomo.py --judge-backend claude-cli
```

**Option B — OpenCode Go only** (no Claude CLI needed; judge is less comparable to published baselines):

```bash
python bench/run_locomo.py --judge-backend opencode
```

**Option C — retrieval only** (zero LLM cost; measures recall quality, not end-to-end QA):

```bash
python bench/run_locomo.py --predict-only --convs 1
```

If you *do* have API keys, `--llm-backend auto` picks `api` when `DEEPSEEK_API_KEY` is set; same for `ANTHROPIC_API_KEY` + judge.

| Variable | Default | Purpose |
|---|---|---|
| `OPENCODE_MODEL` | `opencode-go/deepseek-v4-pro` | OpenCode Go extract/answer model |
| `OPENCODE_JUDGE_MODEL` | same as above | OpenCode judge model |
| `CLAUDE_MODEL` | `sonnet` | Claude Code judge alias |
| `DEEPSEEK_API_KEY` | — | Only if `--llm-backend api` |
| `ANTHROPIC_API_KEY` | — | Only if `--judge-backend api` |
| `AIMEM_VAULT_DIR` | `vault/` | Override vault path |
| `AIMEM_CHROMA_DIR` | `.chroma/` | Override chroma path |

LOCOMO uses throwaway dirs under `bench/.tmp/` — the real vault is never touched.

## What it measures

LOCOMO pipeline per conversation:

1. Wipe temp vault/chroma
2. DeepSeek (via opencode) extracts atomic facts from each session → `remember()` → `index()`
3. Per question: `recall(k)` → DeepSeek answers → Claude judges vs gold

Outputs `bench/results.json` with accuracy (overall + per category), recall latency (p50/p95), and mean context chars injected.

## Caveats (read before comparing numbers)

1. **Judge ≠ leaderboard config.** Published mem0/supermemory numbers use `gpt-4o-mini` as judge; we use Claude Sonnet via subscription. Treat published baselines as rough, not strict same-config.
2. **LOCOMO ≠ dev-assistant notes.** A good LOCOMO score doesn't fully transfer to personal/project memory. Run `run_custom.py` for the number that matters to you.
3. **Benchmark = extractor + retrieval.** ai-memory has no auto-extraction; the add-phase LLM extractor is part of what's measured, matching real usage (Claude calls `remember`).
4. **Full run is slow.** Each question spawns `opencode run` + `claude -p` subprocesses. Budget time and subscription usage accordingly.

## Comparison table

| System | LOCOMO acc | Cost model | Local/Private |
|---|---|---|---|
| mem0 | ~66.9% | paid API | no |
| supermemory | ~83–92% (est.) | paid SaaS | no |
| memclaw (local) | — | free | yes |
| **ai-memory** | *(from `results.json`)* | **$0 embeds** | **yes** |

Honest claim: Pareto tradeoff on cost, privacy, latency, and token efficiency — not necessarily highest accuracy.