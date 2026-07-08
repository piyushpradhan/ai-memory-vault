# Cross-harness AI memory: wire up + automate the existing ai-memory system

## Context

You already built 90% of this: `~/projects/ai-memory` is a markdown vault (Obsidian-editable, git-tracked) + ChromaDB semantic index (local embeddings, `all-MiniLM-L6-v2` — zero API token cost) + FastAPI server on port 8420 + `aimemory` CLI, with launchd plists for auto-start and 5-min reindex.

**But nothing is connected to it.** The server is stopped, launchd jobs aren't loaded, `mcpServers` in `~/.claude.json` is empty, `opencode.jsonc` is bare, and there's no Grok config. The vault also can't answer "what's done / pending / planned" per project — it stores facts, not working state.

There is also a **real bug**: in `src/aimemory/retriever.py` `_scan_vault()` (lines ~183–201), `chunks.append(...)` is indented inside `if heading_match:` — so any note (or first section) without a `## ` heading is **never indexed**. A chunk of your vault is currently invisible to search.

Goals:
1. One memory backend reachable from Claude Code CLI, Claude Code Desktop, Opencode, and Grok Build; remembers done/pending/planned across projects and sessions.
2. **Zero manual invocation** — retrieval happens automatically from the user's prompt; writes happen automatically as the model works.
3. Minimal token cost on every provider.

## Storage: markdown + vector DB (both, as today)

Markdown files stay the **source of truth**: human-editable in Obsidian, git-diffable, portable, no lock-in, survives any tool dying. ChromaDB (already a vector database) is the **derived index** — rebuildable from markdown at any time, used only for semantic recall. A DB-only store would still need embeddings anyway and would lose inspectability. Same architecture supermemory uses internally (doc store + embedding index). No change needed, just the bug fix.

## Key design decisions

- **One long-running server, MCP over HTTP.** Mount an MCP endpoint (streamable HTTP at `/mcp`) onto the *existing* FastAPI app. Embedding model loads once; no ChromaDB multi-process conflicts; launchd setup already written.
- **Automatic retrieval via hooks, not tool calls.** In Claude Code a `UserPromptSubmit` hook sends the user's prompt to `POST /query` and injects top matches into context — every prompt, no model round-trip, no tool tokens. Opencode (TS plugin) and Grok Build (hooks) get the same treatment. Claude Desktop has no hooks → MCP tools + instructions make the model self-serve there.
- **Automatic writes via protocol, not prompts.** Global instructions in each harness tell the model to upsert the project status note at milestones and save durable facts *unprompted*. The user never types "remember this".
- **3 tiny MCP tools only** (~300 tokens schema): `recall`, `remember`, `read_note` — used by Claude Desktop always, and by the other harnesses only when the hook-injected context isn't enough.
- **Project state = a convention.** One note `projects/status-<project>.md` per project with `## Done / ## In progress / ## Next / ## Decisions`. Deterministically readable via `read_note(slug)`; auto-injected by the SessionStart hook.
- **Upsert by slug** in `remember` — same title overwrites instead of spawning duplicates (vault already has 5 near-duplicate yank notes, 3 piyush-pradhan notes).

## Phase 1 — Fix the retriever (`src/aimemory/retriever.py`)

1. Fix the `_scan_vault` indentation bug so headingless sections are indexed too.
2. Deterministic access: `get_note(slug)` (read file straight from vault) and make `add_memory` a true upsert — currently `add_memory` chunk IDs (path-hash) differ from `index()` IDs (path#i-hash), leaving stale chunks; unify by reindexing the written file's chunks after write.
3. `MAX_CONTENT_LENGTH` 2000 → 1200; recall default n=3. Add a relevance floor (~0.35) so low-quality matches are dropped instead of injected.

## Phase 2 — Server: MCP endpoint + hook endpoint (`src/aimemory/server.py`)

1. Add dependency `mcp` (official Python SDK). `FastMCP` with 3 tools backed by the same `MemoryRetriever`:
   - `recall(query, n=3, category=None)` → compact `{title, content, relevance}` list
   - `remember(title, content, category="facts", tags=[])` → upsert; description: recall first to avoid dupes; use `status-<project>` titles for working state
   - `read_note(slug)` → exact note body
2. Mount at `/mcp` on the existing FastAPI app; keep REST endpoints (hooks and CLI use them).
3. Add `GET /context?q=<prompt>&project=<name>` — single endpoint for hooks: returns status note + top relevant memories above the relevance floor, pre-formatted as compact plaintext, empty response if nothing clears the floor (so irrelevant prompts inject zero tokens).
4. Drop `reload=True` in `main()` (dev-mode watcher under launchd is waste).

## Phase 3 — Start backend + register MCP everywhere

1. Run `scripts/setup-macos.sh` (loads launchd server + reindex jobs), verify `/health`.
2. **Claude Code CLI:** `claude mcp add --scope user --transport http aimemory http://localhost:8420/mcp`.
3. **Claude Desktop:** add to `~/Library/Application Support/Claude/claude_desktop_config.json` (via `mcp-remote` shim if that build needs stdio).
4. **Opencode:** `"mcp": {"aimemory": {"type": "remote", "url": "http://localhost:8420/mcp"}}` in `~/.config/opencode/opencode.jsonc`.
5. **Grok Build:** reads Claude's MCP config out of the box; verify with `grok inspect`, else declare in `~/.grok/config.toml`.

## Phase 4 — Automatic retrieval + automatic writes per harness

1. **Claude Code (CLI + Desktop app sessions):** in `~/.claude/settings.json`:
   - `SessionStart` hook → `curl -s "localhost:8420/context?project=$(basename $PWD)"` → status note lands in context at session open.
   - `UserPromptSubmit` hook → small script reads the prompt from hook stdin JSON, calls `/context?q=...`, prints matches (or nothing). Fully automatic per-prompt RAG, ~0–400 tokens, no tool calls. Curl timeout 1–2s so a dead server never blocks prompts.
2. **Opencode:** TS plugin in `~/.config/opencode/plugins/` using its message hook to inject `/context` results the same way; plus the protocol snippet in `~/.config/opencode/AGENTS.md`.
3. **Grok Build:** equivalent hook via its hooks system (confirm exact hook names with `grok inspect` / docs during implementation); protocol snippet in its global instruction file.
4. **Claude Desktop (standalone app):** no hooks — protocol instructions + MCP tool descriptions drive automatic `recall`/`remember` by the model itself.
5. **Automatic writes (all harnesses):** rewrite `scripts/context-prompt.txt` into an ~8-line protocol installed in each harness's global instruction file (`~/.claude/CLAUDE.md`, `~/.config/opencode/AGENTS.md`, Grok's global file): *update `status-<project>` after completing/deciding anything significant; save durable user facts/preferences when learned; recall before remember; bullets, one topic per note.* The model writes memory as a side effect of working — user never asks.

## Phase 5 (optional, one-time) — Vault dedup

Merge obvious duplicates (5× yank, 3× piyush-pradhan, 2× mediocre…) so recall slots aren't wasted on near-identical chunks. Can be done by hand in Obsidian later.

## Token cost summary (per session)

| Cost | Amount |
|---|---|
| Hook-injected context (SessionStart + per-prompt) | 0–400 tokens per prompt, only when relevant (floor filter); **no tool-call round trips** |
| MCP schema (3 tools) | ~300 tokens once per session |
| One `remember`/status upsert | ~150–300 tokens (content the model produces anyway) |
| Embedding, indexing, retrieval compute | $0 — all local |

## Verification

1. `aimemory index` → chunk count jumps after bug fix (headingless notes now indexed).
2. `curl "localhost:8420/context?q=what%20is%20yank&project=yank"` returns compact context; an unrelated query returns empty.
3. Fresh Claude Code session in another project: hooks visibly inject status/context (check with `/context` in transcript); mention a durable fact while working → model saves it unprompted → retrievable in a *second* fresh session and from Opencode.
4. `claude mcp list` shows aimemory connected; `grok inspect` lists it too.
5. Kill server → prompts still work (hook times out silently); launchd restarts it (`launchctl list | grep aimemory`).
