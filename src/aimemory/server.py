from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from .config import VAULT_PATH, CHROMA_PATH, MAX_RESULTS, TOKEN_BUDGET
from .retriever import MemoryRetriever

retriever = MemoryRetriever(VAULT_PATH, CHROMA_PATH)

mcp = FastMCP("aimemory", streamable_http_path="/")


@mcp.tool()
def recall(query: str, n: int = 3, category: str | None = None) -> list[dict]:
    """Semantic search over the memory vault. Call this before answering to recall
    facts, preferences, or project state about the user. Returns compact
    {title, content, relevance} matches above the relevance floor; empty list
    if nothing relevant (irrelevant queries cost zero context)."""
    return retriever.query(query, n=n, category=category)


@mcp.tool()
def remember(
    title: str,
    content: str,
    category: str = "facts",
    tags: list[str] | None = None,
) -> str:
    """Upsert a durable memory note (same title overwrites — slug-keyed, no
    duplicates). RECALL FIRST to avoid dupes. For project working state use a
    title like 'status-<project>' with sections '## Done / ## In progress / ##
    Next / ## Decisions'. Content is markdown bullets, one topic per note.
    Categories: facts, people, projects, concepts."""
    path = retriever.add_memory(title, content, category, tags or [])
    overlaps = retriever.check_overlaps(title, content)
    msg = f"Saved: {path}"
    if overlaps:
        parts = [f"{o['slug']} ({o['similarity']:.2f})" for o in overlaps]
        msg += ". Overlaps: " + "; ".join(parts)
        msg += " — read_note + merge + forget the loser."
    return msg


@mcp.tool()
def forget(slug: str) -> str:
    """Soft-delete a note: moves it to vault/.trash/, removes chunks from the
    index, and rebuilds the search index. Returns a confirmation or error."""
    ok = retriever.delete_note(slug)
    if ok:
        return f"Forgot '{slug}' — moved to .trash/ and removed from index."
    return f"Note not found: '{slug}'"


@mcp.tool()
def read_note(slug: str) -> str:
    """Read a note's exact body by slug (filename stem), e.g. 'status-yank' or
    'ai-preferences'. Use for deterministic project-state reads. Empty string
    if the note does not exist."""
    return retriever.get_note(slug) or ""


mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="AI Memory",
    description="Permanent cross-provider memory via markdown + semantic search",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/mcp", mcp_app)


class QueryRequest(BaseModel):
    query: str
    n: int = MAX_RESULTS
    category: Optional[str] = None


class MemoryRequest(BaseModel):
    title: str
    content: str
    category: str = "facts"
    tags: list[str] = []


class QueryResponse(BaseModel):
    results: list[dict]
    count: int


class IndexResponse(BaseModel):
    status: str
    chunk_count: int


class AddResponse(BaseModel):
    status: str
    path: str


class ForgetResponse(BaseModel):
    status: str
    message: str


@app.post("/query", response_model=QueryResponse)
def query_memories(req: QueryRequest):
    results = retriever.query(req.query, n=req.n, category=req.category)
    return QueryResponse(results=results, count=len(results))


@app.post("/index", response_model=IndexResponse)
def index_vault():
    n = retriever.index()
    return IndexResponse(status="ok", chunk_count=n)


@app.post("/memory", response_model=AddResponse)
def add_memory(req: MemoryRequest):
    path = retriever.add_memory(
        req.title, req.content, req.category, req.tags
    )
    return AddResponse(status="ok", path=path)


@app.post("/forget", response_model=ForgetResponse)
def forget_note(slug: str = ""):
    if not slug:
        return ForgetResponse(status="error", message="slug is required")
    ok = retriever.delete_note(slug)
    if ok:
        return ForgetResponse(
            status="ok", message=f"Forgot '{slug}'"
        )
    return ForgetResponse(
        status="error", message=f"Note not found: '{slug}'"
    )


@app.get("/context", response_class=PlainTextResponse)
def context(q: str = "", project: Optional[str] = None):
    """Single endpoint for hooks: status note + top relevant memories, compact
    plaintext. Enforces TOKEN_BUDGET char limit. Empty body when nothing
    clears the relevance floor (irrelevant prompts inject zero tokens)."""
    budget = TOKEN_BUDGET
    parts: list[str] = []
    remaining = budget

    if project:
        body = retriever.get_note(f"status-{project}")
        if body:
            header = f"# Project status: {project}\n\n"
            if len(header) + len(body) <= remaining:
                parts.append(header + body)
                remaining -= len(header) + len(body)
            else:
                truncated = body[: remaining - len(header)] + "\n...(truncated)"
                parts.append(header + truncated)
                remaining = 0

    if q and remaining > 0:
        results = retriever.query(q, n=5)
        if results:
            lines = ["# Relevant memories"]
            for r in results:
                if remaining <= 0:
                    break
                header = f"\n## {r['title']} ({r['relevance']:.0%})"
                entry = header + "\n" + r["content"]
                if len(entry) <= remaining:
                    lines.append(entry)
                    remaining -= len(entry)
                elif remaining > len(header):
                    avail = remaining - len(header)
                    truncated = r["content"][:avail] + "\n...(truncated)"
                    lines.append(header + "\n" + truncated)
                    remaining = 0
                    break
                else:
                    break
            parts.append("\n".join(lines))

    return "\n\n".join(parts).strip()


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": retriever.count()}


def main():
    uvicorn.run("aimemory.server:app", host="127.0.0.1", port=8420)
