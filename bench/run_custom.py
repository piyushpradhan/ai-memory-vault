#!/usr/bin/env python3
"""Custom sanity eval: did the right note surface in top-3 for real vault queries?

This is the only number that measures ai-memory's actual use case (dev-assistant
notes), complementing the LOCOMO conversational-QA benchmark.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aimemory.config import CHROMA_PATH, VAULT_PATH
from aimemory.retriever import MemoryRetriever

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BENCH_DIR / "custom_results.json"

# Each entry: query, expected slug stem (filename without .md), optional category
CUSTOM_QUESTIONS = [
    {
        "query": "What tech stack does the user prefer for backend and frontend?",
        "expected_slug": "tech-stack-preferences",
    },
    {
        "query": "How should AI assistants communicate with this user?",
        "expected_slug": "ai-preferences",
    },
    {
        "query": "What are the user's preferences for concise communication?",
        "expected_slug": "preferences-concise-communication",
    },
    {
        "query": "What is the user's pricing philosophy for SaaS products?",
        "expected_slug": "pricing-philosophy",
    },
    {
        "query": "What motivates the user to build open source software?",
        "expected_slug": "open-source-motivation",
    },
    {
        "query": "What are the pain points with Redux state management?",
        "expected_slug": "redux-pain-points",
    },
    {
        "query": "What are the pain points with Zustand?",
        "expected_slug": "zustand-pain-points",
    },
    {
        "query": "What macOS development tools and environment does the user use?",
        "expected_slug": "macos-development-environment",
    },
    {
        "query": "What are the user's code style preferences?",
        "expected_slug": "code-style",
    },
    {
        "query": "What AI providers does the user use?",
        "expected_slug": "ai-providers",
    },
    {
        "query": "What are the goals for the AI memory system?",
        "expected_slug": "memory-goals",
    },
    {
        "query": "What is Yank's tech stack?",
        "expected_slug": "yank-tech-stack",
    },
    {
        "query": "Who are Yank's competitors?",
        "expected_slug": "yank-competitors",
    },
    {
        "query": "What is the user's testing philosophy?",
        "expected_slug": "testing-preferences",
    },
    {
        "query": "What version control workflow does the user prefer?",
        "expected_slug": "version-control-preferences",
    },
    {
        "query": "What are indie SaaS revenue benchmarks in 2026?",
        "expected_slug": "indie-saas-benchmarks-2026",
    },
    {
        "query": "What is the user's favorite development tools?",
        "expected_slug": "favorite-tools",
    },
    {
        "query": "How does the user prefer to receive feedback?",
        "expected_slug": "how-piyush-likes-to-receive-feedback",
        "category": "concepts",
    },
    {
        "query": "What is local-first software design philosophy?",
        "expected_slug": "local-first-software-design",
        "category": "concepts",
    },
    {
        "query": "What is token-efficient memory design?",
        "expected_slug": "token-efficient-memory-design",
        "category": "concepts",
    },
]


def slug_from_source(source: str) -> str:
    """Extract slug from retriever source path like 'facts/tech-stack-preferences.md'."""
    return Path(source).stem


def run_eval(n: int = 3) -> dict:
    retriever = MemoryRetriever(VAULT_PATH, CHROMA_PATH)
    chunk_count = retriever.count()
    if chunk_count == 0:
        print("Vault not indexed — running index first...")
        chunk_count = retriever.index()
        print(f"Indexed {chunk_count} chunks")

    results = []
    hits = 0
    latencies = []

    for q in CUSTOM_QUESTIONS:
        t0 = time.perf_counter()
        recalled = retriever.query(
            q["query"], n=n, category=q.get("category"), min_relevance=0.0
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        top_slugs = [slug_from_source(r["source"]) for r in recalled]
        hit = q["expected_slug"] in top_slugs
        if hit:
            hits += 1

        rank = (
            top_slugs.index(q["expected_slug"]) + 1
            if hit
            else None
        )
        results.append(
            {
                "query": q["query"],
                "expected_slug": q["expected_slug"],
                "hit": hit,
                "rank": rank,
                "top_slugs": top_slugs,
                "latency_ms": round(latency_ms, 1),
            }
        )
        mark = "✓" if hit else "✗"
        print(f"  {mark} {q['expected_slug']:40s} (rank {rank or '-'})")

    accuracy = hits / len(CUSTOM_QUESTIONS)
    summary = {
        "total": len(CUSTOM_QUESTIONS),
        "hits_top3": hits,
        "accuracy_top3": round(accuracy, 4),
        "chunks_indexed": chunk_count,
        "recall_latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 1),
        "questions": results,
    }
    return summary


def main() -> None:
    print(f"Custom eval against real vault: {VAULT_PATH}")
    print(f"Testing top-3 recall for {len(CUSTOM_QUESTIONS)} hand-written queries\n")
    summary = run_eval()
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\nTop-3 hit rate: {summary['accuracy_top3']:.1%} "
        f"({summary['hits_top3']}/{summary['total']})"
    )
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()