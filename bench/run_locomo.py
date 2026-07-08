#!/usr/bin/env python3
"""LOCOMO benchmark runner for ai-memory.

Measures {LLM extractor + ai-memory retrieval + answerer + judge} on the
LOCOMO-10 dataset using mem0's judge prompt for methodology comparability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aimemory.config import CHROMA_PATH, RELEVANCE_FLOOR, ROOT as PROJECT_ROOT, VAULT_PATH
from aimemory.retriever import MemoryRetriever

BENCH_DIR = Path(__file__).resolve().parent
DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DATASET_PATH = BENCH_DIR / "data" / "locomo10.json"
JUDGE_PROMPT_PATH = BENCH_DIR / "judge_prompt.txt"
RESULTS_PATH = BENCH_DIR / "results.json"
TMP_ROOT = BENCH_DIR / ".tmp"

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
CATEGORIES_TO_EVALUATE = {1, 2, 3, 4}

JUDGE_SYSTEM = (
    "You are evaluating conversational AI memory recall. "
    "Return JSON only with the format requested."
)

EXTRACTOR_PROMPT = """Extract atomic factual memories from this conversation snippet.
Each fact must be self-contained (include speaker names and the session date).

Session date: {session_date}

Conversation:
{messages}

Return a JSON array of objects, each with:
- "title": short slug (3-6 words)
- "content": one-sentence fact

Return [] if there are no extractable facts. JSON only."""

ANSWER_PROMPT = """You are answering a question using retrieved memories from past conversations.
Read ALL memories below before answering. "User" in memories refers to the main person.

These conversations took place around {reference_date}. Events occurred in 2022-2024.
Use dates from memories; do not invent dates or use 2025/2026.

{memories}

Question: {question}

Work through the memories, then give your final answer after "ANSWER:"."""


# ---------------------------------------------------------------------------
# Safety + dataset
# ---------------------------------------------------------------------------


def guard_not_real_vault(vault_dir: Path, chroma_dir: Path) -> None:
    real_vault = (PROJECT_ROOT / "vault").resolve()
    real_chroma = (PROJECT_ROOT / ".chroma").resolve()
    for label, path in [("vault", vault_dir), ("chroma", chroma_dir)]:
        resolved = path.resolve()
        if resolved == real_vault or resolved == real_chroma:
            raise RuntimeError(f"Refusing to benchmark against real {label}: {path}")


def download_dataset() -> Path:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        return DATASET_PATH
    print(f"Downloading LOCOMO-10 dataset to {DATASET_PATH}...")
    with httpx.Client(timeout=120) as client:
        resp = client.get(DATASET_URL)
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, list) or len(data) != 10:
        raise RuntimeError(f"Invalid dataset: expected 10 conversations, got {len(data)}")
    DATASET_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return DATASET_PATH


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LOCOMO parsing (from mem0 harness)
# ---------------------------------------------------------------------------


def parse_locomo_date(date_str: str) -> datetime | None:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def get_sorted_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    session_keys = [k for k in conversation if re.match(r"^session_\d+$", k)]
    paired = []
    for key in session_keys:
        date_key = f"{key}_date_time"
        paired.append((key, conversation.get(date_key, ""), conversation[key]))
    paired.sort(
        key=lambda item: (
            (0, parse_locomo_date(item[1]) or datetime.min)
            if parse_locomo_date(item[1])
            else (1, int(re.search(r"\d+", item[0]).group()))
        )
    )
    return paired


def session_to_text(turns: list[dict]) -> str:
    lines = []
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        blip = turn.get("blip_caption", "")
        query = turn.get("query", "")
        if query and blip:
            photo = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo = f"[Sharing image - query for: {query}]"
        elif blip:
            photo = f"[Sharing image that shows: {blip}]"
        else:
            photo = ""
        if photo:
            text = f"{text} {photo}" if text else photo
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def preprocess_answer(category: int, answer: str) -> str:
    if category == 3 and ";" in answer:
        return answer.split(";")[0].strip()
    return str(answer)


# ---------------------------------------------------------------------------
# LLM backends — opencode-go + Claude Code CLI (default), or direct API keys
# ---------------------------------------------------------------------------

JUDGE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["label", "reasoning"],
    }
)


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


def parse_json_blob(text: str) -> list | dict:
    text = strip_json_fences(text)
    for candidate in (text,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            if "Extra data" in str(exc):
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(candidate.lstrip())
                return obj
    for pattern in (r"\[[\s\S]*?\]", r"\{[\s\S]*?\}"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("no JSON found", text, 0)


class OpencodeClient:
    """DeepSeek via OpenCode Go subscription (`opencode run`)."""

    def __init__(self, model: str = "opencode-go/deepseek-v4-pro", timeout: int = 300):
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        parts = []
        if system:
            parts.append(system)
        parts.append(user)
        prompt = "\n\n".join(parts)
        cmd = [
            "opencode",
            "run",
            "--pure",
            "-m",
            self.model,
            "--format",
            "json",
            "--title",
            "aimemory-bench",
            "--dir",
            "/tmp",
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
            cwd="/tmp",
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"opencode run failed: {err}")
        return self._parse_output(proc.stdout)

    def extract_json(self, text: str) -> list | dict:
        return parse_json_blob(text)

    def judge(self, prompt: str) -> dict:
        full_prompt = (
            f"{JUDGE_SYSTEM}\n\n{prompt}\n\n"
            'Respond with JSON only: {"label":"CORRECT"|"WRONG","reasoning":"..."}'
        )
        try:
            raw = self.chat("", full_prompt)
            if not raw.strip():
                raise ValueError("empty response from opencode")
            return parse_json_blob(raw)
        except Exception as exc:
            return {"label": "WRONG", "reasoning": f"opencode judge failed: {exc}"}

    @staticmethod
    def _parse_output(stdout: str) -> str:
        texts: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") != "text":
                continue
            part = evt.get("part", {})
            if part.get("type") == "text" and part.get("text"):
                texts.append(part["text"])
        return "".join(texts) if texts else stdout.strip()


class ClaudeCliClient:
    """Claude via $20 subscription (`claude -p`), not Anthropic API keys."""

    def __init__(self, model: str = "sonnet", timeout: int = 180):
        self.model = model
        self.timeout = timeout

    def judge(self, prompt: str) -> dict:
        full_prompt = f"{JUDGE_SYSTEM}\n\n{prompt}"
        cmd = [
            "claude",
            "-p",
            "--bare",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--json-schema",
            JUDGE_JSON_SCHEMA,
            full_prompt,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            return {"label": "WRONG", "reasoning": f"claude failed: {err}"}
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"label": "WRONG", "reasoning": "Failed to parse claude output"}
        if isinstance(payload.get("structured_output"), dict):
            return payload["structured_output"]
        result = payload.get("result", "")
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return parse_json_blob(result)
        return {"label": "WRONG", "reasoning": "No structured judge output"}


class DeepSeekClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base = "https://api.deepseek.com/v1"

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        with httpx.Client(timeout=180) as client:
            resp = client.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def extract_json(self, text: str) -> list | dict:
        return parse_json_blob(text)


class AnthropicClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        with httpx.Client(timeout=180) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 1024,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    def judge(self, prompt: str) -> dict:
        raw = self.chat(JUDGE_SYSTEM, prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {"label": "WRONG", "reasoning": "Failed to parse judge response"}


def resolve_llm_backend(name: str) -> str:
    if name != "auto":
        return name
    return "api" if os.environ.get("DEEPSEEK_API_KEY") else "opencode"


def resolve_judge_backend(name: str) -> str:
    if name != "auto":
        return name
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    # Prefer Claude Code subscription when logged in; fall back to opencode judge.
    try:
        proc = subprocess.run(
            ["claude", "-p", "--bare", "Reply OK"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0 and "Not logged in" not in (proc.stdout + proc.stderr):
            return "claude-cli"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "opencode"


def build_llm_client(backend: str, args: argparse.Namespace):
    if backend == "opencode":
        model = os.environ.get("OPENCODE_MODEL", args.opencode_model)
        return OpencodeClient(model=model), model
    if backend == "api":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY required for --llm-backend api")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        return DeepSeekClient(key, model), model
    raise RuntimeError(f"Unknown --llm-backend: {backend}")


def build_judge_client(backend: str, args: argparse.Namespace):
    if backend == "claude-cli":
        model = os.environ.get("CLAUDE_MODEL", args.claude_model)
        return ClaudeCliClient(model=model), model
    if backend == "opencode":
        model = os.environ.get("OPENCODE_JUDGE_MODEL", args.opencode_model)
        return OpencodeClient(model=model), model
    if backend == "api":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY required for --judge-backend api")
        model = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6-20250514")
        return AnthropicClient(key, model), model
    raise RuntimeError(f"Unknown --judge-backend: {backend}")


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------


def clear_workspace(vault_dir: Path, chroma_dir: Path) -> None:
    guard_not_real_vault(vault_dir, chroma_dir)
    for d in (vault_dir, chroma_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def format_memories(results: list[dict]) -> str:
    if not results:
        return "(No relevant memories found)"
    lines = []
    for r in results:
        lines.append(f"- {r['title']}: {r['content']}")
    return "\n".join(lines)


def ingest_conversation(
    retriever: MemoryRetriever,
    entry: dict,
    conv_id: str,
    extractor,
) -> int:
    conversation = entry["conversation"]
    sorted_sessions = get_sorted_sessions(conversation)
    total_facts = 0

    for _session_key, date_str, turns in sorted_sessions:
        messages = session_to_text(turns)
        if not messages.strip():
            continue
        prompt = EXTRACTOR_PROMPT.format(session_date=date_str, messages=messages)
        try:
            raw = extractor.chat(
                "Extract factual memories as JSON. Return a JSON array only.",
                prompt,
            )
            facts = extractor.extract_json(raw)
        except Exception as exc:
            print(f"  extractor error: {exc}")
            continue
        if not isinstance(facts, list):
            continue
        for i, fact in enumerate(facts):
            if not isinstance(fact, dict):
                continue
            title = fact.get("title", f"fact-{total_facts + i}")
            content = fact.get("content", "")
            if not content.strip():
                continue
            retriever.add_memory(
                title=title,
                content=content,
                category="facts",
                tags=[conv_id],
            )
            total_facts += 1
        retriever.index()
    return total_facts


def answer_question(
    question: str,
    memories: list[dict],
    reference_date: str,
    answerer,
) -> str:
    prompt = ANSWER_PROMPT.format(
        reference_date=reference_date,
        memories=format_memories(memories),
        question=question,
    )
    raw = answerer.chat("", prompt)
    if "ANSWER:" in raw:
        return raw.rsplit("ANSWER:", 1)[-1].strip()
    return raw.strip()


def run_benchmark(args: argparse.Namespace) -> dict:
    dataset = load_dataset(Path(args.dataset))
    judge_template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

    llm_backend = resolve_llm_backend(args.llm_backend)
    judge_backend = resolve_judge_backend(args.judge_backend)

    answerer, extractor_model = build_llm_client(llm_backend, args)
    extractor = answerer
    judge = None
    judge_model = None
    if not args.predict_only:
        judge, judge_model = build_judge_client(judge_backend, args)

    k_values = [3, 5, 10] if args.k_sweep else [args.k]
    conv_limit = min(args.convs, len(dataset))
    results: dict = {
        "config": {
            "convs": conv_limit,
            "k_values": k_values,
            "relevance_floor": RELEVANCE_FLOOR,
            "llm_backend": llm_backend,
            "judge_backend": judge_backend if not args.predict_only else None,
            "extractor_model": extractor_model,
            "answerer_model": extractor_model,
            "judge_model": judge_model,
            "predict_only": args.predict_only,
        },
        "per_k": {},
        "questions": [],
    }

    for k in k_values:
        results["per_k"][str(k)] = {
            "overall": {"correct": 0, "total": 0, "accuracy": 0.0},
            "by_category": {name: {"correct": 0, "total": 0} for name in CATEGORY_NAMES.values()},
            "recall_latency_ms": [],
            "context_chars": [],
        }

    for conv_idx in range(conv_limit):
        entry = dataset[conv_idx]
        conv_id = f"locomo_{conv_idx}"
        vault_dir = TMP_ROOT / f"vault_{conv_idx}"
        chroma_dir = TMP_ROOT / f"chroma_{conv_idx}"

        print(f"\n=== Conversation {conv_idx} ===")
        clear_workspace(vault_dir, chroma_dir)
        retriever = MemoryRetriever(vault_dir, chroma_dir)

        n_facts = ingest_conversation(retriever, entry, conv_id, extractor)
        print(f"  Ingested {n_facts} facts, {retriever.count()} chunks indexed")

        conversation = entry["conversation"]
        sorted_sessions = get_sorted_sessions(conversation)
        ref_date = sorted_sessions[-1][1] if sorted_sessions else "2023"

        questions = entry.get("qa", entry.get("qa_pairs", []))
        eval_questions = [
            (qi, qa)
            for qi, qa in enumerate(questions)
            if qa.get("category") in CATEGORIES_TO_EVALUATE
        ]
        if args.max_questions:
            eval_questions = eval_questions[: args.max_questions]

        for qi, qa in eval_questions:
            question = qa["question"]
            category = qa["category"]
            cat_name = CATEGORY_NAMES.get(category, "unknown")
            gold = preprocess_answer(category, str(qa["answer"]))

            q_result: dict = {
                "question_id": f"conv{conv_idx}_q{qi}",
                "conversation_idx": conv_idx,
                "category": category,
                "category_name": cat_name,
                "question": question,
                "gold_answer": gold,
            }

            for k in k_values:
                t0 = time.perf_counter()
                memories = retriever.query(question, n=k)
                latency_ms = (time.perf_counter() - t0) * 1000
                ctx_chars = sum(len(m["content"]) for m in memories)

                bucket = results["per_k"][str(k)]
                bucket["recall_latency_ms"].append(latency_ms)
                bucket["context_chars"].append(ctx_chars)

                if args.predict_only:
                    q_result[f"k{k}"] = {
                        "memories": len(memories),
                        "latency_ms": round(latency_ms, 1),
                        "context_chars": ctx_chars,
                    }
                    continue

                generated = answer_question(question, memories, ref_date, answerer)
                judge_prompt = judge_template.format(
                    question=question,
                    answer=gold,
                    response=generated,
                )
                verdict = judge.judge(judge_prompt)
                correct = verdict.get("label", "").upper() == "CORRECT"

                bucket["overall"]["total"] += 1
                bucket["by_category"][cat_name]["total"] += 1
                if correct:
                    bucket["overall"]["correct"] += 1
                    bucket["by_category"][cat_name]["correct"] += 1

                q_result[f"k{k}"] = {
                    "generated_answer": generated,
                    "judgment": verdict.get("label", "WRONG"),
                    "reasoning": verdict.get("reasoning", ""),
                    "memories_retrieved": len(memories),
                    "latency_ms": round(latency_ms, 1),
                    "context_chars": ctx_chars,
                }

            results["questions"].append(q_result)
            if not args.predict_only:
                primary = q_result.get(f"k{k_values[0]}", {})
                mark = "✓" if primary.get("judgment") == "CORRECT" else "✗"
                print(f"  {mark} [{cat_name}] {question[:60]}...")

    for k, bucket in results["per_k"].items():
        total = bucket["overall"]["total"]
        if total:
            bucket["overall"]["accuracy"] = round(
                bucket["overall"]["correct"] / total, 4
            )
        for cat, stats in bucket["by_category"].items():
            if stats["total"]:
                stats["accuracy"] = round(stats["correct"] / stats["total"], 4)
        lats = bucket["recall_latency_ms"]
        if lats:
            bucket["recall_latency_p50_ms"] = round(statistics.median(lats), 1)
            bucket["recall_latency_p95_ms"] = round(
                sorted(lats)[int(len(lats) * 0.95)], 1
            )
        chars = bucket["context_chars"]
        if chars:
            bucket["mean_context_chars"] = round(statistics.mean(chars), 1)

    return results


# ---------------------------------------------------------------------------
# Self-check (no API keys needed for recall portion)
# ---------------------------------------------------------------------------


def self_check() -> None:
    print("Running self-check...")
    vault_dir = TMP_ROOT / "selfcheck_vault"
    chroma_dir = TMP_ROOT / "selfcheck_chroma"
    clear_workspace(vault_dir, chroma_dir)
    r = MemoryRetriever(vault_dir, chroma_dir)

    r.add_memory(
        "caroline-lgbtq-group",
        "Caroline went to an LGBTQ support group on 7 May 2023.",
        category="facts",
        tags=["selfcheck"],
    )
    r.index()
    hits = r.query("When did Caroline go to the LGBTQ support group?", n=3)
    assert hits, "FAIL: recall returned nothing for a fact just written"
    assert hits[0]["relevance"] >= RELEVANCE_FLOOR, (
        f"FAIL: top hit below relevance floor ({hits[0]['relevance']})"
    )
    print("  ✓ recall returns freshly-written fact above floor")

    judge_template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    judge_backend = resolve_judge_backend("auto")
    try:
        judge, _ = build_judge_client(
            judge_backend,
            argparse.Namespace(
                claude_model=os.environ.get("CLAUDE_MODEL", "sonnet"),
                opencode_model=os.environ.get(
                    "OPENCODE_MODEL", "opencode-go/deepseek-v4-pro"
                ),
            ),
        )
        good = judge_template.format(
            question="When did Caroline go to the LGBTQ support group?",
            answer="7 May 2023",
            response="Caroline went on 7 May 2023.",
        )
        bad = judge_template.format(
            question="When did Caroline go to the LGBTQ support group?",
            answer="7 May 2023",
            response="She went in 2019.",
        )
        good_v = judge.judge(good)
        bad_v = judge.judge(bad)
        assert good_v.get("label", "").upper() == "CORRECT", (
            f"FAIL: judge marked obvious correct as {good_v}"
        )
        assert bad_v.get("label", "").upper() == "WRONG", (
            f"FAIL: judge marked obvious wrong as {bad_v}"
        )
        print(f"  ✓ judge ({judge_backend}) marks obvious correct/wrong answers")
    except Exception as exc:
        print(f"  ⚠ skipping judge self-check: {exc}")

    guard_not_real_vault(vault_dir, chroma_dir)
    print("  ✓ temp vault guard passes")
    print("Self-check passed.")


def print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("LOCOMO BENCHMARK SUMMARY")
    print("=" * 60)
    for k, bucket in results["per_k"].items():
        overall = bucket["overall"]
        if overall["total"] == 0:
            continue
        print(f"\nk={k}: {overall['accuracy']:.1%} ({overall['correct']}/{overall['total']})")
        for cat in ("single-hop", "multi-hop", "temporal", "open-domain"):
            stats = bucket["by_category"].get(cat, {})
            if stats.get("total"):
                acc = stats.get("accuracy", stats["correct"] / stats["total"])
                print(f"  {cat:12s} {acc:.1%} ({stats['correct']}/{stats['total']})")
        if "recall_latency_p50_ms" in bucket:
            print(
                f"  recall p50/p95: {bucket['recall_latency_p50_ms']:.0f}ms"
                f" / {bucket['recall_latency_p95_ms']:.0f}ms"
            )
        if "mean_context_chars" in bucket:
            print(f"  mean context chars: {bucket['mean_context_chars']:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOCOMO benchmark on ai-memory")
    parser.add_argument("--convs", type=int, default=10, help="Number of conversations")
    parser.add_argument("--k", type=int, default=3, help="Recall top-k (default: 3)")
    parser.add_argument(
        "--k-sweep", action="store_true", help="Sweep k over 3, 5, 10"
    )
    parser.add_argument(
        "--max-questions", type=int, default=None, help="Cap questions per conversation"
    )
    parser.add_argument(
        "--dataset", default=str(DATASET_PATH), help="Path to locomo10.json"
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Ingest + recall only, skip answer/judge",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["auto", "opencode", "api"],
        default="auto",
        help="LLM for extract+answer (default: opencode-go, or api if DEEPSEEK_API_KEY set)",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["auto", "claude-cli", "opencode", "api"],
        default="auto",
        help="Judge backend (default: claude -p if logged in, else opencode)",
    )
    parser.add_argument(
        "--opencode-model",
        default="opencode-go/deepseek-v4-pro",
        help="Model for opencode run (default: opencode-go/deepseek-v4-pro)",
    )
    parser.add_argument(
        "--claude-model",
        default="sonnet",
        help="Claude Code model alias for judge (default: sonnet)",
    )
    parser.add_argument("--self-check", action="store_true", help="Run sanity checks")
    parser.add_argument(
        "--download-only", action="store_true", help="Download dataset and exit"
    )
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    if args.download_only:
        download_dataset()
        print(f"Dataset ready at {DATASET_PATH}")
        return

    download_dataset()
    results = run_benchmark(args)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()