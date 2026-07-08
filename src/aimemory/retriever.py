from __future__ import annotations

import re
import math
import hashlib
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import chromadb
import frontmatter
from chromadb.utils import embedding_functions

from .config import (
    EMBEDDING_MODEL,
    MAX_CONTENT_LENGTH,
    MAX_RESULTS,
    RELEVANCE_FLOOR,
    CANDIDATE_K,
)

CATEGORY_DIRS = {
    "people": "people",
    "projects": "projects",
    "concepts": "concepts",
    "facts": "facts",
}


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.N: int = 0

    def index(self, docs: list[tuple[str, str]]) -> None:
        self.corpus = []
        self.doc_ids = []
        df: dict[str, int] = defaultdict(int)
        for cid, text in docs:
            tokens = self._tokenize(text)
            self.corpus.append(tokens)
            self.doc_ids.append(cid)
            for t in set(tokens):
                df[t] += 1
        self.N = len(self.corpus)
        if self.N == 0:
            self.avgdl = 0.0
            self.idf = {}
            return
        self.avgdl = sum(len(d) for d in self.corpus) / self.N
        self.idf = {}
        for t, f in df.items():
            self.idf[t] = math.log((self.N - f + 0.5) / (f + 0.5) + 1)

    def search(self, query: str) -> list[tuple[str, float]]:
        qtokens = self._tokenize(query)
        if not qtokens or self.N == 0:
            return []
        scores: list[tuple[str, float]] = []
        for i, doc_tokens in enumerate(self.corpus):
            score = 0.0
            dl = len(doc_tokens)
            for t in qtokens:
                if t not in self.idf:
                    continue
                tf = doc_tokens.count(t)
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[t] * num / denom
            if score > 0:
                scores.append((self.doc_ids[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class MemoryChunk:
    id: str
    title: str
    content: str
    category: str
    tags: list[str] = field(default_factory=list)
    source_file: str = ""


class MemoryRetriever:
    def __init__(self, vault_path: Path, chroma_path: Path):
        self.vault_path = vault_path
        self._lock = threading.RLock()
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name="memories",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = BM25()
        self._chunks: dict[str, MemoryChunk] = {}
        self._reranker = None
        self._load_chunks()

    def _load_chunks(self) -> None:
        chunks = self._scan_vault()
        for c in chunks:
            self._chunks[c.id] = c
        self._rebuild_bm25()

    def index(self) -> int:
        chunks = self._scan_vault()

        with self._lock:
            try:
                self.client.delete_collection(name=self.collection.name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name="memories",
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )

            self._chunks.clear()
            if not chunks:
                self.bm25.index([])
                return 0

            self.collection.add(
                documents=[c.content for c in chunks],
                metadatas=[self._meta(c) for c in chunks],
                ids=[c.id for c in chunks],
            )

            for c in chunks:
                self._chunks[c.id] = c

            self._rebuild_bm25()
        return len(chunks)

    def query(
        self,
        text: str,
        n: int = MAX_RESULTS,
        category: str | None = None,
        min_relevance: float = RELEVANCE_FLOOR,
    ) -> list[dict]:
        where = {"category": category} if category else None

        with self._lock:
            vec_results = self.collection.query(
                query_texts=[text],
                n_results=CANDIDATE_K,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        vec_ids: list[str] = (
            vec_results["ids"][0] if vec_results.get("ids") and vec_results["ids"][0] else []
        )
        vec_distances = (
            vec_results["distances"][0] if vec_results.get("distances") else []
        )

        vec_sim: dict[str, float] = {}
        for cid, dist in zip(vec_ids, vec_distances):
            vec_sim[cid] = max(0.0, 1.0 - dist)

        bm25_results = self.bm25.search(text)[:CANDIDATE_K]
        bm25_ids = [cid for cid, _ in bm25_results]

        fused_ids = self._rrf_fuse(vec_ids, bm25_ids, CANDIDATE_K)
        if not fused_ids:
            return []

        candidates = [self._chunks[cid] for cid in fused_ids if cid in self._chunks]
        if not candidates:
            return []

        seen_sources: set[str] = set()
        out: list[dict] = []
        for c in candidates:
            if c.source_file in seen_sources:
                continue
            seen_sources.add(c.source_file)

            relevance = vec_sim.get(c.id, 0.0)
            if relevance < min_relevance:
                continue

            content = c.content
            if len(content) > MAX_CONTENT_LENGTH:
                content = self._truncate_sentence(content, MAX_CONTENT_LENGTH)

            out.append(
                {
                    "title": c.title,
                    "content": content,
                    "category": c.category,
                    "tags": c.tags,
                    "relevance": round(relevance, 4),
                    "source": c.source_file,
                }
            )

            if len(out) >= n:
                break

        return out

    def add_memory(
        self,
        title: str,
        content: str,
        category: str = "facts",
        tags: list[str] | None = None,
    ) -> str:
        tags = tags or []
        subdir = CATEGORY_DIRS.get(category, "facts")
        dir_path = self.vault_path / subdir
        dir_path.mkdir(parents=True, exist_ok=True)

        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        filepath = dir_path / f"{slug}.md"

        fm = frontmatter.Post(
            content,
            title=title,
            category=category,
            tags=tags,
            created=datetime.now().strftime("%Y-%m-%d"),
        )
        filepath.write_text(frontmatter.dumps(fm))

        rel = filepath.relative_to(self.vault_path)
        with self._lock:
            stale = self.collection.get(where={"source": str(rel)})
            if stale and stale.get("ids"):
                self.collection.delete(ids=stale["ids"])

            chunks = self._scan_file(filepath)
            if chunks:
                self.collection.upsert(
                    documents=[c.content for c in chunks],
                    metadatas=[self._meta(c) for c in chunks],
                    ids=[c.id for c in chunks],
                )
                for c in chunks:
                    self._chunks[c.id] = c

            self._rebuild_bm25()
        return str(filepath)

    def check_overlaps(self, title: str, content: str, threshold: float = 0.82) -> list[dict]:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        candidates = self._search_candidates(content, n=10)
        overlaps: list[dict] = []
        if not candidates:
            return overlaps

        reranker = self._get_reranker()
        for c in candidates:
            c_slug = Path(c.source_file).stem
            if c_slug == slug:
                continue
            pairs = [(content, c.content)]
            score = float(reranker.predict(pairs)[0])
            prob = 1.0 / (1.0 + math.exp(-score))
            if prob > threshold:
                overlaps.append(
                    {
                        "slug": c_slug,
                        "title": c.title,
                        "similarity": round(prob, 4),
                    }
                )
        return overlaps

    def delete_note(self, slug: str) -> bool:
        slug_norm = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
        for subdir in CATEGORY_DIRS.values():
            candidate = self.vault_path / subdir / f"{slug_norm}.md"
            if not candidate.exists():
                continue
            trash_dir = self.vault_path / ".trash"
            trash_dir.mkdir(exist_ok=True)
            rel = candidate.relative_to(self.vault_path)
            dest = trash_dir / f"{slug_norm}.md"
            if dest.exists():
                dest.unlink()
            candidate.rename(dest)

            with self._lock:
                stale = self.collection.get(where={"source": str(rel)})
                if stale and stale.get("ids"):
                    self.collection.delete(ids=stale["ids"])
                    for cid in stale["ids"]:
                        self._chunks.pop(cid, None)
                self._rebuild_bm25()
            return True
        return False

    def get_note(self, slug: str) -> str | None:
        slug_norm = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
        for subdir in CATEGORY_DIRS.values():
            candidate = self.vault_path / subdir / f"{slug_norm}.md"
            if candidate.exists():
                try:
                    return frontmatter.load(candidate).content.strip()
                except Exception:
                    return None
        for md_file in self.vault_path.rglob(f"{slug_norm}.md"):
            if md_file.parent.name in (".obsidian", "templates"):
                continue
            try:
                return frontmatter.load(md_file).content.strip()
            except Exception:
                continue
        return None

    def count(self) -> int:
        return self.collection.count()

    def _scan_vault(self) -> list[MemoryChunk]:
        chunks: list[MemoryChunk] = []
        for md_file in sorted(self.vault_path.rglob("*.md")):
            rel = md_file.relative_to(self.vault_path)
            if rel.parts and rel.parts[0] in (".obsidian", "templates", ".trash"):
                continue
            chunks.extend(self._scan_file(md_file))
        return chunks

    def _scan_file(self, md_file: Path) -> list[MemoryChunk]:
        rel = md_file.relative_to(self.vault_path)
        category = self._infer_category(rel)
        try:
            post = frontmatter.load(md_file)
        except Exception:
            return []

        title = post.get("title", md_file.stem)
        tags = post.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        body = post.content.strip()
        if not body:
            return []

        sections = re.split(r"\n(?=## )", body)
        chunks: list[MemoryChunk] = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            sec_title = title
            heading_match = re.match(r"^## (.+)$", section, re.MULTILINE)
            if heading_match:
                sec_title = f"{title} / {heading_match.group(1).strip()}"
            chunk_id = hashlib.sha256(f"{rel}#{i}".encode()).hexdigest()[:16]
            chunks.append(
                MemoryChunk(
                    id=chunk_id,
                    title=sec_title,
                    content=self._clean_wikilinks(section).strip(),
                    category=category,
                    tags=tags,
                    source_file=str(rel),
                )
            )
        return chunks

    def _rebuild_bm25(self) -> None:
        docs = [(cid, c.content) for cid, c in self._chunks.items()]
        self.bm25.index(docs)

    def _search_candidates(self, text: str, n: int = 10) -> list[MemoryChunk]:
        with self._lock:
            vec_results = self.collection.query(
                query_texts=[text],
                n_results=n,
                include=["metadatas"],
            )
        if not vec_results.get("ids") or not vec_results["ids"][0]:
            return []
        return [self._chunks[cid] for cid in vec_results["ids"][0] if cid in self._chunks]

    def _rrf_fuse(
        self, vec_ids: list[str], bm25_ids: list[str], n: int
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for cid in vec_ids[:n]:
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
        for cid in bm25_ids[:n]:
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
        return result[:n]

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def _truncate_sentence(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        last_period = truncated.rfind(". ")
        last_newline = truncated.rfind("\n")
        cut = max(last_period, last_newline)
        if cut > max_len * 0.5:
            return text[: cut + 1] + "\n\n...(truncated)"
        return truncated + "\n\n...(truncated)"

    @staticmethod
    def _meta(c: MemoryChunk) -> dict:
        return {
            "title": c.title,
            "category": c.category,
            "tags": ",".join(c.tags),
            "source": c.source_file,
            "char_count": len(c.content),
        }

    _WIKILINK_RE = re.compile(r"\[\[([^\]]+?)(?:\|[^\]]+)?\]\]")

    @classmethod
    def _clean_wikilinks(cls, text: str) -> str:
        return cls._WIKILINK_RE.sub(r"\1", text)

    @staticmethod
    def _infer_category(rel: Path) -> str:
        if len(rel.parts) > 1:
            for cat, dirname in CATEGORY_DIRS.items():
                if rel.parts[0] == dirname:
                    return cat
        return "facts"
