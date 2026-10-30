"""Hybrid RAG engine: OWASP LLM01 prompt-injection sanitizer, recursive markdown
chunking, Hugging Face SentenceTransformer (BAAI/bge-small-en-v1.5) 384-dim
L2-normalized dense embeddings with deterministic offline fallback, FAISS
IndexFlatIP vector search, and Reciprocal Rank Fusion (RRF) with Okapi BM25."""
from __future__ import annotations

import hashlib
import math
import os
import pathlib
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

import faiss
import numpy as np

from finpulse.schemas import (
    DocumentInput,
    EvalCaseResult,
    ModeEvalSummary,
    RAGEvalResponse,
    RetrievedChunk,
)

EMBEDDING_DIM = 384
DEFAULT_HF_EMBEDDING_MODEL = os.getenv("FINPULSE_HF_MODEL", "BAAI/bge-small-en-v1.5")

_HF_MODEL_LOCK = threading.Lock()
_HF_MODEL_INSTANCE: Any | None = None
_HF_LOAD_ATTEMPTED: bool = False
_ACTIVE_EMBEDDING_BACKEND: str = "uninitialized"

_INJECTION_PATTERNS = [
    re.compile(r"<\s*/?\s*(?:system|instructions|prompt|assistant|developer|tool_call)\s*>", re.I),
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.I),
]

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
    "what", "how", "why", "when", "does", "can", "should", "into", "over",
})


def sanitize_prompt_input(text: str) -> str:
    """Strip OWASP LLM01 prompt-injection delimiters and override directives."""
    cleaned = text
    for _ in range(3):
        prev = cleaned
        for pat in _INJECTION_PATTERNS:
            cleaned = pat.sub(" ", cleaned)
        if cleaned == prev:
            break
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer with stop-word removal."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _get_hf_embedder() -> Any | None:
    """Thread-safe lazy loader for Hugging Face SentenceTransformer (`BAAI/bge-small-en-v1.5`).
    Loads from local Hugging Face cache first (`local_files_only=True`) so corporate
    firewalls or offline environments never block or hang; falls back cleanly to None."""
    global _HF_MODEL_INSTANCE, _HF_LOAD_ATTEMPTED, _ACTIVE_EMBEDDING_BACKEND

    backend_pref = os.getenv("FINPULSE_EMBEDDING_BACKEND", "auto").strip().lower()
    if backend_pref == "hash":
        _ACTIVE_EMBEDDING_BACKEND = f"hash-{EMBEDDING_DIM}"
        return None

    with _HF_MODEL_LOCK:
        if _HF_LOAD_ATTEMPTED:
            return _HF_MODEL_INSTANCE
        _HF_LOAD_ATTEMPTED = True
        try:
            from sentence_transformers import SentenceTransformer

            try:
                model = SentenceTransformer(DEFAULT_HF_EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                if os.getenv("FINPULSE_ALLOW_HF_DOWNLOAD", "0") == "1":
                    model = SentenceTransformer(DEFAULT_HF_EMBEDDING_MODEL)
                else:
                    raise
            _HF_MODEL_INSTANCE = model
            _ACTIVE_EMBEDDING_BACKEND = f"huggingface:{DEFAULT_HF_EMBEDDING_MODEL}"
        except Exception:
            _HF_MODEL_INSTANCE = None
            _ACTIVE_EMBEDDING_BACKEND = f"hash-{EMBEDDING_DIM}"
        return _HF_MODEL_INSTANCE


def get_active_embedding_backend() -> str:
    """Return identifier of the active dense embedding backend (`huggingface:...` or `hash-384`)."""
    if _ACTIVE_EMBEDDING_BACKEND == "uninitialized":
        _get_hf_embedder()
    return _ACTIVE_EMBEDDING_BACKEND


def encode_texts_hash(texts: list[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic offline fallback encoder producing L2-normalized float32 vectors
    via signed multi-probe feature hashing over unigrams, bigrams, and character trigrams."""
    mat = np.zeros((len(texts), dim), dtype=np.float32)

    for row_idx, raw_text in enumerate(texts):
        tokens = tokenize(raw_text)
        if not tokens:
            mat[row_idx, 0] = 1.0
            continue

        features: dict[str, float] = {}
        for i, tok in enumerate(tokens):
            features[f"u:{tok}"] = features.get(f"u:{tok}", 0.0) + 2.0
            if i + 1 < len(tokens):
                bg = f"b:{tok}_{tokens[i + 1]}"
                features[bg] = features.get(bg, 0.0) + 1.5
            padded = f"#{tok}#"
            for j in range(len(padded) - 2):
                cg = f"c:{padded[j : j + 3]}"
                features[cg] = features.get(cg, 0.0) + 0.35

        vec = np.zeros(dim, dtype=np.float32)
        for feat, count in features.items():
            weight = 1.0 + math.log(count)
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h1 = int.from_bytes(digest[:4], "little")
            h2 = int.from_bytes(digest[4:], "little")
            idx1 = h1 % dim
            idx2 = h2 % dim
            sign1 = 1.0 if (h1 & 1) == 0 else -1.0
            sign2 = 1.0 if (h2 & 1) == 0 else -1.0
            vec[idx1] += sign1 * weight
            vec[idx2] += sign2 * (0.5 * weight)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            mat[row_idx] = vec / norm
        else:
            mat[row_idx, 0] = 1.0

    return np.ascontiguousarray(mat, dtype=np.float32)


_VECTOR_CACHE_LOCK = threading.Lock()
_VECTOR_CACHE: dict[str, np.ndarray] = {}


def encode_texts_dense(texts: list[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Encode texts into L2-normalized float32 dense vectors (shape: [N, dim]).
    Uses Hugging Face `SentenceTransformer('BAAI/bge-small-en-v1.5')` with thread-safe
    vector caching when available, and automatically falls back to `encode_texts_hash`
    in offline/CI environments."""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    if dim == EMBEDDING_DIM:
        hf_model = _get_hf_embedder()
        if hf_model is not None:
            with _VECTOR_CACHE_LOCK:
                missing = [t for t in dict.fromkeys(texts) if t not in _VECTOR_CACHE]
            if missing:
                encoded = np.ascontiguousarray(
                    hf_model.encode(
                        missing,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
                with _VECTOR_CACHE_LOCK:
                    for t_str, row_vec in zip(missing, encoded):
                        _VECTOR_CACHE[t_str] = row_vec
            with _VECTOR_CACHE_LOCK:
                mat = np.stack([_VECTOR_CACHE[t] for t in texts], axis=0)
            return np.ascontiguousarray(mat, dtype=np.float32)

    return encode_texts_hash(texts, dim=dim)


def chunk_document(
    doc: DocumentInput,
    chunk_size: int = 420,
    chunk_overlap: int = 60,
) -> list[dict[str, str | None]]:
    """Split a markdown or text document by section headings and character windows."""
    sections = re.split(r"(?m)^(?=##?\s+)", doc.content.strip())
    chunks: list[dict[str, str | None]] = []
    seq = 0

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= chunk_size:
            seq += 1
            chunks.append({
                "chunk_id": f"{doc.doc_id}#c{seq}",
                "doc_id": doc.doc_id,
                "title": doc.title,
                "category": doc.category,
                "ticker": doc.ticker,
                "text": sec,
            })
        else:
            step = max(chunk_size - chunk_overlap, 60)
            for start in range(0, len(sec), step):
                window = sec[start : start + chunk_size].strip()
                if len(window) < 30:
                    continue
                seq += 1
                chunks.append({
                    "chunk_id": f"{doc.doc_id}#c{seq}",
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "category": doc.category,
                    "ticker": doc.ticker,
                    "text": window,
                })
                if start + chunk_size >= len(sec):
                    break

    return chunks


@dataclass
class _StoredChunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    ticker: str | None
    text: str
    tokens: list[str]
    content_hash: str


_QUERY_SYNONYM_MAP: dict[str, str] = {
    "vol crush": "implied volatility IV crush vertical debit spread Vega",
    "iv crush": "implied volatility IV crush vertical debit spread Vega",
    "time decay": "Theta daily time decay DTE option Greeks",
    "convexity": "Gamma convexity ATM expiry Delta",
    "stop loss": "ATR stop-loss position sizing per-trade risk floor never widen",
    "position size": "ATR volatility-normalized position sizing MaxRupeeRisk",
    "drawdown": "daily drawdown wall -2.0 R risk governance",
    "slippage": "Net-R accounting transaction charges brokerage GST",
    "fees": "Net-R accounting round-trip transaction charges brokerage GST",
    "breakout": "Opening Range Breakout ORB RVOL relative volume VWAP",
    "pullback": "VWAP standard deviation band institutional accumulation RVOL",
    "injection": "OWASP LLM01 prompt-injection sanitizer delimiter grounding",
}


def expand_financial_query(query: str) -> str:
    """Deterministic domain synonym/acronym query expansion for Corrective RAG retries."""
    q_low = query.lower()
    expansions: list[str] = []
    for phrase, expanded in _QUERY_SYNONYM_MAP.items():
        if phrase in q_low:
            expansions.append(expanded)
    if not expansions:
        expansions.append("VWAP ATR Black-Scholes Greeks risk governance")
    return f"{query} {' '.join(expansions)}".strip()


class HybridFAISSStore:
    """Thread-safe FAISS IndexFlatIP dense vector index paired with Okapi BM25
    lexical scoring, content-hash deduplication, and Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self._lock = threading.RLock()
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: list[_StoredChunk] = []
        self._doc_freqs: dict[str, int] = {}
        self._seen_hashes: set[str] = set()

    @property
    def total_vectors(self) -> int:
        with self._lock:
            return int(self.index.ntotal)

    @property
    def embedding_backend(self) -> str:
        return get_active_embedding_backend()

    def ingest_documents(
        self,
        documents: list[DocumentInput],
        chunk_size: int = 420,
        chunk_overlap: int = 60,
    ) -> int:
        """Chunk documents, deduplicate via SHA-256 content digest, embed into
        L2-normalized vectors, and thread-safely add to FAISS + BM25 index."""
        raw_chunks: list[dict[str, str | None]] = []
        for doc in documents:
            raw_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

        if not raw_chunks:
            return 0

        with self._lock:
            deduped_chunks: list[tuple[dict[str, str | None], str]] = []
            for c in raw_chunks:
                raw_key = f"{c['doc_id']}::{c['chunk_id']}::{c['text']}"
                digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
                if digest in self._seen_hashes:
                    continue
                self._seen_hashes.add(digest)
                deduped_chunks.append((c, digest))

            if not deduped_chunks:
                return 0

            texts = [f"{c['title']} {c['text']}" for c, _ in deduped_chunks]
            vectors = encode_texts_dense(texts, dim=self.dim)
            self.index.add(vectors)

            for c, digest in deduped_chunks:
                toks = tokenize(f"{c['title']} {c['text']}")
                for unique_tok in set(toks):
                    self._doc_freqs[unique_tok] = self._doc_freqs.get(unique_tok, 0) + 1
                self._chunks.append(
                    _StoredChunk(
                        chunk_id=str(c["chunk_id"]),
                        doc_id=str(c["doc_id"]),
                        title=str(c["title"]),
                        category=str(c["category"]),
                        ticker=c["ticker"],
                        text=str(c["text"]),
                        tokens=toks,
                        content_hash=digest,
                    )
                )
            return len(deduped_chunks)

    def _bm25_score(self, query_tokens: list[str], chunk: _StoredChunk, avg_dl: float) -> float:
        if not query_tokens or not chunk.tokens:
            return 0.0
        n_docs = max(len(self._chunks), 1)
        k1 = 1.5
        b = 0.75
        dl = len(chunk.tokens)
        score = 0.0
        tf_map: dict[str, int] = {}
        for t in chunk.tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        for qt in set(query_tokens):
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue
            df = self._doc_freqs.get(qt, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / max(avg_dl, 1.0)))
            score += idf * ((tf * (k1 + 1.0)) / denom)
        return score

    def search(
        self,
        query: str,
        top_k: int = 4,
        category: str | None = None,
        ticker: str | None = None,
        use_hybrid_rrf: bool = True,
        retrieval_mode: Literal["hybrid_rrf", "dense_only", "lexical_only"] = "hybrid_rrf",
    ) -> tuple[str, list[RetrievedChunk]]:
        """Execute FAISS dense search + BM25 lexical search fused via Reciprocal Rank Fusion."""
        clean_q = sanitize_prompt_input(query)
        effective_mode: Literal["hybrid_rrf", "dense_only", "lexical_only"] = (
            "dense_only" if (not use_hybrid_rrf and retrieval_mode == "hybrid_rrf") else retrieval_mode
        )

        with self._lock:
            if self.total_vectors == 0 or not self._chunks:
                return clean_q, []

            q_vec = encode_texts_dense([clean_q], dim=self.dim)
            search_k = min(self.total_vectors, max(top_k * 4, 16))
            scores, indices = self.index.search(q_vec, search_k)

            dense_map: dict[int, float] = {}
            for sim, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    dense_map[int(idx)] = float(sim)

            q_tokens = tokenize(clean_q)
            avg_dl = sum(len(c.tokens) for c in self._chunks) / max(len(self._chunks), 1)

            candidate_indices: list[int] = []
            for idx, ch in enumerate(self._chunks):
                if category and ch.category.lower() != category.lower():
                    continue
                if ticker and (ch.ticker or "").upper() != ticker.upper():
                    continue
                candidate_indices.append(idx)

            if not candidate_indices:
                return clean_q, []

            dense_ranked = sorted(
                candidate_indices,
                key=lambda i: dense_map.get(i, -1.0),
                reverse=True,
            )
            lex_scores = {
                i: self._bm25_score(q_tokens, self._chunks[i], avg_dl) for i in candidate_indices
            }
            lex_ranked = sorted(candidate_indices, key=lambda i: lex_scores[i], reverse=True)

            dense_rank_pos = {idx: r + 1 for r, idx in enumerate(dense_ranked)}
            lex_rank_pos = {idx: r + 1 for r, idx in enumerate(lex_ranked)}

            rrf_k = 60.0
            scored_items: list[tuple[int, float]] = []
            for idx in candidate_indices:
                d_rank = dense_rank_pos[idx]
                l_rank = lex_rank_pos[idx]
                if effective_mode == "hybrid_rrf":
                    rrf = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + l_rank))
                    rrf += 0.01 * max(dense_map.get(idx, 0.0), 0.0) + 0.005 * lex_scores[idx]
                elif effective_mode == "lexical_only":
                    rrf = (1.0 / (rrf_k + l_rank)) + 0.01 * lex_scores[idx]
                else:
                    rrf = max(dense_map.get(idx, 0.0), 0.0)
                scored_items.append((idx, rrf))

            scored_items.sort(key=lambda item: item[1], reverse=True)

            results: list[RetrievedChunk] = []
            for idx, rrf_val in scored_items[:top_k]:
                ch = self._chunks[idx]
                results.append(
                    RetrievedChunk(
                        chunk_id=ch.chunk_id,
                        doc_id=ch.doc_id,
                        title=ch.title,
                        category=ch.category,
                        ticker=ch.ticker,
                        text=ch.text,
                        dense_score=round(max(dense_map.get(idx, 0.0), 0.0), 4),
                        lexical_score=round(lex_scores[idx], 4),
                        rrf_score=round(rrf_val, 5),
                    )
                )
            return clean_q, results


DEFAULT_EVAL_BENCHMARK: list[dict[str, object]] = [
    {
        "query": "Institutional VWAP standard deviation bands and RVOL pullback accumulation",
        "expected_doc_ids": ["intraday_microstructure"],
        "expected_chunk_ids": ["intraday_microstructure#c2", "intraday_microstructure#c3"],
    },
    {
        "query": "ATR(14) True Range stop loss distance and floor(MaxRupeeRisk) position sizing",
        "expected_doc_ids": ["intraday_microstructure"],
        "expected_chunk_ids": ["intraday_microstructure#c4", "intraday_microstructure#c5"],
    },
    {
        "query": "Opening Range Breakout ORB RVOL 1.8 threshold and extension from VWAP",
        "expected_doc_ids": ["intraday_microstructure"],
        "expected_chunk_ids": ["intraday_microstructure#c6"],
    },
    {
        "query": "Black-Scholes-Merton d1 d2 call and put pricing equation",
        "expected_doc_ids": ["options_greeks_and_iv"],
        "expected_chunk_ids": ["options_greeks_and_iv#c2"],
    },
    {
        "query": "Gamma convexity near expiry and daily Theta decay for ATM options",
        "expected_doc_ids": ["options_greeks_and_iv"],
        "expected_chunk_ids": ["options_greeks_and_iv#c3", "options_greeks_and_iv#c4"],
    },
    {
        "query": "Implied volatility IV crush ahead of binary events and vertical debit spread Vega",
        "expected_doc_ids": ["options_greeks_and_iv"],
        "expected_chunk_ids": ["options_greeks_and_iv#c5", "options_greeks_and_iv#c6"],
    },
    {
        "query": "Daily drawdown wall -2.0 R and never widen a validated stop-loss guardrail",
        "expected_doc_ids": ["risk_and_execution_governance"],
        "expected_chunk_ids": ["risk_and_execution_governance#c2", "risk_and_execution_governance#c3"],
    },
    {
        "query": "Net-R accounting after brokerage, SEBI turnover charges, stamp duty, and 18% GST",
        "expected_doc_ids": ["risk_and_execution_governance"],
        "expected_chunk_ids": ["risk_and_execution_governance#c4", "risk_and_execution_governance#c5"],
    },
    {
        "query": "OWASP LLM01 prompt injection delimiter sanitization and citation grounding",
        "expected_doc_ids": ["risk_and_execution_governance"],
        "expected_chunk_ids": ["risk_and_execution_governance#c6", "risk_and_execution_governance#c7"],
    },
]


def evaluate_retrieval_benchmark(
    store: HybridFAISSStore,
    top_k: int = 3,
    include_per_query_cases: bool = True,
) -> RAGEvalResponse:
    """Run an Information Retrieval ablation study comparing `hybrid_rrf`,
    `dense_only`, and `lexical_only` on Recall@K, MRR@K, NDCG@K, and latency."""
    modes: tuple[Literal["hybrid_rrf", "dense_only", "lexical_only"], ...] = (
        "hybrid_rrf",
        "dense_only",
        "lexical_only",
    )
    summaries: dict[str, ModeEvalSummary] = {}

    for mode in modes:
        recalls: list[float] = []
        rrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []
        cases: list[EvalCaseResult] = []

        for item in DEFAULT_EVAL_BENCHMARK:
            q_str = str(item["query"])
            expected_chunks = set(str(x) for x in item.get("expected_chunk_ids", []))  # type: ignore[arg-type]
            expected_docs = [str(x) for x in item.get("expected_doc_ids", [])]  # type: ignore[arg-type]

            t0 = time.perf_counter()
            _, results = store.search(
                query=q_str,
                top_k=top_k,
                use_hybrid_rrf=(mode == "hybrid_rrf"),
                retrieval_mode=mode,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            retrieved_chunks = [r.chunk_id for r in results]

            # Section-level chunk hit: 1.0 if any chunk from the target section is in top_k
            recall = 1.0 if any(cid in expected_chunks for cid in retrieved_chunks) else 0.0

            rr = 0.0
            rels: list[float] = []
            for rank_idx, r in enumerate(results, start=1):
                rel = 1.0 if r.chunk_id in expected_chunks else (0.25 if r.doc_id in expected_docs else 0.0)
                rels.append(rel)
                if rel >= 1.0 and rr == 0.0:
                    rr = 1.0 / rank_idx

            dcg = sum(rel / math.log2(idx + 1.0) for idx, rel in enumerate(rels, start=1))
            ideal_rels = sorted(rels + ([1.0] if recall < 1.0 else []), reverse=True)[:top_k]
            idcg = sum(rel / math.log2(idx + 1.0) for idx, rel in enumerate(ideal_rels, start=1))
            ndcg = min(1.0, dcg / idcg) if idcg > 0 else 0.0

            recalls.append(recall)
            rrs.append(rr)
            ndcgs.append(ndcg)

            if include_per_query_cases:
                cases.append(
                    EvalCaseResult(
                        query=q_str,
                        expected_doc_ids=list(expected_chunks),
                        retrieved_doc_ids=retrieved_chunks,
                        recall_at_k=round(recall, 4),
                        reciprocal_rank=round(rr, 4),
                        ndcg_at_k=round(ndcg, 4),
                        latency_ms=round(elapsed_ms, 3),
                    )
                )

        lat_arr = np.array(latencies, dtype=np.float64)
        summaries[mode] = ModeEvalSummary(
            retrieval_mode=mode,
            top_k=top_k,
            num_queries=len(DEFAULT_EVAL_BENCHMARK),
            mean_recall_at_k=round(float(np.mean(recalls)), 4),
            mrr_at_k=round(float(np.mean(rrs)), 4),
            mean_ndcg_at_k=round(float(np.mean(ndcgs)), 4),
            mean_latency_ms=round(float(np.mean(lat_arr)), 3),
            p95_latency_ms=round(float(np.percentile(lat_arr, 95)), 3),
            cases=cases,
        )

    return RAGEvalResponse(
        top_k=top_k,
        corpus_chunks=store.total_vectors,
        modes=summaries,
    )


def build_default_store(corpus_dir: pathlib.Path | None = None) -> HybridFAISSStore:
    """Initialize a HybridFAISSStore preloaded with the built-in markdown corpus."""
    store = HybridFAISSStore()
    if corpus_dir is None:
        corpus_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "corpus"

    if not corpus_dir.exists():
        return store

    docs: list[DocumentInput] = []
    category_map = {
        "intraday_microstructure": "microstructure",
        "options_greeks_and_iv": "derivatives",
        "risk_and_execution_governance": "risk",
    }
    for md_path in sorted(corpus_dir.glob("*.md")):
        stem = md_path.stem
        text = md_path.read_text(encoding="utf-8")
        first_line = text.splitlines()[0].lstrip("# ").strip() if text.splitlines() else stem
        docs.append(
            DocumentInput(
                doc_id=stem,
                title=first_line,
                content=text,
                category=category_map.get(stem, "general"),
            )
        )

    if docs:
        store.ingest_documents(docs)
    return store
