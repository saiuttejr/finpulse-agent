"""Hybrid RAG engine: OWASP LLM01 prompt-injection sanitizer, recursive markdown
chunking, 384-dim L2-normalized dense embeddings, FAISS IndexFlatIP vector search,
and Reciprocal Rank Fusion (RRF) with lexical BM25-style scoring."""
from __future__ import annotations

import hashlib
import math
import pathlib
import re
from dataclasses import dataclass

import faiss
import numpy as np

from finpulse.schemas import DocumentInput, RetrievedChunk

EMBEDDING_DIM = 384

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


def encode_texts_dense(texts: list[str], dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Encode texts into L2-normalized float32 dense vectors (shape: [N, dim]).
    Uses signed multi-probe feature hashing over unigrams, bigrams, and character
    trigrams with sublinear TF scaling so semantically & lexically related financial
    terms map to high-cosine-similarity vectors deterministically without network I/O."""
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


class HybridFAISSStore:
    """FAISS IndexFlatIP dense vector index paired with BM25 lexical scoring
    and Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: list[_StoredChunk] = []
        self._doc_freqs: dict[str, int] = {}

    @property
    def total_vectors(self) -> int:
        return int(self.index.ntotal)

    def ingest_documents(
        self,
        documents: list[DocumentInput],
        chunk_size: int = 420,
        chunk_overlap: int = 60,
    ) -> int:
        """Chunk documents, embed into L2-normalized vectors, and add to FAISS."""
        raw_chunks: list[dict[str, str | None]] = []
        for doc in documents:
            raw_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

        if not raw_chunks:
            return 0

        texts = [ f"{c['title']} {c['text']}" for c in raw_chunks ]
        vectors = encode_texts_dense(texts, dim=self.dim)
        self.index.add(vectors)

        for c in raw_chunks:
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
                )
            )
        return len(raw_chunks)

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
    ) -> tuple[str, list[RetrievedChunk]]:
        """Execute FAISS dense search + BM25 lexical search fused via Reciprocal Rank Fusion."""
        clean_q = sanitize_prompt_input(query)
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
            if use_hybrid_rrf:
                rrf = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + l_rank))
                # Tie-break with normalized dense + lexical score
                rrf += 0.01 * max(dense_map.get(idx, 0.0), 0.0) + 0.005 * lex_scores[idx]
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
