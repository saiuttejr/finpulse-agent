# Architectural Decision Records (ADRs) & Engineering Tradeoffs

This document records the core architectural decisions, mathematical tradeoffs, and production scaling path for `FinPulse AI Agent` (`v0.2.0`).

---

## ADR-001: Hybrid FAISS `IndexFlatIP` + Okapi BM25 via Reciprocal Rank Fusion ($k=60$)

### ADR-001 Context

Financial research queries combine dense semantic concepts (*"volatility crush ahead of binary catalysts"*) with exact lexical tokens (`NIFTY`, `BANKNIFTY`, `24400 CE`, `RVOL >= 1.8`, `OWASP LLM01`, `-2.0 R`). Pure dense bi-encoders (`BAAI/bge-small-en-v1.5`) compress text into a fixed 384-dimensional vector and can dilute exact strike numbers or rare regulatory/ticker identifiers. Conversely, pure lexical BM25 fails on vocabulary mismatch (e.g., querying *"vol collapse"* when the document says *"IV crush"*).

### ADR-001 Decision

Combine 384-dimensional unit $L_2$-normalized dense embeddings in `faiss.IndexFlatIP(384)` with Okapi BM25 ($k_1 = 1.5, b = 0.75$) using **Reciprocal Rank Fusion (RRF, $k=60$)** in `HybridFAISSStore.search` (`src/finpulse/rag.py`).

### Why RRF Instead of Linear Score Combination

- **Scale invariance:** Cosine similarity is bounded in $[-1, 1]$, whereas Okapi BM25 scores are unbounded in $[0, \infty)$ and vary with query length and inverse document frequency. Linearly blending $\alpha \cdot s_{\text{dense}} + (1 - \alpha) \cdot s_{\text{bm25}}$ requires fragile per-query min-max normalization that breaks on single-candidate or skewed distributions.
- **Non-parametric rank stability:** RRF ($\sum 1 / (60 + r_i)$) depends only on rank ordering, preventing an outlier BM25 spike on a common keyword from drowning out high-cosine semantic matches.
- **Ablation verification (`POST /v1/rag/evaluate`):** On the 9-query Golden IR Benchmark (`DEFAULT_EVAL_BENCHMARK`), `hybrid_rrf` achieves `Recall@3 = 1.0000`, `MRR@3 = 1.0000`, and `NDCG@3 = 0.9938` (vs. `0.9876` for `dense_only`), while maintaining sub-millisecond cached query latency (`~0.61 ms` mean).

---

## ADR-002: Conditional `LangGraph` Topology with Corrective RAG & Tool Bypass

### ADR-002 Context

A naive multi-step agent implemented as a static linear chain (`classify -> retrieve -> execute_tools -> synthesize`) suffers from two flaws:

1. **Unnecessary tool latency on pure knowledge queries:** Queries such as *"Explain the OWASP LLM01 prompt injection delimiter policy"* do not require computing 16-bar Pandas/NumPy indicators or Black-Scholes Greeks.
2. **Silent failure on low-confidence retrieval:** Short or colloquial queries (*"How do fees affect slippage?"*) may yield low initial dense cosine similarity against formal documentation (*"Net-R accounting after round-trip transaction charges"*).

### ADR-002 Decision

Implement a conditional `StateGraph` in `FinPulseLangGraphAgent._build_graph` (`src/finpulse/agent.py`) governed by `_route_after_retrieval`:

- **Corrective RAG Loop (`rewrite_query`):** If the top retrieved chunk's dense cosine score is below `min_dense_confidence` (`0.14` default) and `rewrite_count < 1`, `_route_after_retrieval` routes to `rewrite_query`, which expands financial acronyms/synonyms (`expand_financial_query`) and loops back to `retrieve_context`. Bounding `rewrite_count < 1` guarantees termination (no infinite graph cycles).
- **Graph-Level Tool Bypass:** When `routed_intent == "rag_knowledge"` and no explicit `bars`, `option_params`, or `risk_params` payloads are provided, `_route_after_retrieval` routes directly from `retrieve_context` to `synthesize_and_grade`, bypassing `execute_quant_tools`.

---

## ADR-003: Deterministic `@tool` Execution + Numerical Faithfulness Verification Gate

### ADR-003 Context

Generative LLMs are probabilistic token predictors and frequently hallucinate floating-point arithmetic, cumulative volume-weighted averages, or standard normal CDF ($\Phi(d_1)$) evaluations. In quantitative finance and pre-trade risk governance, a hallucinated stop price or lot size is unacceptable.

### ADR-003 Decision

1. **Strict separation of computation and prose:** All numerical indicators (`compute_intraday_indicators`), option pricing/Greeks (`compute_option_greeks`), and risk/VaR sizing (`compute_position_sizing_and_risk`) are executed exclusively inside deterministic Python `@tool` functions (`src/finpulse/analytics.py`).
2. **Post-synthesis faithfulness gate (`verify_numerical_faithfulness`):** When `_synthesize_and_grade` invokes an external LLM (`GEMINI_API_KEY` or `OPENAI_API_KEY` via `synthesize_with_llm`), the returned text must contain the exact numerical anchors emitted by the tools (`latest_close`, `vwap`, `atr_14`, `theoretical_price`, `delta`, `recommended_units`, `net_reward_to_risk`). If any number is altered or omitted by the LLM, the agent automatically discards the unfaithful LLM text and falls back to `format_deterministic_synthesis`.

---

## ADR-004: Cache-First Offline Hugging Face Loading + Deterministic Hash Fallback

### ADR-004 Context

Standard calls to `SentenceTransformer("BAAI/bge-small-en-v1.5")` issue outbound HTTP requests to `huggingface.co` on initialization. In enterprise networks with strict egress proxies or air-gapped CI runners, outbound model-hub calls can block startup or fail with TLS interception errors.

### ADR-004 Decision

In `_get_hf_embedder` (`src/finpulse/rag.py`):

1. **Local cache first (`local_files_only=True`):** Attempt to load `BAAI/bge-small-en-v1.5` strictly from the local disk cache with zero network I/O.
2. **Opt-in hub download (`FINPULSE_ALLOW_HF_DOWNLOAD=1`):** Only attempt remote download when explicitly enabled (e.g., in GitHub Actions CI).
3. **Thread-safe vector cache (`_VECTOR_CACHE`):** Cache L2-normalized float32 embeddings by text key under `_VECTOR_CACHE_LOCK` so repeated corpus chunks and benchmark queries bypass redundant transformer forward passes.
4. **Deterministic `blake2b` 2-probe hash fallback (`encode_texts_hash`):** If PyTorch/model weights are unavailable or `FINPULSE_EMBEDDING_BACKEND=hash` is set, fall back to signed 2-probe feature hashing over unigrams, bigrams, and character trigrams into $\mathbb{R}^{384}$.

---

## ADR-005: Concurrency, Idempotency & Production Scaling Roadmap

### Current Single-Instance Guarantees

| Subsystem | Concurrency / Integrity Mechanism | Implementation Reference |
| --- | --- | --- |
| **FAISS + BM25 Store** | `threading.RLock` around all `ingest_documents` and `search` operations | `HybridFAISSStore` (`src/finpulse/rag.py`) |
| **Idempotent Ingestion** | SHA-256 content digest (`doc_id::chunk_id::text`) tracked in `_seen_hashes` | `HybridFAISSStore.ingest_documents` (`src/finpulse/rag.py`) |
| **Embedding Cache** | `threading.Lock` protecting `_VECTOR_CACHE` dictionary reads/writes | `encode_texts_dense` (`src/finpulse/rag.py`) |
| **Rate Limiting** | Thread-safe per-client token bucket (`25 req/sec`, `50` burst) | `TokenBucketRateLimiter` (`src/finpulse/api.py`) |
| **Telemetry** | Bounded `collections.deque(maxlen=1000)` protected by `threading.Lock` | `ServiceTelemetry` (`src/finpulse/api.py`) |

### Horizontal Scaling Path ($10^3 \rightarrow 10^7$ Chunks & Multi-Replica Deployment)

When scaling beyond a single container replica or beyond $10^5$ indexed chunks:

1. **ANN Index Upgrade (`IndexFlatIP` $\rightarrow$ `IndexHNSWFlat` / `IndexIVFPQ`):**
   - Exact `IndexFlatIP` is $\mathcal{O}(N \cdot d)$ per query, which is optimal up to $\sim 10^5$ vectors ($< 2\text{ ms}$). Beyond $10^6$ vectors, replace `IndexFlatIP(384)` with `faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)` or a partitioned external vector store (Qdrant / Milvus / pgvector) for sub-linear $\mathcal{O}(\log N)$ search.
2. **Distributed Rate Limiting & Shared State:**
   - Replace in-process `TokenBucketRateLimiter` with a Redis Lua atomic token-bucket script so rate limits apply globally across Gunicorn/Uvicorn worker processes and Kubernetes pods.
3. **Asynchronous Ingestion Pipeline:**
   - Offload large document batch chunking and `SentenceTransformer` GPU/CPU batch encoding from the synchronous `POST /v1/rag/ingest` request thread to a background worker queue (Celery / Dramatiq / Kafka) with durable object-store snapshot persistence (`faiss.write_index`).
