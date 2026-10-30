# FinPulse AI Agent (`finpulse-ai-agent`)

Containerized **Python + FastAPI + LangChain + LangGraph + Hugging Face (`sentence-transformers`) + FAISS** microservice and **Streamlit** evaluation dashboard for financial document **Retrieval-Augmented Generation (RAG)** and **Pandas / NumPy** quantitative market-data & options Greek analysis.

---

## Documentation Index

- **[System Architecture & LangGraph Workflow](docs/ARCHITECTURE.md)** — Conditional `StateGraph` design, Hugging Face `BAAI/bge-small-en-v1.5` + FAISS `IndexFlatIP` + BM25 Reciprocal Rank Fusion pipeline, and OWASP LLM01 / JWT RBAC security model.
- **[REST API Reference](docs/API_REFERENCE.md)** — Request/response JSON schemas, authentication headers, and curl examples for all 11 FastAPI endpoints.
- **[Quantitative & Retrieval Math](docs/QUANT_AND_RAG_MATH.md)** — Exact equations for session VWAP ($\pm 1\sigma$), ATR(14), Wilder RSI(14), Empirical Bayes touch probability, Black-Scholes Greeks, Newton-Raphson IV solver, and BM25 + RRF.

---

## System Flow

```mermaid
flowchart LR
    UI["Client / Streamlit UI"] -->|OAuth2 Scoped JWT| FastAPI["FastAPI + Rate Limiter"]
    FastAPI --> Route["1. classify_and_route\n(OWASP LLM01 Guard)"]
    Route --> Retrieve["2. retrieve_context\n(FAISS Cosine + BM25 RRF)"]
    Retrieve -->|Low Confidence| Rewrite["2b. rewrite_query\n(Corrective RAG)"]
    Rewrite --> Retrieve
    Retrieve -->|Quant / Options Intent| Quant["3. execute_quant_tools\n(Indicators, Greeks, VaR/Risk)"]
    Retrieve -->|Pure Knowledge Intent| Grade["4. synthesize_and_grade\n(Citation & Faithfulness Gate)"]
    Quant --> Grade
```

---

## Architecture & Key Capabilities

1. **Thread-Safe Hybrid Vector + Lexical RAG & IR Evaluation (`src/finpulse/rag.py`):**
   - Recursive markdown section chunking with document/category/ticker provenance metadata and SHA-256 content-hash deduplication.
   - **Hugging Face `SentenceTransformer("BAAI/bge-small-en-v1.5")`** 384-dimensional $L_2$-normalized dense vector embeddings (with thread-safe vector caching and automatic offline/CI fallback) indexed in **FAISS (`faiss.IndexFlatIP`)** for exact cosine similarity search, protected by `threading.RLock`.
   - **Reciprocal Rank Fusion (RRF, $k=60$)** combining dense **FAISS** cosine retrieval with **Okapi BM25** lexical scoring and metadata filtering.
   - **Golden IR Evaluation Benchmark (`/v1/rag/evaluate`):** Automated ablation across `hybrid_rrf`, `dense_only`, and `lexical_only` computing **$\text{Recall}@K$**, **$\text{MRR}@K$**, **$\text{NDCG}@K$**, and **p95 latency**.
   - **OWASP LLM01 Prompt-Injection Defense:** Iterative stripping of `<system>`, `<instructions>`, and override directives prior to embedding and prompt synthesis.
2. **Conditional & Self-Correcting `LangGraph` Agent (`src/finpulse/agent.py`):**
   - Built on `langgraph.graph.StateGraph` with `add_conditional_edges` and `langchain_core.tools.tool`:
     - **Corrective RAG Loop:** Routes low-confidence retrievals through `rewrite_query` for domain synonym/acronym expansion before re-querying FAISS + BM25.
     - **Conditional Tool Bypass:** Pure `rag_knowledge` queries bypass `execute_quant_tools` at the graph-edge level.
     - **Grounded LLM Synthesis + Numerical Faithfulness Verifier:** Supports **Gemini (`GEMINI_API_KEY`)** and **OpenAI (`OPENAI_API_KEY`)** REST synthesis with automatic deterministic fallback and `verify_numerical_faithfulness` cross-checking every synthesized number against tool outputs (`QuantIndicatorSummary`, `OptionGreeksResponse`, `PositionSizingResponse`).
     - **Server-Sent Events (SSE) Streaming (`/v1/agent/stream`):** Streams real-time per-node execution events.
3. **Vectorized `Pandas` & `NumPy` Quant + Pre-Trade Risk Engine (`src/finpulse/analytics.py`):**
   - Intraday session **VWAP** ($\pm 1\sigma$ volume-weighted bands), **ATR(14)**, **Wilder/EWMA RSI(14)**, **EMA(9/20)**, **RVOL**, and Empirical Bayes barrier-crossing **touch-probability** estimator.
   - **Black-Scholes-Merton** option pricing, 1st & 2nd order analytical Greeks (**Delta, Gamma, daily Theta, Vega, Vanna, daily Charm**), **Put-Call Parity residual check**, and a **Newton-Raphson + Bisection Implied Volatility (IV) solver**.
   - **Pre-Trade Risk Governance (`/v1/quant/risk`):** ATR volatility-normalized position sizing, **Gross-R vs. Net-R** after transaction charges, **Half-Kelly fraction**, **95% VaR & CVaR (Expected Shortfall)**, and hard $-2.0R$ daily drawdown wall enforcement.
4. **OAuth2 RBAC Scopes, Rate Limiting & Observability (`src/finpulse/api.py`, `src/finpulse/auth.py`):**
   - HS256 **JWT Bearer** authentication (`/v1/auth/token`, `PyJWT`) with fine-grained OAuth2 scope enforcement (`rag:write`, `agent:run`), constant-time `hmac.compare_digest` verification, and SHA-256 `appIdHash` signature generation.
   - Thread-safe **Token-Bucket Rate Limiter** (`429 Too Many Requests`), `X-Request-ID` & `X-Process-Time-Ms` correlation middleware, and `/metrics` latency percentile telemetry (`p50`/`p95`).

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pytest suite
pytest -v

# 3. Start the FastAPI server
uvicorn finpulse.api:app --reload --app-dir src --port 8000

# 4. Launch the Streamlit evaluation UI
PYTHONPATH=src streamlit run ui.py
```

---

## Project Layout

```text
finpulse-ai-agent/
├── .github/workflows/ci.yml        # GitHub Actions CI running pytest
├── data/corpus/                    # Built-in financial & derivatives markdown corpus
│   ├── intraday_microstructure.md
│   ├── options_greeks_and_iv.md
│   └── risk_and_execution_governance.md
├── docs/                           # Detailed architecture, API, and mathematical docs
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE.md
│   └── QUANT_AND_RAG_MATH.md
├── src/finpulse/
│   ├── __init__.py
│   ├── agent.py                    # LangChain @tool functions + 4-node LangGraph StateGraph
│   ├── analytics.py                # Pandas/NumPy indicators, touch probability & Black-Scholes Greeks/IV
│   ├── api.py                      # FastAPI application & OpenAPI routes
│   ├── auth.py                     # OAuth2 / HS256 JWT token issuance & SHA-256 appIdHash helper
│   ├── rag.py                      # OWASP LLM01 sanitizer, chunker, FAISS IndexFlatIP + BM25 RRF
│   └── schemas.py                  # Pydantic v2 request/response contracts
├── tests/
│   └── test_finpulse.py            # End-to-end pytest suite (RAG, FAISS, Quant, LangGraph, FastAPI, JWT)
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── ui.py                           # Streamlit evaluation dashboard
```
