# REST API Reference — FinPulse AI Agent

Base URL (local development): `http://localhost:8000`  
Interactive OpenAPI / Swagger UI: `http://localhost:8000/docs`  
ReDoc Specification: `http://localhost:8000/redoc`

---

## Endpoint Summary

| Method | Path | Auth / Scope Required | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | No | Service liveness, version, active embedding model, and FAISS index size |
| `GET` | `/metrics` | No | Operational telemetry: request counts, status codes, and `p50`/`p95` latency (`ms`) |
| `POST` | `/v1/auth/token` | Client credentials | Issue an `HS256` Bearer JWT access token with OAuth2 scopes |
| `POST` | `/v1/rag/ingest` | `Bearer <JWT>` (`rag:write`) | Chunk, deduplicate (SHA-256), embed, and index documents into FAISS + BM25 |
| `POST` | `/v1/rag/query` | No | Execute hybrid FAISS + BM25 Reciprocal Rank Fusion search |
| `POST` | `/v1/rag/evaluate` | No | Run Golden IR Benchmark (`Recall@K`, `MRR@K`, `NDCG@K`) across 3 retrieval modes |
| `POST` | `/v1/quant/indicators` | No | Compute vectorized Pandas/NumPy intraday indicators & touch probability |
| `POST` | `/v1/quant/greeks` | No | Compute Black-Scholes option price, solved IV, 1st/2nd-order Greeks & parity check |
| `POST` | `/v1/quant/risk` | No | Compute ATR position sizing, Net-R after fees, Half-Kelly, 95% VaR/CVaR & governance |
| `POST` | `/v1/agent/analyze` | `Bearer <JWT>` (`agent:run`) | Run the conditional `LangGraph` agent across RAG + quantitative tools |
| `POST` | `/v1/agent/stream` | `Bearer <JWT>` (`agent:run`) | Stream live `LangGraph` node execution events via Server-Sent Events (`text/event-stream`) |

---

## Middleware Headers, OAuth2 Scopes & Standard Errors

### Correlation & Rate-Limit Headers

Every HTTP request processed by `observability_and_rate_limit_middleware` (`src/finpulse/api.py`) includes:

| Header | Direction | Description |
| --- | --- | --- |
| `X-Request-ID` | Request / Response | Caller-supplied or server-generated UUIDv4 request correlation identifier |
| `X-Process-Time-Ms` | Response | End-to-end server execution time in milliseconds (`float` formatted to 3 decimals) |
| `Retry-After` | Response (`429` only) | Recommended backoff interval in seconds when the per-client token bucket is exhausted |

### OAuth2 Scope-Based RBAC (`require_scopes`)

Tokens issued by `POST /v1/auth/token` carry a space-delimited `scope` claim (`rag:read rag:write quant:execute agent:run` by default). Protected routes verify both signature validity (`HS256`, `iss = "finpulse-ai-agent"`) and required scope membership:

| HTTP Status | Condition | Example Response Payload |
| --- | --- | --- |
| `401 Unauthorized` | Missing, expired, or invalid `Authorization: Bearer <token>` header | `{"detail": "Missing Bearer authentication token"}` |
| `403 Forbidden` | Valid JWT that lacks the route's required scope (e.g., `rag:write` or `agent:run`) | `{"detail": "Insufficient OAuth2 scope; missing: rag:write"}` |
| `422 Unprocessable Entity` | Pydantic v2 schema or domain invariant failure (e.g., `high < low` in `OHLCVBar`) | `{"detail": [{"loc": ["body", "bars", 0], "msg": "Value error, high must be >= low"}]}` |
| `429 Too Many Requests` | Client IP exceeded token-bucket capacity (`rate_per_sec = 25.0`, `capacity = 50.0`) | `{"detail": "Rate limit exceeded"}` |

---

## 1. `GET /health`

Returns liveness metadata, active dense embedding backend (`huggingface:BAAI/bge-small-en-v1.5` or `hash-384`), and current FAISS index size.

### Health Response (`200 OK`)

```json
{
  "status": "ok",
  "service": "finpulse-ai-agent",
  "version": "0.2.0",
  "indexed_vectors": 20,
  "embedding_dim": 384,
  "embedding_model": "huggingface:BAAI/bge-small-en-v1.5"
}
```

---

## 2. `GET /metrics`

Returns thread-safe operational telemetry collected by `ServiceTelemetry` (`src/finpulse/api.py`), including per-route request counts, HTTP status code distribution, and rolling latency percentiles (`mean`, `p50`, `p95` in `ms`).

### Metrics Response (`200 OK`)

```json
{
  "total_requests": 14,
  "requests_by_route": {
    "/health": 2,
    "/v1/auth/token": 2,
    "/v1/rag/query": 4,
    "/v1/quant/risk": 3,
    "/v1/agent/analyze": 3
  },
  "status_codes": {
    "200": 13,
    "403": 1
  },
  "mean_latency_ms": 6.412,
  "p50_latency_ms": 4.185,
  "p95_latency_ms": 18.92,
  "indexed_vectors": 20,
  "embedding_dim": 384,
  "embedding_model": "huggingface:BAAI/bge-small-en-v1.5"
}
```

---

## 3. `POST /v1/auth/token`

Validates `client_id` and `client_secret` in constant time via `hmac.compare_digest` (defaults: `finpulse-client` / `finpulse-secret-2026`, configurable via `FINPULSE_CLIENT_ID` and `FINPULSE_CLIENT_SECRET`) and issues a signed `HS256` JWT.

### Token Request Body

```json
{
  "client_id": "finpulse-client",
  "client_secret": "finpulse-secret-2026",
  "scope": "rag:read rag:write quant:execute agent:run"
}
```

### Token Response (`200 OK`)

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "rag:read rag:write quant:execute agent:run"
}
```

---

## 4. `POST /v1/rag/ingest`

Requires header: `Authorization: Bearer <access_token>` with scope `rag:write`.  
Splits documents along Markdown headers and sliding windows, computes SHA-256 content hashes to skip duplicate chunks idempotently, encodes 384-dim unit $L_2$ vectors, and updates the FAISS `IndexFlatIP` and BM25 tables under `threading.RLock`.

### Ingest Request Body

```json
{
  "chunk_size": 420,
  "chunk_overlap": 60,
  "documents": [
    {
      "doc_id": "banknifty_expiry_note",
      "title": "BANKNIFTY Weekly Expiry Pinning",
      "content": "## Pinning Mechanics\nLarge call and put open interest walls at round strikes compress realized volatility into the final hour.",
      "category": "derivatives",
      "ticker": "BANKNIFTY"
    }
  ]
}
```

### Ingest Response (`200 OK`)

```json
{
  "ingested_documents": 1,
  "indexed_chunks": 1,
  "total_vectors_in_index": 21
}
```

---

## 5. `POST /v1/rag/query`

Runs OWASP LLM01 sanitization followed by FAISS cosine search + Okapi BM25 lexical search fused via Reciprocal Rank Fusion (`retrieval_mode: "hybrid_rrf"`, `"dense_only"`, or `"lexical_only"`).

### RAG Query Request Body

```json
{
  "query": "Implied volatility crush and vertical debit spread Vega",
  "top_k": 2,
  "category": "derivatives",
  "use_hybrid_rrf": true,
  "retrieval_mode": "hybrid_rrf"
}
```

### RAG Query Response (`200 OK`)

```json
{
  "sanitized_query": "Implied volatility crush and vertical debit spread Vega",
  "total_candidates": 20,
  "results": [
    {
      "chunk_id": "options_greeks_and_iv#c5",
      "doc_id": "options_greeks_and_iv",
      "title": "Options Pricing, Black-Scholes Greeks & Volatility Mechanics",
      "category": "derivatives",
      "ticker": null,
      "text": "## Implied Volatility (IV) Crush & Debit Spreads\nAhead of scheduled binary catalysts...",
      "dense_score": 0.7842,
      "lexical_score": 8.9412,
      "rrf_score": 0.08541
    }
  ]
}
```

---

## 6. `POST /v1/rag/evaluate`

Runs the 9-query Golden Information Retrieval Benchmark (`DEFAULT_EVAL_BENCHMARK` in `src/finpulse/rag.py`) across `hybrid_rrf`, `dense_only`, and `lexical_only` modes, computing `Recall@K`, `MRR@K`, graded `NDCG@K`, and `p95` latency (`ms`).

### RAG Evaluation Request Body

```json
{
  "top_k": 3,
  "include_per_query_cases": false
}
```

### RAG Evaluation Response (`200 OK`)

```json
{
  "top_k": 3,
  "corpus_chunks": 20,
  "modes": {
    "hybrid_rrf": {
      "retrieval_mode": "hybrid_rrf",
      "top_k": 3,
      "num_queries": 9,
      "mean_recall_at_k": 1.0,
      "mrr_at_k": 1.0,
      "mean_ndcg_at_k": 0.9938,
      "mean_latency_ms": 0.612,
      "p95_latency_ms": 0.945,
      "cases": []
    },
    "dense_only": {
      "retrieval_mode": "dense_only",
      "top_k": 3,
      "num_queries": 9,
      "mean_recall_at_k": 1.0,
      "mrr_at_k": 1.0,
      "mean_ndcg_at_k": 0.9876,
      "mean_latency_ms": 0.584,
      "p95_latency_ms": 0.892,
      "cases": []
    },
    "lexical_only": {
      "retrieval_mode": "lexical_only",
      "top_k": 3,
      "num_queries": 9,
      "mean_recall_at_k": 1.0,
      "mrr_at_k": 1.0,
      "mean_ndcg_at_k": 0.9938,
      "mean_latency_ms": 0.531,
      "p95_latency_ms": 0.785,
      "cases": []
    }
  }
}
```

---

## 7. `POST /v1/quant/indicators`

Computes session `VWAP` ($\pm 1\sigma$), `ATR(14)`, `RSI(14)`, `EMA(9/20)`, `RVOL`, and Empirical Bayes target-before-stop touch probability over an OHLCV bar series (`min_length = 5`).

### Indicators Request Body

```json
{
  "symbol": "NIFTY",
  "target_atr_multiple": 1.5,
  "stop_atr_multiple": 1.0,
  "bars": [
    {"timestamp": "2026-10-30T09:15:00Z", "open": 24200, "high": 24220, "low": 24190, "close": 24212, "volume": 150000},
    {"timestamp": "2026-10-30T09:20:00Z", "open": 24212, "high": 24235, "low": 24205, "close": 24228, "volume": 165000},
    {"timestamp": "2026-10-30T09:25:00Z", "open": 24228, "high": 24245, "low": 24218, "close": 24240, "volume": 180000},
    {"timestamp": "2026-10-30T09:30:00Z", "open": 24240, "high": 24260, "low": 24232, "close": 24255, "volume": 195000},
    {"timestamp": "2026-10-30T09:35:00Z", "open": 24255, "high": 24275, "low": 24248, "close": 24270, "volume": 240000}
  ]
}
```

### Indicators Response (`200 OK`)

```json
{
  "symbol": "NIFTY",
  "bar_count": 5,
  "latest_close": 24270.0,
  "vwap": 24242.18,
  "vwap_upper_1sigma": 24261.84,
  "vwap_lower_1sigma": 24222.52,
  "atr_14": 28.4,
  "rsi_14": 82.14,
  "ema_9": 24246.85,
  "ema_20": 24235.42,
  "rvol": 1.39,
  "regime": "bullish_above_vwap",
  "target_price": 24312.6,
  "stop_price": 24241.6,
  "empirical_touch_probability": 0.4
}
```

---

## 8. `POST /v1/quant/greeks`

Computes Black-Scholes-Merton theoretical price, 1st-order Greeks (`Delta`, `Gamma`, daily `Theta`, `Vega`), 2nd-order Greeks (`Vanna`, daily `Charm`), Put-Call Parity residual check ($|(C - P) - (S - Ke^{-rT})|$), moneyness (`ITM`/`ATM`/`OTM`), and Newton-Raphson solved `implied_volatility` when `market_premium` is supplied.

### Option Greeks Request Body

```json
{
  "spot": 24350.0,
  "strike": 24400.0,
  "dte_days": 5.0,
  "option_type": "CE",
  "volatility": 0.18,
  "risk_free_rate": 0.068
}
```

### Option Greeks Response (`200 OK`)

```json
{
  "option_type": "CE",
  "spot": 24350.0,
  "strike": 24400.0,
  "dte_days": 5.0,
  "theoretical_price": 191.42,
  "implied_volatility": 0.18,
  "delta": 0.4812,
  "gamma": 0.000772,
  "theta_per_day": -20.8415,
  "vega_per_pct": 11.3724,
  "vanna": 0.001482,
  "charm_per_day": -0.002914,
  "put_call_parity_residual": 0.0,
  "moneyness": "ATM"
}
```

---

## 9. `POST /v1/quant/risk`

Computes ATR/stop-based volatility-normalized lot sizing, `Gross-R` vs. `Net-R` after round-trip transaction fees, `Half-Kelly` optimal capital fraction, `95% VaR` & `CVaR (Expected Shortfall)`, and enforces pre-trade governance rules ($1.0R$ per-trade risk cap, $-2.0R$ daily drawdown wall, and $1.50\times$ minimum Net-R expectancy).

### Position Sizing & Risk Request Body

```json
{
  "account_equity": 1000000.0,
  "max_risk_pct": 0.01,
  "entry_price": 24350.0,
  "stop_price": 24325.0,
  "target_price": 24410.0,
  "lot_size": 25,
  "round_trip_charges": 65.0,
  "current_daily_r": -0.5,
  "win_probability": 0.55
}
```

### Position Sizing & Risk Response (`200 OK`)

```json
{
  "account_equity": 1000000.0,
  "one_r_capital": 10000.0,
  "stop_distance": 25.0,
  "recommended_units": 375,
  "recommended_lots": 15,
  "capital_deployed": 9131250.0,
  "max_loss_at_stop": 9440.0,
  "gross_reward_to_risk": 2.4,
  "net_reward_to_risk": 2.376,
  "half_kelly_fraction": 0.1802,
  "var_95_pct": 0.1689,
  "cvar_95_pct": 0.2118,
  "governance_approved": true,
  "rejection_reasons": []
}
```

---

## 10. `POST /v1/agent/analyze`

Requires header: `Authorization: Bearer <access_token>` with scope `agent:run`.  
Executes the conditional `LangGraph` workflow (`classify_and_route`, `retrieve_context`, optional `rewrite_query`, conditional `execute_quant_tools`, and `synthesize_and_grade`) and returns the grounded synthesis, retrieved citations, quantitative outputs, faithfulness/hallucination verification status, and per-node execution trace.

### Agent Analyze Request Body

```json
{
  "query": "Check NIFTY VWAP, ATR stop sizing, and 24400 CE Theta decay with risk governance",
  "symbol": "NIFTY",
  "top_k": 2
}
```

### Agent Analyze Response (`200 OK`)

```json
{
  "query": "Check NIFTY VWAP, ATR stop sizing, and 24400 CE Theta decay with risk governance",
  "sanitized_query": "Check NIFTY VWAP, ATR stop sizing, and 24400 CE Theta decay with risk governance",
  "routed_intent": "hybrid_research",
  "answer": "[Quant Analytics - NIFTY] Close=24338.5 vs VWAP=24286.41 ... Grounded Corpus Evidence: [options_greeks_and_iv#c3] ...",
  "citations": [
    {
      "chunk_id": "options_greeks_and_iv#c3",
      "doc_id": "options_greeks_and_iv",
      "title": "Options Pricing, Black-Scholes Greeks & Volatility Mechanics",
      "category": "derivatives",
      "ticker": null,
      "text": "## First- and Second-Order Option Greeks...",
      "dense_score": 0.7418,
      "lexical_score": 6.412,
      "rrf_score": 0.06812
    }
  ],
  "quant_summary": {
    "symbol": "NIFTY",
    "bar_count": 16,
    "latest_close": 24338.5,
    "vwap": 24286.41,
    "vwap_upper_1sigma": 24322.9,
    "vwap_lower_1sigma": 24249.91,
    "atr_14": 30.86,
    "rsi_14": 100.0,
    "ema_9": 24304.53,
    "ema_20": 24279.09,
    "rvol": 0.98,
    "regime": "bullish_above_vwap",
    "target_price": 24384.79,
    "stop_price": 24307.64,
    "empirical_touch_probability": 0.7525
  },
  "option_greeks": {
    "option_type": "CE",
    "spot": 24338.5,
    "strike": 24400.0,
    "dte_days": 5.0,
    "theoretical_price": 164.31,
    "implied_volatility": 0.16,
    "delta": 0.4567,
    "gamma": 0.000863,
    "theta_per_day": -18.4912,
    "vega_per_pct": 11.2981,
    "vanna": 0.002214,
    "charm_per_day": -0.003812,
    "put_call_parity_residual": 0.0,
    "moneyness": "ATM"
  },
  "risk_summary": {
    "account_equity": 1000000.0,
    "one_r_capital": 10000.0,
    "stop_distance": 30.86,
    "recommended_units": 321,
    "recommended_lots": 321,
    "capital_deployed": 7812658.5,
    "max_loss_at_stop": 9971.06,
    "gross_reward_to_risk": 1.5,
    "net_reward_to_risk": 1.484,
    "half_kelly_fraction": 0.25,
    "var_95_pct": 0.2086,
    "cvar_95_pct": 0.2615,
    "governance_approved": false,
    "rejection_reasons": [
      "Net reward-to-risk (1.48x) is below the 1.50x minimum expectancy threshold."
    ]
  },
  "hallucination_check_passed": true,
  "numerical_faithfulness_passed": true,
  "confidence_score": 0.81,
  "execution_trace": [
    {"node": "classify_and_route", "status": "ok", "latency_ms": 0.082, "detail": "Sanitized query and routed to intent='hybrid_research'"},
    {"node": "retrieve_context", "status": "ok", "latency_ms": 0.641, "detail": "Retrieved 2 chunks via FAISS + BM25 Reciprocal Rank Fusion"},
    {"node": "execute_quant_tools", "status": "ok", "latency_ms": 2.914, "detail": "Executed LangChain tools: compute_intraday_indicators_tool, compute_option_greeks_tool, compute_position_risk_tool"},
    {"node": "synthesize_and_grade", "status": "ok", "latency_ms": 0.145, "detail": "Provider=deterministic_fallback, verified citations=2, numerical_faithfulness=True, hallucination_check_passed=True"}
  ]
}
```

---

## 11. `POST /v1/agent/stream`

Requires header: `Authorization: Bearer <access_token>` with scope `agent:run`.  
Streams real-time `LangGraph` node execution events using Server-Sent Events (`Content-Type: text/event-stream`). Each completed node emits a `node_completed` JSON frame, followed by a terminal `final_response` frame containing the full `AgentAnalyzeResponse`.

### Stream Request (`curl`)

```bash
curl -N -X POST http://localhost:8000/v1/agent/stream \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "Check NIFTY VWAP and 24400 CE Theta decay", "symbol": "NIFTY", "top_k": 2}'
```

### SSE Stream Wire Format (`200 OK`, `text/event-stream`)

```text
data: {"event": "node_completed", "node": "classify_and_route", "step": {"node": "classify_and_route", "status": "ok", "latency_ms": 0.074, "detail": "Sanitized query and routed to intent='hybrid_research'"}}

data: {"event": "node_completed", "node": "retrieve_context", "step": {"node": "retrieve_context", "status": "ok", "latency_ms": 0.618, "detail": "Retrieved 2 chunks via FAISS + BM25 Reciprocal Rank Fusion"}}

data: {"event": "node_completed", "node": "execute_quant_tools", "step": {"node": "execute_quant_tools", "status": "ok", "latency_ms": 2.411, "detail": "Executed LangChain tools: compute_intraday_indicators_tool, compute_option_greeks_tool"}}

data: {"event": "node_completed", "node": "synthesize_and_grade", "step": {"node": "synthesize_and_grade", "status": "ok", "latency_ms": 0.129, "detail": "Provider=deterministic_fallback, verified citations=2, numerical_faithfulness=True, hallucination_check_passed=True"}}

data: {"event": "final_response", "data": {"query": "Check NIFTY VWAP and 24400 CE Theta decay", "routed_intent": "hybrid_research", "hallucination_check_passed": true, "numerical_faithfulness_passed": true, "confidence_score": 0.81, "...": "..."}}
```
