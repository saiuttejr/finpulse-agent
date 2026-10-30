# REST API Reference — FinPulse AI Agent

Base URL (local development): `http://localhost:8000`
Interactive OpenAPI / Swagger UI: `http://localhost:8000/docs`

---

## Endpoint Summary

| Method | Path | Auth Required | Description |
|---|---|---|---|
| `GET` | `/health` | No | Service health, version, indexed FAISS vector count, and embedding dimension |
| `POST` | `/v1/auth/token` | Client credentials | Issue an `HS256` Bearer JWT access token |
| `POST` | `/v1/rag/ingest` | `Bearer <JWT>` | Chunk, embed, and index new documents into the FAISS + BM25 store |
| `POST` | `/v1/rag/query` | No | Execute hybrid FAISS + BM25 Reciprocal Rank Fusion search |
| `POST` | `/v1/quant/indicators` | No | Compute vectorized Pandas/NumPy intraday indicators & touch probability |
| `POST` | `/v1/quant/greeks` | No | Compute Black-Scholes option price, solved IV, and analytical Greeks |
| `POST` | `/v1/agent/analyze` | `Bearer <JWT>` | Run the 4-node `LangGraph` agent across RAG + quantitative tools |

---

## 1. `GET /health`

Returns liveness metadata and current FAISS index size.

### Response (`200 OK`)
```json
{
  "status": "ok",
  "service": "finpulse-ai-agent",
  "version": "0.1.0",
  "indexed_vectors": 10,
  "embedding_dim": 384
}
```

---

## 2. `POST /v1/auth/token`

Validates `client_id` and `client_secret` (defaults: `finpulse-client` / `finpulse-secret-2026`, configurable via `FINPULSE_CLIENT_ID` and `FINPULSE_CLIENT_SECRET`) and issues a signed `HS256` JWT.

### Request Body
```json
{
  "client_id": "finpulse-client",
  "client_secret": "finpulse-secret-2026",
  "scope": "rag:read rag:write quant:execute agent:run"
}
```

### Response (`200 OK`)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "rag:read rag:write quant:execute agent:run"
}
```

---

## 3. `POST /v1/rag/ingest`

Requires header: `Authorization: Bearer <access_token>`

### Request Body
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

### Response (`200 OK`)
```json
{
  "ingested_documents": 1,
  "indexed_chunks": 1,
  "total_vectors_in_index": 11
}
```

---

## 4. `POST /v1/rag/query`

Runs OWASP LLM01 sanitization followed by FAISS cosine search + BM25 lexical search fused via Reciprocal Rank Fusion (`use_hybrid_rrf: true`).

### Request Body
```json
{
  "query": "Implied volatility crush and vertical debit spread Vega",
  "top_k": 2,
  "category": "derivatives",
  "use_hybrid_rrf": true
}
```

### Response (`200 OK`)
```json
{
  "sanitized_query": "Implied volatility crush and vertical debit spread Vega",
  "total_candidates": 11,
  "results": [
    {
      "chunk_id": "options_greeks_and_iv#c3",
      "doc_id": "options_greeks_and_iv",
      "title": "Options Pricing, Black-Scholes Greeks & Volatility Mechanics",
      "category": "derivatives",
      "ticker": null,
      "text": "## Implied Volatility (IV) Crush & Debit Spreads\nAhead of scheduled binary catalysts...",
      "dense_score": 0.6142,
      "lexical_score": 8.9412,
      "rrf_score": 0.0837
    }
  ]
}
```

---

## 5. `POST /v1/quant/indicators`

Computes session `VWAP` ($\pm 1\sigma$), `ATR(14)`, `RSI(14)`, `EMA(9/20)`, `RVOL`, and empirical target-before-stop touch probability over an OHLCV bar series.

### Request Body
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

---

## 6. `POST /v1/quant/greeks`

Computes Black-Scholes-Merton theoretical price, analytical Greeks (`Delta`, `Gamma`, daily `Theta`, `Vega`), moneyness (`ITM`/`ATM`/`OTM`), and Newton-Raphson solved `implied_volatility` when `market_premium` is supplied.

### Request Body
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

---

## 7. `POST /v1/agent/analyze`

Requires header: `Authorization: Bearer <access_token>`

Executes the full 4-node `LangGraph` workflow and returns the grounded synthesis, retrieved citations, quantitative outputs, hallucination check status, and per-node execution trace.

### Request Body
```json
{
  "query": "Check NIFTY VWAP, ATR stop sizing, and 24400 CE Theta decay",
  "symbol": "NIFTY",
  "top_k": 2
}
```
