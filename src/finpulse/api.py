"""FastAPI REST microservice exposing FinPulse AI Agent endpoints:
- GET  /health
- GET  /metrics
- POST /v1/auth/token
- POST /v1/rag/ingest
- POST /v1/rag/query
- POST /v1/rag/evaluate
- POST /v1/quant/indicators
- POST /v1/quant/greeks
- POST /v1/quant/risk
- POST /v1/agent/analyze
- POST /v1/agent/stream
"""
from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
import numpy as np

from finpulse.agent import FinPulseLangGraphAgent
from finpulse.analytics import (
    compute_intraday_indicators,
    compute_option_greeks,
    compute_position_sizing_and_risk,
)
from finpulse.auth import authenticate_client, create_access_token, require_scopes
from finpulse.rag import build_default_store, evaluate_retrieval_benchmark
from finpulse.schemas import (
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    IngestRequest,
    IngestResponse,
    OptionGreeksRequest,
    OptionGreeksResponse,
    PositionSizingRequest,
    PositionSizingResponse,
    QuantIndicatorRequest,
    QuantIndicatorSummary,
    RAGEvalRequest,
    RAGEvalResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    TokenRequest,
    TokenResponse,
)


class TokenBucketRateLimiter:
    """Thread-safe in-memory per-client token-bucket rate limiter."""

    def __init__(self, rate_per_sec: float = 25.0, capacity: float = 50.0) -> None:
        self.rate_per_sec = rate_per_sec
        self.capacity = capacity
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, client_key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last_ts = self._buckets.get(client_key, (self.capacity, now))
            elapsed = max(0.0, now - last_ts)
            tokens = min(self.capacity, tokens + elapsed * self.rate_per_sec)
            if tokens < cost:
                self._buckets[client_key] = (tokens, now)
                return False
            self._buckets[client_key] = (tokens - cost, now)
            return True


class ServiceTelemetry:
    """Thread-safe operational metrics collector for request counts and latency percentiles."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.requests_by_route: dict[str, int] = {}
        self.status_codes: dict[str, int] = {}
        self._latencies_ms: deque[float] = deque(maxlen=max_samples)

    def record(self, route: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self.requests_by_route[route] = self.requests_by_route.get(route, 0) + 1
            code_key = str(status_code)
            self.status_codes[code_key] = self.status_codes.get(code_key, 0) + 1
            self._latencies_ms.append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._latencies_ms)
            total = self.total_requests
            by_route = dict(self.requests_by_route)
            by_status = dict(self.status_codes)

        if samples:
            arr = np.asarray(samples, dtype=np.float64)
            p50 = round(float(np.percentile(arr, 50)), 3)
            p95 = round(float(np.percentile(arr, 95)), 3)
            mean_lat = round(float(np.mean(arr)), 3)
        else:
            p50 = p95 = mean_lat = 0.0

        return {
            "total_requests": total,
            "requests_by_route": by_route,
            "status_codes": by_status,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
        }


app = FastAPI(
    title="FinPulse AI Agent API",
    version="0.2.0",
    description=(
        "Enterprise Financial RAG & Quantitative Multi-Agent Service built with "
        "FastAPI, LangChain Core, LangGraph, FAISS, Pandas, NumPy, and OAuth2/JWT."
    ),
)

vector_store = build_default_store()
langgraph_agent = FinPulseLangGraphAgent(store=vector_store)
rate_limiter = TokenBucketRateLimiter(rate_per_sec=25.0, capacity=50.0)
telemetry = ServiceTelemetry()


@app.middleware("http")
async def observability_and_rate_limit_middleware(request: Request, call_next) -> Response:
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    client_host = request.client.host if request.client else "local"
    if not rate_limiter.allow(client_host):
        telemetry.record(request.url.path, status.HTTP_429_TOO_MANY_REQUESTS, 0.0)
        return Response(
            content='{"detail":"Rate limit exceeded"}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
            headers={"X-Request-ID": req_id, "Retry-After": "1"},
        )

    t0 = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    telemetry.record(request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "finpulse-ai-agent",
        "version": "0.2.0",
        "indexed_vectors": vector_store.total_vectors,
        "embedding_dim": vector_store.dim,
        "embedding_model": vector_store.embedding_backend,
    }


@app.get("/metrics")
def service_metrics() -> dict[str, Any]:
    snap = telemetry.snapshot()
    snap["indexed_vectors"] = vector_store.total_vectors
    snap["embedding_dim"] = vector_store.dim
    snap["embedding_model"] = vector_store.embedding_backend
    return snap


@app.post("/v1/auth/token", response_model=TokenResponse)
def issue_oauth2_token(req: TokenRequest) -> TokenResponse:
    if not authenticate_client(req.client_id, req.client_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client_id or client_secret",
        )
    token = create_access_token(subject=req.client_id, scope=req.scope)
    return TokenResponse(access_token=token, scope=req.scope)


@app.post("/v1/rag/ingest", response_model=IngestResponse)
def ingest_documents(
    req: IngestRequest,
    _claims: dict[str, Any] = Depends(require_scopes("rag:write")),
) -> IngestResponse:
    added = vector_store.ingest_documents(
        req.documents,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )
    return IngestResponse(
        ingested_documents=len(req.documents),
        indexed_chunks=added,
        total_vectors_in_index=vector_store.total_vectors,
    )


@app.post("/v1/rag/query", response_model=RAGQueryResponse)
def query_rag(req: RAGQueryRequest) -> RAGQueryResponse:
    clean_q, results = vector_store.search(
        query=req.query,
        top_k=req.top_k,
        category=req.category,
        ticker=req.ticker,
        use_hybrid_rrf=req.use_hybrid_rrf,
        retrieval_mode=req.retrieval_mode,
    )
    return RAGQueryResponse(
        sanitized_query=clean_q,
        total_candidates=vector_store.total_vectors,
        results=results,
    )


@app.post("/v1/rag/evaluate", response_model=RAGEvalResponse)
def evaluate_rag(req: RAGEvalRequest) -> RAGEvalResponse:
    return evaluate_retrieval_benchmark(
        store=vector_store,
        top_k=req.top_k,
        include_per_query_cases=req.include_per_query_cases,
    )


@app.post("/v1/quant/indicators", response_model=QuantIndicatorSummary)
def calculate_indicators(req: QuantIndicatorRequest) -> QuantIndicatorSummary:
    return compute_intraday_indicators(
        symbol=req.symbol,
        bars=req.bars,
        target_atr_multiple=req.target_atr_multiple,
        stop_atr_multiple=req.stop_atr_multiple,
    )


@app.post("/v1/quant/greeks", response_model=OptionGreeksResponse)
def calculate_option_greeks(req: OptionGreeksRequest) -> OptionGreeksResponse:
    return compute_option_greeks(req)


@app.post("/v1/quant/risk", response_model=PositionSizingResponse)
def calculate_position_risk(req: PositionSizingRequest) -> PositionSizingResponse:
    return compute_position_sizing_and_risk(req)


@app.post("/v1/agent/analyze", response_model=AgentAnalyzeResponse)
def run_agent_analysis(
    req: AgentAnalyzeRequest,
    _claims: dict[str, Any] = Depends(require_scopes("agent:run")),
) -> AgentAnalyzeResponse:
    return langgraph_agent.run(req)


@app.post("/v1/agent/stream")
def stream_agent_analysis(
    req: AgentAnalyzeRequest,
    _claims: dict[str, Any] = Depends(require_scopes("agent:run")),
) -> StreamingResponse:
    return StreamingResponse(
        langgraph_agent.stream_events(req),
        media_type="text/event-stream",
    )
