"""FastAPI REST microservice exposing FinPulse AI Agent endpoints:
- GET  /health
- POST /v1/auth/token
- POST /v1/rag/ingest
- POST /v1/rag/query
- POST /v1/quant/indicators
- POST /v1/quant/greeks
- POST /v1/agent/analyze
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status

from finpulse.agent import FinPulseLangGraphAgent
from finpulse.analytics import compute_intraday_indicators, compute_option_greeks
from finpulse.auth import authenticate_client, create_access_token, require_jwt_claims
from finpulse.rag import build_default_store
from finpulse.schemas import (
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    IngestRequest,
    IngestResponse,
    OptionGreeksRequest,
    OptionGreeksResponse,
    QuantIndicatorRequest,
    QuantIndicatorSummary,
    RAGQueryRequest,
    RAGQueryResponse,
    TokenRequest,
    TokenResponse,
)

app = FastAPI(
    title="FinPulse AI Agent API",
    version="0.1.0",
    description=(
        "Enterprise Financial RAG & Quantitative Multi-Agent Service built with "
        "FastAPI, LangChain Core, LangGraph, FAISS, Pandas, NumPy, and OAuth2/JWT."
    ),
)

vector_store = build_default_store()
langgraph_agent = FinPulseLangGraphAgent(store=vector_store)


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "finpulse-ai-agent",
        "version": "0.1.0",
        "indexed_vectors": vector_store.total_vectors,
        "embedding_dim": vector_store.dim,
    }


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
    _claims: dict[str, Any] = Depends(require_jwt_claims),
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
    )
    return RAGQueryResponse(
        sanitized_query=clean_q,
        total_candidates=vector_store.total_vectors,
        results=results,
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


@app.post("/v1/agent/analyze", response_model=AgentAnalyzeResponse)
def run_agent_analysis(
    req: AgentAnalyzeRequest,
    _claims: dict[str, Any] = Depends(require_jwt_claims),
) -> AgentAnalyzeResponse:
    return langgraph_agent.run(req)
