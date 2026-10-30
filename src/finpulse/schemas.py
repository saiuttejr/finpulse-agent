"""Pydantic v2 request/response schemas for FinPulse AI Agent."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


class TokenRequest(BaseModel):
    client_id: str = Field(..., min_length=2, description="OAuth2 client identifier")
    client_secret: str = Field(..., min_length=6, description="OAuth2 client secret")
    scope: str = Field(default="rag:read rag:write quant:execute agent:run")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    scope: str


class DocumentInput(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=10)
    category: str = Field(default="general", max_length=64)
    ticker: str | None = Field(default=None, max_length=32)


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(..., min_length=1)
    chunk_size: int = Field(default=420, ge=120, le=2000)
    chunk_overlap: int = Field(default=60, ge=0, le=400)


class IngestResponse(BaseModel):
    ingested_documents: int
    indexed_chunks: int
    total_vectors_in_index: int


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=20)
    category: str | None = Field(default=None)
    ticker: str | None = Field(default=None)
    use_hybrid_rrf: bool = Field(default=True)
    retrieval_mode: Literal["hybrid_rrf", "dense_only", "lexical_only"] = Field(
        default="hybrid_rrf",
        description="Retrieval scoring strategy for ablation or production search",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    category: str
    ticker: str | None = None
    text: str
    dense_score: float
    lexical_score: float
    rrf_score: float


class RAGQueryResponse(BaseModel):
    sanitized_query: str
    total_candidates: int
    results: list[RetrievedChunk]


class EvalCaseResult(BaseModel):
    query: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    latency_ms: float


class ModeEvalSummary(BaseModel):
    retrieval_mode: Literal["hybrid_rrf", "dense_only", "lexical_only"]
    top_k: int
    num_queries: int
    mean_recall_at_k: float
    mrr_at_k: float
    mean_ndcg_at_k: float
    mean_latency_ms: float
    p95_latency_ms: float
    cases: list[EvalCaseResult]


class RAGEvalRequest(BaseModel):
    top_k: int = Field(default=3, ge=1, le=10)
    include_per_query_cases: bool = Field(default=True)


class RAGEvalResponse(BaseModel):
    top_k: int
    corpus_chunks: int
    modes: dict[str, ModeEvalSummary]


class OHLCVBar(BaseModel):
    timestamp: str
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_high_low(self) -> OHLCVBar:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        return self


class QuantIndicatorRequest(BaseModel):
    symbol: str = Field(default="NIFTY", min_length=1, max_length=32)
    bars: list[OHLCVBar] = Field(..., min_length=5)
    target_atr_multiple: float = Field(default=1.5, gt=0, le=10.0)
    stop_atr_multiple: float = Field(default=1.0, gt=0, le=10.0)


class QuantIndicatorSummary(BaseModel):
    symbol: str
    bar_count: int
    latest_close: float
    vwap: float
    vwap_upper_1sigma: float
    vwap_lower_1sigma: float
    atr_14: float
    rsi_14: float
    ema_9: float
    ema_20: float
    rvol: float
    regime: Literal["bullish_above_vwap", "bearish_below_vwap", "balanced_near_vwap"]
    target_price: float
    stop_price: float
    empirical_touch_probability: float


class OptionGreeksRequest(BaseModel):
    spot: float = Field(..., gt=0, description="Underlying spot price S")
    strike: float = Field(..., gt=0, description="Option strike price K")
    dte_days: float = Field(..., gt=0, le=3650, description="Days to expiration")
    option_type: Literal["CE", "PE"] = Field(default="CE")
    volatility: float = Field(default=0.18, gt=0.01, le=5.0, description="Annualized IV (e.g. 0.18 = 18%)")
    risk_free_rate: float = Field(default=0.068, ge=0.0, le=0.5)
    market_premium: float | None = Field(default=None, gt=0, description="Optional market LTP to solve IV")


class OptionGreeksResponse(BaseModel):
    option_type: Literal["CE", "PE"]
    spot: float
    strike: float
    dte_days: float
    theoretical_price: float
    implied_volatility: float
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_pct: float
    vanna: float = 0.0
    charm_per_day: float = 0.0
    put_call_parity_residual: float = 0.0
    moneyness: Literal["ITM", "ATM", "OTM"]


class PositionSizingRequest(BaseModel):
    account_equity: float = Field(default=1000000.0, gt=0, description="Total trading capital")
    max_risk_pct: float = Field(default=0.01, gt=0, le=0.10, description="Max risk per trade (1.0 R = 1%)")
    entry_price: float = Field(..., gt=0)
    stop_price: float = Field(..., gt=0)
    target_price: float = Field(..., gt=0)
    lot_size: int = Field(default=1, ge=1)
    round_trip_charges: float = Field(default=65.0, ge=0.0, description="Brokerage + statutory fees")
    current_daily_r: float = Field(default=0.0, le=10.0, description="Session realized/unrealized R")
    win_probability: float = Field(default=0.52, gt=0.0, lt=1.0)
    returns_sample: list[float] | None = Field(default=None, description="Optional bar returns for VaR/CVaR")


class PositionSizingResponse(BaseModel):
    account_equity: float
    one_r_capital: float
    stop_distance: float
    recommended_units: int
    recommended_lots: int
    capital_deployed: float
    max_loss_at_stop: float
    gross_reward_to_risk: float
    net_reward_to_risk: float
    half_kelly_fraction: float
    var_95_pct: float
    cvar_95_pct: float
    governance_approved: bool
    rejection_reasons: list[str]


class AgentAnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1500)
    symbol: str = Field(default="NIFTY", max_length=32)
    top_k: int = Field(default=3, ge=1, le=10)
    bars: list[OHLCVBar] | None = None
    option_params: OptionGreeksRequest | None = None
    risk_params: PositionSizingRequest | None = None


class NodeExecutionStep(BaseModel):
    node: str
    status: Literal["ok", "skipped"]
    latency_ms: float
    detail: str


class AgentAnalyzeResponse(BaseModel):
    query: str
    sanitized_query: str
    routed_intent: Literal["quant_metrics", "options_greeks", "rag_knowledge", "hybrid_research"]
    answer: str
    citations: list[RetrievedChunk]
    quant_summary: QuantIndicatorSummary | None = None
    option_greeks: OptionGreeksResponse | None = None
    risk_summary: PositionSizingResponse | None = None
    hallucination_check_passed: bool
    numerical_faithfulness_passed: bool = True
    confidence_score: float
    execution_trace: list[NodeExecutionStep]
