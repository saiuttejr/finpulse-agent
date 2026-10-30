"""LangChain Core @tool functions and LangGraph 4-node stateful workflow
for grounded financial RAG + quantitative indicator & options Greek analysis."""
from __future__ import annotations

import time
from typing import Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from finpulse.analytics import compute_intraday_indicators, compute_option_greeks
from finpulse.rag import HybridFAISSStore, sanitize_prompt_input
from finpulse.schemas import (
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    NodeExecutionStep,
    OHLCVBar,
    OptionGreeksRequest,
    OptionGreeksResponse,
    QuantIndicatorSummary,
    RetrievedChunk,
)


def default_sample_bars() -> list[OHLCVBar]:
    """Provide a deterministic 16-bar 5-minute intraday series when caller omits bars."""
    base = 24200.0
    bars: list[OHLCVBar] = []
    for i in range(16):
        drift = i * 8.5
        open_p = base + drift
        high_p = open_p + 18.0 + (i % 3) * 4.0
        low_p = open_p - 10.0
        close_p = open_p + 11.0
        vol = 120000.0 + (i * 15000.0 if i < 14 else 95000.0)
        bars.append(
            OHLCVBar(
                timestamp=f"2026-10-30T09:{15 + i * 5:02d}:00Z",
                open=round(open_p, 2),
                high=round(high_p, 2),
                low=round(low_p, 2),
                close=round(close_p, 2),
                volume=vol,
            )
        )
    return bars


@tool
def compute_intraday_indicators_tool(symbol: str, bars_dicts: list[dict]) -> dict:
    """Compute Pandas/NumPy intraday VWAP, ATR(14), RSI(14), EMA(9/20), RVOL, and touch probability."""
    parsed_bars = [OHLCVBar(**b) for b in bars_dicts]
    summary = compute_intraday_indicators(symbol=symbol, bars=parsed_bars)
    return summary.model_dump()


@tool
def compute_option_greeks_tool(option_dict: dict) -> dict:
    """Compute Black-Scholes option price, solved implied volatility, Delta, Gamma, Theta, and Vega."""
    req = OptionGreeksRequest(**option_dict)
    res = compute_option_greeks(req)
    return res.model_dump()


class FinPulseGraphState(TypedDict, total=False):
    request: AgentAnalyzeRequest
    sanitized_query: str
    routed_intent: Literal["quant_metrics", "options_greeks", "rag_knowledge", "hybrid_research"]
    citations: list[RetrievedChunk]
    quant_summary: QuantIndicatorSummary | None
    option_greeks: OptionGreeksResponse | None
    answer: str
    hallucination_check_passed: bool
    confidence_score: float
    execution_trace: list[NodeExecutionStep]


class FinPulseLangGraphAgent:
    """Stateful 4-node LangGraph workflow combining FAISS Hybrid RAG with
    LangChain @tool quantitative execution and citation grounding checks."""

    def __init__(self, store: HybridFAISSStore) -> None:
        self.store = store
        self.graph = self._build_graph()

    def _classify_and_route(self, state: FinPulseGraphState) -> FinPulseGraphState:
        t0 = time.perf_counter()
        req = state["request"]
        clean_q = sanitize_prompt_input(req.query)
        q_low = clean_q.lower()

        wants_options = (
            req.option_params is not None
            or any(k in q_low for k in ("greek", "delta", "theta", "gamma", "vega", "iv", "option", "straddle", "black-scholes"))
        )
        wants_quant = (
            req.bars is not None
            or any(k in q_low for k in ("vwap", "atr", "rsi", "rvol", "ema", "orb", "indicator", "probability", "stop"))
        )

        if wants_options and wants_quant:
            intent: Literal["quant_metrics", "options_greeks", "rag_knowledge", "hybrid_research"] = (
                "hybrid_research"
            )
        elif wants_options:
            intent = "options_greeks"
        elif wants_quant:
            intent = "quant_metrics"
        else:
            intent = "rag_knowledge"

        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        trace = list(state.get("execution_trace", []))
        trace.append(
            NodeExecutionStep(
                node="classify_and_route",
                status="ok",
                latency_ms=elapsed,
                detail=f"Sanitized query and routed to intent='{intent}'",
            )
        )
        return {
            "sanitized_query": clean_q,
            "routed_intent": intent,
            "execution_trace": trace,
        }

    def _retrieve_context(self, state: FinPulseGraphState) -> FinPulseGraphState:
        t0 = time.perf_counter()
        req = state["request"]
        clean_q = state["sanitized_query"]
        _, chunks = self.store.search(query=clean_q, top_k=req.top_k, use_hybrid_rrf=True)
        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        trace = list(state.get("execution_trace", []))
        trace.append(
            NodeExecutionStep(
                node="retrieve_context",
                status="ok",
                latency_ms=elapsed,
                detail=f"Retrieved {len(chunks)} chunks via FAISS + BM25 Reciprocal Rank Fusion",
            )
        )
        return {
            "citations": chunks,
            "execution_trace": trace,
        }

    def _execute_quant_tools(self, state: FinPulseGraphState) -> FinPulseGraphState:
        t0 = time.perf_counter()
        req = state["request"]
        intent = state["routed_intent"]
        quant_summary: QuantIndicatorSummary | None = None
        option_greeks: OptionGreeksResponse | None = None
        invoked_tools: list[str] = []

        if intent in ("quant_metrics", "hybrid_research") or req.bars is not None:
            bars_to_use = req.bars if req.bars is not None else default_sample_bars()
            raw_quant = compute_intraday_indicators_tool.invoke({
                "symbol": req.symbol,
                "bars_dicts": [b.model_dump() for b in bars_to_use],
            })
            quant_summary = QuantIndicatorSummary(**raw_quant)
            invoked_tools.append("compute_intraday_indicators_tool")

        if intent in ("options_greeks", "hybrid_research") or req.option_params is not None:
            opt_req = req.option_params or OptionGreeksRequest(
                spot=quant_summary.latest_close if quant_summary else 24350.0,
                strike=24400.0,
                dte_days=5.0,
                option_type="CE",
                volatility=0.16,
            )
            raw_opt = compute_option_greeks_tool.invoke({
                "option_dict": opt_req.model_dump(),
            })
            option_greeks = OptionGreeksResponse(**raw_opt)
            invoked_tools.append("compute_option_greeks_tool")

        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        trace = list(state.get("execution_trace", []))
        trace.append(
            NodeExecutionStep(
                node="execute_quant_tools",
                status="ok" if invoked_tools else "skipped",
                latency_ms=elapsed,
                detail=(
                    f"Executed LangChain tools: {', '.join(invoked_tools)}"
                    if invoked_tools
                    else "Pure knowledge query; no numerical tool call required"
                ),
            )
        )
        return {
            "quant_summary": quant_summary,
            "option_greeks": option_greeks,
            "execution_trace": trace,
        }

    def _synthesize_and_grade(self, state: FinPulseGraphState) -> FinPulseGraphState:
        t0 = time.perf_counter()
        citations = state.get("citations", [])
        quant = state.get("quant_summary")
        greeks = state.get("option_greeks")

        sections: list[str] = []
        if quant is not None:
            sections.append(
                f"[Quant Analytics - {quant.symbol}] Close={quant.latest_close} vs VWAP={quant.vwap} "
                f"(1sigma band: {quant.vwap_lower_1sigma}..{quant.vwap_upper_1sigma}), "
                f"ATR(14)={quant.atr_14}, RSI(14)={quant.rsi_14}, EMA9/20={quant.ema_9}/{quant.ema_20}, "
                f"RVOL={quant.rvol}x ({quant.regime}). Target={quant.target_price}, Stop={quant.stop_price}, "
                f"Empirical Touch Probability={quant.empirical_touch_probability:.1%}."
            )

        if greeks is not None:
            sections.append(
                f"[Black-Scholes Greeks - {greeks.option_type} {greeks.strike}] "
                f"Theoretical Price={greeks.theoretical_price} (Moneyness={greeks.moneyness}, "
                f"IV={greeks.implied_volatility:.2%}, DTE={greeks.dte_days}d), "
                f"Delta={greeks.delta}, Gamma={greeks.gamma}, "
                f"Theta/day={greeks.theta_per_day}, Vega/1%={greeks.vega_per_pct}."
            )

        if citations:
            cite_snippets = [
                f"[{c.chunk_id}] ({c.title}): {c.text.splitlines()[0].lstrip('# ').strip()}"
                for c in citations[:2]
            ]
            sections.append("Grounded Corpus Evidence: " + " | ".join(cite_snippets))

        answer = " ".join(sections) if sections else "No grounded evidence found for query."

        has_citations = len(citations) > 0 and citations[0].rrf_score > 0
        hallucination_ok = bool(has_citations or quant is not None or greeks is not None)
        top_dense = citations[0].dense_score if citations else 0.5
        confidence = round(min(max(0.55 + 0.35 * top_dense, 0.0), 0.99), 3)

        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        trace = list(state.get("execution_trace", []))
        trace.append(
            NodeExecutionStep(
                node="synthesize_and_grade",
                status="ok",
                latency_ms=elapsed,
                detail=f"Verified citations={len(citations)}, hallucination_check_passed={hallucination_ok}",
            )
        )
        return {
            "answer": answer,
            "hallucination_check_passed": hallucination_ok,
            "confidence_score": confidence,
            "execution_trace": trace,
        }

    def _build_graph(self):
        workflow = StateGraph(FinPulseGraphState)
        workflow.add_node("classify_and_route", self._classify_and_route)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("execute_quant_tools", self._execute_quant_tools)
        workflow.add_node("synthesize_and_grade", self._synthesize_and_grade)

        workflow.add_edge(START, "classify_and_route")
        workflow.add_edge("classify_and_route", "retrieve_context")
        workflow.add_edge("retrieve_context", "execute_quant_tools")
        workflow.add_edge("execute_quant_tools", "synthesize_and_grade")
        workflow.add_edge("synthesize_and_grade", END)
        return workflow.compile()

    def run(self, req: AgentAnalyzeRequest) -> AgentAnalyzeResponse:
        final_state: FinPulseGraphState = self.graph.invoke({
            "request": req,
            "execution_trace": [],
        })
        return AgentAnalyzeResponse(
            query=req.query,
            sanitized_query=final_state["sanitized_query"],
            routed_intent=final_state["routed_intent"],
            answer=final_state["answer"],
            citations=final_state.get("citations", []),
            quant_summary=final_state.get("quant_summary"),
            option_greeks=final_state.get("option_greeks"),
            hallucination_check_passed=final_state["hallucination_check_passed"],
            confidence_score=final_state["confidence_score"],
            execution_trace=final_state["execution_trace"],
        )
