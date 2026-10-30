"""LangChain Core @tool functions and conditional LangGraph stateful workflow
with Corrective RAG query rewriting, quantitative tool execution,
Gemini / OpenAI LLM synthesis, and numerical faithfulness + citation verification."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any, Literal, TypedDict

import httpx
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from finpulse.analytics import (
    compute_intraday_indicators,
    compute_option_greeks,
    compute_position_sizing_and_risk,
)
from finpulse.rag import HybridFAISSStore, expand_financial_query, sanitize_prompt_input
from finpulse.schemas import (
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    NodeExecutionStep,
    OHLCVBar,
    OptionGreeksRequest,
    OptionGreeksResponse,
    PositionSizingRequest,
    PositionSizingResponse,
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
    """Compute Black-Scholes option price, solved implied volatility, Delta, Gamma, Theta, Vega, Vanna, Charm, and parity residual."""
    req = OptionGreeksRequest(**option_dict)
    res = compute_option_greeks(req)
    return res.model_dump()


@tool
def compute_position_risk_tool(risk_dict: dict) -> dict:
    """Compute ATR position sizing, Net-R after transaction costs, Half-Kelly, 95% VaR/CVaR, and pre-trade governance checks."""
    req = PositionSizingRequest(**risk_dict)
    res = compute_position_sizing_and_risk(req)
    return res.model_dump()


def verify_numerical_faithfulness(
    answer: str,
    quant: QuantIndicatorSummary | None,
    greeks: OptionGreeksResponse | None,
    risk: PositionSizingResponse | None = None,
) -> bool:
    """Verify that deterministic numerical anchors produced by tools appear faithfully in the synthesized report."""
    expected_tokens: list[str] = []
    if quant is not None:
        expected_tokens.extend([str(quant.latest_close), str(quant.vwap), str(quant.atr_14)])
    if greeks is not None:
        expected_tokens.extend([str(greeks.theoretical_price), str(greeks.delta)])
    if risk is not None:
        expected_tokens.extend([str(risk.recommended_units), str(risk.net_reward_to_risk)])
    return all(tok in answer for tok in expected_tokens)


def format_deterministic_synthesis(
    citations: list[RetrievedChunk],
    quant: QuantIndicatorSummary | None,
    greeks: OptionGreeksResponse | None,
    risk: PositionSizingResponse | None = None,
) -> str:
    """Build a deterministic, citation-grounded synthesis from tool outputs and retrieved chunks."""
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
            f"Theta/day={greeks.theta_per_day}, Vega/1%={greeks.vega_per_pct}, "
            f"Vanna={greeks.vanna}, Charm/day={greeks.charm_per_day}."
        )

    if risk is not None:
        gate_lbl = "APPROVED" if risk.governance_approved else "REJECTED"
        sections.append(
            f"[Pre-Trade Risk Governance - {gate_lbl}] Units={risk.recommended_units} "
            f"(1R={risk.one_r_capital}, MaxLoss={risk.max_loss_at_stop}), "
            f"Net-R={risk.net_reward_to_risk}x, Half-Kelly={risk.half_kelly_fraction:.2%}, "
            f"VaR95={risk.var_95_pct}%, CVaR95={risk.cvar_95_pct}%."
        )

    if citations:
        cite_snippets = [
            f"[{c.chunk_id}] ({c.title}): {c.text.splitlines()[0].lstrip('# ').strip()}"
            for c in citations[:2]
        ]
        sections.append("Grounded Corpus Evidence: " + " | ".join(cite_snippets))

    return " ".join(sections) if sections else "No grounded evidence found for query."


def synthesize_with_llm(
    query: str,
    citations: list[RetrievedChunk],
    quant: QuantIndicatorSummary | None,
    greeks: OptionGreeksResponse | None,
    risk: PositionSizingResponse | None = None,
    http_client: httpx.Client | None = None,
) -> tuple[str | None, str]:
    """Invoke Gemini (`GEMINI_API_KEY`) or OpenAI-compatible (`OPENAI_API_KEY`) REST API
    to synthesize a grounded financial analysis report. Returns `(text, provider_name)`
    or `(None, 'deterministic_fallback')` when no API key is configured or on network error."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not gemini_key and not openai_key:
        return None, "deterministic_fallback"

    context_blocks = [f"[{c.chunk_id}] ({c.title}): {c.text}" for c in citations[:3]]
    tool_facts = {
        "quant_summary": quant.model_dump() if quant else None,
        "option_greeks": greeks.model_dump() if greeks else None,
        "risk_summary": risk.model_dump() if risk else None,
    }
    system_prompt = (
        "You are FinPulse AI, a quantitative financial research assistant. "
        "Synthesize a concise analysis grounded strictly in the provided corpus chunks "
        "and deterministic tool JSON. Preserve exact numerical figures (Close, VWAP, ATR, "
        "Theoretical Price, Delta, Recommended Units, Net-R) and cite [chunk_id] brackets."
    )
    user_prompt = (
        f"Query: {query}\n\n"
        f"Deterministic Tool Outputs (JSON):\n{json.dumps(tool_facts)}\n\n"
        f"Retrieved Corpus Chunks:\n" + "\n".join(context_blocks)
    )

    close_client = False
    client = http_client
    if client is None:
        client = httpx.Client(timeout=8.0)
        close_client = True

    try:
        if gemini_key:
            model_name = os.getenv("FINPULSE_GEMINI_MODEL", "gemini-2.0-flash")
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={gemini_key}"
            )
            resp = client.post(
                url,
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 450},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = (
                data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if data.get("candidates")
                else ""
            )
            if text:
                return text, f"gemini:{model_name}"

        if openai_key:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            model_name = os.getenv("FINPULSE_OPENAI_MODEL", "gpt-4o-mini")
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": model_name,
                    "temperature": 0.0,
                    "max_tokens": 450,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            if text:
                return text, f"openai:{model_name}"
    except Exception:
        return None, "deterministic_fallback"
    finally:
        if close_client:
            client.close()

    return None, "deterministic_fallback"


class FinPulseGraphState(TypedDict, total=False):
    request: AgentAnalyzeRequest
    sanitized_query: str
    active_search_query: str
    rewrite_count: int
    routed_intent: Literal["quant_metrics", "options_greeks", "rag_knowledge", "hybrid_research"]
    citations: list[RetrievedChunk]
    quant_summary: QuantIndicatorSummary | None
    option_greeks: OptionGreeksResponse | None
    risk_summary: PositionSizingResponse | None
    answer: str
    hallucination_check_passed: bool
    numerical_faithfulness_passed: bool
    confidence_score: float
    execution_trace: list[NodeExecutionStep]


class FinPulseLangGraphAgent:
    """Conditional & self-correcting LangGraph workflow combining FAISS Hybrid RAG,
    Corrective RAG query rewriting, LangChain @tool quantitative execution,
    and numerical faithfulness + citation grounding checks."""

    def __init__(self, store: HybridFAISSStore, min_dense_confidence: float = 0.14) -> None:
        self.store = store
        self.min_dense_confidence = min_dense_confidence
        self.graph = self._build_graph()

    def _classify_and_route(self, state: FinPulseGraphState) -> FinPulseGraphState:
        t0 = time.perf_counter()
        req = state["request"]
        clean_q = sanitize_prompt_input(req.query)
        q_low = clean_q.lower()

        wants_options = (
            req.option_params is not None
            or any(
                k in q_low
                for k in (
                    "greek", "delta", "theta", "gamma", "vega", "vanna", "charm",
                    "iv", "option", "straddle", "black-scholes", "parity",
                )
            )
        )
        wants_quant = (
            req.bars is not None
            or req.risk_params is not None
            or any(
                k in q_low
                for k in (
                    "vwap", "atr", "rsi", "rvol", "ema", "orb", "indicator",
                    "probability", "stop", "var", "cvar", "kelly", "position size",
                )
            )
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
            "active_search_query": clean_q,
            "rewrite_count": 0,
            "routed_intent": intent,
            "execution_trace": trace,
        }

    def _retrieve_context(self, state: FinPulseGraphState) -> FinPulseGraphState:
        t0 = time.perf_counter()
        req = state["request"]
        search_q = state.get("active_search_query") or state["sanitized_query"]
        _, chunks = self.store.search(query=search_q, top_k=req.top_k, use_hybrid_rrf=True)
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

    def _rewrite_query(self, state: FinPulseGraphState) -> FinPulseGraphState:
        """Corrective RAG node: expands low-confidence queries with financial domain terms."""
        t0 = time.perf_counter()
        base_q = state["sanitized_query"]
        expanded_q = expand_financial_query(base_q)
        rewrite_count = state.get("rewrite_count", 0) + 1
        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        trace = list(state.get("execution_trace", []))
        trace.append(
            NodeExecutionStep(
                node="rewrite_query",
                status="ok",
                latency_ms=elapsed,
                detail=f"Low initial retrieval confidence; expanded query (attempt {rewrite_count})",
            )
        )
        return {
            "active_search_query": expanded_q,
            "rewrite_count": rewrite_count,
            "execution_trace": trace,
        }

    def _route_after_retrieval(
        self, state: FinPulseGraphState
    ) -> Literal["rewrite_query", "execute_quant_tools", "synthesize_and_grade"]:
        """Conditional edge router after retrieval:
        1. Route to `rewrite_query` if initial retrieval confidence is below threshold.
        2. Route directly to `synthesize_and_grade` for pure `rag_knowledge` queries with no tool inputs.
        3. Otherwise route to `execute_quant_tools`."""
        citations = state.get("citations", [])
        top_dense = citations[0].dense_score if citations else 0.0
        if top_dense < self.min_dense_confidence and state.get("rewrite_count", 0) < 1:
            return "rewrite_query"

        req = state["request"]
        if (
            state["routed_intent"] == "rag_knowledge"
            and req.bars is None
            and req.option_params is None
            and req.risk_params is None
        ):
            return "synthesize_and_grade"
        return "execute_quant_tools"

    def _execute_quant_tools(self, state: FinPulseGraphState) -> FinPulseGraphState:
        t0 = time.perf_counter()
        req = state["request"]
        intent = state["routed_intent"]
        quant_summary: QuantIndicatorSummary | None = None
        option_greeks: OptionGreeksResponse | None = None
        risk_summary: PositionSizingResponse | None = None
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

        q_low = state["sanitized_query"].lower()
        wants_risk = req.risk_params is not None or any(
            k in q_low for k in ("risk", "kelly", "var", "cvar", "position size", "drawdown")
        )
        if wants_risk:
            risk_req = req.risk_params or PositionSizingRequest(
                account_equity=1000000.0,
                max_risk_pct=0.01,
                entry_price=quant_summary.latest_close if quant_summary else 24338.5,
                stop_price=quant_summary.stop_price if quant_summary else 24312.5,
                target_price=quant_summary.target_price if quant_summary else 24395.0,
                win_probability=(
                    quant_summary.empirical_touch_probability if quant_summary else 0.55
                ),
            )
            raw_risk = compute_position_risk_tool.invoke({
                "risk_dict": risk_req.model_dump(),
            })
            risk_summary = PositionSizingResponse(**raw_risk)
            invoked_tools.append("compute_position_risk_tool")

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
            "risk_summary": risk_summary,
            "execution_trace": trace,
        }

    def _synthesize_and_grade(self, state: FinPulseGraphState) -> FinPulseGraphState:
        t0 = time.perf_counter()
        citations = state.get("citations", [])
        quant = state.get("quant_summary")
        greeks = state.get("option_greeks")
        risk = state.get("risk_summary")
        clean_q = state.get("sanitized_query", "")

        llm_answer, provider_used = synthesize_with_llm(
            query=clean_q,
            citations=citations,
            quant=quant,
            greeks=greeks,
            risk=risk,
        )
        if llm_answer and verify_numerical_faithfulness(llm_answer, quant, greeks, risk):
            answer = llm_answer
        else:
            answer = format_deterministic_synthesis(citations, quant, greeks, risk)
            provider_used = "deterministic_fallback"

        has_citations = len(citations) > 0 and citations[0].rrf_score > 0
        faithfulness_ok = verify_numerical_faithfulness(answer, quant, greeks, risk)
        hallucination_ok = bool(
            (has_citations or quant is not None or greeks is not None) and faithfulness_ok
        )
        top_dense = citations[0].dense_score if citations else 0.5
        confidence = round(min(max(0.55 + 0.35 * top_dense, 0.0), 0.99), 3)

        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        trace = list(state.get("execution_trace", []))
        trace.append(
            NodeExecutionStep(
                node="synthesize_and_grade",
                status="ok",
                latency_ms=elapsed,
                detail=(
                    f"Provider={provider_used}, verified citations={len(citations)}, "
                    f"numerical_faithfulness={faithfulness_ok}, "
                    f"hallucination_check_passed={hallucination_ok}"
                ),
            )
        )
        return {
            "answer": answer,
            "hallucination_check_passed": hallucination_ok,
            "numerical_faithfulness_passed": faithfulness_ok,
            "confidence_score": confidence,
            "execution_trace": trace,
        }

    def _build_graph(self):
        workflow = StateGraph(FinPulseGraphState)
        workflow.add_node("classify_and_route", self._classify_and_route)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("execute_quant_tools", self._execute_quant_tools)
        workflow.add_node("synthesize_and_grade", self._synthesize_and_grade)

        workflow.add_edge(START, "classify_and_route")
        workflow.add_edge("classify_and_route", "retrieve_context")
        workflow.add_conditional_edges(
            "retrieve_context",
            self._route_after_retrieval,
            {
                "rewrite_query": "rewrite_query",
                "execute_quant_tools": "execute_quant_tools",
                "synthesize_and_grade": "synthesize_and_grade",
            },
        )
        workflow.add_edge("rewrite_query", "retrieve_context")
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
            risk_summary=final_state.get("risk_summary"),
            hallucination_check_passed=final_state["hallucination_check_passed"],
            numerical_faithfulness_passed=final_state.get("numerical_faithfulness_passed", True),
            confidence_score=final_state["confidence_score"],
            execution_trace=final_state["execution_trace"],
        )

    def stream_events(self, req: AgentAnalyzeRequest) -> Iterator[str]:
        """Yield Server-Sent Events (SSE) JSON payloads for each LangGraph node execution
        followed by the final synthesized `AgentAnalyzeResponse`."""
        acc_state: dict[str, Any] = {"request": req, "execution_trace": []}
        seen_steps = 0

        for update in self.graph.stream({"request": req, "execution_trace": []}):
            for node_name, node_state in update.items():
                acc_state.update(node_state)
                trace = acc_state.get("execution_trace", [])
                if len(trace) > seen_steps:
                    latest_step: NodeExecutionStep = trace[-1]
                    seen_steps = len(trace)
                    payload = {
                        "event": "node_completed",
                        "node": node_name,
                        "step": latest_step.model_dump(),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

        final_resp = AgentAnalyzeResponse(
            query=req.query,
            sanitized_query=acc_state["sanitized_query"],
            routed_intent=acc_state["routed_intent"],
            answer=acc_state["answer"],
            citations=acc_state.get("citations", []),
            quant_summary=acc_state.get("quant_summary"),
            option_greeks=acc_state.get("option_greeks"),
            risk_summary=acc_state.get("risk_summary"),
            hallucination_check_passed=acc_state["hallucination_check_passed"],
            numerical_faithfulness_passed=acc_state.get("numerical_faithfulness_passed", True),
            confidence_score=acc_state["confidence_score"],
            execution_trace=acc_state["execution_trace"],
        )
        yield f"data: {json.dumps({'event': 'final_response', 'data': final_resp.model_dump()})}\n\n"
