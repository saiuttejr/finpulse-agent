"""Streamlit Evaluation & Research Dashboard for FinPulse AI Agent."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from finpulse.agent import FinPulseLangGraphAgent
from finpulse.rag import HybridFAISSStore, build_default_store, evaluate_retrieval_benchmark
from finpulse.schemas import AgentAnalyzeRequest, OptionGreeksRequest, PositionSizingRequest


@st.cache_resource
def get_store_and_agent() -> tuple[HybridFAISSStore, FinPulseLangGraphAgent]:
    store = build_default_store()
    return store, FinPulseLangGraphAgent(store=store)


def main() -> None:
    st.set_page_config(
        page_title="FinPulse AI — RAG & Quant Agent",
        layout="wide",
    )
    st.title("FinPulse AI — Hybrid RAG, Quant & Risk Governance Agent")
    st.caption(
        "FastAPI · LangChain Core · Conditional LangGraph · FAISS (IndexFlatIP + BM25 RRF) · "
        "Pandas & NumPy · Black-Scholes Greeks · VaR/CVaR Risk Governance"
    )

    store, agent = get_store_and_agent()

    tab_agent, tab_eval = st.tabs([
        "LangGraph Agent & Risk Governance",
        "RAG Retrieval Benchmark (Recall@K / MRR@K / NDCG@K)",
    ])

    with st.sidebar:
        st.subheader("Parameters")
        symbol = st.text_input("Ticker Symbol", value="NIFTY")
        top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=6, value=3)
        include_options = st.checkbox("Include Option Greeks Input", value=True)
        spot = st.number_input("Spot Price (S)", value=24350.0, step=10.0)
        strike = st.number_input("Strike Price (K)", value=24400.0, step=50.0)
        dte = st.number_input("Days to Expiry (DTE)", value=5.0, min_value=0.5, step=1.0)
        opt_type = st.selectbox("Option Type", options=["CE", "PE"])
        iv = st.slider("Implied Volatility (IV)", min_value=0.08, max_value=0.60, value=0.18, step=0.01)
        include_risk = st.checkbox("Include Pre-Trade Risk Sizing", value=True)
        equity = st.number_input("Account Equity", value=1000000.0, step=50000.0)
        daily_r = st.number_input("Current Session R", value=-0.5, min_value=-5.0, max_value=5.0, step=0.25)

    with tab_agent:
        query = st.text_area(
            "Research or Quantitative Query",
            value=(
                "Evaluate NIFTY VWAP, ATR(14) stop sizing, 24400 CE Theta/Vega IV-crush risk, "
                "and pre-trade position sizing with VaR/CVaR."
            ),
            height=90,
        )

        if st.button("Run LangGraph Agent", type="primary"):
            opt_req = (
                OptionGreeksRequest(
                    spot=spot,
                    strike=strike,
                    dte_days=dte,
                    option_type=opt_type,
                    volatility=iv,
                )
                if include_options
                else None
            )
            risk_req = (
                PositionSizingRequest(
                    account_equity=equity,
                    max_risk_pct=0.01,
                    entry_price=spot,
                    stop_price=round(spot - 25.0, 2),
                    target_price=round(spot + 55.0, 2),
                    current_daily_r=daily_r,
                )
                if include_risk
                else None
            )
            req = AgentAnalyzeRequest(
                query=query,
                symbol=symbol,
                top_k=top_k,
                option_params=opt_req,
                risk_params=risk_req,
            )
            res = agent.run(req)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Routed Intent", res.routed_intent)
            c2.metric("Confidence Score", f"{res.confidence_score:.1%}")
            c3.metric("Hallucination Gate", "PASSED" if res.hallucination_check_passed else "FLAGGED")
            c4.metric(
                "Numerical Faithfulness",
                "VERIFIED" if res.numerical_faithfulness_passed else "MISMATCH",
            )

            st.subheader("Synthesized Agent Report")
            st.info(res.answer)

            if res.quant_summary is not None:
                st.subheader("Pandas / NumPy Intraday Indicators")
                st.dataframe(pd.DataFrame([res.quant_summary.model_dump()]), use_container_width=True)

            if res.option_greeks is not None:
                st.subheader("Black-Scholes Option Greeks (1st & 2nd Order + Parity Check)")
                st.dataframe(pd.DataFrame([res.option_greeks.model_dump()]), use_container_width=True)

            if res.risk_summary is not None:
                st.subheader("Pre-Trade Risk Governance, Net-R & 95% VaR / CVaR")
                st.dataframe(pd.DataFrame([res.risk_summary.model_dump()]), use_container_width=True)

            st.subheader("FAISS + BM25 Hybrid RRF Retrieved Chunks")
            for c in res.citations:
                with st.expander(
                    f"{c.chunk_id} — {c.title} (Category: {c.category} | RRF: {c.rrf_score:.4f} | Cosine: {c.dense_score:.3f})"
                ):
                    st.markdown(c.text)

            st.subheader("LangGraph Execution Trace")
            st.dataframe(
                pd.DataFrame([step.model_dump() for step in res.execution_trace]),
                use_container_width=True,
            )

    with tab_eval:
        st.subheader("Hybrid RRF vs. Dense FAISS vs. Okapi BM25 Ablation Study")
        eval_k = st.slider("Evaluation Top-K", min_value=1, max_value=5, value=3)
        eval_res = evaluate_retrieval_benchmark(store, top_k=eval_k, include_per_query_cases=True)

        summary_rows = []
        for mode_name, summary in eval_res.modes.items():
            summary_rows.append({
                "Retrieval Mode": mode_name,
                f"Recall@{eval_k}": summary.mean_recall_at_k,
                f"MRR@{eval_k}": summary.mrr_at_k,
                f"NDCG@{eval_k}": summary.mean_ndcg_at_k,
                "Mean Latency (ms)": summary.mean_latency_ms,
                "p95 Latency (ms)": summary.p95_latency_ms,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.subheader("Per-Query Golden Benchmark Results (Hybrid RRF)")
        st.dataframe(
            pd.DataFrame([c.model_dump() for c in eval_res.modes["hybrid_rrf"].cases]),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
