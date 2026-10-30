"""Streamlit Evaluation & Research Dashboard for FinPulse AI Agent."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from finpulse.agent import FinPulseLangGraphAgent
from finpulse.rag import build_default_store
from finpulse.schemas import AgentAnalyzeRequest, OptionGreeksRequest


@st.cache_resource
def get_agent() -> FinPulseLangGraphAgent:
    store = build_default_store()
    return FinPulseLangGraphAgent(store=store)


def main() -> None:
    st.set_page_config(
        page_title="FinPulse AI — RAG & Quant Agent",
        layout="wide",
    )
    st.title("FinPulse AI — Hybrid RAG & Quantitative LangGraph Agent")
    st.caption(
        "FastAPI · LangChain Core · LangGraph · FAISS (IndexFlatIP + BM25 RRF) · Pandas & NumPy · Black-Scholes Greeks"
    )

    agent = get_agent()

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

    query = st.text_area(
        "Research or Quantitative Query",
        value="Evaluate NIFTY VWAP, ATR(14) stop sizing, and 24400 CE Theta/Vega IV-crush risk.",
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
        req = AgentAnalyzeRequest(
            query=query,
            symbol=symbol,
            top_k=top_k,
            option_params=opt_req,
        )
        res = agent.run(req)

        col1, col2, col3 = st.columns(3)
        col1.metric("Routed Intent", res.routed_intent)
        col2.metric("Confidence Score", f"{res.confidence_score:.1%}")
        col3.metric("Hallucination Gate", "PASSED" if res.hallucination_check_passed else "FLAGGED")

        st.subheader("Synthesized Agent Report")
        st.info(res.answer)

        if res.quant_summary is not None:
            st.subheader("Pandas / NumPy Intraday Indicators")
            st.dataframe(pd.DataFrame([res.quant_summary.model_dump()]), use_container_width=True)

        if res.option_greeks is not None:
            st.subheader("Black-Scholes Option Greeks")
            st.dataframe(pd.DataFrame([res.option_greeks.model_dump()]), use_container_width=True)

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


if __name__ == "__main__":
    main()
