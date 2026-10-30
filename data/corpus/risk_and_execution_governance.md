# Algorithmic Risk Guardrails & Execution Governance

## Hard Per-Trade & Daily Risk Caps
Every algorithmic or discretionary execution proposal must pass pre-trade risk validation before order routing:
1. **Per-Trade Risk Floor:** Maximum loss at the hard stop (`|Entry - Stop| * Quantity + EstimatedCharges`) must never exceed the configured per-trade risk cap (`1.0 R`).
2. **Daily Drawdown Wall:** If realized + open unrealized net loss reaches `-2.0 R` in a single session, new entries are blocked for the remainder of the trading day.
3. **Never Widen a Stop-Loss:** Once a position is opened with a validated stop price, moving the stop further away from entry is strictly prohibited. Stops may only trail toward breakeven or profit lock (`T1` / `T2`).

## Net-R Accounting After Transaction Charges
Gross R-multiples overstate edge on small accounts because brokerage, exchange turnover fees, SEBI charges, stamp duty, and 18% GST create a fixed drag per round-trip order. Expectancy must be evaluated on **Net-R**:
`Net_PnL = Gross_PnL - RoundTripCharges`
`Net_R = Net_PnL / InitialRiskRupees`
A setup requires a minimum `1.8x` to `2.0x` net reward-to-risk ratio after transaction costs to maintain positive expectancy at a 45%-50% win rate.

## Prompt-Injection & Hallucination Safeguards in Financial AI Agents
Financial research agents consuming external filings or user prompts must enforce:
1. **Input Delimiter Sanitization (OWASP LLM01):** Strip XML/markdown system-override tags (`<system>`, `<instructions>`, `IGNORE PREVIOUS INSTRUCTIONS`) before embedding or LLM prompt construction.
2. **Retrieval Grounding & Citation Verification:** Every synthesized risk or microstructure claim must cite a retrieved `chunk_id` above the minimum similarity threshold, and numerical metrics must come directly from deterministic `Pandas`/`NumPy` tool outputs rather than free-form LLM arithmetic.
