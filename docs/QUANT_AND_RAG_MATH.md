# Quantitative & Retrieval Mathematical Reference

This document specifies the exact mathematical formulations implemented in `src/finpulse/analytics.py` and `src/finpulse/rag.py`.

---

## 1. Intraday Market Microstructure Indicators (`src/finpulse/analytics.py`)

### 1.1 Session VWAP & Volume-Weighted $\pm 1\sigma$ Bands
For intraday bar $i \in \{1, \dots, t\}$ with high $H_i$, low $L_i$, close $C_i$, and volume $V_i$:
$$\text{TP}_i = \frac{H_i + L_i + C_i}{3}, \qquad \text{VWAP}_t = \frac{\sum_{i=1}^t \text{TP}_i \cdot V_i}{\sum_{i=1}^t V_i}$$
The volume-weighted variance around session VWAP is computed via the second moment:
$$\sigma_{\text{VWAP}, t}^2 = \max\!\left(0,\; \frac{\sum_{i=1}^t \text{TP}_i^2 \cdot V_i}{\sum_{i=1}^t V_i} - \text{VWAP}_t^2\right)$$
giving upper and lower $1\sigma$ bands $\text{VWAP}_t \pm \sigma_{\text{VWAP}, t}$.

### 1.2 True Range ($\text{TR}$) and $\text{ATR}(14)$
$$\text{TR}_t = \max\!\big(H_t - L_t,\; |H_t - C_{t-1}|,\; |L_t - C_{t-1}|\big)$$
$$\text{ATR}_{14, t} = \frac{1}{w} \sum_{k=t-w+1}^t \text{TR}_k, \qquad w = \min(14, t)$$

### 1.3 Wilder / EWMA Relative Strength Index ($\text{RSI}(14)$)
With bar-to-bar close change $\Delta C_t = C_t - C_{t-1}$, upward move $U_t = \max(\Delta C_t, 0)$, and downward move $D_t = \max(-\Delta C_t, 0)$ smoothed with $\alpha = 1/14$:
$$\text{RS}_t = \frac{\text{EWMA}_\alpha(U)_t}{\text{EWMA}_\alpha(D)_t}, \qquad \text{RSI}_{14, t} = 100 - \frac{100}{1 + \text{RS}_t}$$

### 1.4 Empirical Bayes Barrier-Crossing Touch Probability
Given target distance $+k \cdot \text{ATR}$ and stop distance $-m \cdot \text{ATR}$, the random-walk barrier-crossing prior is:
$$P_0 = \frac{m}{k + m}$$
Across $N$ historical rolling forward horizons in the session, if $W$ trials reach $C_i + k \cdot \text{ATR}_i$ before breaching $C_i - m \cdot \text{ATR}_i$, the empirical rate $\hat{p} = W / N$ is shrunk toward $P_0$ via Empirical Bayes weighting:
$$P_{\text{touch}} = w \cdot \hat{p} + (1 - w) \cdot P_0, \qquad w = \min\!\left(\frac{N}{N + 10},\; 0.85\right)$$

---

## 2. Black-Scholes-Merton Option Pricing, Greeks & IV Solver (`src/finpulse/analytics.py`)

For spot price $S$, strike $K$, annualized risk-free rate $r$, annualized implied volatility $\sigma$, and time to expiry $T = \text{DTE} / 365$:
$$d_1 = \frac{\ln(S / K) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

### 2.1 Option Prices & First/Second-Order Greeks
With standard normal CDF $\Phi(x)$ and PDF $\phi(x)$:
- **Call / Put Theoretical Price:**
  $$C = S\Phi(d_1) - Ke^{-rT}\Phi(d_2), \qquad P = Ke^{-rT}\Phi(-d_2) - S\Phi(-d_1)$$
- **Delta ($\Delta$):**
  $$\Delta_{\text{call}} = \Phi(d_1), \qquad \Delta_{\text{put}} = \Phi(d_1) - 1$$
- **Gamma ($\Gamma$):**
  $$\Gamma = \frac{\phi(d_1)}{S\sigma\sqrt{T}}$$
- **Theta per Calendar Day ($\Theta_{\text{day}}$):**
  $$\Theta_{\text{call, day}} = \frac{1}{365}\left(-\frac{S\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}\Phi(d_2)\right)$$
- **Vega per $1\%$ Vol Move ($\nu_{1\%}$):**
  $$\nu_{1\%} = \frac{S\phi(d_1)\sqrt{T}}{100}$$
- **Second-Order Greeks (Vanna & Daily Charm) and Put-Call Parity Check:**
  $$\text{Vanna}_{1\%} = -\frac{\phi(d_1) d_2}{100 \sigma}, \qquad \text{Charm}_{\text{day}} = -\frac{\phi(d_1)\big(2rT - d_2\sigma\sqrt{T}\big)}{365 \cdot 2T\sigma\sqrt{T}}$$
  $$\varepsilon_{\text{parity}} = \left|(C - P) - \left(S - Ke^{-rT}\right)\right| < 10^{-6}$$

### 2.2 Hybrid Newton-Raphson + Bisection Implied Volatility Solver (`solve_implied_volatility`)
Given observed market premium $V_{\text{mkt}}$, we solve $f(\sigma) = V_{\text{BS}}(\sigma) - V_{\text{mkt}} = 0$ over bracket $[\sigma_{\text{low}}, \sigma_{\text{high}}] = [0.01, 4.00]$:
$$\sigma_{n+1} = \sigma_n - \frac{V_{\text{BS}}(\sigma_n) - V_{\text{mkt}}}{\partial V_{\text{BS}} / \partial \sigma}$$
Whenever the Newton step jumps outside the tightened bracket $[\sigma_{\text{low}}, \sigma_{\text{high}}]$ or Vega approaches zero in deep OTM/ITM tails, the solver falls back to the midpoint $\frac{1}{2}(\sigma_{\text{low}} + \sigma_{\text{high}})$, guaranteeing convergence within 40 iterations to $|f(\sigma)| < 10^{-4}$.

### 2.3 Pre-Trade Risk Governance, Net-R, Half-Kelly & VaR / CVaR (`compute_position_sizing_and_risk`)
- **Volatility-Normalized Lot Sizing:** Given account equity $E$, per-trade risk fraction $r_{\max}$ ($1.0R = E \cdot r_{\max}$), round-trip transaction charges $F$, and stop distance $\Delta_{\text{stop}} = |P_{\text{entry}} - P_{\text{stop}}|$:
  $$Q = \left\lfloor \frac{\max(1.0R - F, 0)}{\Delta_{\text{stop}} \cdot L} \right\rfloor \cdot L$$
- **Net-R vs. Gross-R & Half-Kelly Fraction:**
  $$b_{\text{net}} = \frac{\max(Q \cdot |P_{\text{target}} - P_{\text{entry}}| - F,\; 0)}{Q \cdot \Delta_{\text{stop}} + F}, \qquad f_{\text{Kelly}/2}^* = \frac{1}{2}\max\!\left(0,\; p - \frac{1 - p}{b_{\text{net}}}\right)$$
- **95% Value-at-Risk ($\text{VaR}_{95\%}$) & Expected Shortfall ($\text{CVaR}_{95\%}$):**
  $$\text{VaR}_{95\%} = -Q_{0.05}(R), \qquad \text{CVaR}_{95\%} = -\mathbb{E}\big[R \;\big|\; R \le -\text{VaR}_{95\%}\big]$$

---

## 3. Hybrid Vector + Lexical Retrieval & IR Metrics (`src/finpulse/rag.py`)

### 3.1 Okapi BM25 Lexical Score
For query $Q$ and chunk $d$ with token length $|d|$ and corpus average chunk length $\text{avgdl}$ ($k_1 = 1.5, b = 0.75$):
$$\text{BM25}(Q, d) = \sum_{q \in Q} \ln\!\left(1 + \frac{N - \text{df}(q) + 0.5}{\text{df}(q) + 0.5}\right) \cdot \frac{\text{tf}(q, d)\cdot(k_1 + 1)}{\text{tf}(q, d) + k_1\cdot\left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

### 3.2 Reciprocal Rank Fusion (RRF)
Combining rank $r_{\text{dense}}(d)$ from `faiss.IndexFlatIP` cosine similarity and rank $r_{\text{bm25}}(d)$ from Okapi BM25:
$$\text{RRF}(d) = \frac{1}{60 + r_{\text{dense}}(d)} + \frac{1}{60 + r_{\text{bm25}}(d)}$$

### 3.3 Information Retrieval Evaluation Metrics ($\text{Recall}@K$, $\text{MRR}@K$, $\text{NDCG}@K$)
$$\text{MRR}@K = \frac{1}{|Q|}\sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}, \qquad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}, \qquad \text{DCG}@K = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}$$
