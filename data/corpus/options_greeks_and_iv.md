# Options Pricing, Black-Scholes Greeks & Volatility Mechanics

## Black-Scholes-Merton Pricing Model
For a European option on an underlying at spot `S`, strike `K`, risk-free rate `r`, annualized volatility `sigma`, and time to expiry `T` (in years):
`d1 = (ln(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))`
`d2 = d1 - sigma * sqrt(T)`
Call price: `C = S * N(d1) - K * exp(-r * T) * N(d2)`
Put price: `P = K * exp(-r * T) * N(-d2) - S * N(-d1)`

## First- and Second-Order Option Greeks
- **Delta (`Δ`):** Sensitivity of option premium to a 1-point move in the underlying (`N(d1)` for calls, `N(d1) - 1` for puts). Near-the-money (ATM) options carry `|Δ| ≈ 0.50`; deep out-of-the-money (OTM) options (`|Δ| < 0.20`) suffer poor risk-adjusted expectancy due to low probability of expiring in the money.
- **Gamma (`Γ`):** Rate of change of Delta per 1-point move in spot (`N'(d1) / (S * sigma * sqrt(T))`). Gamma peaks for ATM options as `T -> 0` (expiry day), creating explosive convexity for buyers and severe tail risk for naked sellers.
- **Theta (`Θ`):** Daily time-value decay (`dV / dt`, typically expressed per calendar day). Long option positions bleed Theta every day; Theta accelerates non-linearly inside the final 5 DTE.
- **Vega (`ν`):** Sensitivity of option price to a 1-percentage-point (`0.01`) change in Implied Volatility (`S * N'(d1) * sqrt(T)`).

## Implied Volatility (IV) Crush & Debit Spreads
Ahead of scheduled binary catalysts (earnings, central-bank policy announcements, budget events), option market makers inflate Implied Volatility (`IV`). Immediately after the event resolves, `IV` collapses toward historical realized volatility (`IV crush`). Naked long calls or puts bought at elevated `IV` can lose 30%-60% of their premium via negative Vega (`ν * ΔIV`) even when the directional call is right. A vertical debit spread (buying an ATM/ITM leg and selling an OTM leg at the same expiry) offsets Vega and Theta exposure while capping maximum loss to the net debit paid.
