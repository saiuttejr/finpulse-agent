# Intraday Market Microstructure & Indicator Playbook

## Volume-Weighted Average Price (VWAP) & Standard Deviation Bands
VWAP represents the volume-weighted average transaction price from the session open:
`VWAP_t = sum(TypicalPrice_i * Volume_i) / sum(Volume_i)` where `TypicalPrice = (High + Low + Close) / 3`.
Institutional execution algorithms benchmark fills against session VWAP. When price holds above a rising VWAP with `RVOL >= 1.5`, pullbacks to the first upper standard-deviation band or VWAP itself act as institutional accumulation zones (VWAP reclaim / higher-low setup). Conversely, repeated rejections below a declining VWAP signal distribution.

## Average True Range (ATR) & Volatility-Normalized Position Sizing
True Range (`TR`) captures both intrabar range and overnight or interbar gaps:
`TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)`.
`ATR(14)` is the 14-period smoothed average of `TR`. Stop-loss distances should be anchored to market structure and checked against `ATR(14)` (typically `1.0x` to `1.5x ATR(14)` on 5-minute bars) so normal intrabar noise does not prematurely trigger stops. Position size in shares is strictly `floor(MaxRupeeRisk / |Entry - Stop|)`.

## Relative Volume (RVOL) & Opening Range Breakout (ORB)
Relative Volume (`RVOL`) compares cumulative session volume against the historical average volume at the exact same time of day. An Opening Range Breakout (15-minute or 30-minute ORB) requires `RVOL >= 1.8`, a clean candle close outside the opening range, and price extension within `1.2x ATR` of VWAP to avoid chasing exhausted breakouts.

## Relative Strength Index (RSI) & Exponential Moving Averages (EMA 9 / EMA 20)
`RSI(14)` measures momentum velocity on a 0-100 scale using Wilder's smoothing of upward and downward closes. In trend regimes, `RSI(14)` holding between `55` and `75` alongside `EMA(9) > EMA(20) > VWAP` confirms sustained directional momentum, whereas `RSI > 80` at `+2.5x ATR` above VWAP warns of mean-reversion risk.
