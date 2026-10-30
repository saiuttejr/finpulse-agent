"""Vectorized Pandas & NumPy market-data preprocessing, technical indicators,
empirical touch-probability modeling, and Black-Scholes-Merton Greeks + IV solver."""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from finpulse.schemas import (
    OHLCVBar,
    OptionGreeksRequest,
    OptionGreeksResponse,
    QuantIndicatorSummary,
)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function N(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function N'(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bars_to_dataframe(bars: list[OHLCVBar]) -> pd.DataFrame:
    """Convert validated OHLCVBar schemas into a cleaned, sorted Pandas DataFrame."""
    records = [b.model_dump() for b in bars]
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
    return df


def compute_intraday_indicators(
    symbol: str,
    bars: list[OHLCVBar],
    target_atr_multiple: float = 1.5,
    stop_atr_multiple: float = 1.0,
) -> QuantIndicatorSummary:
    """Compute vectorized VWAP (+1/-1 sigma), ATR(14), RSI(14), EMA(9/20), RVOL,
    and empirical target-before-stop touch probability using Pandas and NumPy."""
    df = bars_to_dataframe(bars)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = np.maximum(df["volume"].to_numpy(dtype=np.float64), 1.0)

    typical_price = (high + low + close) / 3.0
    cum_tpv = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    vwap_series = cum_tpv / cum_vol

    # Volume-weighted variance around session VWAP for +/- 1 sigma bands
    cum_tp2v = np.cumsum((typical_price ** 2) * volume)
    variance = np.maximum((cum_tp2v / cum_vol) - (vwap_series ** 2), 0.0)
    vwap_std = float(np.sqrt(variance[-1]))
    vwap_val = float(vwap_series[-1])

    # True Range and ATR(14)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    window = min(14, len(tr))
    atr_series = pd.Series(tr).rolling(window=window, min_periods=1).mean().to_numpy()
    atr_val = max(float(atr_series[-1]), 1e-4)

    # Wilder / EWMA RSI(14)
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = pd.Series(gains).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy()[-1]
    avg_loss = pd.Series(losses).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy()[-1]
    if avg_loss <= 1e-12:
        rsi_val = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = float(avg_gain / avg_loss)
        rsi_val = 100.0 - (100.0 / (1.0 + rs))

    # EMA 9 and EMA 20
    close_s = pd.Series(close)
    ema_9 = float(close_s.ewm(span=min(9, len(close)), adjust=False).mean().iloc[-1])
    ema_20 = float(close_s.ewm(span=min(20, len(close)), adjust=False).mean().iloc[-1])

    # Relative Volume (latest bar vs rolling mean volume)
    mean_vol = float(np.mean(volume[:-1])) if len(volume) > 1 else float(volume[-1])
    rvol_val = float(volume[-1] / max(mean_vol, 1.0))

    latest_close = float(close[-1])
    if latest_close > vwap_val + 0.15 * atr_val:
        regime: Literal["bullish_above_vwap", "bearish_below_vwap", "balanced_near_vwap"] = (
            "bullish_above_vwap"
        )
    elif latest_close < vwap_val - 0.15 * atr_val:
        regime = "bearish_below_vwap"
    else:
        regime = "balanced_near_vwap"

    target_price = round(latest_close + target_atr_multiple * atr_val, 2)
    stop_price = round(max(latest_close - stop_atr_multiple * atr_val, 0.01), 2)

    touch_prob = estimate_empirical_touch_probability(
        high=high,
        low=low,
        close=close,
        atr=atr_series,
        target_atr=target_atr_multiple,
        stop_atr=stop_atr_multiple,
    )

    return QuantIndicatorSummary(
        symbol=symbol.upper(),
        bar_count=len(df),
        latest_close=round(latest_close, 2),
        vwap=round(vwap_val, 2),
        vwap_upper_1sigma=round(vwap_val + vwap_std, 2),
        vwap_lower_1sigma=round(vwap_val - vwap_std, 2),
        atr_14=round(atr_val, 4),
        rsi_14=round(float(rsi_val), 2),
        ema_9=round(ema_9, 2),
        ema_20=round(ema_20, 2),
        rvol=round(rvol_val, 2),
        regime=regime,
        target_price=target_price,
        stop_price=stop_price,
        empirical_touch_probability=round(touch_prob, 4),
    )


def estimate_empirical_touch_probability(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    target_atr: float = 1.5,
    stop_atr: float = 1.0,
    horizon: int = 8,
) -> float:
    """Compute empirical probability that price reaches +target_atr*ATR before
    hitting -stop_atr*ATR over forward rolling windows, blended with random-walk prior."""
    n = len(close)
    prior = stop_atr / (target_atr + stop_atr)
    if n < 6:
        return prior

    wins = 0
    trials = 0
    for i in range(n - 2):
        cur_atr = max(float(atr[i]), 1e-4)
        entry = float(close[i])
        tgt = entry + target_atr * cur_atr
        stp = entry - stop_atr * cur_atr
        end_idx = min(n, i + 1 + horizon)
        fwd_high = high[i + 1 : end_idx]
        fwd_low = low[i + 1 : end_idx]

        hit_tgt = np.where(fwd_high >= tgt)[0]
        hit_stp = np.where(fwd_low <= stp)[0]
        if len(hit_tgt) == 0 and len(hit_stp) == 0:
            max_excursion = float(np.max(fwd_high) - entry)
            min_excursion = float(entry - np.min(fwd_low))
            if max_excursion / target_atr >= min_excursion / stop_atr:
                wins += 1
            trials += 1
        elif len(hit_tgt) > 0 and (len(hit_stp) == 0 or hit_tgt[0] <= hit_stp[0]):
            wins += 1
            trials += 1
        else:
            trials += 1

    if trials == 0:
        return prior
    empirical = wins / trials
    # Empirical Bayes shrinkage toward analytical barrier-crossing prior
    weight = min(trials / (trials + 10.0), 0.85)
    return float(weight * empirical + (1.0 - weight) * prior)


def black_scholes_price_and_greeks(
    spot: float,
    strike: float,
    dte_days: float,
    volatility: float,
    risk_free_rate: float = 0.068,
    option_type: Literal["CE", "PE"] = "CE",
) -> tuple[float, float, float, float, float]:
    """Return (price, delta, gamma, theta_per_day, vega_per_pct)."""
    t = max(dte_days / 365.0, 1.0 / (365.0 * 24.0))
    sigma = max(volatility, 1e-4)
    sqrt_t = math.sqrt(t)

    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    discount = math.exp(-risk_free_rate * t)
    pdf_d1 = _norm_pdf(d1)

    if option_type == "CE":
        price = spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta_annual = (
            -(spot * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - risk_free_rate * strike * discount * _norm_cdf(d2)
        )
    else:
        price = strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        theta_annual = (
            -(spot * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + risk_free_rate * strike * discount * _norm_cdf(-d2)
        )

    gamma = pdf_d1 / (spot * sigma * sqrt_t)
    theta_per_day = theta_annual / 365.0
    vega_per_pct = (spot * pdf_d1 * sqrt_t) / 100.0
    return price, delta, gamma, theta_per_day, vega_per_pct


def solve_implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    dte_days: float,
    risk_free_rate: float = 0.068,
    option_type: Literal["CE", "PE"] = "CE",
    initial_guess: float = 0.20,
) -> float:
    """Solve implied volatility via Newton-Raphson with bisection bounds fallback."""
    sigma = max(min(initial_guess, 3.0), 0.02)
    low_vol, high_vol = 0.01, 4.0

    for _ in range(40):
        price, _, _, _, vega_pct = black_scholes_price_and_greeks(
            spot=spot,
            strike=strike,
            dte_days=dte_days,
            volatility=sigma,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
        )
        diff = price - market_price
        if abs(diff) < 1e-4:
            return round(sigma, 4)

        if diff > 0:
            high_vol = min(high_vol, sigma)
        else:
            low_vol = max(low_vol, sigma)

        vega_raw = vega_pct * 100.0
        if vega_raw > 1e-6:
            candidate = sigma - diff / vega_raw
            if low_vol < candidate < high_vol:
                sigma = candidate
            else:
                sigma = 0.5 * (low_vol + high_vol)
        else:
            sigma = 0.5 * (low_vol + high_vol)

    return round(sigma, 4)


def compute_option_greeks(req: OptionGreeksRequest) -> OptionGreeksResponse:
    """Compute option theoretical price, solved IV (if market_premium supplied),
    Delta, Gamma, daily Theta, Vega per 1% IV, and moneyness classification."""
    iv = req.volatility
    if req.market_premium is not None:
        iv = solve_implied_volatility(
            market_price=req.market_premium,
            spot=req.spot,
            strike=req.strike,
            dte_days=req.dte_days,
            risk_free_rate=req.risk_free_rate,
            option_type=req.option_type,
            initial_guess=req.volatility,
        )

    price, delta, gamma, theta_day, vega_pct = black_scholes_price_and_greeks(
        spot=req.spot,
        strike=req.strike,
        dte_days=req.dte_days,
        volatility=iv,
        risk_free_rate=req.risk_free_rate,
        option_type=req.option_type,
    )

    rel_diff = abs(req.spot - req.strike) / req.spot
    if rel_diff <= 0.0075:
        moneyness: Literal["ITM", "ATM", "OTM"] = "ATM"
    elif (req.option_type == "CE" and req.spot > req.strike) or (
        req.option_type == "PE" and req.spot < req.strike
    ):
        moneyness = "ITM"
    else:
        moneyness = "OTM"

    return OptionGreeksResponse(
        option_type=req.option_type,
        spot=round(req.spot, 2),
        strike=round(req.strike, 2),
        dte_days=round(req.dte_days, 2),
        theoretical_price=round(price, 2),
        implied_volatility=round(iv, 4),
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta_per_day=round(theta_day, 4),
        vega_per_pct=round(vega_pct, 4),
        moneyness=moneyness,
    )
