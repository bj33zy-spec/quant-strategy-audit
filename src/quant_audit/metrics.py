"""Basic return-series metrics used by the audit tools."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_returns(returns: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Return finite simple returns as a one-dimensional NumPy array."""
    values = np.asarray(returns, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("At least one finite return observation is required.")
    if np.any(values <= -1.0):
        raise ValueError("Simple returns must be greater than -1.0 (-100%).")
    return values


def annualized_sharpe(
    returns: pd.Series | np.ndarray | list[float],
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualized Sharpe from periodic simple returns.

    The risk-free rate is an annual decimal rate, e.g. 0.04 for 4%.
    The default assumes a zero risk-free rate, which is convenient for research
    demonstrations but should be changed for a real application if appropriate.
    """
    values = clean_returns(returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    excess = values - annual_risk_free_rate / periods_per_year
    volatility = np.std(excess, ddof=1) if len(excess) > 1 else 0.0
    if not np.isfinite(volatility) or volatility <= np.finfo(float).eps:
        return float("nan")
    return float(np.mean(excess) / volatility * np.sqrt(periods_per_year))


def annualized_volatility(
    returns: pd.Series | np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    """Annualize the sample standard deviation of periodic returns."""
    values = clean_returns(returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) * np.sqrt(periods_per_year))


def total_return(returns: pd.Series | np.ndarray | list[float]) -> float:
    """Compound a simple-return series into a cumulative return."""
    values = clean_returns(returns)
    return float(np.prod(1.0 + values) - 1.0)


def max_drawdown(returns: pd.Series | np.ndarray | list[float]) -> float:
    """Return the most negative peak-to-trough drawdown of an equity curve."""
    values = clean_returns(returns)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    return float(np.min(drawdowns))
