"""Statistical inference helpers for strategy-return series.

These methods are diagnostics, not proof that a strategy will remain profitable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import annualized_sharpe, clean_returns


def newey_west_mean_test(
    returns: pd.Series | np.ndarray | list[float],
    max_lags: int | None = None,
) -> dict[str, float | int]:
    """Test whether the mean return differs from zero with a HAC standard error.

    Uses a Bartlett-kernel Newey-West estimate of the long-run variance. The
    reported two-sided p-value uses a t reference distribution as a practical
    finite-sample approximation; it should not be read as an exact guarantee.
    """
    values = clean_returns(returns)
    n = len(values)
    if n < 3:
        raise ValueError("At least three observations are needed for the mean test.")
    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    if max_lags < 0:
        raise ValueError("max_lags cannot be negative.")
    max_lags = min(int(max_lags), n - 1)

    centered = values - np.mean(values)
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        autocov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * weight * autocov

    # Rounding can make the long-run variance slightly negative in edge cases.
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = float(np.sqrt(long_run_variance / n))
    mean_return = float(np.mean(values))
    if standard_error == 0.0:
        t_stat = 0.0 if mean_return == 0 else float(np.sign(mean_return) * np.inf)
        p_value = 1.0 if mean_return == 0 else 0.0
    else:
        t_stat = mean_return / standard_error
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))

    return {
        "n_observations": n,
        "mean_return": mean_return,
        "standard_error": standard_error,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "max_lags": max_lags,
    }


def adjust_pvalues(p_values: Sequence[float], method: str = "holm") -> np.ndarray:
    """Adjust p-values using Bonferroni, Holm, or Benjamini-Hochberg.

    Returns values in the same order as the input. Holm controls family-wise
    error; Benjamini-Hochberg controls the false-discovery rate under its usual
    assumptions. Non-finite p-values are treated as 1.0.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if np.any((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)):
        raise ValueError("Finite p-values must lie between 0 and 1.")
    p = np.where(np.isfinite(p), p, 1.0)
    m = len(p)
    if m == 0:
        return p
    method = method.lower()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    if method not in {"holm", "bh", "fdr_bh"}:
        raise ValueError("method must be 'bonferroni', 'holm', or 'bh'.")

    order = np.argsort(p)
    sorted_p = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    if method == "holm":
        adjusted_sorted = np.maximum.accumulate((m - ranks + 1) * sorted_p)
    else:
        adjusted_sorted = np.minimum.accumulate((sorted_p * m / ranks)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray | list[float],
    benchmark_annual_sharpe: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Estimate P(true Sharpe exceeds a benchmark) using the PSR approximation.

    Skewness and Pearson kurtosis enter the standard-error adjustment. The
    benchmark is expressed as an annualized Sharpe and converted to the return
    frequency before applying the formula. This is a diagnostic and does not
    correct on its own for choosing the best result from many trials.
    """
    values = clean_returns(returns)
    n = len(values)
    if n < 3:
        raise ValueError("At least three observations are needed for PSR.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    sd = float(np.std(values, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")

    sr = float(np.mean(values) / sd)
    benchmark = float(benchmark_annual_sharpe / np.sqrt(periods_per_year))
    skew = float(stats.skew(values, bias=False))
    pearson_kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    denominator_squared = 1.0 - skew * sr + ((pearson_kurtosis - 1.0) / 4.0) * sr**2
    if not np.isfinite(denominator_squared) or denominator_squared <= 0:
        return float("nan")
    z_score = (sr - benchmark) * np.sqrt(n - 1.0) / np.sqrt(denominator_squared)
    return float(stats.norm.cdf(z_score))


def block_bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray | list[float],
    n_bootstrap: int = 500,
    block_size: int = 10,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    seed: int = 2026,
) -> tuple[float, float]:
    """Circular moving-block bootstrap interval for annualized Sharpe.

    Blocks help preserve short-range dependence better than resampling each
    return independently. The interval is still conditional on the provided
    sample and is not a selection-adjusted interval for the best of many models.
    """
    values = clean_returns(returns)
    n = len(values)
    if n < 10:
        raise ValueError("At least 10 observations are needed for block bootstrap.")
    if n_bootstrap < 50:
        raise ValueError("Use at least 50 bootstrap samples for a useful interval.")
    if block_size < 1 or block_size > n:
        raise ValueError("block_size must be between 1 and the sample length.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        estimates[i] = annualized_sharpe(values[indices], periods_per_year)
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)
