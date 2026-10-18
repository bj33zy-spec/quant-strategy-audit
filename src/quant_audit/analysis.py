"""Combine the metric and inference modules into a repeatable audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import (
    adjust_pvalues,
    block_bootstrap_sharpe_ci,
    newey_west_mean_test,
    probabilistic_sharpe_ratio,
)
from .metrics import annualized_sharpe, annualized_volatility, max_drawdown, total_return
from .pbo import compute_pbo


def audit_strategies(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    pbo_splits: int = 4,
    bootstrap_samples: int = 250,
    bootstrap_block_size: int = 10,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute per-candidate metrics, inferential diagnostics, and PBO."""
    clean = returns.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(axis=0, how="any")
    if clean.shape[1] < 2:
        raise ValueError("Provide at least two numeric candidate-return columns.")
    minimum_rows = max(30, 2 * pbo_splits * 5)
    if len(clean) < minimum_rows:
        raise ValueError(f"Need at least {minimum_rows} complete observations for this audit.")

    records: list[dict[str, object]] = []
    p_values: list[float] = []
    for column in clean.columns:
        values = clean[column]
        mean_test = newey_west_mean_test(values)
        ci_low, ci_high = block_bootstrap_sharpe_ci(
            values,
            n_bootstrap=bootstrap_samples,
            block_size=min(bootstrap_block_size, len(values)),
            periods_per_year=periods_per_year,
            seed=seed + len(records),
        )
        record = {
            "strategy": str(column),
            "observations": int(values.shape[0]),
            "total_return": total_return(values),
            "annualized_sharpe": annualized_sharpe(values, periods_per_year),
            "annualized_volatility": annualized_volatility(values, periods_per_year),
            "max_drawdown": max_drawdown(values),
            "mean_period_return": float(values.mean()),
            "newey_west_t_stat": mean_test["t_statistic"],
            "raw_mean_p_value": mean_test["p_value"],
            "psr_vs_zero": probabilistic_sharpe_ratio(values, 0.0, periods_per_year),
            "bootstrap_sharpe_ci_low": ci_low,
            "bootstrap_sharpe_ci_high": ci_high,
        }
        records.append(record)
        p_values.append(float(mean_test["p_value"]))

    results = pd.DataFrame(records)
    results["holm_adjusted_p_value"] = adjust_pvalues(p_values, "holm")
    results["bh_adjusted_q_value"] = adjust_pvalues(p_values, "bh")
    results = results.sort_values("annualized_sharpe", ascending=False, na_position="last").reset_index(drop=True)
    pbo = compute_pbo(clean, n_splits=pbo_splits, periods_per_year=periods_per_year)
    return results, pbo
