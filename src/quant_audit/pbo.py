"""Combinatorially symmetric cross-validation (CSCV) estimate of PBO."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .metrics import annualized_sharpe


def compute_pbo(
    returns: pd.DataFrame,
    n_splits: int = 4,
    periods_per_year: int = 252,
) -> dict[str, object]:
    """Estimate Probability of Backtest Overfitting from candidate returns.

    The timeline is divided into 2 * n_splits contiguous blocks. For every
    choice of n_splits blocks as in-sample, the best in-sample Sharpe candidate
    is ranked against all candidates on the complementary out-of-sample blocks.
    PBO is the fraction of partitions where the selected candidate ranks at or
    below the median out of sample.

    The result depends on the candidate set, sample length, and block design.
    It is a diagnostic, not a universal probability that a live strategy fails.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame with one column per strategy.")
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if clean.shape[1] < 2:
        raise ValueError("PBO requires at least two candidate strategy columns.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    n_blocks = 2 * int(n_splits)
    if len(clean) < n_blocks * 5:
        raise ValueError(f"Need at least {n_blocks * 5} complete rows for this split setting.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    block_indices = [block for block in np.array_split(np.arange(len(clean)), n_blocks) if len(block)]
    n_strategies = clean.shape[1]
    values = clean.to_numpy(dtype=float)
    details: list[dict[str, object]] = []
    logits: list[float] = []
    strategy_names = list(clean.columns)

    for is_block_ids in combinations(range(len(block_indices)), n_splits):
        is_set = set(is_block_ids)
        oos_block_ids = [i for i in range(len(block_indices)) if i not in is_set]
        is_rows = np.concatenate([block_indices[i] for i in is_block_ids])
        oos_rows = np.concatenate([block_indices[i] for i in oos_block_ids])
        is_matrix = values[is_rows, :]
        oos_matrix = values[oos_rows, :]

        is_sharpes = np.array([
            annualized_sharpe(is_matrix[:, j], periods_per_year) for j in range(n_strategies)
        ])
        # A constant-return candidate has undefined Sharpe; do not let NaN win.
        is_sharpes = np.where(np.isfinite(is_sharpes), is_sharpes, -np.inf)
        winner_idx = int(np.argmax(is_sharpes))
        oos_sharpes = np.array([
            annualized_sharpe(oos_matrix[:, j], periods_per_year) for j in range(n_strategies)
        ])
        oos_sharpes = np.where(np.isfinite(oos_sharpes), oos_sharpes, -np.inf)
        ranks = rankdata(oos_sharpes, method="average")  # 1 = worst, N = best
        omega = float(ranks[winner_idx] / (n_strategies + 1.0))
        logit = float(np.log(omega / (1.0 - omega)))
        logits.append(logit)
        details.append({
            "in_sample_blocks": ";".join(map(str, is_block_ids)),
            "selected_strategy": strategy_names[winner_idx],
            "in_sample_sharpe": float(is_sharpes[winner_idx]),
            "out_of_sample_sharpe": float(oos_sharpes[winner_idx]),
            "out_of_sample_rank": float(ranks[winner_idx]),
            "out_of_sample_rank_fraction": omega,
            "logit_rank": logit,
        })

    logits_array = np.asarray(logits, dtype=float)
    pbo = float(np.mean(logits_array <= 0.0))
    detail_df = pd.DataFrame(details)
    return {
        "pbo": pbo,
        "n_partitions": int(len(details)),
        "n_splits": int(n_splits),
        "n_observations": int(len(clean)),
        "n_strategies": int(n_strategies),
        "median_logit_rank": float(np.median(logits_array)),
        "partition_details": detail_df,
    }
