"""Reproducible synthetic experiments for demonstrating selection bias."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import annualized_sharpe


def simulate_null_returns(
    n_periods: int = 756,
    n_strategies: int = 30,
    seed: int = 2026,
    annual_volatility: float = 0.16,
    common_factor_correlation: float = 0.25,
) -> pd.DataFrame:
    """Generate correlated candidate returns with zero expected alpha.

    These are synthetic daily returns, not observed market data. A shared noise
    component gives candidate strategies modest cross-correlation; the design
    intentionally assigns no strategy a positive expected return.
    """
    if n_periods < 20:
        raise ValueError("n_periods must be at least 20.")
    if n_strategies < 2:
        raise ValueError("n_strategies must be at least 2.")
    if annual_volatility <= 0:
        raise ValueError("annual_volatility must be positive.")
    if not 0.0 <= common_factor_correlation < 1.0:
        raise ValueError("common_factor_correlation must be in [0, 1).")

    rng = np.random.default_rng(seed)
    daily_sigma = annual_volatility / np.sqrt(252.0)
    common = rng.normal(0.0, daily_sigma, size=(n_periods, 1))
    idiosyncratic = rng.normal(0.0, daily_sigma, size=(n_periods, n_strategies))
    returns = (
        np.sqrt(common_factor_correlation) * common
        + np.sqrt(1.0 - common_factor_correlation) * idiosyncratic
    )
    columns = [f"candidate_{i:03d}" for i in range(1, n_strategies + 1)]
    return pd.DataFrame(returns, columns=columns)


def run_selection_bias_experiment(
    candidate_counts: tuple[int, ...] = (5, 10, 25, 50, 100),
    repetitions: int = 100,
    n_train: int = 504,
    n_test: int = 252,
    seed: int = 2026,
) -> pd.DataFrame:
    """Show how picking the best in-sample result can inflate apparent Sharpe.

    Each repetition simulates a pool of zero-alpha candidates, selects the top
    in-sample Sharpe from increasing trial counts, and measures the same
    selected candidate over a later, untouched test segment.
    """
    counts = tuple(sorted(set(int(c) for c in candidate_counts)))
    if not counts or counts[0] < 2:
        raise ValueError("candidate_counts must contain integers of at least 2.")
    if repetitions < 10:
        raise ValueError("Use at least 10 repetitions for a stable demo summary.")
    if n_train < 30 or n_test < 30:
        raise ValueError("n_train and n_test must both be at least 30.")

    rng = np.random.default_rng(seed)
    max_candidates = max(counts)
    records: list[dict[str, float | int]] = []
    for rep in range(repetitions):
        # One common factor plus idiosyncratic noise creates correlated candidates.
        total_n = n_train + n_test
        sigma = 0.16 / np.sqrt(252.0)
        common = rng.normal(0.0, sigma, size=(total_n, 1))
        idio = rng.normal(0.0, sigma, size=(total_n, max_candidates))
        pool = np.sqrt(0.25) * common + np.sqrt(0.75) * idio
        # Randomize candidate order so smaller trial counts are not always the same columns.
        pool = pool[:, rng.permutation(max_candidates)]
        train, test = pool[:n_train], pool[n_train:]
        train_mean = train.mean(axis=0)
        train_std = train.std(axis=0, ddof=1)
        train_sharpes = np.divide(
            train_mean, train_std, out=np.full_like(train_mean, np.nan), where=train_std > 0
        ) * np.sqrt(252.0)
        test_mean = test.mean(axis=0)
        test_std = test.std(axis=0, ddof=1)
        test_sharpes = np.divide(
            test_mean, test_std, out=np.full_like(test_mean, np.nan), where=test_std > 0
        ) * np.sqrt(252.0)

        for count in counts:
            best_idx = int(np.nanargmax(train_sharpes[:count]))
            records.append({
                "repetition": rep,
                "candidate_count": count,
                "selected_in_sample_sharpe": float(train_sharpes[best_idx]),
                "selected_out_of_sample_sharpe": float(test_sharpes[best_idx]),
                "selected_out_of_sample_return": float(np.prod(1.0 + test[:, best_idx]) - 1.0),
            })

    raw = pd.DataFrame(records)
    summary = raw.groupby("candidate_count", as_index=False).agg(
        median_in_sample_sharpe=("selected_in_sample_sharpe", "median"),
        median_out_of_sample_sharpe=("selected_out_of_sample_sharpe", "median"),
        share_positive_out_of_sample_sharpe=(
            "selected_out_of_sample_sharpe", lambda x: float(np.mean(np.asarray(x) > 0.0))
        ),
        median_out_of_sample_return=("selected_out_of_sample_return", "median"),
    )
    summary["share_positive_out_of_sample_sharpe"] *= 100.0
    return summary
