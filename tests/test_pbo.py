import numpy as np
import pandas as pd

from quant_audit.pbo import compute_pbo
from quant_audit.simulation import simulate_null_returns, run_selection_bias_experiment


def test_pbo_returns_valid_probability_and_partitions():
    returns = simulate_null_returns(n_periods=160, n_strategies=6, seed=3)
    result = compute_pbo(returns, n_splits=2)
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["n_partitions"] == 6
    assert len(result["partition_details"]) == 6


def test_simulator_is_reproducible():
    first = simulate_null_returns(n_periods=80, n_strategies=4, seed=12)
    second = simulate_null_returns(n_periods=80, n_strategies=4, seed=12)
    pd.testing.assert_frame_equal(first, second)


def test_selection_bias_summary_has_expected_candidate_counts():
    summary = run_selection_bias_experiment(
        candidate_counts=(3, 8), repetitions=10, n_train=60, n_test=60, seed=4
    )
    assert summary["candidate_count"].tolist() == [3, 8]
    assert np.all(np.isfinite(summary["median_in_sample_sharpe"]))
