import numpy as np
import pytest

from quant_audit.inference import (
    adjust_pvalues,
    block_bootstrap_sharpe_ci,
    newey_west_mean_test,
    probabilistic_sharpe_ratio,
)


def test_holm_adjustment_is_in_original_order_and_monotone_by_rank():
    raw = np.array([0.04, 0.001, 0.02])
    adjusted = adjust_pvalues(raw, "holm")
    assert adjusted[1] == pytest.approx(0.003)
    assert adjusted[2] == pytest.approx(0.04)
    assert adjusted[0] == pytest.approx(0.04)
    assert np.all(adjusted >= raw)


def test_bh_adjustment_is_bounded():
    adjusted = adjust_pvalues([0.01, 0.02, 0.5], "bh")
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_newey_west_test_returns_valid_fields():
    rng = np.random.default_rng(7)
    result = newey_west_mean_test(rng.normal(0.001, 0.01, 200))
    assert result["n_observations"] == 200
    assert 0 <= result["p_value"] <= 1
    assert result["standard_error"] >= 0


def test_psr_probability_is_bounded():
    rng = np.random.default_rng(8)
    probability = probabilistic_sharpe_ratio(rng.normal(0.0005, 0.01, 250))
    assert 0 <= probability <= 1


def test_block_bootstrap_is_reproducible():
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0005, 0.01, 200)
    first = block_bootstrap_sharpe_ci(returns, n_bootstrap=50, seed=22)
    second = block_bootstrap_sharpe_ci(returns, n_bootstrap=50, seed=22)
    assert first == pytest.approx(second)
    assert first[0] <= first[1]
