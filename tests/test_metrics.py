import numpy as np
import pytest

from quant_audit.metrics import annualized_sharpe, max_drawdown, total_return


def test_total_return_compounds_simple_returns():
    assert total_return([0.10, -0.10]) == pytest.approx(-0.01)


def test_max_drawdown_includes_initial_equity_peak():
    assert max_drawdown([-0.10, 0.05]) == pytest.approx(-0.10)


def test_sharpe_is_nan_for_constant_returns():
    assert np.isnan(annualized_sharpe([0.01] * 20))


def test_invalid_simple_return_is_rejected():
    with pytest.raises(ValueError):
        total_return([0.01, -1.0])
