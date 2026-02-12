from __future__ import annotations

import numpy as np
import pytest

from dtcalib.metrics import Metrics


def test_mse_zero_when_identical() -> None:
    y = np.array([0.0, 1.0, -2.0, 3.5], dtype=float)
    assert Metrics.mse(y, y) == 0.0


def test_rmse_zero_when_identical() -> None:
    y = np.array([10.0, 10.0, 10.0], dtype=float)
    assert Metrics.rmse(y, y) == 0.0


def test_mse_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([2.0, 2.0, 2.0], dtype=float)
    # errors: [-1, 0, 1] -> squared: [1, 0, 1] -> mean = 2/3
    expected = 2.0 / 3.0
    assert Metrics.mse(y_true, y_pred) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_rmse_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([2.0, 2.0, 2.0], dtype=float)
    # mse = 2/3 -> rmse = sqrt(2/3)
    expected = np.sqrt(2.0 / 3.0)
    assert Metrics.rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_nmse_matches_definition() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=1000)
    y_pred = y_true + 0.1  # constant bias

    mse = Metrics.mse(y_true, y_pred)
    var = float(np.var(y_true))
    expected = mse / var

    assert Metrics.nmse(y_true, y_pred) == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_nmse_is_finite_for_constant_signal_due_to_eps() -> None:
    y_true = np.ones(50, dtype=float) * 3.0  # variance = 0
    y_pred = np.ones(50, dtype=float) * 4.0  # mse = 1

    # With eps guard, nmse should be mse / eps and be finite
    nmse = Metrics.nmse(y_true, y_pred, eps=1e-6)
    assert np.isfinite(nmse)
    assert nmse == pytest.approx(1.0 / 1e-6, rel=1e-12, abs=0.0)


def test_compute_returns_consistent_values() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([0.0, 1.5, 1.5, 3.0], dtype=float)

    res = Metrics.compute(y_true, y_pred)

    # Sanity: res.mse and res.rmse consistent
    assert res.rmse == pytest.approx(np.sqrt(res.mse), rel=1e-12, abs=0.0)
    # nmse consistent with definition (with eps default)
    expected_nmse = res.mse / max(float(np.var(y_true)), 1e-12)
    assert res.nmse == pytest.approx(expected_nmse, rel=1e-12, abs=0.0)


def test_raises_on_shape_mismatch() -> None:
    y_true = np.array([0.0, 1.0], dtype=float)
    y_pred = np.array([0.0, 1.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="Shape mismatch"):
        Metrics.mse(y_true, y_pred)


def test_raises_on_non_1d_arrays() -> None:
    y_true = np.zeros((10, 1), dtype=float)
    y_pred = np.zeros((10, 1), dtype=float)

    with pytest.raises(ValueError, match="Expected 1D arrays"):
        Metrics.rmse(y_true, y_pred)


def test_raises_on_empty_arrays() -> None:
    y_true = np.array([], dtype=float)
    y_pred = np.array([], dtype=float)

    with pytest.raises(ValueError, match="Empty arrays"):
        Metrics.nmse(y_true, y_pred)
