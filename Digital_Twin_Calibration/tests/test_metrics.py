from __future__ import annotations

import numpy as np
import pytest

from dtcalib.metrics import Metrics


# ============================================================
# Basic deterministic metrics
# ============================================================

def test_mse_zero_when_identical() -> None:
    y = np.array([0.0, 1.0, -2.0, 3.5], dtype=float)
    assert Metrics.mse(y, y) == 0.0


def test_rmse_zero_when_identical() -> None:
    y = np.array([10.0, 10.0, 10.0], dtype=float)
    assert Metrics.rmse(y, y) == 0.0


def test_mse_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([2.0, 2.0, 2.0], dtype=float)

    assert Metrics.mse(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_rmse_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([2.0, 2.0, 2.0], dtype=float)

    assert Metrics.rmse(y_true, y_pred) == pytest.approx(np.sqrt(2.0 / 3.0))


def test_nmse_matches_definition() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=1000)
    y_pred = y_true + 0.1

    expected = Metrics.mse(y_true, y_pred) / float(np.var(y_true))

    assert Metrics.nmse(y_true, y_pred) == pytest.approx(expected)


def test_nmse_is_finite_for_constant_signal_due_to_eps() -> None:
    y_true = np.ones(50, dtype=float) * 3.0
    y_pred = np.ones(50, dtype=float) * 4.0

    nmse = Metrics.nmse(y_true, y_pred, eps=1e-6)

    assert np.isfinite(nmse)
    assert nmse == pytest.approx(1.0 / 1e-6)


def test_compute_returns_consistent_values() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([0.0, 1.5, 1.5, 3.0], dtype=float)

    res = Metrics.compute(y_true, y_pred)

    assert res.mse == pytest.approx(Metrics.mse(y_true, y_pred))
    assert res.rmse == pytest.approx(np.sqrt(res.mse))
    assert res.nmse == pytest.approx(Metrics.nmse(y_true, y_pred))


def test_mae_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([2.0, 2.0, 1.0], dtype=float)

    # abs errors = [1, 0, 2] -> mean = 1
    assert Metrics.mae(y_true, y_pred) == pytest.approx(1.0)


def test_mape_percent_known_value() -> None:
    y_true = np.array([100.0, 200.0], dtype=float)
    y_pred = np.array([110.0, 180.0], dtype=float)

    # relative errors = [10%, 10%]
    assert Metrics.mape_percent(y_true, y_pred) == pytest.approx(10.0)


def test_mape_percent_handles_zero_target_with_eps() -> None:
    y_true = np.array([0.0], dtype=float)
    y_pred = np.array([1.0], dtype=float)

    value = Metrics.mape_percent(y_true, y_pred, eps=1e-6)

    assert np.isfinite(value)
    assert value == pytest.approx(100.0 / 1e-6)


# ============================================================
# Correlation
# ============================================================

def test_safe_corrcoef_perfect_positive_correlation() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=float)
    b = np.array([2.0, 4.0, 6.0], dtype=float)

    assert Metrics.safe_corrcoef(a, b) == pytest.approx(1.0)


def test_safe_corrcoef_perfect_negative_correlation() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=float)
    b = np.array([3.0, 2.0, 1.0], dtype=float)

    assert Metrics.safe_corrcoef(a, b) == pytest.approx(-1.0)


def test_safe_corrcoef_returns_nan_for_constant_signal() -> None:
    a = np.ones(10, dtype=float)
    b = np.arange(10, dtype=float)

    assert np.isnan(Metrics.safe_corrcoef(a, b))


def test_safe_corrcoef_returns_nan_for_too_few_samples() -> None:
    a = np.array([1.0], dtype=float)
    b = np.array([2.0], dtype=float)

    assert np.isnan(Metrics.safe_corrcoef(a, b))


# ============================================================
# Probabilistic metrics
# ============================================================

def test_coverage_from_samples_known_case() -> None:
    y_true = np.array([0.0, 10.0], dtype=float)

    samples = np.array(
        [
            [-1.0, 0.0, 1.0],
            [20.0, 21.0, 22.0],
        ],
        dtype=float,
    )

    coverage = Metrics.coverage_from_samples(
        y_true,
        samples,
        levels=(0.68,),
    )

    assert 0.68 in coverage
    assert coverage[0.68] == pytest.approx(0.5)


def test_mean_interval_width_known_case() -> None:
    samples = np.array(
        [
            [0.0, 1.0, 2.0],
            [10.0, 12.0, 14.0],
        ],
        dtype=float,
    )

    width = Metrics.mean_interval_width(samples, level=1.0)

    # Full interval widths: 2 and 4 -> mean = 3
    assert width == pytest.approx(3.0)


def test_gaussian_nll_matches_manual_formula() -> None:
    y_true = np.array([0.0], dtype=float)
    mu = np.array([0.0], dtype=float)
    std = np.array([1.0], dtype=float)

    expected = 0.5 * np.log(2.0 * np.pi)

    assert Metrics.gaussian_nll(y_true, mu, std) == pytest.approx(expected)


def test_gaussian_nll_is_finite_for_zero_std_due_to_eps() -> None:
    y_true = np.array([0.0], dtype=float)
    mu = np.array([0.0], dtype=float)
    std = np.array([0.0], dtype=float)

    value = Metrics.gaussian_nll(y_true, mu, std, eps=1e-6)

    assert np.isfinite(value)


def test_calibration_error_from_samples_perfect_uniform_like_case() -> None:
    y_true = np.array([0.0, 1.0, 2.0], dtype=float)

    samples = np.array(
        [
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=float,
    )

    ce = Metrics.calibration_error_from_samples(
        y_true,
        samples,
        probs=np.array([1.0 / 3.0, 2.0 / 3.0]),
    )

    assert np.isfinite(ce)
    assert ce >= 0.0


# ============================================================
# Validation errors
# ============================================================

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