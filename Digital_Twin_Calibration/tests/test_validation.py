from __future__ import annotations

import numpy as np
import pytest

from dtcalib.data import Experiment, ExperimentsDataset
from dtcalib.validation import LeaveOneExperimentOutCV, CrossValidationResult, FoldResult
from dtcalib.simulation import Simulator, SimulationResult
from dtcalib.calibration import CalibrationReport
from dtcalib.metrics import MetricsResult


# ------------------------------------------------------------------
# Fake simulator: yhat = theta[0] * u
# ------------------------------------------------------------------

class FakeSimulator(Simulator):
    def simulate(
        self,
        t: np.ndarray,
        u: np.ndarray,
        theta: np.ndarray,
    ) -> SimulationResult:
        y = float(theta[0]) * u
        return SimulationResult(y=y, aux={})


# ------------------------------------------------------------------
# Fake calibrator: always returns theta = [2.0]
# ------------------------------------------------------------------

class FakeCalibrator:
    def calibrate(self, experiments, *, theta0, bounds=None, max_nfev=None):
        return CalibrationReport(
            theta_hat=np.array([2.0], dtype=float),
            cost=0.0,
            success=True,
            message="mock",
            nfev=1,
            per_experiment_metrics=[],
        )


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _make_dataset(n_exps: int = 3) -> ExperimentsDataset:
    experiments = []

    for i in range(n_exps):
        t = np.linspace(0.0, 1.0, 10)
        u = np.ones_like(t)
        y = 2.0 * u

        experiments.append(
            Experiment(
                name=f"exp_{i}",
                t=t,
                u=u,
                y=y,
                meta={},
            )
        )

    return ExperimentsDataset(experiments)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_number_of_folds_equals_number_of_experiments():
    dataset = _make_dataset(4)

    cv = LeaveOneExperimentOutCV(
        simulator=FakeSimulator(),
        calibrator=FakeCalibrator(),
    )

    result = cv.run(dataset, theta0=np.array([1.0], dtype=float))

    assert len(result.folds) == 4


def test_each_experiment_is_held_out_once():
    dataset = _make_dataset(3)

    cv = LeaveOneExperimentOutCV(
        simulator=FakeSimulator(),
        calibrator=FakeCalibrator(),
    )

    result = cv.run(dataset, theta0=np.array([1.0], dtype=float))

    held_out_names = [fold.held_out for fold in result.folds]

    assert set(held_out_names) == {"exp_0", "exp_1", "exp_2"}


def test_fold_contains_expected_fields():
    dataset = _make_dataset(2)

    cv = LeaveOneExperimentOutCV(
        simulator=FakeSimulator(),
        calibrator=FakeCalibrator(),
    )

    result = cv.run(dataset, theta0=np.array([1.0], dtype=float))
    fold = result.folds[0]

    assert isinstance(fold.held_out, str)
    assert isinstance(fold.theta_hat, np.ndarray)
    assert isinstance(fold.train_report, CalibrationReport)
    assert isinstance(fold.test_metrics, MetricsResult)

    assert fold.theta_hat.shape == (1,)
    assert fold.theta_hat[0] == pytest.approx(2.0)


def test_metrics_are_zero_for_perfect_model():
    dataset = _make_dataset(3)

    cv = LeaveOneExperimentOutCV(
        simulator=FakeSimulator(),
        calibrator=FakeCalibrator(),
    )

    result = cv.run(dataset, theta0=np.array([1.0], dtype=float))

    for fold in result.folds:
        assert fold.test_metrics.rmse == pytest.approx(0.0)
        assert fold.test_metrics.mse == pytest.approx(0.0)
        assert fold.test_metrics.nmse == pytest.approx(0.0)


def test_summary_computes_statistics():
    dataset = _make_dataset(3)

    cv = LeaveOneExperimentOutCV(
        simulator=FakeSimulator(),
        calibrator=FakeCalibrator(),
    )

    result = cv.run(dataset, theta0=np.array([1.0], dtype=float))

    summary = result.summary()

    assert set(summary.keys()) == {
        "rmse_mean",
        "rmse_std",
        "nmse_mean",
        "nmse_std",
    }

    assert summary["rmse_mean"] == pytest.approx(0.0)
    assert summary["rmse_std"] == pytest.approx(0.0)
    assert summary["nmse_mean"] == pytest.approx(0.0)
    assert summary["nmse_std"] == pytest.approx(0.0)


def test_summary_std_is_zero_with_single_fold():
    fold = FoldResult(
        held_out="exp_0",
        theta_hat=np.array([2.0], dtype=float),
        train_report=CalibrationReport(
            theta_hat=np.array([2.0], dtype=float),
            cost=0.0,
            success=True,
            message="mock",
            nfev=1,
            per_experiment_metrics=[],
        ),
        test_metrics=MetricsResult(
            rmse=1.0,
            nmse=2.0,
            mse=1.0,
        ),
    )

    result = CrossValidationResult(folds=[fold])
    summary = result.summary()

    assert summary["rmse_mean"] == pytest.approx(1.0)
    assert summary["rmse_std"] == pytest.approx(0.0)
    assert summary["nmse_mean"] == pytest.approx(2.0)
    assert summary["nmse_std"] == pytest.approx(0.0)