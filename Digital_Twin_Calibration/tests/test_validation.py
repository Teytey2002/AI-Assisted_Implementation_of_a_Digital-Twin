from __future__ import annotations

import numpy as np
import pytest

from dtcalib.data import Experiment, ExperimentsDataset
from dtcalib.validation import LeaveOneExperimentOutCV
from dtcalib.metrics import Metrics
from dtcalib.simulation import Simulator, SimulationResult
from dtcalib.calibration import CalibrationReport


# ------------------------------------------------------------------
# Fake simulator: yhat = theta[0] * u
# ------------------------------------------------------------------

class FakeSimulator(Simulator):
    def simulate(self, t: np.ndarray, u: np.ndarray, theta: np.ndarray) -> SimulationResult:
        y = theta[0] * u
        return SimulationResult(y=y, aux={})


# ------------------------------------------------------------------
# Fake calibrator: always returns theta = [2.0]
# ------------------------------------------------------------------

class FakeCalibrator:
    def calibrate(self, experiments, *, theta0, bounds=None, max_nfev=None):
        # per_experiment_metrics must exist in CalibrationReport
        return CalibrationReport(
            theta_hat=np.array([2.0], dtype=float),
            cost=0.0,
            success=True,
            message="mock",
            nfev=1,
            per_experiment_metrics=[],
        )



# ------------------------------------------------------------------
# Helper to create small dataset
# ------------------------------------------------------------------

def _make_dataset(n_exps: int = 3) -> ExperimentsDataset:
    experiments = []

    for i in range(n_exps):
        t = np.linspace(0, 1, 10)
        u = np.ones_like(t)
        y = 2.0 * u  # perfect model for theta=2.0

        experiments.append(
            Experiment(
                name=f"exp_{i}",
                t=t,
                u=u,
                y=y,
                meta={}
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
        calibrator=FakeCalibrator()
    )

    result = cv.run(dataset, theta0=np.array([1.0]))

    assert len(result.folds) == 4


def test_each_experiment_is_held_out_once():
    dataset = _make_dataset(3)

    cv = LeaveOneExperimentOutCV(
        simulator=FakeSimulator(),
        calibrator=FakeCalibrator()
    )

    result = cv.run(dataset, theta0=np.array([1.0]))

    held_out_names = [fold.held_out for fold in result.folds]

    assert set(held_out_names) == {"exp_0", "exp_1", "exp_2"}


def test_metrics_are_zero_for_perfect_model():
    dataset = _make_dataset(3)

    cv = LeaveOneExperimentOutCV(
        simulator=FakeSimulator(),
        calibrator=FakeCalibrator()
    )

    result = cv.run(dataset, theta0=np.array([1.0]))

    for fold in result.folds:
        assert fold.test_metrics.rmse == pytest.approx(0.0)
        assert fold.test_metrics.mse == pytest.approx(0.0)


def test_summary_computes_statistics():
    dataset = _make_dataset(3)

    cv = LeaveOneExperimentOutCV(
        simulator=FakeSimulator(),
        calibrator=FakeCalibrator()
    )

    result = cv.run(dataset, theta0=np.array([1.0]))

    summary = result.summary()

    assert "rmse_mean" in summary
    assert "rmse_std" in summary
    assert "nmse_mean" in summary
    assert "nmse_std" in summary

    assert summary["rmse_mean"] == pytest.approx(0.0)
