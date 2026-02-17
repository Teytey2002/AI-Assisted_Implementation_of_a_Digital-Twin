from __future__ import annotations

import numpy as np
import pytest

from dtcalib.data import Experiment
from dtcalib.calibration import LeastSquaresCalibrator, BayesianMAPCalibrator
from dtcalib.simulation import ExampleRCCircuitSimulator


# ------------------------------------------------------------
# Helper: generate synthetic experiment with known tau
# ------------------------------------------------------------

def _generate_experiment(tau: float, n: int = 200) -> Experiment:
    t = np.linspace(0.0, 1.0, n)
    u = np.sin(2 * np.pi * 1.0 * t)  # sinus input

    sim = ExampleRCCircuitSimulator(use_tau=True)
    y = sim.simulate(t, u, theta=np.array([tau])).y

    return Experiment(
        name="synthetic",
        t=t,
        u=u,
        y=y,
        meta={}
    )


# ------------------------------------------------------------
# Test: recover known tau
# ------------------------------------------------------------

def test_calibration_recovers_true_tau():
    true_tau = 0.15
    exp = _generate_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([0.05]),   # wrong initial guess
        bounds=(np.array([0.01]), np.array([1.0]))
    )

    assert report.success
    assert report.theta_hat.shape == (1,)
    assert report.theta_hat[0] == pytest.approx(true_tau, rel=1e-2)


# ------------------------------------------------------------
# Test: weights mismatch error
# ------------------------------------------------------------

def test_weights_length_mismatch_raises():
    exp = _generate_experiment(0.1)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    with pytest.raises(ValueError, match="weights must match"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1]),
            weights=[1.0, 2.0]
        )


# ------------------------------------------------------------
# Test: error if no experiments
# ------------------------------------------------------------

def test_no_experiments_raises():
    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    with pytest.raises(ValueError, match="Need at least one experiment"):
        calibrator.calibrate(
            experiments=[],
            theta0=np.array([0.1])
        )


# ------------------------------------------------------------
# Test: bounds enforce positivity
# ------------------------------------------------------------

def test_bounds_enforce_positive_tau():
    true_tau = 0.2
    exp = _generate_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([0.12]),
        bounds=(np.array([0.1]), np.array([0.3]))
    )

    assert report.theta_hat[0] >= 0.1
    assert report.theta_hat[0] <= 0.3
    
    
def test_bounds_raise_if_theta0_outside():
    exp = _generate_experiment(0.2)
    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    with pytest.raises(ValueError, match="x0.*infeasible"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.05]),
            bounds=(np.array([0.1]), np.array([0.3])),
        )



# ------------------------------------------------------------
# Test 1: MAP ≈ Least Squares if prior is very weak
# ------------------------------------------------------------

def test_map_matches_least_squares_when_prior_is_weak():
    true_tau = 0.15
    exp = _generate_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)

    ls_cal = LeastSquaresCalibrator(simulator)
    map_cal = BayesianMAPCalibrator(
        simulator,
        prior_mean=np.array([0.0]),
        prior_std=np.array([1e6]),  # extremely weak prior
        sigma_y=1.0
    )

    ls_report = ls_cal.calibrate(
        experiments=[exp],
        theta0=np.array([0.05]),
        bounds=(np.array([0.01]), np.array([1.0]))
    )

    map_report = map_cal.calibrate(
        experiments=[exp],
        theta0=np.array([0.05]),
        bounds=(np.array([0.01]), np.array([1.0]))
    )

    assert map_report.success
    assert map_report.theta_hat[0] == pytest.approx(
        ls_report.theta_hat[0],
        rel=1e-3
    )


# ------------------------------------------------------------
# Test 2: Strong prior pulls solution toward prior_mean
# ------------------------------------------------------------

def test_strong_prior_influences_solution():
    true_tau = 0.2
    exp = _generate_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)

    strong_prior_mean = np.array([0.05])
    strong_prior_std = np.array([1e-4])  # very strong prior

    map_cal = BayesianMAPCalibrator(
        simulator,
        prior_mean=strong_prior_mean,
        prior_std=strong_prior_std,
        sigma_y=1.0
    )

    report = map_cal.calibrate(
        experiments=[exp],
        theta0=np.array([0.05]),
        bounds=(np.array([0.01]), np.array([1.0]))
    )

    # Solution should be closer to prior than true value
    assert abs(report.theta_hat[0] - strong_prior_mean[0]) < 0.01


# ------------------------------------------------------------
# Test 3: prior dimension mismatch raises
# ------------------------------------------------------------

def test_prior_dimension_mismatch_raises():
    simulator = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError):
        BayesianMAPCalibrator(
            simulator,
            prior_mean=np.array([0.1]),
            prior_std=np.array([0.1, 0.2])  # wrong size
        )
