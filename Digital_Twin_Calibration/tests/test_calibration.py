from __future__ import annotations

import numpy as np
import pytest

from dtcalib.data import Experiment
from dtcalib.calibration import (
    LeastSquaresCalibrator,
    BayesianMAPCalibrator,
    GeneticAlgorithmCalibrator,
    ParticleSwarmCalibrator,
)
from dtcalib.simulation import ExampleRCCircuitSimulator, LowPassR1CR2Simulator


# ============================================================
# Helpers
# ============================================================

def _generate_tau_experiment(tau: float, n: int = 200) -> Experiment:
    t = np.linspace(0.0, 1.0, n)
    u = np.sin(2 * np.pi * 1.0 * t)

    sim = ExampleRCCircuitSimulator(use_tau=True)
    y = sim.simulate(t, u, theta=np.array([tau], dtype=float)).y

    return Experiment(name="synthetic_tau", t=t, u=u, y=y, meta={})


def _generate_lowpass_experiment(
    *,
    R1: float = 10_000.0,
    R2: float = 10_000.0,
    C: float = 1e-6,
    n: int = 1000,
) -> Experiment:
    t = np.linspace(0.0, 0.05, n)
    u = np.sin(2 * np.pi * 100.0 * t)

    sim = LowPassR1CR2Simulator(
        calibrated_params=("R1", "R2", "C"),
        fixed_params={},
        y0_mode="dc_from_u0",
    )

    y = sim.simulate(
        t,
        u,
        theta=np.array([R1, R2, C], dtype=float),
    ).y

    return Experiment(name="synthetic_lowpass", t=t, u=u, y=y, meta={})


# ============================================================
# Least Squares
# ============================================================

def test_least_squares_calibration_recovers_true_tau():
    true_tau = 0.15
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
    )

    assert report.success
    assert report.theta_hat.shape == (1,)
    assert report.theta_hat[0] == pytest.approx(true_tau, rel=1e-2)


def test_least_squares_recovers_lowpass_capacitance():
    true_C = 1e-6
    exp = _generate_lowpass_experiment(C=true_C)

    simulator = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
        y0_mode="dc_from_u0",
    )

    calibrator = LeastSquaresCalibrator(simulator)

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([3e-6], dtype=float),
        bounds=(np.array([1e-9], dtype=float), np.array([1e-2], dtype=float)),
    )

    assert report.success
    assert report.theta_hat.shape == (1,)
    assert report.theta_hat[0] == pytest.approx(true_C, rel=1e-2)


def test_least_squares_recovers_lowpass_r2_and_c():
    true_R2 = 10_000.0
    true_C = 1e-6
    exp = _generate_lowpass_experiment(R2=true_R2, C=true_C)

    simulator = LowPassR1CR2Simulator(
        calibrated_params=("R2", "C"),
        fixed_params={"R1": 10_000.0},
        y0_mode="dc_from_u0",
    )

    calibrator = LeastSquaresCalibrator(simulator)

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([5_000.0, 3e-6], dtype=float),
        bounds=(
            np.array([1e2, 1e-9], dtype=float),
            np.array([1e7, 1e-2], dtype=float),
        ),
        max_nfev=5000,
    )

    assert report.success
    assert report.theta_hat.shape == (2,)
    assert report.theta_hat[0] == pytest.approx(true_R2, rel=5e-2)
    assert report.theta_hat[1] == pytest.approx(true_C, rel=5e-2)


def test_least_squares_weights_length_mismatch_raises():
    exp = _generate_tau_experiment(0.1)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    with pytest.raises(ValueError, match="weights must match"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1], dtype=float),
            weights=[1.0, 2.0],
        )


def test_least_squares_no_experiments_raises():
    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    with pytest.raises(ValueError, match="Need at least one experiment"):
        calibrator.calibrate(
            experiments=[],
            theta0=np.array([0.1], dtype=float),
        )


def test_least_squares_bounds_enforce_positive_tau():
    true_tau = 0.2
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([0.12], dtype=float),
        bounds=(np.array([0.1], dtype=float), np.array([0.3], dtype=float)),
    )

    assert report.success
    assert 0.1 <= report.theta_hat[0] <= 0.3


def test_least_squares_report_contains_metrics():
    exp = _generate_tau_experiment(0.15)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = LeastSquaresCalibrator(simulator)

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
    )

    assert isinstance(report.cost, float)
    assert isinstance(report.nfev, int)
    assert len(report.per_experiment_metrics) == 1
    assert report.per_experiment_metrics[0][0] == "synthetic_tau"


# ============================================================
# Bayesian MAP
# ============================================================

def test_map_matches_least_squares_when_prior_is_weak():
    true_tau = 0.15
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)

    ls_cal = LeastSquaresCalibrator(simulator)
    map_cal = BayesianMAPCalibrator(
        simulator,
        prior_mean=np.array([0.0], dtype=float),
        prior_std=np.array([1e6], dtype=float),
        sigma_y=1.0,
    )

    ls_report = ls_cal.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
    )

    map_report = map_cal.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
    )

    assert map_report.success
    assert map_report.theta_hat[0] == pytest.approx(ls_report.theta_hat[0], rel=1e-3)


def test_strong_prior_influences_solution():
    true_tau = 0.2
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)

    strong_prior_mean = np.array([0.05], dtype=float)
    strong_prior_std = np.array([1e-4], dtype=float)

    map_cal = BayesianMAPCalibrator(
        simulator,
        prior_mean=strong_prior_mean,
        prior_std=strong_prior_std,
        sigma_y=1.0,
    )

    report = map_cal.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
    )

    assert report.success
    assert abs(report.theta_hat[0] - strong_prior_mean[0]) < 0.01


def test_prior_dimension_mismatch_raises():
    simulator = ExampleRCCircuitSimulator(use_tau=True)

    with pytest.raises(ValueError):
        BayesianMAPCalibrator(
            simulator,
            prior_mean=np.array([0.1], dtype=float),
            prior_std=np.array([0.1, 0.2], dtype=float),
        )


def test_map_theta0_dimension_mismatch_raises():
    exp = _generate_tau_experiment(0.2)
    simulator = ExampleRCCircuitSimulator(use_tau=True)

    calibrator = BayesianMAPCalibrator(
        simulator,
        prior_mean=np.array([0.1], dtype=float),
        prior_std=np.array([0.1], dtype=float),
    )

    with pytest.raises(ValueError, match="theta0 shape"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1, 0.2], dtype=float),
            bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        )


# ============================================================
# Genetic Algorithm
# ============================================================

def test_ga_calibration_recovers_true_tau():
    true_tau = 0.15
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = GeneticAlgorithmCalibrator(
        simulator,
        population_size=40,
        n_generations=60,
        seed=42,
        polish=True,
    )

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        max_nfev=4000,
    )

    assert report.success
    assert report.theta_hat.shape == (1,)
    assert report.theta_hat[0] == pytest.approx(true_tau, rel=5e-2)


def test_ga_recovers_lowpass_capacitance():
    true_C = 1e-6
    exp = _generate_lowpass_experiment(C=true_C)

    simulator = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
        y0_mode="dc_from_u0",
    )

    calibrator = GeneticAlgorithmCalibrator(
        simulator,
        population_size=30,
        n_generations=40,
        seed=42,
        polish=True,
    )

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([3e-6], dtype=float),
        bounds=(np.array([1e-9], dtype=float), np.array([1e-2], dtype=float)),
        max_nfev=3000,
    )

    assert report.success
    assert report.theta_hat[0] == pytest.approx(true_C, rel=1e-1)


def test_ga_no_experiments_raises():
    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = GeneticAlgorithmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="Need at least one experiment"):
        calibrator.calibrate(
            experiments=[],
            theta0=np.array([0.1], dtype=float),
            bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        )


def test_ga_weights_length_mismatch_raises():
    exp = _generate_tau_experiment(0.1)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = GeneticAlgorithmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="weights must match"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1], dtype=float),
            bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
            weights=[1.0, 2.0],
        )


def test_ga_requires_explicit_bounds():
    exp = _generate_tau_experiment(0.2)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = GeneticAlgorithmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="requires explicit bounds"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1], dtype=float),
            bounds=None,
        )


def test_ga_requires_theta0_inside_bounds():
    exp = _generate_tau_experiment(0.2)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = GeneticAlgorithmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="theta0 must lie inside bounds"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.05], dtype=float),
            bounds=(np.array([0.1], dtype=float), np.array([0.3], dtype=float)),
        )


def test_ga_requires_strictly_positive_lower_bounds():
    exp = _generate_tau_experiment(0.2)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = GeneticAlgorithmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="strictly positive lower bounds"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1], dtype=float),
            bounds=(np.array([0.0], dtype=float), np.array([1.0], dtype=float)),
        )


def test_ga_negative_weights_raise():
    exp = _generate_tau_experiment(0.2)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = GeneticAlgorithmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="weights must be non-negative"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1], dtype=float),
            bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
            weights=[-1.0],
        )


def test_ga_is_reproducible_with_fixed_seed():
    true_tau = 0.18
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)

    cal1 = GeneticAlgorithmCalibrator(
        simulator,
        population_size=30,
        n_generations=40,
        seed=123,
        polish=False,
    )
    cal2 = GeneticAlgorithmCalibrator(
        simulator,
        population_size=30,
        n_generations=40,
        seed=123,
        polish=False,
    )

    report1 = cal1.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        max_nfev=3000,
    )
    report2 = cal2.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        max_nfev=3000,
    )

    assert report1.theta_hat[0] == pytest.approx(report2.theta_hat[0], rel=1e-12, abs=1e-12)


# ============================================================
# Particle Swarm Optimization
# ============================================================

def test_pso_calibration_recovers_true_tau():
    true_tau = 0.15
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = ParticleSwarmCalibrator(
        simulator,
        swarm_size=30,
        n_iterations=60,
        seed=42,
        polish=True,
    )

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        max_nfev=4000,
    )

    assert report.success
    assert report.theta_hat.shape == (1,)
    assert report.theta_hat[0] == pytest.approx(true_tau, rel=5e-2)


def test_pso_recovers_lowpass_capacitance():
    true_C = 1e-6
    exp = _generate_lowpass_experiment(C=true_C)

    simulator = LowPassR1CR2Simulator(
        calibrated_params=("C",),
        fixed_params={"R1": 10_000.0, "R2": 10_000.0},
        y0_mode="dc_from_u0",
    )

    calibrator = ParticleSwarmCalibrator(
        simulator,
        swarm_size=30,
        n_iterations=50,
        seed=42,
        polish=True,
    )

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([3e-6], dtype=float),
        bounds=(np.array([1e-9], dtype=float), np.array([1e-2], dtype=float)),
        max_nfev=3000,
    )

    assert report.success
    assert report.theta_hat[0] == pytest.approx(true_C, rel=1e-1)


def test_pso_no_experiments_raises():
    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = ParticleSwarmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="Need at least one experiment"):
        calibrator.calibrate(
            experiments=[],
            theta0=np.array([0.1], dtype=float),
            bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        )


def test_pso_weights_length_mismatch_raises():
    exp = _generate_tau_experiment(0.1)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = ParticleSwarmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="weights must match"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1], dtype=float),
            bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
            weights=[1.0, 2.0],
        )


def test_pso_requires_finite_bounds():
    exp = _generate_tau_experiment(0.2)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = ParticleSwarmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="requires finite bounds"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.1], dtype=float),
            bounds=None,
        )


def test_pso_raises_if_theta0_outside_bounds():
    exp = _generate_tau_experiment(0.2)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = ParticleSwarmCalibrator(simulator, seed=42)

    with pytest.raises(ValueError, match="Initial guess is outside of provided bounds"):
        calibrator.calibrate(
            experiments=[exp],
            theta0=np.array([0.05], dtype=float),
            bounds=(np.array([0.1], dtype=float), np.array([0.3], dtype=float)),
        )


def test_pso_is_reproducible_with_fixed_seed():
    true_tau = 0.18
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)

    cal1 = ParticleSwarmCalibrator(
        simulator,
        swarm_size=30,
        n_iterations=40,
        seed=123,
        polish=False,
    )
    cal2 = ParticleSwarmCalibrator(
        simulator,
        swarm_size=30,
        n_iterations=40,
        seed=123,
        polish=False,
    )

    report1 = cal1.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        max_nfev=3000,
    )
    report2 = cal2.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        max_nfev=3000,
    )

    assert report1.success
    assert report2.success
    assert report1.theta_hat[0] == pytest.approx(report2.theta_hat[0], rel=1e-12, abs=1e-12)


def test_pso_report_contains_metrics():
    true_tau = 0.15
    exp = _generate_tau_experiment(true_tau)

    simulator = ExampleRCCircuitSimulator(use_tau=True)
    calibrator = ParticleSwarmCalibrator(
        simulator,
        swarm_size=30,
        n_iterations=40,
        seed=42,
        polish=True,
    )

    report = calibrator.calibrate(
        experiments=[exp],
        theta0=np.array([0.05], dtype=float),
        bounds=(np.array([0.01], dtype=float), np.array([1.0], dtype=float)),
        max_nfev=3000,
    )

    assert report.success
    assert isinstance(report.cost, float)
    assert isinstance(report.nfev, int)
    assert len(report.per_experiment_metrics) == 1
    assert report.per_experiment_metrics[0][0] == "synthetic_tau"