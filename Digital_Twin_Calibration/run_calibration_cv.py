from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from dtcalib.data import ExperimentsDataset
from dtcalib.simulation import (
    ExampleRCCircuitSimulator,
    LowPassR1CR2Simulator,
    ThreeStageRCLadderSimulator,
)
from dtcalib.calibration import (
    LeastSquaresCalibrator,
    BayesianMAPCalibrator,
    GeneticAlgorithmCalibrator,
    ParticleSwarmCalibrator,
)
from dtcalib.validation import LeaveOneExperimentOutCV


# ---------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------
SCENARIOS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "lowpass_r1cr2": {
        "c_only": {
            "calibrated_params": ("C",),
            "fixed_params": {"R1": 10_000.0, "R2": 10_000.0},
            "theta0": np.array([3e-6], dtype=float),
            "bounds": (
                np.array([1e-9], dtype=float),
                np.array([1e-2], dtype=float),
            ),
            "prior_mean": np.array([1.0032e-6], dtype=float),
            "prior_std": np.array([5e-7], dtype=float),
        },
        "r2_c": {
            "calibrated_params": ("R2", "C"),
            "fixed_params": {"R1": 10_000.0},
            "theta0": np.array([5_000.0, 3e-6], dtype=float),
            "bounds": (
                np.array([1e2, 1e-9], dtype=float),
                np.array([1e7, 1e-2], dtype=float),
            ),
            "prior_mean": np.array([10_000.0, 1e-6], dtype=float),
            "prior_std": np.array([1_000.0, 5e-7], dtype=float),
        },
        "r1_r2_c": {
            "calibrated_params": ("R1", "R2", "C"),
            "fixed_params": {},
            "theta0": np.array([3_000.0, 15_000.0, 3e-6], dtype=float),
            "bounds": (
                np.array([1e2, 1e2, 1e-9], dtype=float),
                np.array([1e7, 1e7, 1e-2], dtype=float),
            ),
            "prior_mean": np.array([10_000.0, 10_000.0, 1e-6], dtype=float),
            "prior_std": np.array([1_000.0, 1_000.0, 5e-7], dtype=float),
        },
    },

    "three_stage_rc": {
        "caps_only": {
            "calibrated_params": ("C1", "C2", "C3"),
            "fixed_params": {
                "R1": 10.0,
                "R2": 47.5,
                "R3": 22.1,
                "R4": 15.0,
                "R5": 33.2,
                "R6": 68.1,
                "R7": 100.0,
            },
            "theta0": np.array([0.5e-6, 5e-6, 8e-6], dtype=float),
            "bounds": (
                np.array([1e-9, 1e-9, 1e-9], dtype=float),
                np.array([1e-2, 1e-2, 1e-2], dtype=float),
            ),
            "prior_mean": np.array([1e-6, 10e-6, 15e-6], dtype=float),
            "prior_std": np.array([0.5e-6, 5e-6, 7.5e-6], dtype=float),
        },
        "resistors_only": {
            "calibrated_params": ("R1", "R2", "R3", "R4", "R5", "R6", "R7"),
            "fixed_params": {
                "C1": 1e-6,
                "C2": 10e-6,
                "C3": 15e-6,
            },
            "theta0": np.array([8.0, 40.0, 20.0, 12.0, 30.0, 60.0, 90.0], dtype=float),
            "bounds": (
                np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3], dtype=float),
                np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6], dtype=float),
            ),
            "prior_mean": np.array([10.0, 47.5, 22.1, 15.0, 33.2, 68.1, 100.0], dtype=float),
            "prior_std": np.array([2.0, 9.5, 4.4, 3.0, 6.6, 13.6, 20.0], dtype=float),
        },
        "all_components": {
            "calibrated_params": ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "C1", "C2", "C3"),
            "fixed_params": {},
            "theta0": np.array([8.0, 40.0, 20.0, 12.0, 30.0, 60.0, 90.0, 0.5e-6, 5e-6, 8e-6], dtype=float),
            "bounds": (
                np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-9, 1e-9, 1e-9], dtype=float),
                np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e-2, 1e-2, 1e-2], dtype=float),
            ),
            "prior_mean": np.array([10.0, 47.5, 22.1, 15.0, 33.2, 68.1, 100.0, 1e-6, 10e-6, 15e-6], dtype=float),
            "prior_std": np.array([2.0, 9.5, 4.4, 3.0, 6.6, 13.6, 20.0, 0.5e-6, 5e-6, 7.5e-6], dtype=float),
        },
    },
}


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------
def get_scenario_config(simulator_name: str, scenario_name: str) -> Dict[str, Any]:
    if simulator_name not in SCENARIOS:
        raise ValueError(
            f"Unknown simulator '{simulator_name}'. "
            f"Available simulators: {list(SCENARIOS.keys())}"
        )

    simulator_scenarios = SCENARIOS[simulator_name]

    if scenario_name not in simulator_scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario_name}' for simulator '{simulator_name}'. "
            f"Available scenarios: {list(simulator_scenarios.keys())}"
        )

    return simulator_scenarios[scenario_name]


def build_simulator(simulator_name: str, config: Dict[str, Any], y0_mode: str):
    if simulator_name == "example_rc":
        return ExampleRCCircuitSimulator(use_tau=True)

    if simulator_name == "lowpass_r1cr2":
        return LowPassR1CR2Simulator(
            calibrated_params=config["calibrated_params"],
            fixed_params=config["fixed_params"],
            y0_mode=y0_mode,
        )

    if simulator_name == "three_stage_rc":
        return ThreeStageRCLadderSimulator(
            calibrated_params=config["calibrated_params"],
            fixed_params=config["fixed_params"],
            y0_mode=y0_mode,
        )

    raise ValueError(f"Unsupported simulator: {simulator_name}")


def build_calibrator(calibrator_name: str, simulator, config: Dict[str, Any], args):
    if calibrator_name == "ls":
        return LeastSquaresCalibrator(
            simulator=simulator,
            method=args.local_method,
            loss=args.loss,
            f_scale=args.f_scale,
        )

    if calibrator_name == "map":
        return BayesianMAPCalibrator(
            simulator=simulator,
            prior_mean=config["prior_mean"],
            prior_std=config["prior_std"],
            sigma_y=args.sigma_y,
            method=args.local_method,
            loss=args.loss,
            f_scale=args.f_scale,
        )

    if calibrator_name == "ga":
        return GeneticAlgorithmCalibrator(
            simulator=simulator,
            population_size=args.population_size,
            n_generations=args.n_generations,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            mutation_scale=args.mutation_scale,
            elite_fraction=args.elite_fraction,
            init_near_theta0_fraction=args.init_near_theta0_fraction,
            init_near_theta0_scale=args.init_near_theta0_scale,
            mutation_mode=args.mutation_mode,
            seed=args.seed,
            polish=args.polish,
            polish_method=args.local_method,
            polish_loss=args.loss,
            polish_f_scale=args.f_scale,
        )

    if calibrator_name == "pso":
        return ParticleSwarmCalibrator(
            simulator=simulator,
            swarm_size=args.swarm_size,
            n_iterations=args.n_iterations,
            inertia=args.inertia,
            cognitive=args.cognitive,
            social=args.social,
            seed=args.seed,
            polish=args.polish,
        )

    raise ValueError(f"Unsupported calibrator: {calibrator_name}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-experiment-out calibration cross-validation."
    )

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        choices=["lowpass_r1cr2", "three_stage_rc"],
    )
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument(
        "--calibrator",
        type=str,
        required=True,
        choices=["ls", "map", "ga", "pso"],
    )

    parser.add_argument("--y0-mode", type=str, default="zero")
    parser.add_argument("--max-nfev", type=int, default=5000)

    parser.add_argument("--local-method", type=str, default="trf")
    parser.add_argument("--loss", type=str, default="linear")
    parser.add_argument("--f-scale", type=float, default=1.0)
    parser.add_argument("--sigma-y", type=float, default=1.0)

    parser.add_argument("--population-size", type=int, default=80)
    parser.add_argument("--n-generations", type=int, default=120)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--mutation-scale", type=float, default=0.15)
    parser.add_argument("--elite-fraction", type=float, default=0.1)
    parser.add_argument("--init-near-theta0-fraction", type=float, default=0.5)
    parser.add_argument("--init-near-theta0-scale", type=float, default=0.25)
    parser.add_argument("--mutation-mode", type=str, default="log", choices=["log", "relative"])

    parser.add_argument("--swarm-size", type=int, default=40)
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--inertia", type=float, default=0.7)
    parser.add_argument("--cognitive", type=float, default=1.5)
    parser.add_argument("--social", type=float, default=1.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--polish", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    data_folder = Path(args.dataset)
    ds = ExperimentsDataset.from_csv_folder(data_folder)

    y0 = ds[0].y
    print("y stats: min=", float(y0.min()), "max=", float(y0.max()), "std=", float(y0.std()))
    print("u stats: min=", float(ds[0].u.min()), "max=", float(ds[0].u.max()), "std=", float(ds[0].u.std()))

    config = get_scenario_config(args.simulator, args.scenario)

    print("\nCalibration setup")
    print("dataset           =", data_folder)
    print("simulator         =", args.simulator)
    print("scenario          =", args.scenario)
    print("calibrator        =", args.calibrator)
    print("calibrated_params =", config["calibrated_params"])
    print("fixed_params      =", config["fixed_params"])
    print("theta0            =", config["theta0"])
    print("bounds            =", config["bounds"])
    print("prior_mean        =", config["prior_mean"])
    print("prior_std         =", config["prior_std"])
    print("y0_mode           =", args.y0_mode)

    simulator = build_simulator(args.simulator, config, args.y0_mode)
    calibrator = build_calibrator(args.calibrator, simulator, config, args)

    cv = LeaveOneExperimentOutCV(simulator, calibrator)

    cv_result = cv.run(
        ds,
        theta0=config["theta0"],
        bounds=config["bounds"],
        max_nfev=args.max_nfev,
    )

    print("\nCV summary:", cv_result.summary())

    print("\nFirst folds:")
    for fold in cv_result.folds[:5]:
        print(
            f"[held-out={fold.held_out}] "
            f"theta_hat={fold.theta_hat} "
            f"rmse={fold.test_metrics.rmse:.6g} "
            f"nmse={fold.test_metrics.nmse:.6g}"
        )


if __name__ == "__main__":
    main()