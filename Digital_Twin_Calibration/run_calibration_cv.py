from __future__ import annotations

from pathlib import Path
import numpy as np

from dtcalib.data import ExperimentsDataset
from dtcalib.simulation import ExampleRCCircuitSimulator, LowPassR1CR2Simulator
from dtcalib.calibration import LeastSquaresCalibrator, BayesianMAPCalibrator, RCNeuralCalibrator, ParticleSwarmCalibrator
from dtcalib.validation import LeaveOneExperimentOutCV
from dtcalib.calibration import GeneticAlgorithmCalibrator


def main() -> None:
    #data_folder = Path("data/LP_Dataset_csv_C_Modified")  
    data_folder = Path("data/ALL_LP_DATASETS_CSV_Deep_learning/dataset_+c_1p0032em06")  
    ds = ExperimentsDataset.from_csv_folder(data_folder)

    # ADD test visu log 
    y0 = ds[0].y
    print("y stats: min=", float(y0.min()), "max=", float(y0.max()), "std=", float(y0.std()))
    print("u stats: min=", float(ds[0].u.min()), "max=", float(ds[0].u.max()), "std=", float(ds[0].u.std()))

    # ------------------------------------------------------------------
    # Calibration setup
    # Choose one scenario by changing calibrated_params / fixed_params
    # ATTENTION : Make sure to follow th same order of parameters in calibrated_params and initialGuess and bounds --> ("R1", "R2", "C")
    # How to choose the prior mean and std for BayesianMAPCalibrator ?
    #   - prior_mean: can be based on datasheet values, physical intuition --> "True" value
    #   - prior_std: can be based on datasheet tolerances, physical intuition.
    #     If R1 = 10 000 Ohm with 5% tolerance, we can set prior_std for R1 to 500 Ohm.
    #     If C = 1e-6 F with 20% tolerance, we can set prior_std for C to 0.2e-6 F.
    #   here we choose 10% tolerance for resistors and 50% tolerance for capacitor
    # ------------------------------------------------------------------

    # --- Scenario 1: calibrate only C ---
    #calibrated_params = ("C",)
    #fixed_params = {
    #    "R1": 10_000.0,
    #    "R2": 10_000.0,
    #}
    #initialGuess = np.array([3e-6], dtype=float)      # Initial guess for C
    #bounds = (np.array([1e-9], dtype=float), np.array([1e-2], dtype=float))     # Bounds for C
    ## For BayesianMAPCalibrator
    #prior_mean = np.array([1.0032e-6], dtype=float)
    #prior_std  = np.array([5e-7], dtype=float)      # If prior_std too big, the prior is almost flat and we recover the LeastSquaresCalibrator results. 
                                                    # If too small, the prior dominates and we get theta_hat close to prior_mean. So we need to find a good balance.


    # --- Scenario 2: calibrate R2 and C ---
    #calibrated_params = ("R2", "C")
    #fixed_params = {
    #    "R1": 10_000.0,
    #}
    #initialGuess = np.array([5_000.0, 3e-6], dtype=float)
    #bounds = (
    #    np.array([1e2, 1e-9], dtype=float),
    #    np.array([1e7, 1e-2], dtype=float),
    #)
    ## For BayesianMAPCalibrator
    #prior_mean = np.array([10_000.0, 1e-6], dtype=float)
    #prior_std  = np.array([1_000.0, 5e-7], dtype=float)

    # --- Scenario 3: calibrate R1, R2, C ---
    calibrated_params = ("R1", "R2", "C")
    fixed_params = {}
    initialGuess = np.array([3_000.0, 15_000.0, 3e-6], dtype=float)
    bounds = (
        np.array([1e2, 1e2, 1e-9], dtype=float),
        np.array([1e7, 1e7, 1e-2], dtype=float),
    )
    # for BayesianMAPCalibrator
    prior_mean = np.array([10_000.0, 10_000.0, 1e-6], dtype=float)
    prior_std  = np.array([1_000.0, 1_000.0, 5e-7], dtype=float)

    print("\nCalibration setup")
    print("calibrated_params =", calibrated_params)
    print("fixed_params =", fixed_params)
    print("initialGuess =", initialGuess)
    print("bounds =", bounds)
    print("prior_mean =", prior_mean)
    print("prior_std =", prior_std)

    # ------------------------------------------------------------------
    # Chose the simulator 
    # ------------------------------------------------------------------

    #simulator = ExampleRCCircuitSimulator(use_tau=True)    # For unit test
    simulator = LowPassR1CR2Simulator(calibrated_params=calibrated_params, fixed_params=fixed_params, y0_mode="dc_from_u0")


    # ------------------------------------------------------------------
    # Chose the calibrator 
    # ------------------------------------------------------------------

    #calibrator = LeastSquaresCalibrator(
    #    simulator=simulator,
    #    method="trf",
    #    loss="linear",
    #)

    #calibrator = BayesianMAPCalibrator(
    #    simulator=simulator,
    #    prior_mean=prior_mean,
    #    prior_std=prior_std,
    #    sigma_y=1.0,
    #)

    calibrator = GeneticAlgorithmCalibrator(
        simulator=simulator,
        population_size=80,
        n_generations=120,
        crossover_rate=0.9,
        mutation_rate=0.2,
        mutation_scale=0.15,
        elite_fraction=0.1,
        init_near_theta0_fraction=0.5,
        init_near_theta0_scale=0.25,
        mutation_mode="log",
        seed=42,
        polish=True,
        polish_method="trf",
        polish_loss="linear",
        polish_f_scale=1.0,
    )

    #calibrator = ParticleSwarmCalibrator(
    #    simulator,
    #    swarm_size=40,
    #    n_iterations=100,
    #    inertia=0.7,
    #    cognitive=1.5,
    #    social=1.5,
    #    seed=42,
    #    polish=True,
    #)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    cv = LeaveOneExperimentOutCV(simulator, calibrator)
    
    cv_result = cv.run(ds, theta0=initialGuess, bounds=bounds, max_nfev=5000)
    
    print("CV summary:", cv_result.summary())
    for fold in cv_result.folds[:5]:
        print(
            f"[held-out={fold.held_out}] "
            f"theta_hat={fold.theta_hat} "
            f"rmse={fold.test_metrics.rmse:.6g} nmse={fold.test_metrics.nmse:.6g}"
        )


if __name__ == "__main__":
    main()
