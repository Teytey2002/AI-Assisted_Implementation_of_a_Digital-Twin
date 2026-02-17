from __future__ import annotations

from pathlib import Path
import numpy as np

from dtcalib.data import ExperimentsDataset
from dtcalib.simulation import ExampleRCCircuitSimulator, LowPassR1CR2Simulator
from dtcalib.calibration import LeastSquaresCalibrator, BayesianMAPCalibrator
from dtcalib.validation import LeaveOneExperimentOutCV


def main() -> None:
    data_folder = Path("data/LP_Dataset_csv_C_Modified")  
    ds = ExperimentsDataset.from_csv_folder(data_folder)

    # ADD test visu log 
    y0 = ds[0].y
    print("y stats: min=", float(y0.min()), "max=", float(y0.max()), "std=", float(y0.std()))
    print("u stats: min=", float(ds[0].u.min()), "max=", float(ds[0].u.max()), "std=", float(ds[0].u.std()))

    # ---------- Chose the simulator ---------- #
    #simulator = ExampleRCCircuitSimulator(use_tau=True)    # For unit test
    simulator = LowPassR1CR2Simulator(R1=10_000.0, R2=10_000.0, use_C=True, y0_mode="dc_from_u0")

    # ---------- Chose the calibrator ---------- #
    #calibrator = LeastSquaresCalibrator(
    #    simulator,
    #    method="trf",
    #    loss="linear",
    #)

    calibrator = BayesianMAPCalibrator(
        simulator=simulator,
        prior_mean=np.array([5e-7]),
        prior_std=np.array([1.5e-6]),  # prior "large" => proche LS
        sigma_y=1.0,
    )

    cv = LeaveOneExperimentOutCV(simulator, calibrator)

    theta0 = np.array([3e-6])  # initial guess de C
    bounds = (np.array([1e-9]), np.array([1e-2]))  # C entre 1 nF et 10 mF (à adapter)

    cv_result = cv.run(ds, theta0=theta0, bounds=bounds, max_nfev=200)

    print("CV summary:", cv_result.summary())
    for fold in cv_result.folds[:5]:
        print(
            f"[held-out={fold.held_out}] "
            f"theta_hat={fold.theta_hat} "
            f"rmse={fold.test_metrics.rmse:.6g} nmse={fold.test_metrics.nmse:.6g}"
        )


if __name__ == "__main__":
    main()
