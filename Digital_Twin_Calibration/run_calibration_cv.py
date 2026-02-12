from __future__ import annotations

from pathlib import Path
import numpy as np

from dtcalib.data import ExperimentsDataset
from dtcalib.simulation import ExampleRCCircuitSimulator
from dtcalib.calibration import LeastSquaresCalibrator
from dtcalib.validation import LeaveOneExperimentOutCV


def main() -> None:
    data_folder = Path("data/LP_Dataset_csv_C_Modified")  
    ds = ExperimentsDataset.from_csv_folder(data_folder)

    # ADD test visu log 
    y0 = ds[0].y
    print("y stats: min=", float(y0.min()), "max=", float(y0.max()), "std=", float(y0.std()))
    print("u stats: min=", float(ds[0].u.min()), "max=", float(ds[0].u.max()), "std=", float(ds[0].u.std()))

    simulator = ExampleRCCircuitSimulator(use_tau=True)

    calibrator = LeastSquaresCalibrator(
        simulator,
        method="trf",
        loss="linear",
    )

    cv = LeaveOneExperimentOutCV(simulator, calibrator)

    theta0 = np.array([0.1], dtype=float)  # initial guess (tau ici)
    bounds = (np.array([1e-6]), np.array([1e6]))  # tau > 0

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
