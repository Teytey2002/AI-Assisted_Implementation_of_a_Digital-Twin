from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Experiment, ExperimentsDataset
from .calibration import LeastSquaresCalibrator, CalibrationReport
from .metrics import Metrics, MetricsResult
from .simulation import Simulator
from tqdm import tqdm


@dataclass(frozen=True)
class FoldResult:
    held_out: str                       # Name of tested experiment 
    theta_hat: np.ndarray               # param estimated on the train
    train_report: CalibrationReport     
    test_metrics: MetricsResult


@dataclass(frozen=True)
class CrossValidationResult:
    folds: List[FoldResult]

    def summary(self) -> Dict[str, float]:
        rmses = np.array([f.test_metrics.rmse for f in self.folds], dtype=float)
        nmses = np.array([f.test_metrics.nmse for f in self.folds], dtype=float)
        return {
            "rmse_mean": float(rmses.mean()),
            "rmse_std": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
            "nmse_mean": float(nmses.mean()),
            "nmse_std": float(nmses.std(ddof=1)) if len(nmses) > 1 else 0.0,
        }


class LeaveOneExperimentOutCV:
    """
    Cross-validation: hold out one experiment, calibrate on the others, evaluate on the held-out.
    """

    def __init__(self, simulator: Simulator, calibrator: LeastSquaresCalibrator) -> None:
        self._sim = simulator
        self._cal = calibrator

    def run(
        self,
        dataset: ExperimentsDataset,
        *,
        theta0: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_nfev: Optional[int] = None,
    ) -> CrossValidationResult:
        folds: List[FoldResult] = []

        for i, test_exp in enumerate(tqdm(dataset.experiments, desc="CV folds")):
            train_exps = [e for j, e in enumerate(dataset.experiments) if j != i]

            train_report = self._cal.calibrate(
                train_exps,
                theta0=theta0,
                bounds=bounds,
                max_nfev=max_nfev,
            )

            yhat_test = self._sim.simulate(test_exp.t, test_exp.u, train_report.theta_hat).y
            test_metrics = Metrics.compute(test_exp.y, yhat_test)

            folds.append(
                FoldResult(
                    held_out=test_exp.name,
                    theta_hat=train_report.theta_hat,
                    train_report=train_report,
                    test_metrics=test_metrics,
                )
            )

        return CrossValidationResult(folds=folds)
