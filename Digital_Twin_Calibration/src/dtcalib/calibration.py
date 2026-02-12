from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .data import Experiment
from .simulation import Simulator
from .metrics import Metrics, MetricsResult


@dataclass(frozen=True)
class CalibrationReport:
    theta_hat: np.ndarray       # Parameter estimated
    cost: float
    success: bool               # Info of convergence (from scipy)
    message: str                # Info of convergence (from scipy)
    nfev: int                   # Info of convergence (from scipy)
    per_experiment_metrics: List[tuple[str, MetricsResult]]     # [RMSE, NMSE, MSE]


class LeastSquaresCalibrator:
    """
    Nonlinear least squares parameter calibration:
      theta_hat = argmin_theta sum_i || y_i - sim(t_i,u_i;theta) ||^2

    Uses scipy.optimize.least_squares (Levenberg-Marquardt or trust-region).
    """

    def __init__(
        self,
        simulator: Simulator,
        *,
        method: str = "trf",
        loss: str = "linear",
        f_scale: float = 1.0,
    ) -> None:
        """
        Parameters:
          method: "trf", "dogbox", or "lm" (lm requires unconstrained)
          loss: robust losses supported by scipy ("linear", "soft_l1", "huber", "cauchy", "arctan")
        """
        self._sim = simulator
        self._method = method
        self._loss = loss
        self._f_scale = float(f_scale)

    def calibrate(
        self,
        experiments: Sequence[Experiment],
        *,
        theta0: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        weights: Optional[Sequence[float]] = None,
        max_nfev: Optional[int] = None,
    ) -> CalibrationReport:
        if len(experiments) == 0:
            raise ValueError("Need at least one experiment.")
        theta0 = np.asarray(theta0, dtype=float)

        if weights is not None and len(weights) != len(experiments):
            raise ValueError("weights must match number of experiments.")
        w = np.ones(len(experiments), dtype=float) if weights is None else np.asarray(weights, dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            res_parts: List[np.ndarray] = []
            for i, exp in enumerate(experiments):
                sim_out = self._sim.simulate(exp.t, exp.u, theta).y
                # Weighted residuals: sqrt(w_i) * (y - yhat)
                res_parts.append(np.sqrt(w[i]) * (exp.y - sim_out))
            return np.concatenate(res_parts, axis=0)

        if bounds is None:
            lb = -np.inf * np.ones_like(theta0)
            ub = np.inf * np.ones_like(theta0)
            bounds = (lb, ub)

        result = least_squares(
            residuals,
            theta0,
            bounds=bounds,
            method=self._method,
            loss=self._loss,
            f_scale=self._f_scale,
            max_nfev=max_nfev,
        )

        # Build per-experiment diagnostics
        per_metrics: List[tuple[str, MetricsResult]] = []
        for exp in experiments:
            yhat = self._sim.simulate(exp.t, exp.u, result.x).y
            per_metrics.append((exp.name, Metrics.compute(exp.y, yhat)))

        return CalibrationReport(
            theta_hat=result.x.astype(float),
            cost=float(result.cost),  # 0.5 * sum(residuals**2)
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            per_experiment_metrics=per_metrics,
        )
