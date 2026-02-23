from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from .data import Experiment
from .simulation import Simulator
from .metrics import Metrics, MetricsResult

from pathlib import Path
import torch

from dtcalib.deep_learning.model import RCInverseCNN


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
            #verbose=2              # Add a lot of information but to much in the terminal. Mayby on log after
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


class BayesianMAPCalibrator(LeastSquaresCalibrator):
    """
    Nonlinear Bayesian MAP calibration.

    We assume:
      - Measurement model: y = sim(t,u;theta) + eps,  eps ~ N(0, sigma_y^2)
      - Prior on parameters: theta ~ N(prior_mean, diag(prior_std^2))

    MAP estimate:
      theta_hat = argmin_theta [ (1/(2*sigma_y^2)) * sum ||y - sim(...)||^2
                                + (1/2) * sum ||(theta - prior_mean)/prior_std||^2 ]

    Implementation trick:
      least_squares minimizes 0.5*sum(r(theta)^2).
      So we concatenate:
        r_data  = (y - yhat)/sigma_y
        r_prior = (theta - prior_mean)/prior_std
    """

    def __init__(
        self,
        simulator: Simulator,
        *,
        prior_mean: np.ndarray,
        prior_std: np.ndarray,
        sigma_y: float = 1.0,
        method: str = "trf",
        loss: str = "linear",
        f_scale: float = 1.0,
    ) -> None:
        super().__init__(simulator, method=method, loss=loss, f_scale=f_scale)

        self._prior_mean = np.asarray(prior_mean, dtype=float)
        self._prior_std = np.asarray(prior_std, dtype=float)

        if self._prior_mean.ndim != 1 or self._prior_std.ndim != 1:
            raise ValueError("prior_mean and prior_std must be 1D arrays.")
        if self._prior_mean.shape != self._prior_std.shape:
            raise ValueError("prior_mean and prior_std must have the same shape.")
        if np.any(self._prior_std <= 0):
            raise ValueError("prior_std must be strictly positive.")

        self._sigma_y = float(sigma_y)
        if self._sigma_y <= 0:
            raise ValueError("sigma_y must be strictly positive.")

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

        if theta0.ndim != 1:
            raise ValueError("theta0 must be a 1D array.")
        if theta0.shape != self._prior_mean.shape:
            raise ValueError(
                f"theta0 shape {theta0.shape} must match prior_mean shape {self._prior_mean.shape}."
            )

        if weights is not None and len(weights) != len(experiments):
            raise ValueError("weights must match number of experiments.")
        w = np.ones(len(experiments), dtype=float) if weights is None else np.asarray(weights, dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            res_parts: List[np.ndarray] = []

            # Data residuals (whitened by sigma_y, weighted per experiment)
            for i, exp in enumerate(experiments):
                sim_out = self._sim.simulate(exp.t, exp.u, theta).y
                r = (exp.y - sim_out) / self._sigma_y
                res_parts.append(np.sqrt(w[i]) * r)

            # Prior residuals (MAP): (theta - mu)/sigma_prior
            r_prior = (theta - self._prior_mean) / self._prior_std
            res_parts.append(r_prior)

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
            # verbose=2
        )

        # Build per-experiment diagnostics (same as LeastSquaresCalibrator)
        per_metrics: List[tuple[str, MetricsResult]] = []
        for exp in experiments:
            yhat = self._sim.simulate(exp.t, exp.u, result.x).y
            per_metrics.append((exp.name, Metrics.compute(exp.y, yhat)))

        return CalibrationReport(
            theta_hat=result.x.astype(float),
            cost=float(result.cost),
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            per_experiment_metrics=per_metrics,
        )






# -----------------------------------------------------------
# ------------- Deep learning Calibration -------------------
# -----------------------------------------------------------

@dataclass(frozen=True)
class NormalizationStats:
    x_mean: torch.Tensor  # shape [2]
    x_std: torch.Tensor   # shape [2]
    y_mean: torch.Tensor  # scalar
    y_std: torch.Tensor   # scalar


class RCNeuralCalibrator:
    """
    Neural inverse calibrator:
      input  : Vin(t), Vout(t)
      output : C_hat

    It encapsulates:
      - model weights
      - normalization stats (train-derived)
      - de-normalization of the output
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stats: NormalizationStats,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        # Move stats to device for fast inference
        self.stats = NormalizationStats(
            x_mean=stats.x_mean.to(self.device),
            x_std=stats.x_std.to(self.device),
            y_mean=stats.y_mean.to(self.device),
            y_std=stats.y_std.to(self.device),
        )

    @staticmethod
    def load(checkpoint_path: Union[str, Path], device: Optional[torch.device] = None) -> "RCNeuralCalibrator":
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        model = RCInverseCNN()
        model.load_state_dict(ckpt["model_state_dict"])

        stats = NormalizationStats(
            x_mean=ckpt["x_mean"].float(),
            x_std=ckpt["x_std"].float(),
            y_mean=ckpt["y_mean"].float(),
            y_std=ckpt["y_std"].float(),
        )

        return RCNeuralCalibrator(model=model, stats=stats, device=device)

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [2, T] or [B, 2, T]
        """
        if x.ndim == 2:
            return (x - self.stats.x_mean[:, None]) / self.stats.x_std[:, None]
        if x.ndim == 3:
            return (x - self.stats.x_mean[None, :, None]) / self.stats.x_std[None, :, None]
        raise ValueError(f"Expected x with shape [2,T] or [B,2,T], got {tuple(x.shape)}")

    def _denormalize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        y_norm: [B] or scalar tensor
        """
        return y_norm * self.stats.y_std + self.stats.y_mean

    def predict(self, vin: Union[np.ndarray, torch.Tensor], vout: Union[np.ndarray, torch.Tensor]) -> float:
        """
        vin, vout: arrays of shape [T]
        returns: C_hat in Farads
        """
        if isinstance(vin, np.ndarray):
            vin_t = torch.tensor(vin, dtype=torch.float32)
        else:
            vin_t = vin.float()

        if isinstance(vout, np.ndarray):
            vout_t = torch.tensor(vout, dtype=torch.float32)
        else:
            vout_t = vout.float()

        if vin_t.ndim != 1 or vout_t.ndim != 1:
            raise ValueError("vin and vout must be 1D arrays (shape [T]).")
        if vin_t.shape[0] != vout_t.shape[0]:
            raise ValueError("vin and vout must have the same length T.")

        x = torch.stack([vin_t, vout_t], dim=0).to(self.device)  # [2, T]
        x = self._normalize_x(x)
        x = x.unsqueeze(0)  # [1, 2, T]

        with torch.no_grad():
            y_norm = self.model(x)          # [1]
            y = self._denormalize_y(y_norm) # [1]

        return float(y.item())
    
    def calibrate(
        self,
        experiments: Sequence[Experiment],
        *,
        aggregate: str = "mean",   # "mean" or "median"
    ) -> CalibrationReport:
        """
        Neural calibration by inference.

        For each experiment:
            - Predict C_i = f_NN(Vin_i, Vout_i)

        Then:
            - Aggregate into a global C_hat (mean or median)
            - Compute empirical variance as a proxy for confidence

        Parameters
        ----------
        experiments : list of Experiment
            Experiments containing (t, u, y)

        aggregate : str
            Aggregation strategy across experiments:
                - "mean"   : arithmetic mean
                - "median" : robust median

        Returns
        -------
        CalibrationReport
            theta_hat  : aggregated C estimate
            cost       : empirical variance across C_i
            success    : always True (no optimization involved)
            message    : description of aggregation
            nfev       : 0 (no function evaluations)
            per_experiment_metrics :
                Here we store predicted C_i and deviation from global C_hat
        """

        if len(experiments) == 0:
            raise ValueError("Need at least one experiment.")

        C_predictions: List[float] = []

        # ---- 1. Predict C for each experiment ----
        for exp in experiments:
            C_i = self.predict(exp.u, exp.y)
            C_predictions.append(C_i)

        C_array = np.asarray(C_predictions, dtype=float)

        # ---- 2. Aggregate ----
        if aggregate == "mean":
            C_hat = float(np.mean(C_array))
        elif aggregate == "median":
            C_hat = float(np.median(C_array))
        else:
            raise ValueError("aggregate must be 'mean' or 'median'.")

        # Empirical variance across experiments
        empirical_variance = float(np.var(C_array))

        # ---- 3. Build per-experiment diagnostics ----
        per_metrics: List[tuple[str, MetricsResult]] = []

        for exp, C_i in zip(experiments, C_array):
            # deviation from global estimate
            deviation = float(abs(C_i - C_hat))

            # We store deviation in a MetricsResult-like structure
            # (we reuse the MetricsResult container for consistency)
            dummy_metrics = MetricsResult(
                rmse=deviation,
                nmse=deviation,
                mse=deviation,
            )

            per_metrics.append((exp.name, dummy_metrics))

        # ---- 4. Return CalibrationReport ----
        return CalibrationReport(
            theta_hat=np.array([C_hat], dtype=float),
            cost=empirical_variance,
            success=True,
            message=f"Neural inference with {aggregate} aggregation over {len(experiments)} experiments.",
            nfev=0,
            per_experiment_metrics=per_metrics,
        )