from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Dict


@dataclass(frozen=True)
class MetricsResult:
    """
    Example of usage :
    
    res = Metrics.compute(y, yhat)
    res.rmse
    res.nmse
    res.mse

    """
    rmse: float
    nmse: float
    mse: float


class Metrics:
    """
    Metrics for comparing measured vs predicted signals.

    Conventions:
      - y_true, y_pred are 1D arrays of same length
      - rmse in same units as y
      - nmse = mse / var(y_true) (with numerical guard)
    """

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        Metrics._validate_shapes(y_true, y_pred)
        err = y_true - y_pred
        return float(np.mean(err * err))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(Metrics.mse(y_true, y_pred)))

    @staticmethod
    def nmse(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
        Metrics._validate_shapes(y_true, y_pred)
        mse = Metrics.mse(y_true, y_pred)
        var = float(np.var(y_true))
        return float(mse / max(var, eps))

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsResult:
        mse = Metrics.mse(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        nmse = Metrics.nmse(y_true, y_pred)
        return MetricsResult(rmse=rmse, nmse=nmse, mse=mse)

    @staticmethod
    def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError(f"Expected 1D arrays, got shapes {y_true.shape} and {y_pred.shape}")
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
        if y_true.shape[0] == 0:
            raise ValueError("Empty arrays are not valid for metrics.")


    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))


    def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
        return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100.0)


    def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if a.size < 2:
            return float("nan")
        if np.std(a) < 1e-15 or np.std(b) < 1e-15:
            return float("nan")

        return float(np.corrcoef(a, b)[0, 1])


    def coverage_from_samples(
        y_true: np.ndarray,
        samples: np.ndarray,
        *,
        levels: tuple[float, ...] = (0.68, 0.90, 0.95),
    ) -> Dict[float, float]:
        """
        y_true:  [N]
        samples: [N, S]
        """
        y_true = np.asarray(y_true, dtype=float)
        samples = np.asarray(samples, dtype=float)

        out: Dict[float, float] = {}

        for level in levels:
            alpha = 1.0 - level
            lo = np.quantile(samples, alpha / 2.0, axis=1)
            hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)
            covered = (y_true >= lo) & (y_true <= hi)
            out[level] = float(np.mean(covered))

        return out


    def mean_interval_width(
        samples: np.ndarray,
        *,
        level: float = 0.95,
    ) -> float:
        samples = np.asarray(samples, dtype=float)

        alpha = 1.0 - level
        lo = np.quantile(samples, alpha / 2.0, axis=1)
        hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)

        return float(np.mean(hi - lo))

    def gaussian_nll(
        y_true: np.ndarray,
        mu: np.ndarray,
        std: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        y_true = np.asarray(y_true, dtype=float)
        mu = np.asarray(mu, dtype=float)
        std = np.asarray(std, dtype=float)

        var = np.maximum(std ** 2, eps)
        nll = 0.5 * (np.log(2.0 * np.pi * var) + ((y_true - mu) ** 2) / var)

        return float(np.mean(nll))

    def calibration_error_from_samples(
        y_true: np.ndarray,
        samples: np.ndarray,
        *,
        probs: np.ndarray | None = None,
    ) -> float:
        """
        Probability calibration error.

        For each target y_true[i], compute empirical CDF value:
            p_i = P(sample <= y_true[i])

        If the predictive distributions are calibrated, p_i should be uniform.
        """
        y_true = np.asarray(y_true, dtype=float)
        samples = np.asarray(samples, dtype=float)

        if probs is None:
            probs = np.linspace(0.1, 0.9, 9)

        p_values = np.mean(samples <= y_true[:, None], axis=1)

        ce = 0.0
        for p in probs:
            empirical = np.mean(p_values <= p)
            ce += float((empirical - p) ** 2)

        return float(ce)
