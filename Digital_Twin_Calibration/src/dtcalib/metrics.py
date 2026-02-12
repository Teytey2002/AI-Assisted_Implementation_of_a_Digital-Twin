from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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
