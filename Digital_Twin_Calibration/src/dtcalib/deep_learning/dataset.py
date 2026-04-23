from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Optional, Any
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TargetSpec:
    calibrated_params: tuple[str, ...]
    transform_map: Dict[str, str]  # ex: {"R1": "log", "R2": "log", "C": "log"}

    def __post_init__(self):
        if len(self.calibrated_params) == 0:
            raise ValueError("calibrated_params cannot be empty.")
        if len(set(self.calibrated_params)) != len(self.calibrated_params):
            raise ValueError("calibrated_params must not contain duplicates.")

        for p in self.calibrated_params:
            mode = self.transform_map.get(p, "identity")
            if mode not in {"identity", "log"}:
                raise ValueError(f"Unsupported transform '{mode}' for parameter '{p}'.")

    @property
    def output_dim(self) -> int:
        return len(self.calibrated_params)

    def transform_vector(self, param_dict: Dict[str, float]) -> np.ndarray:
        values = []
        for p in self.calibrated_params:
            if p not in param_dict:
                raise KeyError(f"Missing parameter '{p}' in param_dict.")
            v = float(param_dict[p])
            mode = self.transform_map.get(p, "identity")
            if mode == "log":
                if v <= 0:
                    raise ValueError(f"Parameter '{p}' must be > 0 for log transform.")
                v = np.log(v)
            values.append(v)
        return np.asarray(values, dtype=np.float32)

    def inverse_transform_vector(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (self.output_dim,):
            raise ValueError(f"Expected shape {(self.output_dim,)}, got {y.shape}.")
        out = []
        for i, p in enumerate(self.calibrated_params):
            v = float(y[i])
            mode = self.transform_map.get(p, "identity")
            if mode == "log":
                v = float(np.exp(v))
            out.append(v)
        return np.asarray(out, dtype=np.float32)

    def to_dict(self, values: np.ndarray) -> Dict[str, float]:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.output_dim,):
            raise ValueError(f"Expected shape {(self.output_dim,)}, got {values.shape}.")
        return {p: float(values[i]) for i, p in enumerate(self.calibrated_params)}

class RCSignalDataset(Dataset):
    """
    Dataset based on a manifest.csv file.

    Expected manifest columns:
      group_name, experiment_name, csv_path, R1, R2, C, fc, freq,
      amplitude, phase, offset, n_periods, samples_per_period, n_samples

    Returns:
      x: [3, T]   = [time, input, output]
      y: [d]      = transformed target vector following TargetSpec
    """

    def __init__(
        self,
        root_dir: Path | str,
        *,
        target_spec: TargetSpec,
        manifest_name: str = "manifest.csv",
    ):
        self.root_dir = Path(root_dir)
        self.target_spec = target_spec
        self.manifest_path = self.root_dir / manifest_name

        self.samples: list[tuple[Path, Dict[str, float]]] = []
        self.cache: Dict[Path, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self.x_mean: Optional[torch.Tensor] = None   # [3]
        self.x_std: Optional[torch.Tensor] = None    # [3]
        self.y_mean: Optional[torch.Tensor] = None   # [d]
        self.y_std: Optional[torch.Tensor] = None    # [d]

        self._build_index_from_manifest()

    def _build_index_from_manifest(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)

        required_cols = {"csv_path", "R1", "R2", "C"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )

        for _, row in df.iterrows():
            csv_rel = Path(str(row["csv_path"]))
            csv_path = self.root_dir / csv_rel

            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file listed in manifest does not exist: {csv_path}")

            param_dict = {
                "R1": float(row["R1"]),
                "R2": float(row["R2"]),
                "C": float(row["C"]),
            }

            self.samples.append((csv_path, param_dict))

        if len(self.samples) == 0:
            raise ValueError("No samples found in manifest.")

    def __len__(self) -> int:
        return len(self.samples)

    def set_normalization(
        self,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> None:
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def compute_normalization(self, indices: Optional[Sequence[int]] = None) -> None:
        xs = []
        ys = []

        idxs = range(len(self.samples)) if indices is None else indices
        for idx in idxs:
            csv_path, param_dict = self.samples[idx]
            df = pd.read_csv(csv_path)

            # Expect first 3 columns to be time, input, output
            time = df.iloc[:, 0].values
            vin = df.iloc[:, 1].values
            vout = df.iloc[:, 2].values

            x = np.stack([time, vin, vout], axis=0)            # [3, T]
            y = self.target_spec.transform_vector(param_dict)  # [d]

            xs.append(x)
            ys.append(y)

        xs = np.concatenate(xs, axis=1)   # [3, total_T]
        ys = np.stack(ys, axis=0)         # [N, d]

        self.x_mean = torch.tensor(xs.mean(axis=1), dtype=torch.float32)
        self.x_std = torch.tensor(xs.std(axis=1) + 1e-8, dtype=torch.float32)

        self.y_mean = torch.tensor(ys.mean(axis=0), dtype=torch.float32)
        self.y_std = torch.tensor(ys.std(axis=0) + 1e-8, dtype=torch.float32)

    def __getitem__(self, idx: int):
        csv_path, param_dict = self.samples[idx]

        if csv_path not in self.cache:
            df = pd.read_csv(csv_path)

            time = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32)
            vin = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
            vout = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)

            self.cache[csv_path] = (time, vin, vout)

        time, vin, vout = self.cache[csv_path]
        x = torch.stack([time, vin, vout], dim=0)  # [3, T]

        y_np = self.target_spec.transform_vector(param_dict)
        y = torch.tensor(y_np, dtype=torch.float32)  # [d]

        if self.x_mean is not None:
            x = (x - self.x_mean[:, None]) / self.x_std[:, None]

        if self.y_mean is not None:
            y = (y - self.y_mean) / self.y_std

        return x, y