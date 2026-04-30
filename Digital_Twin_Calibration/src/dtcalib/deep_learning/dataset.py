from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Optional, Any

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
    domain="time":
        1 sample = 1 CSV experiment
        x = [time, Vin, Vout] -> shape [3, T]

    domain="fft":
        1 sample = 1 CSV experiment
        x = [log(freq), log(|H|), phase] for top-k frequencies -> shape [3, K]

    domain="fft_grouped":
        1 sample = 1 physical system, i.e. one group_name
        x = frequency response across all frequencies of this group -> shape [3, N_FREQ]
    """

    def __init__(
        self,
        root_dir: Path | str,
        *,
        target_spec: TargetSpec,
        manifest_name: str = "manifest.csv",
        domain: str = "time",
        fft_top_k: int = 8,
    ):
        self.root_dir = Path(root_dir)
        self.target_spec = target_spec
        self.manifest_path = self.root_dir / manifest_name
        self.domain = domain
        self.fft_top_k = int(fft_top_k)

        if self.domain not in {"time", "fft", "fft_grouped", "time_fft"}:
            raise ValueError("domain must be one of {'time', 'fft', 'fft_grouped', 'time_fft'}.")

        self.samples: list[tuple[Path, Dict[str, float]]] = []
        self.grouped_samples: list[dict[str, Any]] = []

        self.cache: Dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}

        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        self._build_index_from_manifest()

    def _build_index_from_manifest(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)

        required_cols = {"csv_path", "R1", "R2", "C"}
        if self.domain == "fft_grouped":
            required_cols |= {"group_name", "freq"}

        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        if self.domain != "fft_grouped":
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

            return

        # grouped FFT mode
        for group_name, g in df.groupby("group_name", sort=True):
            g = g.sort_values("freq")

            csv_paths = []
            for rel in g["csv_path"].tolist():
                p = self.root_dir / Path(str(rel))
                if not p.exists():
                    raise FileNotFoundError(f"CSV file listed in manifest does not exist: {p}")
                csv_paths.append(p)

            first = g.iloc[0]
            param_dict = {
                "R1": float(first["R1"]),
                "R2": float(first["R2"]),
                "C": float(first["C"]),
            }

            self.grouped_samples.append(
                {
                    "group_name": str(group_name),
                    "csv_paths": csv_paths,
                    "freqs": g["freq"].to_numpy(dtype=np.float64),
                    "param_dict": param_dict,
                }
            )

        if len(self.grouped_samples) == 0:
            raise ValueError("No grouped samples found in manifest.")

    def __len__(self) -> int:
        if self.domain == "fft_grouped":
            return len(self.grouped_samples)
        return len(self.samples)

    def set_normalization(
        self,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> None:
        self.x_mean = x_mean.float()
        self.x_std = x_std.float()
        self.y_mean = y_mean.float()
        self.y_std = y_std.float()

    def compute_normalization(self, indices: Optional[Sequence[int]] = None) -> None:
        xs = []
        ys = []

        idxs = range(len(self)) if indices is None else indices

        for idx in idxs:
            x_np, y_np = self._load_raw_item(int(idx))
            xs.append(x_np)
            ys.append(y_np)

        xs_cat = np.concatenate(xs, axis=1)
        ys_cat = np.stack(ys, axis=0)

        self.x_mean = torch.tensor(xs_cat.mean(axis=1), dtype=torch.float32)
        self.x_std = torch.tensor(xs_cat.std(axis=1) + 1e-8, dtype=torch.float32)

        self.y_mean = torch.tensor(ys_cat.mean(axis=0), dtype=torch.float32)
        self.y_std = torch.tensor(ys_cat.std(axis=0) + 1e-8, dtype=torch.float32)

    def _load_raw_item(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.domain == "fft_grouped":
            sample = self.grouped_samples[idx]
            x = self._build_grouped_fft_features(sample["csv_paths"])
            y = self.target_spec.transform_vector(sample["param_dict"])
            return x, y

        csv_path, param_dict = self.samples[idx]
        df = pd.read_csv(csv_path)

        time = df.iloc[:, 0].to_numpy(dtype=np.float64)
        vin = df.iloc[:, 1].to_numpy(dtype=np.float64)
        vout = df.iloc[:, 2].to_numpy(dtype=np.float64)

        if self.domain == "time":
            x = np.stack([time, vin, vout], axis=0).astype(np.float32)

        elif self.domain == "fft":
            x = self._build_fft_features(time, vin, vout)

        elif self.domain == "time_fft":
            x_time = np.stack([time, vin, vout], axis=0).astype(np.float32)

            x_fft = self._build_fft_features(time, vin, vout)  # [3, K]

            K = x_fft.shape[1]
            T = x_time.shape[1]

            x_fft_interp = np.zeros((3, T), dtype=np.float32)
            grid_fft = np.arange(K)
            grid_time = np.linspace(0, K - 1, T)

            for i in range(3):
                x_fft_interp[i] = np.interp(grid_time, grid_fft, x_fft[i])

            x = np.concatenate([x_time, x_fft_interp], axis=0).astype(np.float32)

        else:
            raise ValueError(f"Unsupported domain: {self.domain}")

        y = self.target_spec.transform_vector(param_dict)
        return x, y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx in self.cache:
            return self.cache[idx]

        x_np, y_np = self._load_raw_item(idx)

        x = torch.tensor(x_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)

        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean[:, None]) / self.x_std[:, None]

        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / self.y_std

        self.cache[idx] = (x, y)
        return x, y

    def _build_fft_features(self, time, vin, vout) -> np.ndarray:
        dt = float(np.mean(np.diff(time)))
        n = len(time)

        freqs = np.fft.rfftfreq(n, d=dt)
        U = np.fft.rfft(vin)
        Y = np.fft.rfft(vout)

        amp_u = np.abs(U).astype(np.float64)
        amp_u[0] = 0.0

        top_k = min(self.fft_top_k, len(amp_u) - 1)
        if top_k <= 0:
            raise ValueError("FFT requires at least one non-DC frequency bin.")

        idxs = np.argsort(amp_u)[-top_k:]
        idxs = idxs[np.argsort(freqs[idxs])]

        H = Y[idxs] / (U[idxs] + 1e-12)

        f = np.log1p(freqs[idxs])
        mag = np.log1p(np.abs(H))
        phase = np.unwrap(np.angle(H))

        return np.stack([f, mag, phase], axis=0).astype(np.float32)

    def _build_grouped_fft_features(self, csv_paths: list[Path]) -> np.ndarray:
        freqs_out = []
        mags_out = []
        phases_out = []

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)

            time = df.iloc[:, 0].to_numpy(dtype=np.float64)
            vin = df.iloc[:, 1].to_numpy(dtype=np.float64)
            vout = df.iloc[:, 2].to_numpy(dtype=np.float64)

            dt = float(np.mean(np.diff(time)))
            n = len(time)

            freqs = np.fft.rfftfreq(n, d=dt)
            U = np.fft.rfft(vin)
            Y = np.fft.rfft(vout)

            amp_u = np.abs(U).astype(np.float64)
            amp_u[0] = 0.0

            idx = int(np.argmax(amp_u))
            f = float(freqs[idx])
            H = Y[idx] / (U[idx] + 1e-12)

            freqs_out.append(np.log1p(f))
            mags_out.append(np.log1p(np.abs(H)))
            phases_out.append(np.angle(H))

        phases = np.unwrap(np.asarray(phases_out, dtype=np.float64))

        x = np.stack(
            [
                np.asarray(freqs_out, dtype=np.float64),
                np.asarray(mags_out, dtype=np.float64),
                phases,
            ],
            axis=0,
        )

        return x.astype(np.float32)