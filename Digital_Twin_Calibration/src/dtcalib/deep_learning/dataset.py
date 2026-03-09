import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import re
import numpy as np
from dtcalib.deep_learning.splits_utils import parse_samples


class RCSignalDataset(Dataset):
    """
    Torch Dataset for RC inverse learning.
    Expects structure:

    root/
        dataset_+c_1p003em06/
            exp_.../
                results_....csv
    """
    def __init__(self, root_dir: Path, target_transform: str = "C"):
        root_dir = Path(root_dir)

        parsed = parse_samples(root_dir)
        self.samples = [(Path(csv_path), C_value) for csv_path, C_value in parsed]         

        # Target transform mode
        # "C"   -> y = C
        # "logC"-> y = ln(C)
        self.target_transform = "C"
        self.set_target_transform(target_transform)

        # Normalization parameters (set later)
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # Cache to put the dataset 
        self.cache = {}
    
    # ----------------------------------------------------
    # Target transform
    # ----------------------------------------------------
    def set_target_transform(self, mode: str):
        mode = str(mode)
        if mode not in ("C", "logC"):
            raise ValueError(f"Unknown target_transform={mode}. Use 'C' or 'logC'.")
        self.target_transform = mode

    def _transform_y(self, y_value: float) -> float:
        # C_value is strictly positive (as you confirmed)
        if self.target_transform == "logC":
            return float(np.log(y_value))  # natural log
        return float(y_value)

    # ----------------------------------------------------
    # Compute normalization stats (ATTENTION : call only on train set)
    # ----------------------------------------------------
    def compute_normalization(self, indices=None):
        if len(self.samples) == 0:
            raise ValueError(
                "RCSignalDataset has 0 samples. Check folder naming (C_...) and CSV location."
            )

        xs = []
        ys = []

        iter_samples = self.samples if indices is None else [self.samples[i] for i in indices]
        for csv_path, C_value in iter_samples:
            df = pd.read_csv(csv_path)

            time = df.iloc[:, 0].values
            Vin  = df.iloc[:, 1].values
            Vout = df.iloc[:, 2].values

            x = np.stack([time, Vin, Vout], axis=0)
            xs.append(x)
            ys.append(self._transform_y(C_value))

        xs = np.concatenate(xs, axis=1)  # concat along time
        ys = np.array(ys)

        self.x_mean = torch.tensor(xs.mean(axis=1), dtype=torch.float32)
        self.x_std = torch.tensor(xs.std(axis=1) + 1e-8, dtype=torch.float32)

        self.y_mean = torch.tensor(ys.mean(), dtype=torch.float32)
        self.y_std = torch.tensor(ys.std() + 1e-8, dtype=torch.float32)

    # ----------------------------------------------------
    # Set normalization stats (used for val/test)
    # ----------------------------------------------------
    def set_normalization(self, x_mean, x_std, y_mean, y_std):
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, C_value = self.samples[idx]

        if csv_path not in self.cache:
            df = pd.read_csv(csv_path)
            time = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32)
            Vin  = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
            Vout = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)
            self.cache[csv_path] = (time, Vin, Vout)

        time, Vin, Vout = self.cache[csv_path]

        x = torch.stack([time, Vin, Vout], dim=0)  # [3, T]
        y_value = self._transform_y(C_value)
        y = torch.tensor(y_value, dtype=torch.float32)
        
        # Apply normalization if available
        if self.x_mean is not None:
            x = (x - self.x_mean[:, None]) / self.x_std[:, None]

        if self.y_mean is not None:
            y = (y - self.y_mean) / self.y_std

        return x, y

