import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import re
import numpy as np


class RCSignalDataset(Dataset):
    """
    Torch Dataset for RC inverse learning.
    Expects structure:

    root/
        dataset_C_8e-7/
            exp_.../
                results_....csv
    """

    def __init__(self, root_dir: Path):
        self.samples = []

        root_dir = Path(root_dir)

        for c_folder in root_dir.iterdir():
            if not c_folder.is_dir():
                continue

            # Extract C from folder name
            match = re.search(r"C_(.*)", c_folder.name)
            if match is None:
                continue

            c_str = match.group(1).replace("p", ".")
            C_value = float(c_str)

            for exp_folder in c_folder.iterdir():
                if not exp_folder.is_dir():
                    continue

                for csv_file in exp_folder.glob("*.csv"):
                    self.samples.append((csv_file, C_value))
            
        # Normalization parameters (set later)
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    # ----------------------------------------------------
    # Compute normalization stats (ATTENTION : call only on train set)
    # ----------------------------------------------------
    def compute_normalization(self):
        xs = []
        ys = []

        for csv_path, C_value in self.samples:
            df = pd.read_csv(csv_path)

            Vin = df.iloc[:, 1].values
            Vout = df.iloc[:, 2].values

            x = np.stack([Vin, Vout], axis=0)
            xs.append(x)
            ys.append(C_value)

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

        df = pd.read_csv(csv_path)

        Vin = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
        Vout = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)

        x = torch.stack([Vin, Vout], dim=0)  # [2, T]
        y = torch.tensor(C_value, dtype=torch.float32)
        
        # Apply normalization if available
        if self.x_mean is not None:
            x = (x - self.x_mean[:, None]) / self.x_std[:, None]

        if self.y_mean is not None:
            y = (y - self.y_mean) / self.y_std

        return x, y

