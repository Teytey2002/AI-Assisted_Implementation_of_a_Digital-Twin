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
        dataset_+c_1p003em06/
            exp_.../
                results_....csv
    """
    def __init__(self, root_dir: Path, target_transform: str = "C"):
        self.samples = []

        root_dir = Path(root_dir)

        for c_folder in sorted(root_dir.iterdir()):
            #print("c_folder = ", c_folder)
            if not c_folder.is_dir():
                continue

            # Extract C from folder name like: dataset_+c_1p0032em06
            # on capture 1p0032em06 et on convertit en 1.0032e-06
            match = re.search(r"\+c_([0-9p]+e[m|p][0-9]+)", c_folder.name)
            if match is None:
                #print("match none, je continue")
                continue

            c_token = match.group(1)  # ex: "1p0032em06"
            c_str = (c_token.replace("p", ".").replace("em", "e-").replace("ep", "e+"))
            C_value = float(c_str)

            for csv_file in sorted(c_folder.rglob("*.csv")):
                #print("csv_file", csv_file)
                if "results" not in csv_file.name.lower():  # Au cas ou un csv de log ou autre se trouve un mauvais endroit
                    continue
                self.samples.append((csv_file, C_value))

        # For debug        
        #print(f"[RCSignalDataset] root_dir={root_dir}")
        #print(f"[RCSignalDataset] found {len(self.samples)} samples")
        #if len(self.samples) == 0:
        #    # show first few folder names to debug regex
        #    subdirs = [p.name for p in Path(root_dir).iterdir() if p.is_dir()]
        #    print("[RCSignalDataset] first subdirs:", subdirs[:10])
            

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

            Vin = df.iloc[:, 1].values
            Vout = df.iloc[:, 2].values

            x = np.stack([Vin, Vout], axis=0)
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

        df = pd.read_csv(csv_path)

        Vin = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
        Vout = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)

        x = torch.stack([Vin, Vout], dim=0)  # [2, T]
        y_value = self._transform_y(C_value)
        y = torch.tensor(y_value, dtype=torch.float32)
        
        # Apply normalization if available
        if self.x_mean is not None:
            x = (x - self.x_mean[:, None]) / self.x_std[:, None]

        if self.y_mean is not None:
            y = (y - self.y_mean) / self.y_std

        return x, y

