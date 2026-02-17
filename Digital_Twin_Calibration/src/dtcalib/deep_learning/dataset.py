import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import re


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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, C_value = self.samples[idx]

        df = pd.read_csv(csv_path)

        Vin = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
        Vout = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)

        x = torch.stack([Vin, Vout], dim=0)  # [2, T]
        y = torch.tensor(C_value, dtype=torch.float32)

        return x, y
