from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)     #Frozen true to not destroy the dataset
class Experiment:
    """
    One experiment = one time-series recording with:
      - t: time vector (shape: [T])
      - u: input signal (shape: [T])
      - y: measured output (shape: [T])

    Optional metadata can store frequency/amplitude/etc. for reporting.
    """
    name: str
    t: np.ndarray   # Vector
    u: np.ndarray
    y: np.ndarray
    meta: dict      # Data sup like frequency, amplitude, ...


class DatasetFormatError(ValueError):
    """
    Raised when dataset files are missing required columns or malformed.
    
    Example of use :
    try:
        ds = ExperimentsDataset.from_csv_folder(...)
    except DatasetFormatError as e:
        print("Dataset mal formé:", e)

    """
    # By creating a modified class, we can easily catch bug and 
    # we make the errors more explicit and easier to diagnose


class ExperimentsDataset:
    """
    Container for multiple experiments.

    Typical usage:
        ds = ExperimentsDataset.from_csv_folder(folder, ...)
        exps = ds.experiments
    """

    def __init__(self, experiments: Sequence[Experiment]) -> None:
        if len(experiments) == 0:
            raise ValueError("ExperimentsDataset cannot be empty.")
        self._experiments: List[Experiment] = list(experiments)                     # Why store in self. _experiments? Convention : _ = “private” (internal use) 
                                                                                    # We avoid having someone replace the list without control.
    
    @property   # Means Allow to acces like an attribute (len(ds) or ds[0])
    def experiments(self) -> List[Experiment]:
        return self._experiments

    def __len__(self) -> int:
        return len(self._experiments)

    def __getitem__(self, idx: int) -> Experiment:
        return self._experiments[idx]

    @staticmethod      # Means belongs to the class, but does not depend on an instance (self)
    def from_csv_folder(
        folder: Path | str,
        *,
        time_col: str = "TIME",
        input_col: str = "Addition_2.s_out.signal[1]",
        output_col: str = "SensorVoltage_1.v",
        file_glob: str = "*.csv",
        sort_files: bool = True,
        metadata_from_filename: bool = True,
    ) -> "ExperimentsDataset":
        """
        It a function that construct an object 
        Load experiments from a folder of CSV files (one experiment per CSV).        

        Assumptions:
          - Each CSV contains TIME, input_col, output_col.
          - TIME is numeric, increasing (not strictly required but recommended).
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = list(folder.glob(file_glob))
        if sort_files:
            files.sort()

        if not files:
            raise FileNotFoundError(f"No CSV files matching '{file_glob}' in {folder}")

        experiments: List[Experiment] = []
        for f in files:
            df = pd.read_csv(f)
            missing = [c for c in (time_col, input_col, output_col) if c not in df.columns]
            if missing:
                raise DatasetFormatError(
                    f"File {f.name} missing columns: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )

            t = df[time_col].to_numpy(dtype=float)
            u = df[input_col].to_numpy(dtype=float)
            y = df[output_col].to_numpy(dtype=float)

            if not (len(t) == len(u) == len(y)):
                raise DatasetFormatError(f"File {f.name}: t/u/y lengths are inconsistent.")

            meta = {}
            if metadata_from_filename:
                # Minimal heuristic: keep stem; user can parse freq/ampl later if needed
                meta["filename"] = f.name
                meta["stem"] = f.stem

            experiments.append(
                Experiment(name=f.stem, t=t, u=u, y=y, meta=meta)
            )

        return ExperimentsDataset(experiments)

    def with_experiments(self, exps: Sequence[Experiment]) -> "ExperimentsDataset":
        """Create a new dataset from a subset or modified experiments."""
        return ExperimentsDataset(exps)

qhdqhdbqbdhjqbdjhqbzd