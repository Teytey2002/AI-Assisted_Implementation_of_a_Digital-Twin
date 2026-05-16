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

    @staticmethod
    def _extract_time_input_output(
        df: pd.DataFrame,
        file_path: Path,
        *,
        time_col: str,
        input_col: str,
        output_col: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract t/u/y from either:
          1) explicitly provided columns,
          2) generic Python dataset columns: time, input, output,
          3) EcoSimPro columns.
        """

        # Case 1: user-provided columns
        if {time_col, input_col, output_col}.issubset(df.columns):
            return (
                df[time_col].to_numpy(dtype=float),
                df[input_col].to_numpy(dtype=float),
                df[output_col].to_numpy(dtype=float),
            )

        # Case 2: generic simulator-generated format
        generic_cols = ("time", "input", "output")
        if set(generic_cols).issubset(df.columns):
            return (
                df["time"].to_numpy(dtype=float),
                df["input"].to_numpy(dtype=float),
                df["output"].to_numpy(dtype=float),
            )

        # Case 3: EcoSimPro format
        ecosimpro_cols = (
            "TIME",
            "Addition_2.s_out.signal[1]",
            "SensorVoltage_1.v",
        )
        if set(ecosimpro_cols).issubset(df.columns):
            return (
                df["TIME"].to_numpy(dtype=float),
                df["Addition_2.s_out.signal[1]"].to_numpy(dtype=float),
                df["SensorVoltage_1.v"].to_numpy(dtype=float),
            )

        raise DatasetFormatError(
            f"File {file_path.name} missing compatible columns. "
            f"Expected either explicit columns "
            f"[{time_col!r}, {input_col!r}, {output_col!r}], "
            f"generic columns {list(generic_cols)}, "
            f"or EcoSimPro columns {list(ecosimpro_cols)}. "
            f"Available columns: {list(df.columns)}"
        )
    
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

        files = [f for f in folder.glob(file_glob) if f.name != "manifest.csv"]
        if sort_files:
            files.sort()

        if not files:
            raise FileNotFoundError(f"No CSV files matching '{file_glob}' in {folder}")

        experiments: List[Experiment] = []

        for f in files:
            df = pd.read_csv(f)

            t, u, y = ExperimentsDataset._extract_time_input_output(
                df,
                f,
                time_col=time_col,
                input_col=input_col,
                output_col=output_col,
            )

            if not (len(t) == len(u) == len(y)):
                raise DatasetFormatError(f"File {f.name}: t/u/y lengths are inconsistent.")

            if len(t) == 0:
                raise DatasetFormatError(f"File {f.name}: empty time series.")

            if np.any(~np.isfinite(t)) or np.any(~np.isfinite(u)) or np.any(~np.isfinite(y)):
                raise DatasetFormatError(f"File {f.name}: contains NaN or infinite values.")

            if np.any(np.diff(t) <= 0):
                raise DatasetFormatError(f"File {f.name}: time vector must be strictly increasing.")

            meta = {}
            if metadata_from_filename:
                meta["filename"] = f.name
                meta["stem"] = f.stem

            experiments.append(
                Experiment(name=f.stem, t=t, u=u, y=y, meta=meta)
            )

        return ExperimentsDataset(experiments)

    def with_experiments(self, exps: Sequence[Experiment]) -> "ExperimentsDataset":
        """Create a new dataset from a subset or modified experiments."""
        return ExperimentsDataset(exps)