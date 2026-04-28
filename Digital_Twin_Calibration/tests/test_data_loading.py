from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dtcalib.data import ExperimentsDataset, DatasetFormatError


def _write_csv(
    path: Path,
    *,
    time_col: str = "TIME",
    input_col: str = "Addition_2.s_out.signal[1]",
    output_col: str = "SensorVoltage_1.v",
    n: int = 10,
) -> None:
    t = np.linspace(0.0, 1.0, n)
    u = np.sin(2 * np.pi * 5.0 * t)
    y = 0.5 * u

    pd.DataFrame(
        {
            time_col: t,
            input_col: u,
            output_col: y,
        }
    ).to_csv(path, index=False)


def test_from_csv_folder_loads_experiments(tmp_path: Path) -> None:
    _write_csv(tmp_path / "exp_001.csv", n=20)
    _write_csv(tmp_path / "exp_002.csv", n=30)

    ds = ExperimentsDataset.from_csv_folder(tmp_path)

    assert len(ds) == 2

    exp0 = ds[0]
    exp1 = ds[1]

    assert exp0.name in {"exp_001", "exp_002"}
    assert exp1.name in {"exp_001", "exp_002"}
    assert exp0.name != exp1.name

    assert isinstance(exp0.t, np.ndarray)
    assert isinstance(exp0.u, np.ndarray)
    assert isinstance(exp0.y, np.ndarray)

    assert exp0.t.ndim == 1
    assert exp0.u.ndim == 1
    assert exp0.y.ndim == 1

    assert len(exp0.t) == len(exp0.u) == len(exp0.y)
    assert np.all(np.diff(exp0.t) >= 0)

    assert isinstance(exp0.meta, dict)


def test_from_csv_folder_metadata_contains_filename_and_stem(tmp_path: Path) -> None:
    _write_csv(tmp_path / "exp_001.csv", n=20)

    ds = ExperimentsDataset.from_csv_folder(tmp_path)
    exp = ds[0]

    assert "filename" in exp.meta
    assert "stem" in exp.meta
    assert exp.meta["stem"] == exp.name


def test_from_csv_folder_raises_if_folder_missing() -> None:
    with pytest.raises(FileNotFoundError):
        ExperimentsDataset.from_csv_folder("this_folder_should_not_exist_12345")


def test_from_csv_folder_raises_if_no_csv_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ExperimentsDataset.from_csv_folder(tmp_path)


def test_from_csv_folder_raises_if_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "TIME": [0.0, 1.0],
            "WRONG_INPUT": [0.0, 0.0],
            "WRONG_OUTPUT": [0.0, 0.0],
        }
    )
    df.to_csv(tmp_path / "bad.csv", index=False)

    with pytest.raises(DatasetFormatError, match="missing columns"):
        ExperimentsDataset.from_csv_folder(tmp_path)


def test_from_csv_folder_custom_column_names(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "custom.csv",
        time_col="t",
        input_col="u",
        output_col="y",
        n=15,
    )

    ds = ExperimentsDataset.from_csv_folder(
        tmp_path,
        time_col="t",
        input_col="u",
        output_col="y",
    )

    assert len(ds) == 1

    exp = ds[0]
    assert exp.t.shape == (15,)
    assert exp.u.shape == (15,)
    assert exp.y.shape == (15,)


def test_from_csv_folder_values_are_loaded_correctly(tmp_path: Path) -> None:
    _write_csv(tmp_path / "exp.csv", n=10)

    ds = ExperimentsDataset.from_csv_folder(tmp_path)
    exp = ds[0]

    assert np.allclose(exp.y, 0.5 * exp.u)