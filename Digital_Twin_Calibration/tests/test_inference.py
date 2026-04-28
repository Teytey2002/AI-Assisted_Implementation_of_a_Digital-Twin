from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import dtcalib.deep_learning.inference as inference_module
from dtcalib.deep_learning.inference import _aggregate_array, run_inference


# ============================================================
# Basic aggregation
# ============================================================

def test_aggregate_array_mean() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=float)
    assert _aggregate_array(values, "mean") == pytest.approx(2.0)


def test_aggregate_array_median() -> None:
    values = np.array([1.0, 10.0, 3.0], dtype=float)
    assert _aggregate_array(values, "median") == pytest.approx(3.0)


def test_aggregate_array_invalid_mode_raises() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="aggregate must be"):
        _aggregate_array(values, "max")


# ============================================================
# Fake inference components
# ============================================================

@dataclass
class FakeStats:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    calibrated_params: tuple[str, ...]
    transform_map: dict[str, str]

    def y_norm_to_physical(self, y_norm: torch.Tensor) -> torch.Tensor:
        y_transformed = y_norm * self.y_std[None, :] + self.y_mean[None, :]

        out = []
        for i, p in enumerate(self.calibrated_params):
            if self.transform_map.get(p, "identity") == "log":
                out.append(torch.exp(y_transformed[:, i]))
            else:
                out.append(y_transformed[:, i])

        return torch.stack(out, dim=1)


class FakeCalibrator:
    def __init__(
        self,
        *,
        calibrated_params: tuple[str, ...] = ("R1", "R2", "C"),
        transform_map: dict[str, str] | None = None,
        pred_physical: np.ndarray | None = None,
        samples_physical: np.ndarray | None = None,
    ) -> None:
        if transform_map is None:
            transform_map = {p: "log" for p in calibrated_params}

        self.calibrated_params = calibrated_params
        self.stats = FakeStats(
            x_mean=torch.zeros(6),
            x_std=torch.ones(6),
            y_mean=torch.zeros(len(calibrated_params)),
            y_std=torch.ones(len(calibrated_params)),
            calibrated_params=calibrated_params,
            transform_map=transform_map,
        )

        if pred_physical is None:
            pred_physical = np.array([[10_000.0, 10_000.0, 1e-6]], dtype=np.float64)

        self.pred_physical = pred_physical
        self.samples_physical = samples_physical

    def predict_from_x(self, x: torch.Tensor, *, n_samples: int = 0):
        d = len(self.calibrated_params)

        mean_physical = np.asarray(self.pred_physical, dtype=np.float64)
        assert mean_physical.shape == (1, d)

        transform_map = self.stats.transform_map

        mean_norm = []
        for i, p in enumerate(self.calibrated_params):
            value = mean_physical[0, i]
            if transform_map.get(p, "identity") == "log":
                value = np.log(value)
            mean_norm.append(value)

        mean_norm = np.asarray(mean_norm, dtype=np.float64)[None, :]

        std_norm = np.ones((1, d), dtype=np.float64) * 0.1
        std_physical = np.ones((1, d), dtype=np.float64) * 0.01

        samples = None
        if n_samples > 0 and self.samples_physical is not None:
            samples = np.asarray(self.samples_physical, dtype=np.float64)[None, :, :]

        return SimpleNamespace(
            mean_physical=mean_physical,
            mean_norm=mean_norm,
            std_norm=std_norm,
            std_physical=std_physical,
            samples_physical=samples,
        )


class FakeRCSignalDataset:
    def __init__(
        self,
        root_dir,
        *,
        target_spec,
        manifest_name="manifest.csv",
        domain="time_fft",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.target_spec = target_spec
        self.domain = domain

        self.csv0 = self.root_dir / "sample_0.csv"
        self.csv1 = self.root_dir / "sample_1.csv"

        self.samples = [
            (
                self.csv0,
                {
                    "R1": 10_000.0,
                    "R2": 10_000.0,
                    "C": 1e-6,
                },
            ),
            (
                self.csv1,
                {
                    "R1": 20_000.0,
                    "R2": 10_000.0,
                    "C": 2e-6,
                },
            ),
        ]

        self.grouped_samples = []

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def set_normalization(self, x_mean, x_std, y_mean, y_std) -> None:
        self.x_mean = x_mean.float()
        self.x_std = x_std.float()
        self.y_mean = y_mean.float()
        self.y_std = y_std.float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        csv_path, param_dict = self.samples[idx]

        t = torch.linspace(0.0, 1.0, 64)
        vin = torch.sin(2.0 * torch.pi * t)
        vout = 0.5 * vin

        # Fake time_fft input: 6 channels
        x = torch.stack(
            [
                t,
                vin,
                vout,
                torch.log1p(torch.abs(vin)),
                torch.log1p(torch.abs(vout)),
                torch.zeros_like(t),
            ],
            dim=0,
        )

        y_values = []
        for p in self.target_spec.calibrated_params:
            value = float(param_dict[p])
            if self.target_spec.transform_map.get(p, "identity") == "log":
                value = np.log(value)
            y_values.append(value)

        y = torch.tensor(y_values, dtype=torch.float32)

        return x, y


def _write_fake_signal_csv(path: Path) -> None:
    t = np.linspace(0.0, 1.0, 64)
    vin = np.sin(2.0 * np.pi * t)
    vout = 0.5 * vin

    pd.DataFrame(
        {
            "time": t,
            "vin": vin,
            "vout": vout,
        }
    ).to_csv(path, index=False)


def _prepare_fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()

    _write_fake_signal_csv(root / "sample_0.csv")
    _write_fake_signal_csv(root / "sample_1.csv")

    pd.DataFrame(
        {
            "csv_path": ["sample_0.csv", "sample_1.csv"],
            "R1": [10_000.0, 20_000.0],
            "R2": [10_000.0, 10_000.0],
            "C": [1e-6, 2e-6],
            "group_name": ["g0", "g1"],
            "experiment_name": ["e0", "e1"],
            "freq": [10.0, 20.0],
        }
    ).to_csv(root / "manifest.csv", index=False)

    return root


def _fake_split_payload() -> dict:
    return {
        "n_samples": 2,
        "indices": {
            "train": [0],
            "val": [],
            "test": [0, 1],
        },
        "samples": [
            {"R1": 10_000.0, "R2": 10_000.0, "C": 1e-6},
            {"R1": 20_000.0, "R2": 10_000.0, "C": 2e-6},
        ],
    }


# ============================================================
# run_inference with monkeypatched components
# ============================================================

def test_run_inference_creates_expected_csv_files(monkeypatch, tmp_path: Path) -> None:
    root = _prepare_fake_root(tmp_path)
    checkpoint = tmp_path / "fake_checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    split_json = tmp_path / "split.json"
    split_json.write_text("{}")

    fake_calibrator = FakeCalibrator()

    monkeypatch.setattr(
        inference_module.RCNeuralCalibrator,
        "load",
        staticmethod(lambda checkpoint_path, device: fake_calibrator),
    )
    monkeypatch.setattr(inference_module, "RCSignalDataset", FakeRCSignalDataset)
    monkeypatch.setattr(inference_module, "load_split", lambda path: _fake_split_payload())

    run_inference(
        checkpoint_path=checkpoint,
        root_dir=root,
        split_json_path=split_json,
        device="cpu",
        aggregate="mean",
        save_csv=True,
        n_samples=0,
        hybrid_select=False,
    )

    per_sample_csv = checkpoint.with_name(
        checkpoint.stem + "_test_predictions_per_sample.csv"
    )
    per_group_csv = checkpoint.with_name(
        checkpoint.stem + "_test_predictions_per_group_mean.csv"
    )

    assert per_sample_csv.exists()
    assert per_group_csv.exists()

    df_sample = pd.read_csv(per_sample_csv)
    df_group = pd.read_csv(per_group_csv)

    assert len(df_sample) == 2
    assert len(df_group) == 2

    for p in ("R1", "R2", "C"):
        assert f"true_{p}" in df_sample.columns
        assert f"pred_{p}" in df_sample.columns
        assert f"abs_error_{p}" in df_sample.columns
        assert f"rel_error_percent_{p}" in df_sample.columns
        assert f"pred_{p}_agg" in df_group.columns

    assert np.all(np.isfinite(df_sample["pred_R1"]))
    assert np.all(np.isfinite(df_sample["pred_R2"]))
    assert np.all(np.isfinite(df_sample["pred_C"]))


def test_run_inference_raises_if_test_split_empty(monkeypatch, tmp_path: Path) -> None:
    root = _prepare_fake_root(tmp_path)
    checkpoint = tmp_path / "fake_checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    split_json = tmp_path / "split.json"
    split_json.write_text("{}")

    fake_calibrator = FakeCalibrator()

    monkeypatch.setattr(
        inference_module.RCNeuralCalibrator,
        "load",
        staticmethod(lambda checkpoint_path, device: fake_calibrator),
    )
    monkeypatch.setattr(inference_module, "RCSignalDataset", FakeRCSignalDataset)
    monkeypatch.setattr(
        inference_module,
        "load_split",
        lambda path: {
            "n_samples": 2,
            "indices": {"train": [0], "val": [1], "test": []},
        },
    )

    with pytest.raises(ValueError, match="does not contain any test indices"):
        run_inference(
            checkpoint_path=checkpoint,
            root_dir=root,
            split_json_path=split_json,
            device="cpu",
            aggregate="mean",
            save_csv=False,
        )


def test_run_inference_invalid_aggregate_raises(monkeypatch, tmp_path: Path) -> None:
    root = _prepare_fake_root(tmp_path)
    checkpoint = tmp_path / "fake_checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    split_json = tmp_path / "split.json"
    split_json.write_text("{}")

    fake_calibrator = FakeCalibrator()

    monkeypatch.setattr(
        inference_module.RCNeuralCalibrator,
        "load",
        staticmethod(lambda checkpoint_path, device: fake_calibrator),
    )
    monkeypatch.setattr(inference_module, "RCSignalDataset", FakeRCSignalDataset)
    monkeypatch.setattr(inference_module, "load_split", lambda path: _fake_split_payload())

    with pytest.raises(ValueError, match="aggregate must be"):
        run_inference(
            checkpoint_path=checkpoint,
            root_dir=root,
            split_json_path=split_json,
            device="cpu",
            aggregate="invalid",
            save_csv=False,
        )


def test_run_inference_with_probabilistic_samples_writes_quantiles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _prepare_fake_root(tmp_path)
    checkpoint = tmp_path / "fake_checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    split_json = tmp_path / "split.json"
    split_json.write_text("{}")

    samples_physical = np.array(
        [
            [9_000.0, 9_500.0, 0.8e-6],
            [10_000.0, 10_000.0, 1.0e-6],
            [11_000.0, 10_500.0, 1.2e-6],
        ],
        dtype=np.float64,
    )

    fake_calibrator = FakeCalibrator(samples_physical=samples_physical)

    monkeypatch.setattr(
        inference_module.RCNeuralCalibrator,
        "load",
        staticmethod(lambda checkpoint_path, device: fake_calibrator),
    )
    monkeypatch.setattr(inference_module, "RCSignalDataset", FakeRCSignalDataset)
    monkeypatch.setattr(inference_module, "load_split", lambda path: _fake_split_payload())

    run_inference(
        checkpoint_path=checkpoint,
        root_dir=root,
        split_json_path=split_json,
        device="cpu",
        aggregate="median",
        save_csv=True,
        n_samples=3,
        hybrid_select=False,
    )

    per_sample_csv = checkpoint.with_name(
        checkpoint.stem + "_test_predictions_per_sample.csv"
    )
    per_group_csv = checkpoint.with_name(
        checkpoint.stem + "_test_predictions_per_group_median.csv"
    )

    assert per_sample_csv.exists()
    assert per_group_csv.exists()

    df_sample = pd.read_csv(per_sample_csv)

    for p in ("R1", "R2", "C"):
        assert f"samples_mean_{p}" in df_sample.columns
        assert f"samples_std_{p}" in df_sample.columns
        assert f"samples_q025_{p}" in df_sample.columns
        assert f"samples_q500_{p}" in df_sample.columns
        assert f"samples_q975_{p}" in df_sample.columns

    assert np.all(np.isfinite(df_sample["samples_mean_R1"]))
    assert np.all(np.isfinite(df_sample["samples_q500_C"]))


def test_run_inference_hybrid_selection_adds_selected_columns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _prepare_fake_root(tmp_path)
    checkpoint = tmp_path / "fake_checkpoint.pth"
    checkpoint.write_bytes(b"fake")
    split_json = tmp_path / "split.json"
    split_json.write_text("{}")

    samples_physical = np.array(
        [
            [10_000.0, 10_000.0, 1e-6],
            [12_000.0, 10_000.0, 1.5e-6],
            [20_000.0, 10_000.0, 2e-6],
        ],
        dtype=np.float64,
    )

    fake_calibrator = FakeCalibrator(samples_physical=samples_physical)

    monkeypatch.setattr(
        inference_module.RCNeuralCalibrator,
        "load",
        staticmethod(lambda checkpoint_path, device: fake_calibrator),
    )
    monkeypatch.setattr(inference_module, "RCSignalDataset", FakeRCSignalDataset)
    monkeypatch.setattr(inference_module, "load_split", lambda path: _fake_split_payload())

    run_inference(
        checkpoint_path=checkpoint,
        root_dir=root,
        split_json_path=split_json,
        device="cpu",
        aggregate="mean",
        save_csv=True,
        n_samples=3,
        hybrid_select=True,
        hybrid_n_candidates=3,
    )

    per_sample_csv = checkpoint.with_name(
        checkpoint.stem + "_test_predictions_per_sample.csv"
    )

    df_sample = pd.read_csv(per_sample_csv)

    for p in ("R1", "R2", "C"):
        assert f"selected_{p}" in df_sample.columns
        assert f"selected_abs_error_{p}" in df_sample.columns
        assert f"selected_rel_error_percent_{p}" in df_sample.columns

    assert "hybrid_selected_signal_rmse" in df_sample.columns