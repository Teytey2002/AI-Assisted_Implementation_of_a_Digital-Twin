from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dtcalib.deep_learning.dataset import RCSignalDataset, TargetSpec
from dtcalib.deep_learning.splits_utils import load_split, get_indices


DATA_ROOT = Path("./data/LP_DATASET_R1_R2_C")
SPLIT_JSON = Path("./src/dtcalib/deep_learning/splits/rc_r1r2c_nested_fold0.json")


def make_target_spec_identity_c() -> TargetSpec:
    return TargetSpec(
        calibrated_params=("C",),
        transform_map={"C": "identity"},
    )


def make_target_spec_log_c() -> TargetSpec:
    return TargetSpec(
        calibrated_params=("C",),
        transform_map={"C": "log"},
    )


def make_target_spec_log_r1_r2_c() -> TargetSpec:
    return TargetSpec(
        calibrated_params=("R1", "R2", "C"),
        transform_map={"R1": "log", "R2": "log", "C": "log"},
    )


# ==========================================================
# Basic dataset loading
# ==========================================================

def test_dataset_loads_non_empty():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_identity_c(),
        domain="time_fft",
    )
    assert len(ds) > 0


def test_sample_shape_and_types_single_target_time_fft():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_identity_c(),
        domain="time_fft",
    )

    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert x.ndim == 2
    assert x.shape[0] == 6          # time/Vin/Vout + FFT features interpolated
    assert x.shape[1] > 1

    assert y.ndim == 1
    assert y.shape == (1,)


def test_sample_shape_and_types_multi_target_time_fft():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_log_r1_r2_c(),
        domain="time_fft",
    )

    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert x.ndim == 2
    assert x.shape[0] == 6
    assert x.shape[1] > 1

    assert y.ndim == 1
    assert y.shape == (3,)


def test_dataset_time_domain_still_has_three_channels():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_identity_c(),
        domain="time",
    )

    x, y = ds[0]

    assert x.ndim == 2
    assert x.shape[0] == 3
    assert y.shape == (1,)


def test_dataset_fft_domain_has_three_channels_and_top_k_columns():
    fft_top_k = 8

    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_identity_c(),
        domain="fft",
        fft_top_k=fft_top_k,
    )

    x, y = ds[0]

    assert x.ndim == 2
    assert x.shape[0] == 3
    assert 1 <= x.shape[1] <= fft_top_k
    assert y.shape == (1,)


# ==========================================================
# Target transform
# ==========================================================

def test_log_transform_changes_target():
    ds_c = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_identity_c(),
        domain="time_fft",
    )
    ds_log = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_log_c(),
        domain="time_fft",
    )

    _, y_c = ds_c[0]
    _, y_log = ds_log[0]

    assert y_c.shape == (1,)
    assert y_log.shape == (1,)
    assert y_c[0].item() > 0
    assert np.isclose(y_log[0].item(), np.log(y_c[0].item()), atol=1e-6)


def test_target_spec_inverse_transform_roundtrip():
    spec = make_target_spec_log_r1_r2_c()

    param_dict = {
        "R1": 10_000.0,
        "R2": 20_000.0,
        "C": 1e-6,
    }

    y = spec.transform_vector(param_dict)
    recovered = spec.inverse_transform_vector(y)

    assert recovered.shape == (3,)
    assert recovered[0] == pytest.approx(param_dict["R1"], rel=1e-6)
    assert recovered[1] == pytest.approx(param_dict["R2"], rel=1e-6)
    assert recovered[2] == pytest.approx(param_dict["C"], rel=1e-6)


# ==========================================================
# Split JSON integrity
# ==========================================================

def test_split_json_exists_and_is_consistent():
    assert SPLIT_JSON.exists()

    payload = load_split(SPLIT_JSON)
    train_idx, val_idx, test_idx = get_indices(payload)

    total = len(train_idx) + len(val_idx) + len(test_idx)
    assert total == payload["n_samples"]

    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)


def test_split_indices_are_in_dataset_range():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_log_r1_r2_c(),
        domain="time_fft",
    )

    payload = load_split(SPLIT_JSON)
    train_idx, val_idx, test_idx = get_indices(payload)

    all_idx = train_idx + val_idx + test_idx

    assert len(all_idx) > 0
    assert min(all_idx) >= 0
    assert max(all_idx) < len(ds)


def test_split_groups_are_disjoint_when_available():
    payload = load_split(SPLIT_JSON)

    assert "samples" in payload

    train_idx, val_idx, test_idx = get_indices(payload)
    samples = payload["samples"]

    def group_key(i: int) -> tuple[float, float, float]:
        s = samples[i]
        return (float(s["R1"]), float(s["R2"]), float(s["C"]))

    train_groups = {group_key(i) for i in train_idx}
    val_groups = {group_key(i) for i in val_idx}
    test_groups = {group_key(i) for i in test_idx}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


# ==========================================================
# Normalization correctness
# ==========================================================

def test_normalization_train_mean_zero_time_fft():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_log_r1_r2_c(),
        domain="time_fft",
    )

    payload = load_split(SPLIT_JSON)
    train_idx, _, _ = get_indices(payload)

    ds.compute_normalization(indices=train_idx)
    ds.set_normalization(ds.x_mean, ds.x_std, ds.y_mean, ds.y_std)

    xs = []
    ys = []

    for idx in train_idx:
        x, y = ds[idx]
        xs.append(x.numpy())
        ys.append(y.numpy())

    xs = np.concatenate(xs, axis=1)
    ys = np.stack(ys, axis=0)

    assert xs.shape[0] == 6
    assert ys.shape[1] == 3

    assert np.allclose(xs.mean(axis=1), 0.0, atol=1e-2)
    assert np.allclose(ys.mean(axis=0), 0.0, atol=1e-2)


def test_normalization_stats_have_correct_shapes_time_fft():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_log_r1_r2_c(),
        domain="time_fft",
    )

    payload = load_split(SPLIT_JSON)
    train_idx, _, _ = get_indices(payload)

    ds.compute_normalization(indices=train_idx)

    assert ds.x_mean is not None
    assert ds.x_std is not None
    assert ds.y_mean is not None
    assert ds.y_std is not None

    assert ds.x_mean.shape == (6,)
    assert ds.x_std.shape == (6,)
    assert ds.y_mean.shape == (3,)
    assert ds.y_std.shape == (3,)

    assert torch.all(ds.x_std > 0)
    assert torch.all(ds.y_std > 0)


def test_val_not_used_in_train_normalization():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_log_r1_r2_c(),
        domain="time_fft",
    )

    payload = load_split(SPLIT_JSON)
    train_idx, val_idx, _ = get_indices(payload)

    ds.compute_normalization(indices=train_idx)

    train_targets = np.stack(
        [ds.target_spec.transform_vector(ds.samples[i][1]) for i in train_idx],
        axis=0,
    )
    val_targets = np.stack(
        [ds.target_spec.transform_vector(ds.samples[i][1]) for i in val_idx],
        axis=0,
    )

    train_mean = train_targets.mean(axis=0)
    train_std = train_targets.std(axis=0) + 1e-8

    assert np.allclose(ds.y_mean.numpy(), train_mean, atol=1e-6)
    assert np.allclose(ds.y_std.numpy(), train_std, atol=1e-6)

    val_norm = (val_targets - train_mean) / train_std
    assert val_norm.shape[0] == len(val_idx)
    assert val_norm.shape[1] == 3


def test_getitem_cache_does_not_change_values():
    ds = RCSignalDataset(
        DATA_ROOT,
        target_spec=make_target_spec_log_r1_r2_c(),
        domain="time_fft",
    )

    x1, y1 = ds[0]
    x2, y2 = ds[0]

    assert torch.allclose(x1, x2)
    assert torch.allclose(y1, y2)