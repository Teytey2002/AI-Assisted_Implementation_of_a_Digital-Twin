from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import pytest

from dtcalib.deep_learning.dataset import RCSignalDataset
from dtcalib.deep_learning.splits_utils import load_split


DATA_ROOT = Path("./data/ALL_LP_DATASETS_CSV_Deep_learning")
SPLIT_JSON = Path("./src/dtcalib/deep_learning/splits/rc_split_seed42_70_15_15.json")


# ==========================================================
# Basic dataset loading
# ==========================================================

def test_dataset_loads_non_empty():
    ds = RCSignalDataset(DATA_ROOT)
    assert len(ds) > 0


def test_sample_shape_and_types():
    ds = RCSignalDataset(DATA_ROOT)
    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert x.ndim == 2  # [2, T]
    assert x.shape[0] == 2
    assert y.ndim == 0  # scalar


# ==========================================================
# Target transform
# ==========================================================

def test_log_transform_changes_target():
    ds_C = RCSignalDataset(DATA_ROOT, target_transform="C")
    ds_log = RCSignalDataset(DATA_ROOT, target_transform="logC")

    _, y_C = ds_C[0]
    _, y_log = ds_log[0]

    assert y_C.item() > 0
    assert np.isclose(y_log.item(), np.log(y_C.item()), atol=1e-6)


# ==========================================================
# Split JSON integrity
# ==========================================================

def test_split_json_exists_and_is_consistent():
    assert SPLIT_JSON.exists()

    payload = load_split(SPLIT_JSON)

    train_idx = payload["indices"]["train"]
    val_idx = payload["indices"]["val"]
    test_idx = payload["indices"]["test"]

    total = len(train_idx) + len(val_idx) + len(test_idx)
    assert total == payload["n_samples"]

    # No overlap
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0


# ==========================================================
# Normalization correctness
# ==========================================================

def test_normalization_train_mean_zero():
    ds = RCSignalDataset(DATA_ROOT, target_transform="logC")

    payload = load_split(SPLIT_JSON)
    train_idx = payload["indices"]["train"]

    ds.compute_normalization(indices=train_idx)

    # Apply normalization
    ds.set_normalization(ds.x_mean, ds.x_std, ds.y_mean, ds.y_std)

    xs = []
    ys = []

    for idx in train_idx[:200]:  # sample subset for speed
        x, y = ds[idx]
        xs.append(x.numpy())
        ys.append(y.item())

    xs = np.concatenate(xs, axis=1)
    ys = np.array(ys)

    # Mean should be close to 0
    assert np.allclose(xs.mean(axis=1), 0.0, atol=1e-2)
    assert abs(ys.mean()) < 0.05


def test_val_not_used_in_train_normalization():
    ds = RCSignalDataset(DATA_ROOT, target_transform="C")

    payload = load_split(SPLIT_JSON)
    train_idx = payload["indices"]["train"]
    val_idx = payload["indices"]["val"]

    ds.compute_normalization(indices=train_idx)

    # Compute mean manually on train
    train_targets = [ds._transform_y(ds.samples[i][1]) for i in train_idx]
    val_targets = [ds._transform_y(ds.samples[i][1]) for i in val_idx]

    train_mean = np.mean(train_targets)
    val_mean = np.mean(val_targets)

    # Normalization mean must equal TRAIN mean
    assert np.isclose(ds.y_mean.item(), train_mean, atol=1e-6)
