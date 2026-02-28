# src/dtcalib/iterated_racing/train_api.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dtcalib.deep_learning.model import RCInverseCNN
from dtcalib.deep_learning.dataset import RCSignalDataset
from dtcalib.deep_learning.splits_utils import load_split, get_indices


@dataclass(frozen=True)
class FixedSplit:
    train_idx: list[int]
    val_idx: list[int]
    test_idx: list[int]


def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_fixed_split(split_json_path: str | Path) -> FixedSplit:
    payload = load_split(split_json_path)
    train_idx, val_idx, test_idx = get_indices(payload)
    return FixedSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def run_training_job_fixed_epochs(
    *,
    root_dir: str | Path,
    split_json_path: str | Path,
    params: Dict[str, Any],
    seed: int,
    epochs: int,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Training court pour Iterated Racing:
      - split fixe (train/val) chargé depuis JSON
      - normalisation calculée uniquement sur train
      - epochs FIXES
      - retourne val_loss (et train_loss)

    IMPORTANT:
      - pas d'early stopping ici
      - pas de TensorBoard
      - pas de sauvegarde modèle
    """
    set_global_seed(seed)

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    # --------- Params ---------
    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    weight_decay = float(params.get("weight_decay", 0.0))
    optimizer_name = str(params["optimizer"]).lower()
    sched_factor = float(params["scheduler_factor"])
    sched_patience = int(params["scheduler_patience"])
    target_transform = str(params["target_transform"])  # "C" or "logC"

    # --------- Dataset (full) ---------
    ds = RCSignalDataset(Path(root_dir))
    ds.set_target_transform(target_transform)

    # --------- Fixed split ---------
    split = load_fixed_split(split_json_path)
    train_ds = Subset(ds, split.train_idx)
    val_ds = Subset(ds, split.val_idx)

    # --------- Normalisation (TRAIN ONLY) ---------
    ds.compute_normalization(indices=split.train_idx)

    ds.set_normalization(
        ds.x_mean,
        ds.x_std,
        ds.y_mean,
        ds.y_std,
    )

    # --------- Loaders ---------
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # --------- Model ---------
    model = RCInverseCNN().to(device_t)
    criterion = nn.MSELoss()

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_factor, patience=sched_patience
    )

    # --------- Train fixed epochs ---------
    train_loss = 0.0
    val_loss = 0.0

    for _epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device_t), y.to(device_t)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device_t), y.to(device_t)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss_sum += float(loss.item())

        val_loss = val_loss_sum / max(1, len(val_loader))
        scheduler.step(val_loss)

    # cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"val_loss": float(val_loss), "train_loss": float(train_loss)}