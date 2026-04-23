"""
Example of usage from deep_learning directory: 
    python3 train.py --dataset ../../../data/ALL_LP_DATASETS_CSV_Deep_learning --split ./splits/rc_nested_fold0.json --model prob_cnn
"""
from __future__ import annotations
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # Disable oneDNN optimizations to prevent problem with tensorflow
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from model import RCInverseCNN, ProbabilisticRCInverseCNN
from dataset import RCSignalDataset, TargetSpec
from dtcalib.deep_learning.splits_utils import load_split, get_indices
from dtcalib.calibration import NormalizationStats

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# Loss for probabilistic regression
# y | x ~ N(mu(x), exp(log_var(x)))
# ------------------------------------------------------------
def gaussian_nll_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    # 0.5 * [ log(sigma^2) + (y-mu)^2 / sigma^2 ]
    inv_var = torch.exp(-log_var)
    loss = 0.5 * (log_var + (target - mu) ** 2 * inv_var)
    return loss.mean()


# ------------------------------------------------------------
# Model factory
# ------------------------------------------------------------
def build_model(model_name: str, output_dim: int) -> tuple[nn.Module, str, str]:
    """
    Returns:
        model
        model_mode: 'deterministic' or 'probabilistic'
        model_class_name: name saved in checkpoint
    """
    model_name = model_name.lower()

    if model_name == "cnn":
        return RCInverseCNN(output_dim=output_dim), "deterministic", "RCInverseCNN"

    if model_name == "prob_cnn":
        return ProbabilisticRCInverseCNN(output_dim=output_dim), "probabilistic", "ProbabilisticRCInverseCNN"

    raise ValueError(
        f"Unknown model '{model_name}'. Supported values: 'cnn', 'prob_cnn'."
    )


# ------------------------------------------------------------
# One forward step depending on model mode
# ------------------------------------------------------------
def compute_batch_loss_and_pred(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    model_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Returns:
        loss
        pred_for_metrics_norm   # normalized-space prediction used for RMSE on C
        pred_std_norm_or_none   # only for probabilistic model
    """
    if model_mode == "deterministic":
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss, pred, None

    if model_mode == "probabilistic":
        mu, log_var = model(x)
        loss = gaussian_nll_loss(mu, log_var, y)
        pred_std = torch.sqrt(torch.exp(log_var) + 1e-8)
        return loss, mu, pred_std

    raise ValueError(f"Unsupported model_mode: {model_mode}")


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train(
    dataset_root: Path,
    split_json_path: Path,
    model_name: str,
) -> None:
    # -------------------------
    # Configuration
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Dataset root : {dataset_root}")
    print(f"Split JSON   : {split_json_path}")
    print(f"Model        : {model_name}")

    batch_size = 32
    lr = 0.001753818
    weight_decay = 3.27707e-05
    patience = 25
    max_epochs = 300

    calibrated_params = ("R1", "R2", "C")
    transform_map = {"R1": "log", "R2": "log", "C": "log"}
    base_params = {"R1": 10_000.0, "R2": 10_000.0}  # Pour le moment car seul C est encodé dans le dossier
    target_spec = TargetSpec(calibrated_params=calibrated_params, transform_map=transform_map)


    # -------------------------
    # Dataset
    # -------------------------
    dataset = RCSignalDataset(dataset_root, target_spec=target_spec, base_params=base_params)

    payload = load_split(split_json_path)
    train_idx, val_idx, test_idx = get_indices(payload)

    if len(train_idx) == 0:
        raise ValueError("Train split is empty.")
    if len(val_idx) == 0:
        raise ValueError("Validation split is empty.")

    max_idx = max(train_idx + val_idx + test_idx) if len(train_idx + val_idx + test_idx) > 0 else -1
    if max_idx >= len(dataset):
        raise ValueError(
            f"Split index {max_idx} is out of range for dataset of size {len(dataset)}."
        )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # Compute normalization ONLY on train indices
    dataset.compute_normalization(indices=train_idx)

    # Apply same normalization to all subsets through the shared dataset
    dataset.set_normalization(
        dataset.x_mean,
        dataset.x_std,
        dataset.y_mean,
        dataset.y_std,
    )

    stats = NormalizationStats(
        x_mean=dataset.x_mean.clone(),
        x_std=dataset.x_std.clone(),
        y_mean=dataset.y_mean.clone(),
        y_std=dataset.y_std.clone(),
        calibrated_params=target_spec.calibrated_params,
        transform_map=target_spec.transform_map,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    print(
        f"n_total={len(dataset)} | n_train={len(train_idx)} | "
        f"n_val={len(val_idx)} | n_test={len(test_idx)}"
    )

    # -------------------------
    # Model
    # -------------------------
    model, model_mode, model_class_name = build_model(model_name, output_dim=len(calibrated_params))
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3882663,
        patience=3,
    )

    # -------------------------
    # Logging
    # -------------------------
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}_{session_id}"
    writer = SummaryWriter(f"runs/{run_name}")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{run_name}_best.pth"

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(max_epochs):
        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            loss, _, _ = compute_batch_loss_and_pred(model, x, y, model_mode)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0

        preds_all = []
        targets_all = []

        pred_std_all = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                loss, pred_norm, pred_std_norm = compute_batch_loss_and_pred(model, x, y, model_mode)

                val_loss += float(loss.item())

                preds_all.append(pred_norm.cpu())
                targets_all.append(y.cpu())

                if pred_std_norm is not None:
                    pred_std_all.append(pred_std_norm.cpu())

        val_loss /= len(val_loader)

        preds_all_t = torch.cat(preds_all, dim=0).float()
        targets_all_t = torch.cat(targets_all, dim=0).float()

        preds_phys = stats.y_norm_to_physical(preds_all_t)
        targets_phys = stats.y_norm_to_physical(targets_all_t)

        rmse_per_param = torch.sqrt(torch.mean((preds_phys - targets_phys) ** 2, dim=0))
        rel_error_per_param = torch.mean(
            torch.abs((preds_phys - targets_phys) / (torch.abs(targets_phys) + 1e-12)),
            dim=0,
        ) * 100.0

        rmse_global = torch.sqrt(torch.mean((preds_phys - targets_phys) ** 2)).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # -------- Logging --------
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("RMSE_C/val", rmse_global, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        for i, p in enumerate(calibrated_params):
            writer.add_scalar(f"RMSE_val/{p}", rmse_per_param[i].item(), epoch)
            writer.add_scalar(f"RelativeError_percent_val/{p}", rel_error_per_param[i].item(), epoch)

        if model_mode == "probabilistic" and len(pred_std_all) > 0:
            pred_std_all_t = torch.cat(pred_std_all, dim=0).float()
            writer.add_scalar("PredStd_norm/val_mean", pred_std_all_t.mean().item(), epoch)

        print(
            f"Epoch {epoch + 1} | "
            f"TrainLoss={train_loss:.6e} | "
            f"ValLoss={val_loss:.6e} | "
            f"RMSE(C)={rmse_global:.3e} | "
            f"LR={current_lr:.2e}"
        )

        # -------- Early stopping --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "x_mean": dataset.x_mean,
                    "x_std": dataset.x_std,
                    "y_mean": dataset.y_mean,
                    "y_std": dataset.y_std,
                    "model_class": model_class_name,
                    "model_mode": model_mode,
                    "calibrated_params": calibrated_params,
                    "transform_map": transform_map,
                    "dataset_root": str(dataset_root),
                    "split_json": str(split_json_path),
                    "model_name_arg": model_name,
                },
                model_path,
            )

            print(f"Best model saved: {model_path} (val_loss={best_val_loss:.6e})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    writer.close()
    print(f"Training complete. Best model at: {model_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train inverse RC models (deterministic or probabilistic)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Path to split JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn", "prob_cnn"],
        help="Model to train",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train(
        dataset_root=Path(args.dataset),
        split_json_path=Path(args.split),
        model_name=args.model,
    )


if __name__ == "__main__":
    main()