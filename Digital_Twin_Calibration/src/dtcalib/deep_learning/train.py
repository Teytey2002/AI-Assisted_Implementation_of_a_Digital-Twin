import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import numpy as np

from model import RCInverseCNN
from dataset import RCSignalDataset
import random

from dtcalib.deep_learning.splits_utils import load_split, get_indices
from dtcalib.calibration import NormalizationStats


# Important for reproductibility and comparaison
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(root_dir: Path):

    # -------------------------
    # Configuration
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    batch_size = 32
    lr = 0.001753818
    weight_decay = 3.27707e-05
    patience = 25
    max_epochs = 300

    # -------------------------
    # Dataset
    # -------------------------
    dataset = RCSignalDataset(root_dir)
    dataset.set_target_transform("logC") # Fixe log(c) pour une meilleure convergence

    payload = load_split("./splits/rc_nested_fold0.json")
    train_idx, val_idx, test_idx = get_indices(payload)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # IMPORTANT: compute normalization only on train indices
    dataset.compute_normalization(indices=train_idx)

    # Apply same stats to both
    dataset.set_normalization(
        dataset.x_mean,
        dataset.x_std,
        dataset.y_mean,
        dataset.y_std
    )

    stats = NormalizationStats(
        x_mean=dataset.x_mean.clone(),
        x_std=dataset.x_std.clone(),
        y_mean=dataset.y_mean.clone(),
        y_std=dataset.y_std.clone(),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # -------------------------
    # Model
    # -------------------------
    model = RCInverseCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3882663,
        patience=3
    )

    # -------------------------
    # Logging
    # -------------------------
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/rc_inverse_{session_id}")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"rc_inverse_best_{session_id}.pth"

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
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0

        preds_all = []
        targets_all = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred = model(x)

                loss = criterion(pred, y)
                val_loss += loss.item()

                preds_all.append(pred.cpu().numpy())
                targets_all.append(y.cpu().numpy())

        val_loss /= len(val_loader)

        preds_all = torch.tensor(np.concatenate(preds_all), dtype=torch.float32)
        targets_all = torch.tensor(np.concatenate(targets_all), dtype=torch.float32)

        preds_C = stats.y_norm_to_C(preds_all)
        targets_C = stats.y_norm_to_C(targets_all)

        rmse = torch.sqrt(torch.mean((preds_C - targets_C) ** 2)).item()
        rel_error = (
            torch.mean(torch.abs((preds_C - targets_C) / (torch.abs(targets_C) + 1e-12))) * 100
        ).item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # -------- Logging --------
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("RMSE/val", rmse, epoch)
        writer.add_scalar("RelativeError_percent/val", rel_error, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        print(
            f"Epoch {epoch+1} | "
            f"TrainLoss={train_loss:.6e} | "
            f"ValLoss={val_loss:.6e} | "
            f"RMSE={rmse:.3e} | "
            f"RelErr={rel_error:.2f}% | "
            f"LR={current_lr:.2e}"
        )

        # -------- Early stopping --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
            {
                "model_state_dict": model.state_dict(),
                "x_mean": train_set.dataset.x_mean,   # shape [3]
                "x_std": train_set.dataset.x_std,     # shape [3]
                "y_mean": train_set.dataset.y_mean,   # scalar
                "y_std": train_set.dataset.y_std,     # scalar
                "model_class": "RCInverseCNN",
            },
            model_path,
        )
            print(f"✅ Best model saved (val_loss={best_val_loss:.6e})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⏸️ No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            print("🛑 Early stopping triggered")
            break

    writer.close()
    print(f"Training complete. Best model at: {model_path}")


if __name__ == "__main__":
    train("../../../data/ALL_LP_DATASETS_CSV_Deep_learning")