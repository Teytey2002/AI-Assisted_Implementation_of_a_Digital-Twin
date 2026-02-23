import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import numpy as np

from model import RCInverseCNN
from dataset import RCSignalDataset


def train(root_dir: Path):

    # -------------------------
    # Configuration
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    batch_size = 32
    lr = 1e-3
    patience = 25
    max_epochs = 300

    # -------------------------
    # Dataset
    # -------------------------
    dataset = RCSignalDataset(root_dir)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_set, val_set = random_split(dataset, [n_train, n_val])

    # IMPORTANT: compute normalization only on train
    train_set.dataset.compute_normalization()

    # Apply same stats to both
    stats = (
        train_set.dataset.x_mean,
        train_set.dataset.x_std,
        train_set.dataset.y_mean,
        train_set.dataset.y_std
    )

    train_set.dataset.set_normalization(*stats)
    val_set.dataset.set_normalization(*stats)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # -------------------------
    # Model
    # -------------------------
    model = RCInverseCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,
        patience=10,
        verbose=True
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
            x, y = x.to(device), y.to(device)

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
                x, y = x.to(device), y.to(device)
                pred = model(x)

                loss = criterion(pred, y)
                val_loss += loss.item()

                preds_all.append(pred.cpu().numpy())
                targets_all.append(y.cpu().numpy())

        val_loss /= len(val_loader)

        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)

        rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2))
        rel_error = np.mean(np.abs((preds_all - targets_all) / targets_all)) * 100

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
                "x_mean": train_loader.x_mean,   # shape [2]
                "x_std": train_loader.x_std,     # shape [2]
                "y_mean": train_loader.y_mean,   # scalar
                "y_std": train_loader.y_std,     # scalar
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
