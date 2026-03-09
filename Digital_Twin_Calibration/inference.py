from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dtcalib.calibration import RCNeuralCalibrator
from dtcalib.deep_learning.dataset import RCSignalDataset
from dtcalib.deep_learning.splits_utils import load_split, get_indices


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100.0)


def run_inference(
    *,
    checkpoint_path: str | Path,
    root_dir: str | Path,
    split_json_path: str | Path,
    device: str = "cuda",
    aggregate: str = "mean",
    save_csv: bool = True,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    root_dir = Path(root_dir)
    split_json_path = Path(split_json_path)

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Root dir   : {root_dir}")
    print(f"Split JSON : {split_json_path}")
    print(f"Device     : {device_t}")

    # Load trained neural calibrator
    calibrator = RCNeuralCalibrator.load(checkpoint_path, device=device_t)

    # Load dataset metadata and split indices
    dataset = RCSignalDataset(root_dir)
    payload = load_split(split_json_path)
    train_idx, val_idx, test_idx = get_indices(payload)

    if len(test_idx) > 0:
        eval_idx = test_idx
        eval_name = "test"
    else:
        eval_idx = val_idx
        eval_name = "val"

    if len(eval_idx) == 0:
        raise ValueError("The split does not contain any evaluation indices (val/test).")

    print(f"n_train={len(train_idx)} | n_val={len(val_idx)} | n_test={len(test_idx)}")
    print(f"Using '{eval_name}' split for inference: n_eval={len(eval_idx)}")

    rows = []

    for idx in eval_idx:
        csv_path, true_C = dataset.samples[idx]

        df = pd.read_csv(csv_path)
        vin = df.iloc[:, 1].values.astype(np.float32)
        vout = df.iloc[:, 2].values.astype(np.float32)

        pred_C = calibrator.predict(vin, vout)
        pred_logC = calibrator.predict_logC(vin, vout)

        rows.append(
            {
                "index": int(idx),
                "csv_path": str(csv_path),
                "true_C": float(true_C),
                "pred_C": float(pred_C),
                "abs_error_C": float(abs(pred_C - true_C)),
                "rel_error_percent": float(abs(pred_C - true_C) / max(abs(true_C), 1e-30) * 100.0),
                "pred_logC": float(pred_logC),
                "true_logC": float(np.log(true_C)),
            }
        )

    true_C_all = np.array([r["true_C"] for r in rows], dtype=float)
    pred_C_all = np.array([r["pred_C"] for r in rows], dtype=float)

    true_logC_all = np.log(true_C_all)
    pred_logC_all = np.log(np.maximum(pred_C_all, 1e-30))

    rmse_C = rmse(true_C_all, pred_C_all)
    mae_C = mae(true_C_all, pred_C_all)
    mape_C = mape_percent(true_C_all, pred_C_all)

    rmse_logC = rmse(true_logC_all, pred_logC_all)
    mae_logC = mae(true_logC_all, pred_logC_all)

    print("\n[1] Test metrics on individual samples")
    print(f"- RMSE(C)         = {rmse_C:.6e} F")
    print(f"- MAE(C)          = {mae_C:.6e} F")
    print(f"- MAPE(C)         = {mape_C:.3f} %")
    print(f"- RMSE(logC)      = {rmse_logC:.6e}")
    print(f"- MAE(logC)       = {mae_logC:.6e}")

    # Global aggregate estimate on the test set
    if aggregate == "mean":
        global_C_hat = float(np.mean(pred_C_all))
    elif aggregate == "median":
        global_C_hat = float(np.median(pred_C_all))
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    print("\n[2] Global aggregated estimate on test set")
    print(f"- Aggregate       = {aggregate}")
    print(f"- Global C_hat    = {global_C_hat:.6e} F")

    print("\n[3] Predictions summary")
    print(f"- true_C min/max/mean/std = {true_C_all.min():.6e} / {true_C_all.max():.6e} / {true_C_all.mean():.6e} / {true_C_all.std():.6e}")
    print(f"- pred_C min/max/mean/std = {pred_C_all.min():.6e} / {pred_C_all.max():.6e} / {pred_C_all.mean():.6e} / {pred_C_all.std():.6e}")

    if save_csv:
        out_csv = checkpoint_path.with_name(checkpoint_path.stem + "_test_predictions.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n[4] CSV saved: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference on test split for the trained RC inverse CNN.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--split-json", type=str, required=True, help="Path to split JSON")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--no-save-csv", action="store_true", help="Do not save per-sample predictions CSV")
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        split_json_path=args.split_json,
        device=args.device,
        aggregate=args.aggregate,
        save_csv=not args.no_save_csv,
    )


if __name__ == "__main__":
    main()