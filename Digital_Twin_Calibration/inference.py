from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dtcalib.calibration import RCNeuralCalibrator
from dtcalib.deep_learning.splits_utils import load_split, get_indices, parse_samples

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100.0)

def _aggregate_array(values: np.ndarray, mode: str) -> float:
    if mode == "mean":
        return float(np.mean(values))
    if mode == "median":
        return float(np.median(values))
    raise ValueError("aggregate must be 'mean' or 'median'")

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
    print(f"Aggregate  : {aggregate}")

    # Load trained neural calibrator
    calibrator = RCNeuralCalibrator.load(checkpoint_path, device=device_t)

    # Load dataset metadata and split indices
    payload = load_split(split_json_path)
    train_idx, val_idx, test_idx = get_indices(payload)

    if len(test_idx) == 0:
        raise ValueError("The split does not contain any test indices.")
    
    # Same deterministic parsing logic as split generation / dataset
    samples = parse_samples(root_dir)

    print(f"n_train={len(train_idx)} | n_val={len(val_idx)} | n_test={len(test_idx)}")

    rows: list[dict] = []

    for idx in test_idx:
        csv_path, true_C = samples[idx]

        df = pd.read_csv(csv_path)
        time = df.iloc[:, 0].values.astype(np.float32)
        vin  = df.iloc[:, 1].values.astype(np.float32)
        vout = df.iloc[:, 2].values.astype(np.float32)

        pred_C, pred_logC_std = calibrator.predict_distribution(time, vin, vout)
 
        rows.append(
            {
                "index": int(idx),
                "csv_path": str(csv_path),
                "true_C": float(true_C),
                "pred_C": float(pred_C),
                "abs_error_C": float(abs(pred_C - true_C)),
                "rel_error_percent": float(abs(pred_C - true_C) / max(abs(true_C), 1e-30) * 100.0),
                "pred_logC_std": None if pred_logC_std is None else float(pred_logC_std),
            }
        )

    # ------------------------------------------------------------------
    # [1] Sample-level evaluation
    # ------------------------------------------------------------------
    true_C_all = np.array([r["true_C"] for r in rows], dtype=np.float64)
    pred_C_all = np.array([r["pred_C"] for r in rows], dtype=np.float64)


    rmse_C_sample = rmse(true_C_all, pred_C_all)
    mae_C_sample = mae(true_C_all, pred_C_all)
    mape_C_sample = mape_percent(true_C_all, pred_C_all)

    print("\n[1] Sample-level metrics")
    print(f"- RMSE(C)         = {rmse_C_sample:.6e} F")
    print(f"- MAE(C)          = {mae_C_sample:.6e} F")
    print(f"- MAPE(C)         = {mape_C_sample:.3f} %")

    # ------------------------------------------------------------------
    # [2] Capacity-level aggregated evaluation
    # ------------------------------------------------------------------
    df_rows = pd.DataFrame(rows)

    grouped_rows: list[dict] = []
    for true_C_value, group in df_rows.groupby("true_C", sort=True):
        pred_values = group["pred_C"].to_numpy(dtype=np.float64)
        pred_C_agg = _aggregate_array(pred_values, aggregate)
        true_C_scalar = float(true_C_value)

        grouped_rows.append(
            {
                "true_C": true_C_scalar,
                "n_samples": int(len(group)),
                "pred_C_agg": float(pred_C_agg),
                "abs_error_C": float(abs(pred_C_agg - true_C_scalar)),
                "rel_error_percent": float(abs(pred_C_agg - true_C_scalar) / max(abs(true_C_scalar), 1e-30) * 100.0),
            }
        )

    df_grouped = pd.DataFrame(grouped_rows)

    true_C_group = df_grouped["true_C"].to_numpy(dtype=np.float64)
    pred_C_group = df_grouped["pred_C_agg"].to_numpy(dtype=np.float64)


    rmse_C_group = rmse(true_C_group, pred_C_group)
    mae_C_group = mae(true_C_group, pred_C_group)
    mape_C_group = mape_percent(true_C_group, pred_C_group)

    print("\n[2] Capacity-level aggregated metrics")
    print(f"- Aggregation     = {aggregate}")
    print(f"- #Capacities     = {len(df_grouped)}")
    print(f"- RMSE(C)         = {rmse_C_group:.6e} F")
    print(f"- MAE(C)          = {mae_C_group:.6e} F")
    print(f"- MAPE(C)         = {mape_C_group:.3f} %")

    # ------------------------------------------------------------------
    # [3] Prediction summaries
    # ------------------------------------------------------------------
    print("\n[3] Sample-level prediction summary")
    print(
        f"- true_C min/max/mean/std = "
        f"{true_C_all.min():.6e} / {true_C_all.max():.6e} / {true_C_all.mean():.6e} / {true_C_all.std():.6e}"
    )
    print(
        f"- pred_C min/max/mean/std = "
        f"{pred_C_all.min():.6e} / {pred_C_all.max():.6e} / {pred_C_all.mean():.6e} / {pred_C_all.std():.6e}"
    )

    print("\n[4] Aggregated prediction summary by true capacity")
    print(
        f"- true_C min/max/mean/std = "
        f"{true_C_group.min():.6e} / {true_C_group.max():.6e} / {true_C_group.mean():.6e} / {true_C_group.std():.6e}"
    )
    print(
        f"- pred_C min/max/mean/std = "
        f"{pred_C_group.min():.6e} / {pred_C_group.max():.6e} / {pred_C_group.mean():.6e} / {pred_C_group.std():.6e}"
    )

    # ------------------------------------------------------------------
    # [5] Save CSV files
    # ------------------------------------------------------------------
    if save_csv:
        per_sample_csv = checkpoint_path.with_name(checkpoint_path.stem + "_test_predictions_per_sample.csv")
        per_capacity_csv = checkpoint_path.with_name(
            checkpoint_path.stem + f"_test_predictions_per_capacity_{aggregate}.csv"
        )

        df_rows.to_csv(per_sample_csv, index=False)
        df_grouped.to_csv(per_capacity_csv, index=False)

        print(f"\n[5] CSV saved (sample-level)   : {per_sample_csv}")
        print(f"[5] CSV saved (capacity-level) : {per_capacity_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference on the test split for the trained RC inverse CNN."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--split-json", type=str, required=True, help="Path to split JSON")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--no-save-csv", action="store_true", help="Do not save prediction CSV files")
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