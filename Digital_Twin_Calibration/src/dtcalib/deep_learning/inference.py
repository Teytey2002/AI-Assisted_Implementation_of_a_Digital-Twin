"""
Inference script for evaluating a trained RCNeuralCalibrator on the test split.

Usage example:
python3 inference.py \
  --checkpoint models/cnn_2026-04-23_14-14-14_best.pth \
  --root-dir ../../../data/LP_DATASET_R1_R2_C \
  --split-json ./splits/rc_r1r2c_nested_fold0.json \
  --device cuda \
  --aggregate mean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dtcalib.calibration import RCNeuralCalibrator
from dtcalib.deep_learning.dataset import RCSignalDataset, TargetSpec
from dtcalib.deep_learning.splits_utils import load_split, get_indices


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------
def run_inference(
    *,
    checkpoint_path: str | Path,
    root_dir: str | Path,
    split_json_path: str | Path,
    device: str = "cuda",
    aggregate: str = "mean",
    save_csv: bool = True,
    manifest_name: str = "manifest.csv",
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

    # ------------------------------------------------------------------
    # Load trained neural calibrator
    # ------------------------------------------------------------------
    calibrator = RCNeuralCalibrator.load(checkpoint_path, device=device_t)

    calibrated_params = tuple(calibrator.calibrated_params)
    transform_map = dict(calibrator.stats.transform_map)

    print(f"Calibrated params : {calibrated_params}")
    print(f"Transform map     : {transform_map}")

    # ------------------------------------------------------------------
    # Build dataset with the SAME target specification as the checkpoint
    # ------------------------------------------------------------------
    target_spec = TargetSpec(
        calibrated_params=calibrated_params,
        transform_map=transform_map,
    )

    dataset = RCSignalDataset(
        root_dir,
        target_spec=target_spec,
        manifest_name=manifest_name,
    )

    # ------------------------------------------------------------------
    # Load split
    # ------------------------------------------------------------------
    payload = load_split(split_json_path)
    train_idx, val_idx, test_idx = get_indices(payload)

    if len(test_idx) == 0:
        raise ValueError("The split does not contain any test indices.")

    print(f"n_train={len(train_idx)} | n_val={len(val_idx)} | n_test={len(test_idx)}")

    rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Sample-level inference
    # ------------------------------------------------------------------
    for idx in test_idx:
        csv_path, param_dict = dataset.samples[idx]

        df = pd.read_csv(csv_path)
        time = df.iloc[:, 0].values.astype(np.float32)
        vin = df.iloc[:, 1].values.astype(np.float32)
        vout = df.iloc[:, 2].values.astype(np.float32)

        pred_vec: np.ndarray
        pred_std_vec: np.ndarray | None = None

        # Probabilistic model -> mean + std
        # Deterministic model -> mean only
        try:
            pred_vec, pred_std_vec = calibrator.predict_distribution(time, vin, vout)
        except TypeError:
            pred_vec = calibrator.predict(time, vin, vout)
            pred_std_vec = None

        pred_vec = np.asarray(pred_vec, dtype=np.float64)
        if pred_vec.ndim != 1 or pred_vec.shape[0] != len(calibrated_params):
            raise ValueError(
                f"Expected prediction vector of shape ({len(calibrated_params)},), "
                f"got {pred_vec.shape}."
            )

        if pred_std_vec is not None:
            pred_std_vec = np.asarray(pred_std_vec, dtype=np.float64)
            if pred_std_vec.shape != pred_vec.shape:
                raise ValueError(
                    f"pred_std_vec shape {pred_std_vec.shape} does not match pred_vec shape {pred_vec.shape}."
                )

        row: dict[str, Any] = {
            "index": int(idx),
            "csv_path": str(csv_path),
        }

        for i, p in enumerate(calibrated_params):
            true_val = float(param_dict[p])
            pred_val = float(pred_vec[i])

            row[f"true_{p}"] = true_val
            row[f"pred_{p}"] = pred_val
            row[f"abs_error_{p}"] = float(abs(pred_val - true_val))
            row[f"rel_error_percent_{p}"] = float(
                abs(pred_val - true_val) / max(abs(true_val), 1e-30) * 100.0
            )

            if pred_std_vec is not None:
                row[f"pred_std_{p}"] = float(pred_std_vec[i])
            else:
                row[f"pred_std_{p}"] = None

        rows.append(row)

    df_rows = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # [1] Sample-level metrics
    # ------------------------------------------------------------------
    print("\n[1] Sample-level metrics")
    for p in calibrated_params:
        y_true = df_rows[f"true_{p}"].to_numpy(dtype=np.float64)
        y_pred = df_rows[f"pred_{p}"].to_numpy(dtype=np.float64)

        print(f"- {p}:")
        print(f"    RMSE = {rmse(y_true, y_pred):.6e}")
        print(f"    MAE  = {mae(y_true, y_pred):.6e}")
        print(f"    MAPE = {mape_percent(y_true, y_pred):.3f} %")

    # ------------------------------------------------------------------
    # [2] Aggregated evaluation by TRUE parameter combination
    # ------------------------------------------------------------------
    group_cols = [f"true_{p}" for p in calibrated_params]
    grouped_rows: list[dict[str, Any]] = []

    for group_key, group in df_rows.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row_group: dict[str, Any] = {
            "n_samples": int(len(group)),
        }

        for p, key_val in zip(calibrated_params, group_key):
            row_group[f"true_{p}"] = float(key_val)

        for p in calibrated_params:
            pred_values = group[f"pred_{p}"].to_numpy(dtype=np.float64)
            pred_agg = _aggregate_array(pred_values, aggregate)
            true_val = float(group.iloc[0][f"true_{p}"])

            row_group[f"pred_{p}_agg"] = float(pred_agg)
            row_group[f"abs_error_{p}"] = float(abs(pred_agg - true_val))
            row_group[f"rel_error_percent_{p}"] = float(
                abs(pred_agg - true_val) / max(abs(true_val), 1e-30) * 100.0
            )

            if f"pred_std_{p}" in group.columns:
                std_values = pd.to_numeric(group[f"pred_std_{p}"], errors="coerce").to_numpy(dtype=np.float64)
                if np.all(np.isnan(std_values)):
                    row_group[f"pred_std_{p}_agg"] = None
                else:
                    row_group[f"pred_std_{p}_agg"] = float(np.nanmean(std_values))

        grouped_rows.append(row_group)

    df_grouped = pd.DataFrame(grouped_rows)

    # ------------------------------------------------------------------
    # [3] Aggregated metrics
    # ------------------------------------------------------------------
    print("\n[2] Aggregated metrics by true parameter combination")
    print(f"- Aggregation = {aggregate}")
    print(f"- #groups     = {len(df_grouped)}")

    for p in calibrated_params:
        y_true = df_grouped[f"true_{p}"].to_numpy(dtype=np.float64)
        y_pred = df_grouped[f"pred_{p}_agg"].to_numpy(dtype=np.float64)

        print(f"- {p}:")
        print(f"    RMSE = {rmse(y_true, y_pred):.6e}")
        print(f"    MAE  = {mae(y_true, y_pred):.6e}")
        print(f"    MAPE = {mape_percent(y_true, y_pred):.3f} %")

    # ------------------------------------------------------------------
    # [4] Prediction summaries
    # ------------------------------------------------------------------
    print("\n[3] Sample-level prediction summary")
    for p in calibrated_params:
        true_all = df_rows[f"true_{p}"].to_numpy(dtype=np.float64)
        pred_all = df_rows[f"pred_{p}"].to_numpy(dtype=np.float64)

        print(f"- {p}:")
        print(
            f"    true min/max/mean/std = "
            f"{true_all.min():.6e} / {true_all.max():.6e} / {true_all.mean():.6e} / {true_all.std():.6e}"
        )
        print(
            f"    pred min/max/mean/std = "
            f"{pred_all.min():.6e} / {pred_all.max():.6e} / {pred_all.mean():.6e} / {pred_all.std():.6e}"
        )

    print("\n[4] Aggregated prediction summary")
    for p in calibrated_params:
        true_group = df_grouped[f"true_{p}"].to_numpy(dtype=np.float64)
        pred_group = df_grouped[f"pred_{p}_agg"].to_numpy(dtype=np.float64)

        print(f"- {p}:")
        print(
            f"    true min/max/mean/std = "
            f"{true_group.min():.6e} / {true_group.max():.6e} / {true_group.mean():.6e} / {true_group.std():.6e}"
        )
        print(
            f"    pred min/max/mean/std = "
            f"{pred_group.min():.6e} / {pred_group.max():.6e} / {pred_group.mean():.6e} / {pred_group.std():.6e}"
        )

    # ------------------------------------------------------------------
    # [5] Save CSV files
    # ------------------------------------------------------------------
    if save_csv:
        per_sample_csv = checkpoint_path.with_name(
            checkpoint_path.stem + "_test_predictions_per_sample.csv"
        )
        per_group_csv = checkpoint_path.with_name(
            checkpoint_path.stem + f"_test_predictions_per_group_{aggregate}.csv"
        )

        df_rows.to_csv(per_sample_csv, index=False)
        df_grouped.to_csv(per_group_csv, index=False)

        print(f"\n[5] CSV saved (sample-level) : {per_sample_csv}")
        print(f"[5] CSV saved (group-level)  : {per_group_csv}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference on the test split for the trained inverse CNN calibrator."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--split-json", type=str, required=True, help="Path to split JSON")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--manifest-name", type=str, default="manifest.csv", help="Manifest CSV name")
    parser.add_argument("--no-save-csv", action="store_true", help="Do not save prediction CSV files")

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        split_json_path=args.split_json,
        device=args.device,
        aggregate=args.aggregate,
        save_csv=not args.no_save_csv,
        manifest_name=args.manifest_name,
    )


if __name__ == "__main__":
    main()