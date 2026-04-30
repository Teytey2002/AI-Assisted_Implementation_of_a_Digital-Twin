"""
Inference script for evaluating a trained RCNeuralCalibrator on the test split.

Usage example:
python3 inference.py \
  --checkpoint models/cnn_2026-04-23_14-14-14_best.pth \
  --root-dir ../../../data/LP_DATASET_R1_R2_C \
  --split-json ./splits/rc_r1r2c_nested_fold0.json \
  --device cuda \
  --aggregate mean

  Hybrid version with physics-based selection among sampled candidates:
  python3 inference.py \
  --checkpoint models/prob_cnn_2026-04-27_11-14-50_best.pth \
  --root-dir ../../../data/LP_DATASET_R1_R2_C \
  --split-json ./splits/rc_r1r2c_nested_fold0.json \
  --device cuda \
  --aggregate mean \
  --n-samples 200 \
  --hybrid-select \
  --hybrid-n-candidates 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dtcalib import metrics
import numpy as np
import pandas as pd
import torch

from dtcalib.calibration import RCNeuralCalibrator
from dtcalib.simulation import LowPassR1CR2Simulator
from dtcalib.deep_learning.dataset import RCSignalDataset, TargetSpec
from dtcalib.deep_learning.splits_utils import load_split, get_indices
from dtcalib.metrics import Metrics

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
    n_samples: int = 0,
    hybrid_select: bool = False,
    hybrid_n_candidates: int = 100,
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
        domain="time"
    )

    dataset.set_normalization(
        calibrator.stats.x_mean.cpu(),
        calibrator.stats.x_std.cpu(),
        calibrator.stats.y_mean.cpu(),
        calibrator.stats.y_std.cpu(),
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
        if dataset.domain == "fft_grouped":
            group_sample = dataset.grouped_samples[idx]
            param_dict = group_sample["param_dict"]
            csv_path = group_sample["csv_paths"][0]
            sample_name = group_sample["group_name"]
        else:
            csv_path, param_dict = dataset.samples[idx]
            sample_name = str(csv_path)

        x, y_norm_true = dataset[idx]

        pred = calibrator.predict_from_x(
            x,
            n_samples=n_samples if n_samples > 0 else 0,
        )

        pred_vec = pred.mean_physical.squeeze(0)

        samples_vec = None
        if pred.samples_physical is not None:
            samples_vec = pred.samples_physical.squeeze(0)  # [S, d]

        selected_vec = None
        selected_rmse = None

        if hybrid_select and samples_vec is not None:
            fixed_params = {
                p: float(param_dict[p])
                for p in ("R1", "R2", "C")
                if p not in calibrated_params
            }

            simulator = LowPassR1CR2Simulator(
                calibrated_params=calibrated_params,
                fixed_params=fixed_params,
                y0_mode="dc_from_u0",
            )

            candidate_samples = samples_vec[: int(hybrid_n_candidates)]

            best_score = float("inf")
            best_theta = None

            # ------------------------------------------------------------
            # Case 1: fft_grouped
            # One candidate theta must be evaluated on all frequency CSVs
            # belonging to the same physical group.
            # ------------------------------------------------------------
            if dataset.domain == "fft_grouped":
                eval_csv_paths = group_sample["csv_paths"]

                for theta_candidate in candidate_samples:
                    candidate_scores = []

                    for eval_csv_path in eval_csv_paths:
                        try:
                            df_sig = pd.read_csv(eval_csv_path)
                            time = df_sig.iloc[:, 0].to_numpy(dtype=np.float64)
                            vin = df_sig.iloc[:, 1].to_numpy(dtype=np.float64)
                            vout = df_sig.iloc[:, 2].to_numpy(dtype=np.float64)

                            yhat = simulator.simulate(
                                time,
                                vin,
                                np.asarray(theta_candidate, dtype=np.float64),
                            ).y

                            candidate_scores.append(Metrics.rmse(vout, yhat))

                        except Exception:
                            continue

                    if len(candidate_scores) == 0:
                        continue

                    score = float(np.mean(candidate_scores))

                    if score < best_score:
                        best_score = score
                        best_theta = np.asarray(theta_candidate, dtype=np.float64)

            # ------------------------------------------------------------
            # Case 2: time / fft / time_fft
            # One sample corresponds to one CSV experiment.
            # ------------------------------------------------------------
            else:
                df_sig = pd.read_csv(csv_path)
                time = df_sig.iloc[:, 0].to_numpy(dtype=np.float64)
                vin = df_sig.iloc[:, 1].to_numpy(dtype=np.float64)
                vout = df_sig.iloc[:, 2].to_numpy(dtype=np.float64)

                for theta_candidate in candidate_samples:
                    try:
                        yhat = simulator.simulate(
                            time,
                            vin,
                            np.asarray(theta_candidate, dtype=np.float64),
                        ).y

                        score = Metrics.rmse(vout, yhat)

                    except Exception:
                        continue

                    if score < best_score:
                        best_score = float(score)
                        best_theta = np.asarray(theta_candidate, dtype=np.float64)

            if best_theta is not None:
                selected_vec = best_theta
                selected_rmse = best_score
        row: dict[str, Any] = {
            "index": int(idx),
            "sample_name": sample_name,
            "csv_path": str(csv_path),
        }

        # Store normalized ground truth and normalized probabilistic outputs.
        y_norm_true_np = y_norm_true.detach().cpu().numpy().astype(np.float64)
        mu_norm_np = pred.mean_norm.squeeze(0).astype(np.float64)
        std_norm_np = None
        if pred.std_norm is not None:
            std_norm_np = pred.std_norm.squeeze(0).astype(np.float64)

        for i, p in enumerate(calibrated_params):
            true_val = float(param_dict[p])
            pred_val = float(pred_vec[i])

            row[f"true_{p}"] = true_val
            row[f"pred_{p}"] = pred_val
            row[f"abs_error_{p}"] = float(abs(pred_val - true_val))
            row[f"rel_error_percent_{p}"] = float(abs(pred_val - true_val) / max(abs(true_val), 1e-30) * 100.0)
            row[f"true_norm_{p}"] = float(y_norm_true_np[i])
            row[f"pred_norm_{p}"] = float(mu_norm_np[i])

            if std_norm_np is not None:
                row[f"pred_std_norm_{p}"] = float(std_norm_np[i])
            else:
                row[f"pred_std_norm_{p}"] = None

            if pred.std_physical is not None:
                row[f"pred_std_{p}"] = float(pred.std_physical.squeeze(0)[i])
            else:
                row[f"pred_std_{p}"] = None

            if selected_vec is not None:
                selected_val = float(selected_vec[i])
                row[f"selected_{p}"] = selected_val
                row[f"selected_abs_error_{p}"] = float(abs(selected_val - true_val))
                row[f"selected_rel_error_percent_{p}"] = float(abs(selected_val - true_val) / max(abs(true_val), 1e-30) * 100.0)
            else:
                row[f"selected_{p}"] = None
                row[f"selected_abs_error_{p}"] = None
                row[f"selected_rel_error_percent_{p}"] = None

            if samples_vec is not None:
                row[f"samples_mean_{p}"] = float(np.mean(samples_vec[:, i]))
                row[f"samples_std_{p}"] = float(np.std(samples_vec[:, i]))
                row[f"samples_q025_{p}"] = float(np.quantile(samples_vec[:, i], 0.025))
                row[f"samples_q500_{p}"] = float(np.quantile(samples_vec[:, i], 0.500))
                row[f"samples_q975_{p}"] = float(np.quantile(samples_vec[:, i], 0.975))
            else:
                row[f"samples_mean_{p}"] = None
                row[f"samples_std_{p}"] = None
                row[f"samples_q025_{p}"] = None
                row[f"samples_q500_{p}"] = None
                row[f"samples_q975_{p}"] = None

        row["hybrid_selected_signal_rmse"] = selected_rmse

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
        print(f"    RMSE = {Metrics.rmse(y_true, y_pred):.6e}")
        print(f"    MAE  = {Metrics.mae(y_true, y_pred):.6e}")
        print(f"    MAPE = {Metrics.mape_percent(y_true, y_pred):.3f} %")

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
        print(f"    RMSE = {Metrics.rmse(y_true, y_pred):.6e}")
        print(f"    MAE  = {Metrics.mae(y_true, y_pred):.6e}")
        print(f"    MAPE = {Metrics.mape_percent(y_true, y_pred):.3f} %")

    
    # ------------------------------------------------------------------
    # [4] Probabilistic metrics (if available)
    # ------------------------------------------------------------------
    print("\n[Probabilistic evaluation]")

    has_samples = n_samples > 0 and all(f"samples_q025_{p}" in df_rows.columns for p in calibrated_params)

    for p in calibrated_params:
        y_true = df_rows[f"true_{p}"].to_numpy(dtype=np.float64)
        y_pred = df_rows[f"pred_{p}"].to_numpy(dtype=np.float64)

        std_col = f"pred_std_{p}"
        has_std = std_col in df_rows.columns and df_rows[std_col].notna().any()

        print(f"- {p}:")

        if has_std:
            y_std = pd.to_numeric(df_rows[std_col], errors="coerce").to_numpy(dtype=np.float64)
            mask = np.isfinite(y_std)

            abs_err = np.abs(y_pred[mask] - y_true[mask])

            print(f"    mean predicted std = {np.mean(y_std[mask]):.6e}")
            print(f"    corr(abs_error, pred_std) = {Metrics.safe_corrcoef(abs_err, y_std[mask]):.4f}")

        if has_samples:
            q025 = pd.to_numeric(df_rows[f"samples_q025_{p}"], errors="coerce").to_numpy(dtype=np.float64)
            q500 = pd.to_numeric(df_rows[f"samples_q500_{p}"], errors="coerce").to_numpy(dtype=np.float64)
            q975 = pd.to_numeric(df_rows[f"samples_q975_{p}"], errors="coerce").to_numpy(dtype=np.float64)

            covered_95 = np.mean((y_true >= q025) & (y_true <= q975))
            width_95 = np.mean(q975 - q025)

            print(f"    empirical 95% coverage = {covered_95:.3f}")
            print(f"    mean 95% interval width = {width_95:.6e}")

        # Gaussian NLL in normalized target space.
        std_norm_col = f"pred_std_norm_{p}"
        if std_norm_col in df_rows.columns and df_rows[std_norm_col].notna().any():
            y_true_norm = df_rows[f"true_norm_{p}"].to_numpy(dtype=np.float64)
            y_pred_norm = df_rows[f"pred_norm_{p}"].to_numpy(dtype=np.float64)
            y_std_norm = pd.to_numeric(df_rows[std_norm_col], errors="coerce").to_numpy(dtype=np.float64)

            mask = np.isfinite(y_std_norm)
            print(
                f"    Gaussian NLL norm = "
                f"{Metrics.gaussian_nll(y_true_norm[mask], y_pred_norm[mask], y_std_norm[mask]):.6f}"
            )

    if hybrid_select:
        print("\n[Hybrid physics-based selection]")

        for p in calibrated_params:
            selected_col = f"selected_{p}"

            if selected_col not in df_rows.columns or not df_rows[selected_col].notna().any():
                print(f"- {p}: no valid hybrid selection")
                continue

            y_true = df_rows[f"true_{p}"].to_numpy(dtype=np.float64)
            y_selected = pd.to_numeric(df_rows[selected_col], errors="coerce").to_numpy(dtype=np.float64)

            mask = np.isfinite(y_selected)

            print(f"- {p}:")
            print(f"    RMSE selected = {Metrics.rmse(y_true[mask], y_selected[mask]):.6e}")
            print(f"    MAE selected  = {Metrics.mae(y_true[mask], y_selected[mask]):.6e}")
            print(f"    MAPE selected = {Metrics.mape_percent(y_true[mask], y_selected[mask]):.3f} %")

    # ------------------------------------------------------------------
    # [5] Prediction summaries
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
    # [6] Save CSV files
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
    parser.add_argument("--n-samples", type=int, default=0, help="Number of parameter samples for probabilistic models. Use 0 for point prediction only.")
    parser.add_argument("--hybrid-select", action="store_true", help="Use sampled parameters and select the candidate minimizing simulation error.")
    parser.add_argument("--hybrid-n-candidates", type=int, default=100, help="Number of sampled candidates used for hybrid physics-based selection.")

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        split_json_path=args.split_json,
        device=args.device,
        aggregate=args.aggregate,
        save_csv=not args.no_save_csv,
        manifest_name=args.manifest_name,
        n_samples=args.n_samples,
        hybrid_select=args.hybrid_select,
        hybrid_n_candidates=args.hybrid_n_candidates,
    )


if __name__ == "__main__":
    main()