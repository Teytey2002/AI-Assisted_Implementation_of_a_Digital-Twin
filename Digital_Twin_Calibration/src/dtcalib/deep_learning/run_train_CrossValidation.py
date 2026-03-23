"""
Example of usage from deep_learning directory: 
Put --model prob_cnn and put --model cnn for the other model
    python3 run_train_CrossValidation.py \
        --dataset ../../../data/ALL_LP_DATASETS_CSV_Deep_learning \
        --splits-dir ./splits \
        --model prob_cnn \
        --train-script ./train.py \
        --inference-script ../../../inference.py \
        --models-dir ./models \
        --output-dir ./cv_results \
        --device cuda
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))) * 100.0)


# ------------------------------------------------------------
# CSV metric extraction
# ------------------------------------------------------------
def compute_metrics_from_sample_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)

    true_c = df["true_C"].to_numpy(dtype=np.float64)
    pred_c = df["pred_C"].to_numpy(dtype=np.float64)

    return {
        "rmse_sample": rmse(true_c, pred_c),
        "mae_sample": mae(true_c, pred_c),
        "mape_sample": mape_percent(true_c, pred_c),
    }


def compute_metrics_from_capacity_csv(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)

    true_c = df["true_C"].to_numpy(dtype=np.float64)
    pred_c = df["pred_C_agg"].to_numpy(dtype=np.float64)

    return {
        "rmse_capacity": rmse(true_c, pred_c),
        "mae_capacity": mae(true_c, pred_c),
        "mape_capacity": mape_percent(true_c, pred_c),
        "n_capacities": int(len(df)),
    }


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def find_split_files(splits_dir: Path) -> list[Path]:
    split_files = sorted(splits_dir.glob("rc_nested_fold*.json"))
    if not split_files:
        raise FileNotFoundError(f"No split files found in: {splits_dir}")
    return split_files


def find_latest_checkpoint(models_dir: Path, model_name: str, before_files: set[Path]) -> Path:
    candidates = sorted(models_dir.glob(f"{model_name}_*_best.pth"), key=lambda p: p.stat().st_mtime)
    new_candidates = [p for p in candidates if p not in before_files]

    if new_candidates:
        return new_candidates[-1]

    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {models_dir} for model prefix '{model_name}_*_best.pth'"
    )


def expected_csv_paths_from_checkpoint(checkpoint_path: Path, aggregate: str) -> tuple[Path, Path]:
    per_sample_csv = checkpoint_path.with_name(checkpoint_path.stem + "_test_predictions_per_sample.csv")
    per_capacity_csv = checkpoint_path.with_name(
        checkpoint_path.stem + f"_test_predictions_per_capacity_{aggregate}.csv"
    )
    return per_sample_csv, per_capacity_csv


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def summarize_metrics(df: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in metric_cols:
        values = df[col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": col,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Main CV runner
# ------------------------------------------------------------
def run_dl_cv(
    *,
    dataset_root: Path,
    splits_dir: Path,
    model_name: str,
    train_script: Path,
    inference_script: Path,
    models_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    aggregate: str = "mean",
) -> None:
    split_files = find_split_files(splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Deep Learning Cross-Validation Runner")
    print("========================================")
    print(f"Dataset root    : {dataset_root}")
    print(f"Splits dir      : {splits_dir}")
    print(f"Model           : {model_name}")
    print(f"Train script    : {train_script}")
    print(f"Inference script: {inference_script}")
    print(f"Models dir      : {models_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Device          : {device}")
    print(f"Aggregate       : {aggregate}")
    print(f"#folds          : {len(split_files)}")

    fold_rows: list[dict] = []

    for fold_id, split_path in enumerate(split_files):
        print("\n" + "=" * 80)
        print(f"[FOLD {fold_id}] split = {split_path.name}")
        print("=" * 80)

        before_ckpts = set(models_dir.glob(f"{model_name}_*_best.pth"))

        # -------------------------
        # Train
        # -------------------------
        train_cmd = [
            sys.executable,
            str(train_script),
            "--dataset",
            str(dataset_root),
            "--split",
            str(split_path),
            "--model",
            model_name,
        ]
        run_command(train_cmd)

        checkpoint_path = find_latest_checkpoint(models_dir, model_name, before_ckpts)
        print(f"[FOLD {fold_id}] checkpoint: {checkpoint_path}")

        # -------------------------
        # Inference
        # -------------------------
        infer_cmd = [
            sys.executable,
            str(inference_script),
            "--checkpoint",
            str(checkpoint_path),
            "--root-dir",
            str(dataset_root),
            "--split-json",
            str(split_path),
            "--device",
            device,
            "--aggregate",
            aggregate,
        ]
        run_command(infer_cmd)

        per_sample_csv, per_capacity_csv = expected_csv_paths_from_checkpoint(checkpoint_path, aggregate)

        if not per_sample_csv.exists():
            raise FileNotFoundError(f"Expected sample CSV not found: {per_sample_csv}")
        if not per_capacity_csv.exists():
            raise FileNotFoundError(f"Expected capacity CSV not found: {per_capacity_csv}")

        # -------------------------
        # Read metrics
        # -------------------------
        sample_metrics = compute_metrics_from_sample_csv(per_sample_csv)
        capacity_metrics = compute_metrics_from_capacity_csv(per_capacity_csv)

        row = {
            "fold": fold_id,
            "split_json": str(split_path),
            "checkpoint": str(checkpoint_path),
            "sample_csv": str(per_sample_csv),
            "capacity_csv": str(per_capacity_csv),
            **sample_metrics,
            **capacity_metrics,
        }
        fold_rows.append(row)

        print(f"[FOLD {fold_id}] sample RMSE(C)   = {row['rmse_sample']:.6e} F")
        print(f"[FOLD {fold_id}] sample MAE(C)    = {row['mae_sample']:.6e} F")
        print(f"[FOLD {fold_id}] sample MAPE(C)   = {row['mape_sample']:.3f} %")
        print(f"[FOLD {fold_id}] capacity RMSE(C) = {row['rmse_capacity']:.6e} F")
        print(f"[FOLD {fold_id}] capacity MAE(C)  = {row['mae_capacity']:.6e} F")
        print(f"[FOLD {fold_id}] capacity MAPE(C) = {row['mape_capacity']:.3f} %")

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------
    df_folds = pd.DataFrame(fold_rows)

    metrics_to_summarize = [
        "rmse_sample",
        "mae_sample",
        "mape_sample",
        "rmse_capacity",
        "mae_capacity",
        "mape_capacity",
    ]
    df_summary = summarize_metrics(df_folds, metrics_to_summarize)

    folds_csv = output_dir / f"{model_name}_cv_results_per_fold.csv"
    summary_csv = output_dir / f"{model_name}_cv_summary.csv"

    df_folds.to_csv(folds_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("FINAL CV SUMMARY")
    print("=" * 80)

    for _, row in df_summary.iterrows():
        metric = row["metric"]
        mean = row["mean"]
        std = row["std"]

        unit = "F" if "rmse" in metric or "mae" in metric else "%"
        print(f"{metric:16s}: {mean:.6e} ± {std:.6e} {unit}" if unit == "F"
              else f"{metric:16s}: {mean:.3f} ± {std:.3f} {unit}")

    print(f"\nSaved per-fold results : {folds_csv}")
    print(f"Saved summary results  : {summary_csv}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep-learning cross-validation over all rc_nested_fold*.json splits."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--splits-dir", type=str, required=True, help="Directory containing rc_nested_fold*.json")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "prob_cnn"], help="Model name")
    parser.add_argument("--train-script", type=str, default="train.py", help="Path to train.py")
    parser.add_argument("--inference-script", type=str, default="inference.py", help="Path to inference.py")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory where checkpoints are saved by train.py")
    parser.add_argument("--output-dir", type=str, default="cv_results", help="Directory where CV summary CSV files will be written")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu for inference")
    parser.add_argument("--aggregate", type=str, default="mean", choices=["mean", "median"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dl_cv(
        dataset_root=Path(args.dataset),
        splits_dir=Path(args.splits_dir),
        model_name=args.model,
        train_script=Path(args.train_script),
        inference_script=Path(args.inference_script),
        models_dir=Path(args.models_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        aggregate=args.aggregate,
    )


if __name__ == "__main__":
    main()