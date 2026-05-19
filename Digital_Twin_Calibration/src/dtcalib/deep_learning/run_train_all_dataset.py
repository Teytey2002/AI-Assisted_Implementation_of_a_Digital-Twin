"""" 
Example of usage : 
python3 run_train_all_dataset.py \
  --datasets-root ../../../data/DL_DATASETS \
  --splits-dir ./splits \
  --train-script ./train.py \
  --inference-script ./inference.py \
  --models-dir ./models \
  --output-dir ./final_results \
  --device cuda \
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASET_NAMES = [
    "ThreeStageRC_caps_only",
    "ThreeStageRC_all_components",
    "ThreeStageRLC_caps_only",
    "ThreeStageRLC_inductors_only",
    "ThreeStageRLC_caps_inductors",
    "ThreeStageRLC_all_components",
    "DiodeClippedRC_r_c_only",
    "DiodeClippedRC_r_c_diode",
]

MODEL_NAMES = ["cnn", "prob_cnn"]


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def load_metadata(dataset_root: Path) -> dict:
    path = dataset_root / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def find_latest_checkpoint(
    models_dir: Path,
    model_name: str,
    before_files: set[Path],
) -> Path:
    candidates = sorted(
        models_dir.glob(f"{model_name}_*_best.pth"),
        key=lambda p: p.stat().st_mtime,
    )

    new_candidates = [p for p in candidates if p not in before_files]

    if new_candidates:
        return new_candidates[-1]

    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {models_dir} for model '{model_name}'."
    )


def expected_prediction_csv(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(
        checkpoint_path.stem + "_test_predictions_per_sample.csv"
    )


def compute_param_metrics(
    csv_path: Path,
    calibrated_params: Iterable[str],
) -> dict[str, float]:
    df = pd.read_csv(csv_path)

    out: dict[str, float] = {}

    mapes = []
    rmses = []
    maes = []

    for p in calibrated_params:
        true_col = f"true_{p}"
        pred_col = f"pred_{p}"

        if true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(
                f"Missing columns for parameter {p}: "
                f"{true_col}, {pred_col} in {csv_path}"
            )

        y_true = df[true_col].to_numpy(dtype=np.float64)
        y_pred = df[pred_col].to_numpy(dtype=np.float64)

        err = y_pred - y_true

        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        mape = float(
            np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-30)) * 100.0
        )

        out[f"rmse_{p}"] = rmse
        out[f"mae_{p}"] = mae
        out[f"mape_{p}"] = mape

        rmses.append(rmse)
        maes.append(mae)
        mapes.append(mape)

    out["rmse_mean_over_params"] = float(np.mean(rmses))
    out["mae_mean_over_params"] = float(np.mean(maes))
    out["mape_mean_over_params"] = float(np.mean(mapes))
    out["mape_max_over_params"] = float(np.max(mapes))

    return out


def run_all(
    *,
    datasets_root: Path,
    splits_dir: Path,
    train_script: Path,
    inference_script: Path,
    models_dir: Path,
    output_dir: Path,
    device: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for dataset_name in DATASET_NAMES:
        dataset_root = datasets_root / dataset_name
        split_json = splits_dir / f"{dataset_name}_fft_grouped_split.json"

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_root}")
        if not split_json.exists():
            raise FileNotFoundError(f"Split not found: {split_json}")

        metadata = load_metadata(dataset_root)
        calibrated_params = tuple(metadata["calibrated_params"])

        for model_name in MODEL_NAMES:
            print("\n" + "=" * 90)
            print(f"DATASET: {dataset_name}")
            print(f"MODEL  : {model_name}")
            print("=" * 90)

            before_ckpts = set(models_dir.glob(f"{model_name}_*_best.pth"))

            train_cmd = [
                sys.executable,
                str(train_script),
                "--dataset",
                str(dataset_root),
                "--split",
                str(split_json),
                "--model",
                model_name,
            ]

            run_command(train_cmd)

            checkpoint_path = find_latest_checkpoint(
                models_dir=models_dir,
                model_name=model_name,
                before_files=before_ckpts,
            )

            infer_cmd = [
                sys.executable,
                str(inference_script),
                "--checkpoint",
                str(checkpoint_path),
                "--root-dir",
                str(dataset_root),
                "--split-json",
                str(split_json),
                "--device",
                device,
                "--aggregate",
                "mean",
            ]

            run_command(infer_cmd)

            pred_csv = expected_prediction_csv(checkpoint_path)

            if not pred_csv.exists():
                raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

            metrics = compute_param_metrics(
                csv_path=pred_csv,
                calibrated_params=calibrated_params,
            )

            run_dir = output_dir / dataset_name / model_name
            run_dir.mkdir(parents=True, exist_ok=True)

            copied_checkpoint = run_dir / checkpoint_path.name
            copied_predictions = run_dir / pred_csv.name

            shutil.copy2(checkpoint_path, copied_checkpoint)
            shutil.copy2(pred_csv, copied_predictions)

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "checkpoint": str(copied_checkpoint),
                "predictions_csv": str(copied_predictions),
                "split_json": str(split_json),
                "n_calibrated_params": len(calibrated_params),
                "calibrated_params": ",".join(calibrated_params),
                **metrics,
            }

            rows.append(row)

            print(
                f"[OK] {dataset_name} | {model_name} | "
                f"mean MAPE = {row['mape_mean_over_params']:.4f}% | "
                f"max MAPE = {row['mape_max_over_params']:.4f}%"
            )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary_all_runs.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 90)
    print("ALL TRAININGS FINISHED")
    print("=" * 90)
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets-root", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, required=True)
    parser.add_argument("--train-script", type=str, required=True)
    parser.add_argument("--inference-script", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--output-dir", type=str, default="./final_results")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_all(
        datasets_root=Path(args.datasets_root),
        splits_dir=Path(args.splits_dir),
        train_script=Path(args.train_script),
        inference_script=Path(args.inference_script),
        models_dir=Path(args.models_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
    )


if __name__ == "__main__":
    main()