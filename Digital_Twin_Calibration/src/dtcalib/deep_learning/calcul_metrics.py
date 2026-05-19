from __future__ import annotations

import argparse
import re
from pathlib import Path

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


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-30)
        )
        * 100.0
    )


def find_hybrid_csv(results_dir: Path, dataset_name: str) -> Path | None:
    hybrid_dir = results_dir / dataset_name / "hybrid"

    if not hybrid_dir.exists():
        return None

    files = sorted(
        hybrid_dir.glob("*_test_predictions_per_sample.csv"),
        key=lambda p: p.stat().st_mtime,
    )

    if not files:
        return None

    return files[-1]


def extract_runtime_from_log(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}

    pattern = re.compile(r"\[OK\]\s+(.+?)\s+\|\s+([0-9.]+)\s+min")

    runtimes = {}

    text = log_path.read_text(errors="ignore")

    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            dataset = match.group(1).strip()
            minutes = float(match.group(2))
            runtimes[dataset] = minutes

    return runtimes


def compute_metrics_from_csv(csv_path: Path) -> dict[str, float | str | int]:
    df = pd.read_csv(csv_path)

    params = []

    for col in df.columns:
        if col.startswith("true_"):
            p = col.replace("true_", "")
            if f"pred_{p}" in df.columns and f"selected_{p}" in df.columns:
                params.append(p)

    if not params:
        raise ValueError(f"No true/pred/selected parameter columns found in {csv_path}")

    row: dict[str, float | str | int] = {
        "n_test_samples": int(len(df)),
        "calibrated_params": ",".join(params),
        "n_calibrated_params": int(len(params)),
    }

    direct_mapes = []
    hybrid_mapes = []

    for p in params:
        y_true = df[f"true_{p}"].to_numpy(dtype=np.float64)
        y_pred = df[f"pred_{p}"].to_numpy(dtype=np.float64)

        selected = pd.to_numeric(df[f"selected_{p}"], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(selected)

        direct_mape = mape_percent(y_true, y_pred)

        if np.any(mask):
            hybrid_mape = mape_percent(y_true[mask], selected[mask])
        else:
            hybrid_mape = np.nan

        row[f"direct_mape_{p}"] = direct_mape
        row[f"hybrid_mape_{p}"] = hybrid_mape

        direct_mapes.append(direct_mape)
        hybrid_mapes.append(hybrid_mape)

    direct_mapes_np = np.asarray(direct_mapes, dtype=np.float64)
    hybrid_mapes_np = np.asarray(hybrid_mapes, dtype=np.float64)

    row["direct_mape_mean"] = float(np.nanmean(direct_mapes_np))
    row["hybrid_mape_mean"] = float(np.nanmean(hybrid_mapes_np))

    row["direct_mape_max"] = float(np.nanmax(direct_mapes_np))
    row["hybrid_mape_max"] = float(np.nanmax(hybrid_mapes_np))

    row["hybrid_minus_direct_mape_mean"] = float(
        row["hybrid_mape_mean"] - row["direct_mape_mean"]
    )

    row["hybrid_improvement_percent"] = float(
        (row["direct_mape_mean"] - row["hybrid_mape_mean"])
        / max(row["direct_mape_mean"], 1e-30)
        * 100.0
    )

    if "hybrid_selected_signal_rmse" in df.columns:
        rmse = pd.to_numeric(
            df["hybrid_selected_signal_rmse"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        row["hybrid_signal_rmse_mean"] = float(np.nanmean(rmse))
        row["hybrid_signal_rmse_median"] = float(np.nanmedian(rmse))

    return row


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--log-file", type=str, default="hybrid_inference_all.log")
    parser.add_argument("--output-csv", type=str, default="summary_all_runs_hybrid.csv")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    log_path = Path(args.log_file)

    runtimes = extract_runtime_from_log(log_path)

    rows = []

    for dataset_name in DATASET_NAMES:
        csv_path = find_hybrid_csv(results_dir, dataset_name)

        if csv_path is None:
            rows.append(
                {
                    "dataset": dataset_name,
                    "status": "missing_hybrid_csv",
                    "hybrid_csv": "",
                    "runtime_minutes": runtimes.get(dataset_name, np.nan),
                }
            )
            continue

        metrics = compute_metrics_from_csv(csv_path)

        rows.append(
            {
                "dataset": dataset_name,
                "status": "ok",
                "hybrid_csv": str(csv_path),
                "runtime_minutes": runtimes.get(dataset_name, np.nan),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)

    out_path = results_dir / args.output_csv
    df.to_csv(out_path, index=False)

    print(f"Hybrid summary saved to: {out_path}")
    print(df[[
        "dataset",
        "status",
        "direct_mape_mean",
        "hybrid_mape_mean",
        "hybrid_improvement_percent",
        "runtime_minutes",
    ]])


if __name__ == "__main__":
    main()