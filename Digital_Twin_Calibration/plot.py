"""
Example of usage from project root:
    Juste for CNN
    python3 plot.py \
        --sample-csv ./src/dtcalib/deep_learning/models/cnn_2026-03-23_13-00-46_best_test_predictions_per_sample.csv \
        --capacity-csv ./src/dtcalib/deep_learning/models/cnn_2026-03-23_13-00-46_best_test_predictions_per_capacity_mean.csv \
        --outdir ./plots/cnn \
        --label "Inverse CNN"

    Compare CNN vs Probabilistic CNN
    python3 plot.py \
        --sample-csv ./src/dtcalib/deep_learning/models/cnn_2026-03-23_13-00-46_best_test_predictions_per_sample.csv \
        --capacity-csv ./src/dtcalib/deep_learning/models/cnn_2026-03-23_13-00-46_best_test_predictions_per_capacity_mean.csv \
        --label "Inverse CNN" \
        --sample-csv-b ./src/dtcalib/deep_learning/models/prob_cnn_2026-03-23_11-33-12_best_test_predictions_per_sample.csv \
        --capacity-csv-b ./src/dtcalib/deep_learning/models/prob_cnn_2026-03-23_11-33-12_best_test_predictions_per_capacity_mean.csv \
        --label-b "Probabilistic CNN" \
        --outdir ./plots/compare_cnn_probcnn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def infer_label(path: str | Path, fallback: str) -> str:
    p = Path(path)
    name = p.stem.lower()
    if "prob" in name:
        return "Probabilistic CNN"
    if "cnn" in name:
        return "Inverse CNN"
    return fallback


def add_identity_line(ax, values_a: np.ndarray, values_b: np.ndarray) -> None:
    vmin = min(float(np.min(values_a)), float(np.min(values_b)))
    vmax = max(float(np.max(values_a)), float(np.max(values_b)))
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)


def savefig(fig: plt.Figure, outdir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summary_metrics(df_sample: pd.DataFrame) -> dict[str, float]:
    true_c = df_sample["true_C"].to_numpy(dtype=float)
    pred_c = df_sample["pred_C"].to_numpy(dtype=float)
    abs_err = np.abs(pred_c - true_c)
    rel_err = abs_err / np.maximum(np.abs(true_c), 1e-30) * 100.0

    rmse = float(np.sqrt(np.mean((pred_c - true_c) ** 2)))
    mae = float(np.mean(abs_err))
    mape = float(np.mean(rel_err))

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


# ------------------------------------------------------------
# Single-model plots
# ------------------------------------------------------------
def plot_sample_scatter(df_sample: pd.DataFrame, label: str, outdir: Path, suffix: str) -> None:
    true_c = df_sample["true_C"].to_numpy(dtype=float)
    pred_c = df_sample["pred_C"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_c, pred_c, alpha=0.6, s=18)
    add_identity_line(ax, true_c, pred_c)
    ax.set_title(f"{label} - Sample-level: True C vs Predicted C")
    ax.set_xlabel("True C [F]")
    ax.set_ylabel("Predicted C [F]")
    savefig(fig, outdir, f"{suffix}_sample_true_vs_pred.png")


def plot_capacity_scatter(df_capacity: pd.DataFrame, label: str, outdir: Path, suffix: str) -> None:
    true_c = df_capacity["true_C"].to_numpy(dtype=float)
    pred_c = df_capacity["pred_C_agg"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_c, pred_c, s=45)
    add_identity_line(ax, true_c, pred_c)
    ax.set_title(f"{label} - Capacity-level: True C vs Aggregated Predicted C")
    ax.set_xlabel("True C [F]")
    ax.set_ylabel("Aggregated Predicted C [F]")
    savefig(fig, outdir, f"{suffix}_capacity_true_vs_pred.png")


def plot_relative_error_by_capacity(df_capacity: pd.DataFrame, label: str, outdir: Path, suffix: str) -> None:
    df_sorted = df_capacity.sort_values("true_C").copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df_sorted["true_C"].to_numpy(dtype=float),
        df_sorted["rel_error_percent"].to_numpy(dtype=float),
        marker="o",
    )
    ax.set_title(f"{label} - Relative Error by True Capacity")
    ax.set_xlabel("True C [F]")
    ax.set_ylabel("Relative Error [%]")
    savefig(fig, outdir, f"{suffix}_relative_error_by_capacity.png")


def plot_error_histogram(df_sample: pd.DataFrame, label: str, outdir: Path, suffix: str) -> None:
    rel_err = df_sample["rel_error_percent"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(rel_err, bins=30)
    ax.set_title(f"{label} - Histogram of Relative Errors")
    ax.set_xlabel("Relative Error [%]")
    ax.set_ylabel("Count")
    savefig(fig, outdir, f"{suffix}_hist_relative_error.png")


def plot_boxplot_error_by_capacity(df_sample: pd.DataFrame, label: str, outdir: Path, suffix: str) -> None:
    df_sorted = df_sample.sort_values("true_C").copy()
    grouped = list(df_sorted.groupby("true_C", sort=True))

    labels = [f"{true_c:.2e}" for true_c, _ in grouped]
    data = [g["rel_error_percent"].to_numpy(dtype=float) for _, g in grouped]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(f"{label} - Relative Error Distribution by True Capacity")
    ax.set_xlabel("True C [F]")
    ax.set_ylabel("Relative Error [%]")
    ax.tick_params(axis="x", rotation=45)
    savefig(fig, outdir, f"{suffix}_boxplot_relative_error_by_capacity.png")


def plot_uncertainty_if_available(df_sample: pd.DataFrame, label: str, outdir: Path, suffix: str) -> None:
    if "pred_logC_std" not in df_sample.columns:
        return

    df = df_sample.copy()
    df["abs_error_C"] = np.abs(df["pred_C"] - df["true_C"])

    x = df["pred_logC_std"].to_numpy(dtype=float)
    y = df["abs_error_C"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, s=18)
    ax.set_title(f"{label} - Predicted Uncertainty vs Absolute Error")
    ax.set_xlabel("Predicted std in logC")
    ax.set_ylabel("Absolute Error on C [F]")
    savefig(fig, outdir, f"{suffix}_uncertainty_vs_abs_error.png")


def plot_all_single(
    sample_csv: str | Path,
    capacity_csv: str | Path,
    outdir: str | Path,
    label: str | None = None,
    suffix: str = "model",
) -> None:
    df_sample = load_csv(sample_csv)
    df_capacity = load_csv(capacity_csv)
    outdir = ensure_outdir(outdir)

    if label is None:
        label = infer_label(sample_csv, "Model")

    metrics = summary_metrics(df_sample)
    print(f"\n[{label}]")
    print(f"RMSE(C) = {metrics['rmse']:.6e} F")
    print(f"MAE(C)  = {metrics['mae']:.6e} F")
    print(f"MAPE(C) = {metrics['mape']:.3f} %")

    plot_sample_scatter(df_sample, label, outdir, suffix)
    plot_capacity_scatter(df_capacity, label, outdir, suffix)
    plot_relative_error_by_capacity(df_capacity, label, outdir, suffix)
    plot_error_histogram(df_sample, label, outdir, suffix)
    plot_boxplot_error_by_capacity(df_sample, label, outdir, suffix)
    plot_uncertainty_if_available(df_sample, label, outdir, suffix)


# ------------------------------------------------------------
# Comparison plots (2 models)
# ------------------------------------------------------------
def plot_compare_sample_scatter(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> None:
    true_a = df_a["true_C"].to_numpy(dtype=float)
    pred_a = df_a["pred_C"].to_numpy(dtype=float)
    true_b = df_b["true_C"].to_numpy(dtype=float)
    pred_b = df_b["pred_C"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_a, pred_a, alpha=0.5, s=18, label=label_a)
    ax.scatter(true_b, pred_b, alpha=0.5, s=18, label=label_b)
    add_identity_line(ax, np.concatenate([true_a, true_b]), np.concatenate([pred_a, pred_b]))
    ax.set_title("Sample-level: True C vs Predicted C")
    ax.set_xlabel("True C [F]")
    ax.set_ylabel("Predicted C [F]")
    ax.legend()
    savefig(fig, outdir, "compare_sample_true_vs_pred.png")


def plot_compare_capacity_curves(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> None:
    a = df_a.sort_values("true_C")
    b = df_b.sort_values("true_C")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a["true_C"], a["pred_C_agg"], marker="o", label=label_a)
    ax.plot(b["true_C"], b["pred_C_agg"], marker="o", label=label_b)
    ax.plot(a["true_C"], a["true_C"], linestyle="--", label="Ideal")
    ax.set_title("Capacity-level aggregated predictions")
    ax.set_xlabel("True C [F]")
    ax.set_ylabel("Aggregated Predicted C [F]")
    ax.legend()
    savefig(fig, outdir, "compare_capacity_predictions.png")


def plot_compare_relative_error(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> None:
    a = df_a.sort_values("true_C")
    b = df_b.sort_values("true_C")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a["true_C"], a["rel_error_percent"], marker="o", label=label_a)
    ax.plot(b["true_C"], b["rel_error_percent"], marker="o", label=label_b)
    ax.set_title("Relative Error by True Capacity")
    ax.set_xlabel("True C [F]")
    ax.set_ylabel("Relative Error [%]")
    ax.legend()
    savefig(fig, outdir, "compare_relative_error_by_capacity.png")


def plot_compare_histograms(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> None:
    rel_a = df_a["rel_error_percent"].to_numpy(dtype=float)
    rel_b = df_b["rel_error_percent"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rel_a, bins=30, alpha=0.6, label=label_a)
    ax.hist(rel_b, bins=30, alpha=0.6, label=label_b)
    ax.set_title("Histogram of Relative Errors")
    ax.set_xlabel("Relative Error [%]")
    ax.set_ylabel("Count")
    ax.legend()
    savefig(fig, outdir, "compare_hist_relative_error.png")


def plot_compare_bar_metrics(
    df_sample_a: pd.DataFrame,
    df_sample_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> None:
    m_a = summary_metrics(df_sample_a)
    m_b = summary_metrics(df_sample_b)

    metric_names = ["RMSE", "MAE", "MAPE"]
    values_a = [m_a["rmse"], m_a["mae"], m_a["mape"]]
    values_b = [m_b["rmse"], m_b["mae"], m_b["mape"]]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, values_a, width, label=label_a)
    ax.bar(x + width / 2, values_b, width, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_title("Metric Comparison")
    ax.legend()
    savefig(fig, outdir, "compare_metrics_bar.png")


def compare_models(
    sample_csv_a: str | Path,
    capacity_csv_a: str | Path,
    sample_csv_b: str | Path,
    capacity_csv_b: str | Path,
    outdir: str | Path,
    label_a: str | None = None,
    label_b: str | None = None,
) -> None:
    df_sample_a = load_csv(sample_csv_a)
    df_capacity_a = load_csv(capacity_csv_a)
    df_sample_b = load_csv(sample_csv_b)
    df_capacity_b = load_csv(capacity_csv_b)
    outdir = ensure_outdir(outdir)

    if label_a is None:
        label_a = infer_label(sample_csv_a, "Model A")
    if label_b is None:
        label_b = infer_label(sample_csv_b, "Model B")

    print(f"\nComparing: {label_a} vs {label_b}")
    print("A metrics:", summary_metrics(df_sample_a))
    print("B metrics:", summary_metrics(df_sample_b))

    plot_compare_sample_scatter(df_sample_a, df_sample_b, label_a, label_b, outdir)
    plot_compare_capacity_curves(df_capacity_a, df_capacity_b, label_a, label_b, outdir)
    plot_compare_relative_error(df_capacity_a, df_capacity_b, label_a, label_b, outdir)
    plot_compare_histograms(df_sample_a, df_sample_b, label_a, label_b, outdir)
    plot_compare_bar_metrics(df_sample_a, df_sample_b, label_a, label_b, outdir)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot important figures from inference CSV files.")

    parser.add_argument("--sample-csv", type=str, required=True, help="Per-sample CSV for model A")
    parser.add_argument("--capacity-csv", type=str, required=True, help="Per-capacity CSV for model A")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for figures")
    parser.add_argument("--label", type=str, default=None, help="Optional label for model A")

    parser.add_argument("--sample-csv-b", type=str, default=None, help="Per-sample CSV for model B")
    parser.add_argument("--capacity-csv-b", type=str, default=None, help="Per-capacity CSV for model B")
    parser.add_argument("--label-b", type=str, default=None, help="Optional label for model B")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_all_single(
        sample_csv=args.sample_csv,
        capacity_csv=args.capacity_csv,
        outdir=args.outdir,
        label=args.label,
        suffix="model_a",
    )

    if args.sample_csv_b is not None and args.capacity_csv_b is not None:
        plot_all_single(
            sample_csv=args.sample_csv_b,
            capacity_csv=args.capacity_csv_b,
            outdir=args.outdir,
            label=args.label_b,
            suffix="model_b",
        )

        compare_models(
            sample_csv_a=args.sample_csv,
            capacity_csv_a=args.capacity_csv,
            sample_csv_b=args.sample_csv_b,
            capacity_csv_b=args.capacity_csv_b,
            outdir=args.outdir,
            label_a=args.label,
            label_b=args.label_b,
        )

    print(f"\nPlots saved in: {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()