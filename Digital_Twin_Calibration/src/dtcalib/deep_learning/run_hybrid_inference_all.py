from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


DATASET_NAMES = [
    #"ThreeStageRC_caps_only",
    #"ThreeStageRC_all_components",
    #"ThreeStageRLC_caps_only",
    #"ThreeStageRLC_inductors_only",
    #"ThreeStageRLC_caps_inductors",
    #"ThreeStageRLC_all_components",
    "DiodeClippedRC_r_c_only",
    "DiodeClippedRC_r_c_diode",
]


def find_checkpoint(results_dir: Path, dataset_name: str) -> Path:
    ckpts = sorted(
        (results_dir / dataset_name / "prob_cnn").glob("prob_cnn_*_best.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        raise FileNotFoundError(f"No Prob-CNN checkpoint found for {dataset_name}")
    return ckpts[-1]


def run_one_hybrid(
    dataset_name: str,
    datasets_root: str,
    results_dir: str,
    splits_dir: str,
    inference_script: str,
    device: str,
    n_samples: int,
    hybrid_n_candidates: int,
) -> dict:
    start = time.perf_counter()

    datasets_root_p = Path(datasets_root)
    results_dir_p = Path(results_dir)
    splits_dir_p = Path(splits_dir)

    dataset_root = datasets_root_p / dataset_name
    split_json = splits_dir_p / f"{dataset_name}_fft_grouped_split.json"
    checkpoint = find_checkpoint(results_dir_p, dataset_name)

    cmd = [
        sys.executable,
        str(inference_script),
        "--checkpoint", str(checkpoint),
        "--root-dir", str(dataset_root),
        "--split-json", str(split_json),
        "--device", device,
        "--aggregate", "mean",
        "--n-samples", str(n_samples),
        "--hybrid-select",
        "--hybrid-n-candidates", str(hybrid_n_candidates),
    ]

    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    hybrid_dir = results_dir_p / dataset_name / "hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = checkpoint.with_name(
        checkpoint.stem + "_test_predictions_per_sample.csv"
    )
    per_group_csv = checkpoint.with_name(
        checkpoint.stem + "_test_predictions_per_group_mean.csv"
    )

    target_sample_csv = hybrid_dir / per_sample_csv.name
    target_group_csv = hybrid_dir / per_group_csv.name

    per_sample_csv.replace(target_sample_csv)
    per_group_csv.replace(target_group_csv)
    elapsed = time.perf_counter() - start

    return {
        "dataset": dataset_name,
        "checkpoint": str(checkpoint),
        "n_samples": n_samples,
        "hybrid_n_candidates": hybrid_n_candidates,
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60.0,
        "hybrid_predictions_csv": str(target_sample_csv),
        "hybrid_group_csv": str(target_group_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets-root", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, required=True)
    parser.add_argument("--inference-script", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--hybrid-n-candidates", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=3)

    args = parser.parse_args()

    global_start = time.perf_counter()
    runtime_rows = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_one_hybrid,
                dataset_name,
                args.datasets_root,
                args.results_dir,
                args.splits_dir,
                args.inference_script,
                args.device,
                args.n_samples,
                args.hybrid_n_candidates,
            ): dataset_name
            for dataset_name in DATASET_NAMES
        }

        for future in as_completed(futures):
            dataset_name = futures[future]
            try:
                row = future.result()
                runtime_rows.append(row)
                print(
                    f"[OK] {dataset_name} | "
                    f"{row['runtime_minutes']:.2f} min",
                    flush=True,
                )
            except Exception as e:
                print(f"[FAILED] {dataset_name}: {e}", flush=True)
                raise

    total_elapsed = time.perf_counter() - global_start

    runtime_rows.append(
        {
            "dataset": "TOTAL",
            "checkpoint": "",
            "n_samples": args.n_samples,
            "hybrid_n_candidates": args.hybrid_n_candidates,
            "runtime_seconds": total_elapsed,
            "runtime_minutes": total_elapsed / 60.0,
        }
    )

    df = pd.DataFrame(runtime_rows)
    out_csv = Path(args.results_dir) / "summary_all_runs_hybrid.csv"
    df.to_csv(out_csv, index=False)

    print("\nAll hybrid inferences finished.")
    print(f"Runtime summary saved to: {out_csv}")


if __name__ == "__main__":
    main()