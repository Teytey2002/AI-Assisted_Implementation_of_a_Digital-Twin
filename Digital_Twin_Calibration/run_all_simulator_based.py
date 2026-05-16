from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm


RESULTS_DIR = Path("results/simulator_based_2")
SUMMARY_CSV = RESULTS_DIR / "campaign_summary.csv"

JOBS = [
    {
        "dataset": "data/SIMULATOR_BASED_DATASETS/ThreeStageRC",
        "simulator": "three_stage_rc",
        "scenarios": ["caps_only", "all_components"],
    },
    {
        "dataset": "data/SIMULATOR_BASED_DATASETS/ThreeStageRLC",
        "simulator": "three_stage_rlc",
        "scenarios": ["caps_only", "inductors_only", "caps_inductors", "all_components"],
    },
    {
        "dataset": "data/SIMULATOR_BASED_DATASETS/DiodeClippedRC",
        "simulator": "diode_clipped_rc",
        "scenarios": ["r_c_only", "r_c_diode"],
    },
]

CALIBRATORS = ["ls", "map", "ga", "pso"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=5000)
    parser.add_argument("--skip-success", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_tasks():
    tasks = []
    for job in JOBS:
        for scenario in job["scenarios"]:
            for calibrator in CALIBRATORS:
                run_name = f"{job['simulator']}_{scenario}_{calibrator}"
                tasks.append(
                    {
                        "dataset": job["dataset"],
                        "simulator": job["simulator"],
                        "scenario": scenario,
                        "calibrator": calibrator,
                        "run_name": run_name,
                    }
                )
    return tasks


def run_task(task: dict, max_nfev: int, skip_success: bool) -> dict:
    run_dir = RESULTS_DIR / task["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_json = run_dir / "summary.json"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    if skip_success and summary_json.exists():
        return {
            **task,
            "status": "SKIPPED_SUCCESS",
            "runtime_sec": 0.0,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error_message": "",
        }

    cv_mode = "full" if task["calibrator"] in {"ga", "pso"} else "loo"

    cmd = [
        "python3",
        "run_calibration_cv.py",
        "--dataset", task["dataset"],
        "--simulator", task["simulator"],
        "--scenario", task["scenario"],
        "--calibrator", task["calibrator"],
        "--output-dir", str(RESULTS_DIR),
        "--run-name", task["run_name"],
        "--max-nfev", str(max_nfev),
        "--cv-mode", cv_mode,
    ]

    start = time.time()

    try:
        with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
            result = subprocess.run(
                cmd,
                stdout=out,
                stderr=err,
                text=True,
                check=False,
            )

        runtime_sec = time.time() - start
        status = "SUCCESS" if result.returncode == 0 else "FAILED"

        error_message = ""
        if result.returncode != 0:
            error_message = stderr_path.read_text(errors="ignore")[-2000:]

    except Exception as e:
        runtime_sec = time.time() - start
        status = "FAILED"
        error_message = repr(e)

    return {
        **task,
        "status": status,
        "runtime_sec": runtime_sec,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "error_message": error_message,
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    rows = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(run_task, task, args.max_nfev, args.skip_success)
            for task in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulator-based campaign"):
            row = future.result()
            rows.append(row)
            pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)

    print(f"\nCampaign finished. Summary saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()