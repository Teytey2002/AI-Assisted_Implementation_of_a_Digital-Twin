from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from dtcalib.simulation import (
    ThreeStageRCLadderSimulator,
    ThreeStageRLCLadderSimulator,
    DiodeClippedRCSimulator,
)


OUT_ROOT = Path("./Digital_Twin_Calibration/data/SIMULATOR_BASED_DATASETS")

CSV_SEP = ","
FLOAT_FMT = "%.12g"

N_PERIODS = 5
SAMPLES_PER_PERIOD = 500

FREQUENCIES = np.logspace(np.log10(1),np.log10(1000),25)


def build_time_and_input(
    freq_hz: float,
    *,
    amplitude: float,
    offset: float = 0.0,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    period = 1.0 / freq_hz
    total_time = N_PERIODS * period
    n_samples = max(2, N_PERIODS * SAMPLES_PER_PERIOD)

    t = np.linspace(0.0, total_time, n_samples, dtype=float)
    u = offset + amplitude * np.sin(2.0 * math.pi * freq_hz * t + phase)

    return t, u


def write_dataset(
    *,
    dataset_name: str,
    simulator,
    theta: np.ndarray,
    params: dict[str, float],
    amplitude: float,
    frequencies: np.ndarray = FREQUENCIES,
) -> None:
    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for i, freq in enumerate(frequencies, start=1):
        t, u = build_time_and_input(freq, amplitude=amplitude)

        sim = simulator.simulate(t=t, u=u, theta=theta)
        y = sim.y

        exp_name = f"exp_{i:03d}_f_{str(freq).replace('.', 'p')}hz"
        csv_name = f"{exp_name}.csv"
        csv_path = out_dir / csv_name

        df = pd.DataFrame(
            {
                "time": t,
                "input": u,
                "output": y,
            }
        )
        df.to_csv(csv_path, index=False, sep=CSV_SEP, float_format=FLOAT_FMT)

        row = {
            "group_name": dataset_name,
            "experiment_name": exp_name,
            "csv_path": csv_name,
            "freq": float(freq),
            "amplitude": float(amplitude),
            "n_periods": int(N_PERIODS),
            "samples_per_period": int(SAMPLES_PER_PERIOD),
            "n_samples": int(len(t)),
        }
        row.update(params)
        manifest_rows.append(row)

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"OK: {dataset_name}")
    print(f"  CSV files : {len(frequencies)}")
    print(f"  Folder    : {out_dir}")
    print(f"  Manifest  : {manifest_path}")


def generate_three_stage_rc() -> None:
    params = {
        "R1": 10.0,
        "R2": 42.2,
        "R3": 22.1,
        "R4": 15.0,
        "R5": 33.2,
        "R6": 68.1,
        "R7": 100.0,
        "C1": 1e-6,
        "C2": 10e-6,
        "C3": 15e-6,
    }

    calibrated_params = (
        "R1", "R2", "R3", "R4", "R5", "R6", "R7",
        "C1", "C2", "C3",
    )

    theta = np.array([params[p] for p in calibrated_params], dtype=float)

    simulator = ThreeStageRCLadderSimulator(
        calibrated_params=calibrated_params,
        fixed_params={},
        y0_mode="zero",
    )

    write_dataset(
        dataset_name="ThreeStageRC",
        simulator=simulator,
        theta=theta,
        params=params,
        amplitude=1.0,
    )


def generate_three_stage_rlc() -> None:
    params = {
        "R1": 10.0,
        "L1": 10e-3,
        "R2": 42.2,
        "C1": 1e-6,
        "R3": 22.1,
        "L2": 22e-3,
        "R4": 15.0,
        "C2": 10e-6,
        "R5": 33.2,
        "L3": 33e-3,
        "R6": 68.1,
        "C3": 15e-6,
        "R7": 100.0,
    }

    calibrated_params = (
        "R1", "L1", "R2", "C1",
        "R3", "L2", "R4", "C2",
        "R5", "L3", "R6", "C3", "R7",
    )

    theta = np.array([params[p] for p in calibrated_params], dtype=float)

    simulator = ThreeStageRLCLadderSimulator(
        calibrated_params=calibrated_params,
        fixed_params={},
        y0_mode="zero",
    )

    write_dataset(
        dataset_name="ThreeStageRLC",
        simulator=simulator,
        theta=theta,
        params=params,
        amplitude=1.0,
    )


def generate_diode_clipped_rc() -> None:
    params = {
        "R1": 1_000.0,
        "C1": 10e-6,
        "IS": 2.52e-9,
        "N": 1.75,
        "VT": 25.85e-3,
        "RS": 0.568,
    }

    calibrated_params = ("R1", "C1", "IS", "N", "VT", "RS")
    theta = np.array([params[p] for p in calibrated_params], dtype=float)

    simulator = DiodeClippedRCSimulator(
        calibrated_params=calibrated_params,
        fixed_params={},
        y0_mode="zero",
        method="BDF",
    )

    write_dataset(
        dataset_name="DiodeClippedRC",
        simulator=simulator,
        theta=theta,
        params=params,
        amplitude=2.0,
        frequencies=FREQUENCIES
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    generate_three_stage_rc()
    generate_three_stage_rlc()
    generate_diode_clipped_rc()

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()