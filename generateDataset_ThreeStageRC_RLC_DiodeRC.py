from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from dtcalib.simulation import (
    ThreeStageRCLadderSimulator,
    ThreeStageRLCLadderSimulator,
    DiodeClippedRCSimulator,
)

# ============================================================
# OUTPUT
# ============================================================

OUT_ROOT = Path("./Digital_Twin_Calibration/data/DL_DATASETS")

CSV_SEP = ","
FLOAT_FMT = "%.12g"

# ============================================================
# SIGNAL GENERATION
# ============================================================

N_PERIODS = 5
SAMPLES_PER_PERIOD = 500

FREQUENCIES = np.logspace(np.log10(1), np.log10(1000), 25)

# ============================================================
# DATASET SIZE
# ============================================================

N_GROUPS = 1000

# ============================================================
# PARAMETER RANGES
# ============================================================

PARAM_RANGES = {
    # Resistors
    "R1": (1.0, 1e3),
    "R2": (1.0, 1e3),
    "R3": (1.0, 1e3),
    "R4": (1.0, 1e3),
    "R5": (1.0, 1e3),
    "R6": (1.0, 1e3),
    "R7": (1.0, 1e3),

    # Capacitors
    "C1": (1e-8, 1e-4),
    "C2": (1e-8, 1e-4),
    "C3": (1e-8, 1e-4),

    # Inductors
    "L1": (1e-4, 1e-1),
    "L2": (1e-4, 1e-1),
    "L3": (1e-4, 1e-1),

    # Diode
    "IS": (1e-12, 1e-6),
    "N": (1.0, 2.5),
    "VT": (1e-3, 1e-1),
    "RS": (1e-3, 10.0),
}

# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = {
    ("ThreeStageRC", "caps_only"): {
        "calibrated_params": ("C1", "C2", "C3"),
        "all_params": (
            "R1", "R2", "R3", "R4", "R5", "R6", "R7",
            "C1", "C2", "C3",
        ),
        "amplitude": 1.0,
    },

    ("ThreeStageRC", "all_components"): {
        "calibrated_params": (
            "R1", "R2", "R3", "R4", "R5", "R6", "R7",
            "C1", "C2", "C3",
        ),
        "all_params": (
            "R1", "R2", "R3", "R4", "R5", "R6", "R7",
            "C1", "C2", "C3",
        ),
        "amplitude": 1.0,
    },

    ("ThreeStageRLC", "caps_only"): {
        "calibrated_params": ("C1", "C2", "C3"),
        "all_params": (
            "R1", "L1", "R2", "C1",
            "R3", "L2", "R4", "C2",
            "R5", "L3", "R6", "C3", "R7",
        ),
        "amplitude": 1.0,
    },

    ("ThreeStageRLC", "inductors_only"): {
        "calibrated_params": ("L1", "L2", "L3"),
        "all_params": (
            "R1", "L1", "R2", "C1",
            "R3", "L2", "R4", "C2",
            "R5", "L3", "R6", "C3", "R7",
        ),
        "amplitude": 1.0,
    },

    ("ThreeStageRLC", "caps_inductors"): {
        "calibrated_params": (
            "C1", "C2", "C3",
            "L1", "L2", "L3",
        ),
        "all_params": (
            "R1", "L1", "R2", "C1",
            "R3", "L2", "R4", "C2",
            "R5", "L3", "R6", "C3", "R7",
        ),
        "amplitude": 1.0,
    },

    ("ThreeStageRLC", "all_components"): {
        "calibrated_params": (
            "R1", "L1", "R2", "C1",
            "R3", "L2", "R4", "C2",
            "R5", "L3", "R6", "C3", "R7",
        ),
        "all_params": (
            "R1", "L1", "R2", "C1",
            "R3", "L2", "R4", "C2",
            "R5", "L3", "R6", "C3", "R7",
        ),
        "amplitude": 1.0,
    },

    ("DiodeClippedRC", "r_c_only"): {
        "calibrated_params": ("R1", "C1"),
        "all_params": ("R1", "C1", "IS", "N", "VT", "RS"),
        "amplitude": 2.0,
    },

    ("DiodeClippedRC", "r_c_diode"): {
        "calibrated_params": ("R1", "C1", "IS", "N"),
        "all_params": ("R1", "C1", "IS", "N", "VT", "RS"),
        "amplitude": 2.0,
    },
}

# ============================================================
# SIGNAL GENERATION
# ============================================================

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

    u = offset + amplitude * np.sin(
        2.0 * math.pi * freq_hz * t + phase
    )

    return t, u

# ============================================================
# LATIN HYPERCUBE SAMPLING
# ============================================================

def sample_parameter_groups(
    parameter_names: tuple[str, ...],
    n_groups: int,
    seed: int = 42,
) -> list[dict[str, float]]:

    sampler = qmc.LatinHypercube(
        d=len(parameter_names),
        seed=seed,
    )

    X = sampler.random(n=n_groups)

    groups = []

    for row in X:

        params = {}

        for i, p in enumerate(parameter_names):

            low, high = PARAM_RANGES[p]

            # Log-uniform sampling
            log_low = np.log10(low)
            log_high = np.log10(high)

            value = 10 ** (
                log_low + row[i] * (log_high - log_low)
            )

            params[p] = float(value)

        groups.append(params)

    return groups

# ============================================================
# SIMULATOR FACTORY
# ============================================================

def build_simulator(
    circuit_name: str,
    calibrated_params: tuple[str, ...],
):

    if circuit_name == "ThreeStageRC":
        return ThreeStageRCLadderSimulator(
            calibrated_params=calibrated_params,
            fixed_params={},
            y0_mode="zero",
        )

    if circuit_name == "ThreeStageRLC":
        return ThreeStageRLCLadderSimulator(
            calibrated_params=calibrated_params,
            fixed_params={},
            y0_mode="zero",
        )

    if circuit_name == "DiodeClippedRC":
        return DiodeClippedRCSimulator(
            calibrated_params=calibrated_params,
            fixed_params={},
            y0_mode="zero",
            method="BDF",
        )

    raise ValueError(f"Unknown circuit: {circuit_name}")

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    *,
    circuit_name: str,
    scenario_name: str,
) -> None:

    cfg = SCENARIOS[(circuit_name, scenario_name)]

    calibrated_params = cfg["calibrated_params"]
    all_params = cfg["all_params"]
    amplitude = cfg["amplitude"]

    dataset_name = f"{circuit_name}_{scenario_name}"

    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    simulator = build_simulator(
        circuit_name,
        calibrated_params,
    )

    # ========================================================
    # SAMPLE PARAMETER SPACE
    # ========================================================

    sampled_groups = sample_parameter_groups(
        parameter_names=all_params,
        n_groups=N_GROUPS,
    )

    manifest_rows = []

    # ========================================================
    # GROUP LOOP
    # ========================================================

    for group_idx, params in enumerate(sampled_groups):

        group_name = f"group_{group_idx:05d}"

        group_dir = out_dir / group_name
        group_dir.mkdir(exist_ok=True)

        theta = np.array(
            [params[p] for p in calibrated_params],
            dtype=float,
        )

        # ====================================================
        # FREQUENCY LOOP
        # ====================================================

        for i, freq in enumerate(FREQUENCIES, start=1):

            t, u = build_time_and_input(
                freq,
                amplitude=amplitude,
            )

            sim = simulator.simulate(
                t=t,
                u=u,
                theta=theta,
            )

            y = sim.y

            exp_name = (
                f"exp_{i:03d}_f_{str(freq).replace('.', 'p')}hz"
            )

            csv_name = f"{exp_name}.csv"

            csv_path = group_dir / csv_name

            df = pd.DataFrame(
                {
                    "time": t,
                    "input": u,
                    "output": y,
                }
            )

            df.to_csv(
                csv_path,
                index=False,
                sep=CSV_SEP,
                float_format=FLOAT_FMT,
            )

            row = {
                "group_name": group_name,
                "experiment_name": exp_name,
                "csv_path": str(
                    csv_path.relative_to(out_dir)
                ),
                "circuit": circuit_name,
                "scenario": scenario_name,
                "freq": float(freq),
                "amplitude": float(amplitude),
                "n_periods": int(N_PERIODS),
                "samples_per_period": int(SAMPLES_PER_PERIOD),
                "n_samples": int(len(t)),
            }

            row.update(params)

            manifest_rows.append(row)

        print(f"OK: {group_name}")

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    manifest_path = out_dir / "manifest.csv"

    pd.DataFrame(manifest_rows).to_csv(
        manifest_path,
        index=False,
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "circuit": circuit_name,
        "scenario": scenario_name,
        "calibrated_params": calibrated_params,
        "all_params": all_params,
        "n_groups": N_GROUPS,
        "frequencies": FREQUENCIES.tolist(),
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n================================================")
    print(f"DATASET GENERATED: {dataset_name}")
    print("================================================")
    print(f"Groups      : {N_GROUPS}")
    print(f"Experiments : {N_GROUPS * len(FREQUENCIES)}")
    print(f"Manifest    : {manifest_path}")

# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for circuit_name, scenario_name in SCENARIOS.keys():

        generate_dataset(
            circuit_name=circuit_name,
            scenario_name=scenario_name,
        )

    print("\nAll datasets generated successfully.")

if __name__ == "__main__":
    main()