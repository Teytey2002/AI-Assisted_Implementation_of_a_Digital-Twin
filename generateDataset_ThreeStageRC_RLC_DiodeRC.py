from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from dtcalib.simulation import (
    ThreeStageRCLadderSimulator,
    ThreeStageRLCLadderSimulator,
    DiodeClippedRCSimulator,
)

MAX_WORKERS = 4
OUT_ROOT = Path("./Digital_Twin_Calibration/data/DL_DATASETS_2")

CSV_SEP = ","
FLOAT_FMT = "%.12g"

N_PERIODS = 5
SAMPLES_PER_PERIOD = 500

N_GROUPS = 1000
N_FREQUENCIES = 50
FREQ_MARGIN = 10.0


NOMINAL_PARAMS = {
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
    "L1": 10e-3,
    "L2": 22e-3,
    "L3": 33e-3,
    "IS": 2.52e-9,
    "N": 1.75,
    "VT": 25.85e-3,
    "RS": 0.568,
}


PARAM_RANGES = {
    "R1": (10.0 / 3.0, 10.0 * 3.0),
    "R2": (42.2 / 3.0, 42.2 * 3.0),
    "R3": (22.1 / 3.0, 22.1 * 3.0),
    "R4": (15.0 / 3.0, 15.0 * 3.0),
    "R5": (33.2 / 3.0, 33.2 * 3.0),
    "R6": (68.1 / 3.0, 68.1 * 3.0),
    "R7": (100.0 / 3.0, 100.0 * 3.0),

    "C1": (1e-6 / 10.0, 1e-6 * 10.0),
    "C2": (10e-6 / 10.0, 10e-6 * 10.0),
    "C3": (15e-6 / 10.0, 15e-6 * 10.0),

    "L1": (10e-3 / 10.0, 10e-3 * 10.0),
    "L2": (22e-3 / 10.0, 22e-3 * 10.0),
    "L3": (33e-3 / 10.0, 33e-3 * 10.0),

    "IS": (1e-10, 1e-8),
    "N": (1.2, 2.2),

    "VT": (25.85e-3, 25.85e-3),
    "RS": (0.568, 0.568),
}


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
        "calibrated_params": ("C1", "C2", "C3", "L1", "L2", "L3"),
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

def generate_one_group(
    *,
    group_idx: int,
    sampled_params: dict[str, float],
    out_dir: str,
    circuit_name: str,
    scenario_name: str,
    calibrated_params: tuple[str, ...],
    all_params: tuple[str, ...],
    amplitude: float,
    frequencies: list[float],
) -> list[dict[str, object]]:

    out_dir = Path(out_dir)

    params = {
        p: float(NOMINAL_PARAMS[p])
        for p in all_params
    }

    for p, v in sampled_params.items():
        params[p] = float(v)

    group_name = f"group_{group_idx:05d}"
    group_dir = out_dir / group_name
    group_dir.mkdir(exist_ok=True)

    fixed_params = {
        p: params[p]
        for p in all_params
        if p not in calibrated_params
    }

    theta = np.array(
        [params[p] for p in calibrated_params],
        dtype=float,
    )

    simulator = build_simulator(
        circuit_name=circuit_name,
        calibrated_params=calibrated_params,
        fixed_params=fixed_params,
    )

    rows: list[dict[str, object]] = []

    for i, freq in enumerate(frequencies, start=1):
        t, u = build_time_and_input(
            float(freq),
            amplitude=amplitude,
        )

        sim = simulator.simulate(
            t=t,
            u=u,
            theta=theta,
        )

        y = sim.y

        exp_name = (
            f"exp_{i:03d}_f_{freq:.6e}hz"
            .replace(".", "p")
            .replace("+", "")
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

        row: dict[str, object] = {
            "group_name": group_name,
            "experiment_name": exp_name,
            "csv_path": str(csv_path.relative_to(out_dir)),
            "circuit": circuit_name,
            "scenario": scenario_name,
            "freq": float(freq),
            "amplitude": amplitude,
            "n_periods": int(N_PERIODS),
            "samples_per_period": int(SAMPLES_PER_PERIOD),
            "n_samples": int(len(t)),
        }

        for p in all_params:
            row[p] = float(params[p])

        rows.append(row)

    return rows

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


def get_bounds(param_name: str) -> tuple[float, float]:
    if param_name not in PARAM_RANGES:
        raise KeyError(f"No range defined for parameter: {param_name}")
    return PARAM_RANGES[param_name]


def compute_rc_frequency_band(params: tuple[str, ...]) -> tuple[float, float]:
    resistors = [p for p in params if p.startswith("R")]
    capacitors = [p for p in params if p.startswith("C")]

    if not resistors or not capacitors:
        raise ValueError("RC frequency band requires at least one R and one C.")

    r_min = min(get_bounds(p)[0] for p in resistors)
    r_max = max(get_bounds(p)[1] for p in resistors)

    c_min = min(get_bounds(p)[0] for p in capacitors)
    c_max = max(get_bounds(p)[1] for p in capacitors)

    tau_min = r_min * c_min
    tau_max = r_max * c_max

    fc_max = 1.0 / (2.0 * math.pi * tau_min)
    fc_min = 1.0 / (2.0 * math.pi * tau_max)

    return fc_min / FREQ_MARGIN, fc_max * FREQ_MARGIN


def compute_rlc_frequency_band(params: tuple[str, ...]) -> tuple[float, float]:
    f_rc_min, f_rc_max = compute_rc_frequency_band(params)

    inductors = [p for p in params if p.startswith("L")]
    capacitors = [p for p in params if p.startswith("C")]

    if not inductors or not capacitors:
        return f_rc_min, f_rc_max

    l_min = min(get_bounds(p)[0] for p in inductors)
    l_max = max(get_bounds(p)[1] for p in inductors)

    c_min = min(get_bounds(p)[0] for p in capacitors)
    c_max = max(get_bounds(p)[1] for p in capacitors)

    f0_max = 1.0 / (2.0 * math.pi * math.sqrt(l_min * c_min))
    f0_min = 1.0 / (2.0 * math.pi * math.sqrt(l_max * c_max))

    f_min = min(f_rc_min, f0_min / FREQ_MARGIN)
    f_max = max(f_rc_max, f0_max * FREQ_MARGIN)

    return f_min, f_max


def compute_diode_rc_frequency_band() -> tuple[float, float]:
    r_min, r_max = get_bounds("R1")
    c_min, c_max = get_bounds("C1")

    fc_max = 1.0 / (2.0 * math.pi * r_min * c_min)
    fc_min = 1.0 / (2.0 * math.pi * r_max * c_max)

    return fc_min / FREQ_MARGIN, fc_max * FREQ_MARGIN


def build_frequencies_for_scenario(
    circuit_name: str,
    all_params: tuple[str, ...],
) -> np.ndarray:
    if circuit_name == "ThreeStageRC":
        f_min, f_max = compute_rc_frequency_band(all_params)

    elif circuit_name == "ThreeStageRLC":
        f_min, f_max = compute_rlc_frequency_band(all_params)

    elif circuit_name == "DiodeClippedRC":
        f_min, f_max = compute_diode_rc_frequency_band()

    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")

    f_min = max(float(f_min), 1.0)

    # limite basse réaliste
    f_max = min(float(f_max), 1e5)

    # sécurité
    f_max = max(f_max, f_min * 10.0)

    return np.logspace(
        np.log10(f_min),
        np.log10(f_max),
        N_FREQUENCIES,
    )


def sample_parameter_groups(
    parameter_names: tuple[str, ...],
    n_groups: int,
    seed: int = 42,
) -> list[dict[str, float]]:
    if len(parameter_names) == 0:
        raise ValueError("parameter_names cannot be empty.")

    sampler = qmc.LatinHypercube(
        d=len(parameter_names),
        seed=seed,
    )

    x = sampler.random(n=n_groups)
    groups: list[dict[str, float]] = []

    for row in x:
        params: dict[str, float] = {}

        for i, p in enumerate(parameter_names):
            low, high = get_bounds(p)

            if np.isclose(low, high):
                value = float(low)
            else:
                if low <= 0 or high <= 0:
                    raise ValueError(
                        f"Log-uniform sampling requires positive bounds for {p}: "
                        f"({low}, {high})"
                    )

                log_low = np.log10(low)
                log_high = np.log10(high)

                value = float(
                    10.0 ** (log_low + row[i] * (log_high - log_low))
                )

            params[p] = value

        groups.append(params)

    return groups


def build_simulator(
    circuit_name: str,
    calibrated_params: tuple[str, ...],
    fixed_params: dict[str, float],
):
    if circuit_name == "ThreeStageRC":
        return ThreeStageRCLadderSimulator(
            calibrated_params=calibrated_params,
            fixed_params=fixed_params,
            y0_mode="zero",
        )

    if circuit_name == "ThreeStageRLC":
        return ThreeStageRLCLadderSimulator(
            calibrated_params=calibrated_params,
            fixed_params=fixed_params,
            y0_mode="zero",
        )

    if circuit_name == "DiodeClippedRC":
        return DiodeClippedRCSimulator(
            calibrated_params=calibrated_params,
            fixed_params=fixed_params,
            y0_mode="zero",
            method="BDF",
        )

    raise ValueError(f"Unknown circuit: {circuit_name}")


def generate_dataset(
    *,
    circuit_name: str,
    scenario_name: str,
    seed: int = 42,
) -> None:
    cfg = SCENARIOS[(circuit_name, scenario_name)]

    calibrated_params = tuple(cfg["calibrated_params"])
    all_params = tuple(cfg["all_params"])
    amplitude = float(cfg["amplitude"])

    dataset_name = f"{circuit_name}_{scenario_name}"
    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frequencies = build_frequencies_for_scenario(
        circuit_name=circuit_name,
        all_params=all_params,
    )

    print("\n================================================")
    print(f"GENERATING DATASET: {dataset_name}")
    print("================================================")
    print(f"Calibrated params : {calibrated_params}")
    print(f"All params        : {all_params}")
    print(f"Frequency range   : {frequencies[0]:.6e} Hz -> {frequencies[-1]:.6e} Hz")
    print(f"N frequencies     : {len(frequencies)}")
    print(f"N groups          : {N_GROUPS}")

    sampled_groups = sample_parameter_groups(
        parameter_names=calibrated_params,
        n_groups=N_GROUPS,
        seed=seed,
    )

    manifest_rows: list[dict[str, object]] = []

    frequencies_list = [float(f) for f in frequencies]

    max_workers = min(MAX_WORKERS, os.cpu_count() or 1)

    print(f"Parallel generation with max_workers={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for group_idx, sampled_params in enumerate(sampled_groups):
            futures.append(
                executor.submit(
                    generate_one_group,
                    group_idx=group_idx,
                    sampled_params=sampled_params,
                    out_dir=str(out_dir),
                    circuit_name=circuit_name,
                    scenario_name=scenario_name,
                    calibrated_params=calibrated_params,
                    all_params=all_params,
                    amplitude=amplitude,
                    frequencies=frequencies_list,
                )
            )

        for done_idx, future in enumerate(as_completed(futures), start=1):
            rows = future.result()
            manifest_rows.extend(rows)

            if done_idx % 50 == 0 or done_idx == 1:
                print(f"  OK groups {done_idx}/{N_GROUPS}")

    manifest_rows = sorted(
        manifest_rows,
        key=lambda r: (str(r["group_name"]), float(r["freq"])),
    )

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path,
        index=False,
    )

    metadata = {
        "circuit": circuit_name,
        "scenario": scenario_name,
        "calibrated_params": calibrated_params,
        "all_params": all_params,
        "nominal_params": {
            p: float(NOMINAL_PARAMS[p])
            for p in all_params
        },
        "param_ranges": {
            p: [float(PARAM_RANGES[p][0]), float(PARAM_RANGES[p][1])]
            for p in all_params
        },
        "n_groups": int(N_GROUPS),
        "n_frequencies": int(len(frequencies)),
        "frequencies": frequencies.tolist(),
        "frequency_min_hz": float(frequencies[0]),
        "frequency_max_hz": float(frequencies[-1]),
        "frequency_selection": "scenario_based_from_parameter_bounds",
        "frequency_margin": float(FREQ_MARGIN),
        "n_periods": int(N_PERIODS),
        "samples_per_period": int(SAMPLES_PER_PERIOD),
        "seed": int(seed),
        "sampling": "latin_hypercube_log_uniform_on_calibrated_params_only",
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDATASET GENERATED")
    print(f"Folder      : {out_dir}")
    print(f"Manifest    : {manifest_path}")
    print(f"Groups      : {N_GROUPS}")
    print(f"Experiments : {N_GROUPS * len(frequencies)}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for circuit_name, scenario_name in SCENARIOS.keys():
        generate_dataset(
            circuit_name=circuit_name,
            scenario_name=scenario_name,
            seed=42,
        )

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()