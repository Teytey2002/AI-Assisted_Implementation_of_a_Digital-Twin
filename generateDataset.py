from __future__ import annotations

import math
from pathlib import Path
import csv

import numpy as np
import pandas as pd

from dtcalib.simulation import LowPassR1CR2Simulator


# ============================================================
# Configuration
# ============================================================

OUT_DIR = Path("./Digital_Twin_Calibration/data/LP_DATASET_R1_R2_C")

# Parameter ranges
R1_MIN = 5e3
R1_MAX = 2e4
N_R1 = 10          # Number of R1 value in the range

R2_MIN = 5e3
R2_MAX = 2e4
N_R2 = 10

C_MIN = 8e-7
C_MAX = 3.2e-6
N_C = 10

# Number of frequencies per (R1, R2, C) group
N_FREQ = 40

# Frequency band around cutoff
F_MIN_FACTOR = 0.1
F_MAX_FACTOR = 10.0

# Signal generation
N_PERIODS = 5                 # simulate 5 periods of the sine
SAMPLES_PER_PERIOD = 200      # temporal resolution
AMPLITUDE = 5.0
PHASE = 0.0
OFFSET = 0.0

# Initial condition mode used in your simulator
Y0_MODE = "dc_from_u0"

# CSV separator / float formatting
CSV_SEP = ","
FLOAT_FMT = "%.12g"


# ============================================================
# Helpers
# ============================================================

def compute_fc(r1: float, r2: float, c: float) -> float:
    """Cutoff frequency for R1 in series with (R2 || C)."""
    r_eq = (r1 * r2) / (r1 + r2)
    return 1.0 / (2.0 * math.pi * r_eq * c)


def c_tag(c: float) -> str:
    s = f"{c:.2e}"  # e.g. 2.76e-06
    return s.replace(".", "p").replace("-", "m").replace("+", "")


def r_tag(r: float) -> str:
    # very compact tag, e.g. 9340 -> 9k ; 10200 -> 10k
    rk = int(round(r / 1000.0))
    return f"{rk}k"


def freq_tag(f_hz: float) -> str:
    s = f"{f_hz:.2f}"
    return s.replace(".", "p").replace("-", "m")


def build_time_and_input(
    freq_hz: float,
    amplitude: float = AMPLITUDE,
    phase: float = PHASE,
    offset: float = OFFSET,
    n_periods: int = N_PERIODS,
    samples_per_period: int = SAMPLES_PER_PERIOD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a sinusoidal input:
        u(t) = offset + amplitude * sin(2*pi*f*t + phase)
    """
    if freq_hz <= 0:
        raise ValueError(f"freq_hz must be > 0, got {freq_hz}")

    period = 1.0 / freq_hz
    total_time = n_periods * period
    n_samples = max(2, n_periods * samples_per_period)

    t = np.linspace(0.0, total_time, n_samples, dtype=float)
    u = offset + amplitude * np.sin(2.0 * math.pi * freq_hz * t + phase)
    return t, u


def ensure_unique_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# Main generation
# ============================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    simulator = LowPassR1CR2Simulator(
        calibrated_params=("R1", "R2", "C"),
        fixed_params={},
        y0_mode=Y0_MODE,
    )

    r1_values = np.logspace(math.log10(R1_MIN), math.log10(R1_MAX), N_R1)
    r2_values = np.logspace(math.log10(R2_MIN), math.log10(R2_MAX), N_R2)
    c_values = np.logspace(math.log10(C_MIN), math.log10(C_MAX), N_C)

    manifest_rows: list[dict[str, object]] = []
    written = 0

    for i_r1, r1 in enumerate(r1_values, start=1):
        for i_r2, r2 in enumerate(r2_values, start=1):
            for i_c, c in enumerate(c_values, start=1):
                fc = compute_fc(float(r1), float(r2), float(c))
                f_min = fc * F_MIN_FACTOR
                f_max = fc * F_MAX_FACTOR

                if f_min <= 0 or f_max <= 0 or f_min >= f_max:
                    raise ValueError(
                        f"Invalid frequency band for R1={r1}, R2={r2}, C={c}: "
                        f"f_min={f_min}, f_max={f_max}"
                    )

                freqs = np.logspace(math.log10(f_min), math.log10(f_max), N_FREQ)

                r1tag = r_tag(float(r1))
                r2tag = r_tag(float(r2))
                ctag = c_tag(float(c))

                group_name = f"r1_{r1tag}_r2_{r2tag}_c_{ctag}"
                group_dir = OUT_DIR / group_name
                ensure_unique_dir(group_dir)

                for i_f, freq in enumerate(freqs, start=1):
                    t, u = build_time_and_input(float(freq))

                    theta = np.array([float(r1), float(r2), float(c)], dtype=float)
                    sim = simulator.simulate(t, u, theta)
                    y = sim.y

                    exp_name = f"e{i_f:03d}_f{freq_tag(float(freq))}"
                    csv_name = f"{exp_name}.csv"
                    csv_path = group_dir / csv_name

                    df = pd.DataFrame(
                        {
                            "time": t,
                            "input": u,
                            "output": y,
                        }
                    )
                    df.to_csv(csv_path, index=False, sep=CSV_SEP, float_format=FLOAT_FMT)

                    manifest_rows.append(
                        {
                            "group_name": group_name,
                            "experiment_name": exp_name,
                            "csv_path": str(csv_path.relative_to(OUT_DIR)),
                            "R1": float(r1),
                            "R2": float(r2),
                            "C": float(c),
                            "fc": float(fc),
                            "freq": float(freq),
                            "amplitude": float(AMPLITUDE),
                            "phase": float(PHASE),
                            "offset": float(OFFSET),
                            "n_periods": int(N_PERIODS),
                            "samples_per_period": int(SAMPLES_PER_PERIOD),
                            "n_samples": int(len(t)),
                        }
                    )

                    written += 1

    manifest_path = OUT_DIR / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"OK: wrote {written} CSV experiments into:\n  {OUT_DIR}")
    print(f"Manifest saved to:\n  {manifest_path}")


if __name__ == "__main__":
    main()