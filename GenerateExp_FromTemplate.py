import re
import math
from pathlib import Path
import numpy as np

EXS_CONTENT = """<simulation_settings version="1.0">
<setting id="ACTIVE_LOG" value="TRUE"/>
<setting id="ACTIVE_OPTIMISE_CODE" value="FALSE"/>
<setting id="ACTIVE_PPR_H5" value="FALSE"/>
<setting id="ACTIVE_TRACE" value="TRUE"/>
<setting id="ACTIVE_WARN" value="TRUE"/>
<setting id="DEFAULT_REPORT_SEPARATOR" value="\\t"/>
<setting id="LOCAL_EXPERIMENTS_PATH" value=""/>
<setting id="SETTING_SIM_DETECT_BAD_OPERATION_ANY_TIME" value="FALSE"/>
<setting id="SETTING_SIM_DETECT_BAD_OPERATION_REFRESH_TIME" value="SEV_NONE"/>
<setting id="SETTING_SIM_DETECT_NAN_INF" value="SEV_NONE"/>
<setting id="SETTING_SIM_DETECT_RANGE_VIOLATION" value="SEV_WARNING"/>
<setting id="SETTING_SIM_FORCE_STOP_CINT" value="TRUE"/>
</simulation_settings>
"""


# -----------------------------
# 1) Paths (WSL)
# -----------------------------
# Dossier de sortie Windows vu depuis WSL :
#OUT_DIR = Path(
#    "/mnt/c/Program Files/EcosimPro/STANDARD/libs/ELECTRICAL_EXAMPLES/"
#    "experiments/+filter+examples.default_+l+p_+sensor/+l+p_+dataset"
#)
# Erreur, droit d'accès entre windows et wsl. Pour faire simple, on créer le dossier dans le répertoire actuel puis on déplace manuellement au bon endroit. 
OUT_DIR = Path("/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma2/TFE/AI-Assisted_Implementation_of_a_Digital-Twin/LP_Dataset")

# Chemin du template .exp 
TEMPLATE_EXP = Path(
    "/mnt/c/Program Files/EcosimPro/STANDARD/libs/ELECTRICAL_EXAMPLES/"
    "experiments/+filter+examples.default_+l+p_+sensor/+l+p_+dataset/+l+p_+sensor_+template/+l+p_+sensor_+template.exp"
)

# -----------------------------
# 2) Sweep settings
# -----------------------------
FC = 5.0    # Fréquence de coupure
F_MIN = FC / 10.0   # 0.5 Hz
F_MAX = FC * 10.0   # 50 Hz
N = 100     # Nb d'expérience voulu

# -----------------------------
# Helpers
# -----------------------------
def freq_tag(f_hz: float) -> str:
    """Human-safe tag for filenames/experiment names."""
    # 2-3 sig figs is enough; keep consistent
    s = f"{f_hz:.6g}"  # e.g. 0.5, 12.34, 1e-3
    s = s.replace(".", "p").replace("-", "m")
    s = s.replace("e", "e")  # keep e for scientific notation
    # also remove + if any (rare)
    s = s.replace("+", "")
    return s

def replace_experiment_name(text: str, new_name: str) -> str:
    # Replace first occurrence of: EXPERIMENT <name> ON
    return re.sub(r"(?m)^\s*EXPERIMENT\s+\w+\s+ON\s+",
                  f"EXPERIMENT {new_name} ON ",
                  text,
                  count=1)

def replace_low_period(text: str, new_period: float) -> str:
    # Replace Low_freq.Period = <number>
    # Accept spaces, tabs, scientific notation etc.
    pattern = r"(?m)^\s*Low_freq\.Period\s*=\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$"
    repl = f"Low_freq.Period = {new_period:.12g}"
    out, n = re.subn(pattern, repl, text, count=1)
    if n == 0:
        raise ValueError("Could not find a line 'Low_freq.Period = ...' in the template.")
    return out

def write_exs_file(folder: Path, exp_name: str):
    """
    Create the .exs file required by EcosimPro for an experiment.
    Content is fixed and identical for all experiments.
    """
    exs_path = folder / f"{exp_name}.exs.xml"
    exs_path.write_text(EXS_CONTENT, encoding="utf-8")

def main():
    if not TEMPLATE_EXP.exists():
        raise FileNotFoundError(f"Template .exp not found: {TEMPLATE_EXP}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    template_text = TEMPLATE_EXP.read_text(encoding="utf-8", errors="ignore")

    # log-spaced frequencies
    freqs = np.logspace(math.log10(F_MIN), math.log10(F_MAX), N)

    written = 0
    for i, f in enumerate(freqs, start=1):
        period = 1.0 / float(f)

        tag = freq_tag(float(f))
        exp_name = f"exp_lp_{i:03d}_f{tag}hz"
        report_name = f"results_lp_{i:03d}_f{tag}hz.rpt"

        exp_folder = OUT_DIR / exp_name
        exp_folder.mkdir(parents=True, exist_ok=True)

        # EcosimPro convention: <folder>/<folder>.exp
        exp_path = exp_folder / f"{exp_name}.exp"

        text = template_text
        text = replace_experiment_name(text, exp_name)
        text = replace_low_period(text, period)

        # Replace REPORT_TABLE - only if you already have it in template.
        # If your template doesn't include REPORT_TABLE, add it there first.
        # Note: keep "*" export for now.
        text, n = re.subn(
            r'(?m)^\s*REPORT_TABLE\(".*?"\s*,\s*".*?"\s*\)',
            f'REPORT_TABLE("{report_name}", "*")',
            text,
            count=1
        )
        if n == 0:
            raise ValueError("Could not find REPORT_TABLE(...) in the template. Add it once in template.")

        exp_path.write_text(text, encoding="utf-8")
        write_exs_file(exp_folder, exp_name)
        written += 1

    print(f"OK: wrote {written} experiments into:\n  {OUT_DIR}")

if __name__ == "__main__":
    main()
