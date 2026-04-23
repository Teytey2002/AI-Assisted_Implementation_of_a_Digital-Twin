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
OUT_DIR = Path("/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma2/TFE/AI-Assisted_Implementation_of_a_Digital-Twin/LP_Dataset_Deep_Learning_R1-R2-C-F_Variation")

# Chemin du template .exp 
TEMPLATE_EXP = Path(
    "/mnt/c/Program Files/EcosimPro/STANDARD/libs/ELECTRICAL_EXAMPLES/"
    "experiments/+filter+examples.default_+l+p_+sensor/+l+p_+template_+r1+r2+c/+l+p_+template_+r1+r2+c.exp"
)

def compute_fc(R1, R2, C):
    R_eq = (R1 * R2) / (R1 + R2)
    return 1.0 / (2 * math.pi * R_eq * C)

# -----------------------------
# RANGES PARAMETRES
# -----------------------------
R1_MIN = 5e3
R1_MAX = 2e4
N_R1 = 10

R2_MIN = 5e3
R2_MAX = 2e4
N_R2 = 10

# on fait le calcul de Fc dans le main mtn pour calculer la fréquence de coupure en fonction du couple R1, R2, C  
N = 40     # Nb d'expérience voulu

C_MIN = 8e-7
C_MAX = 3.2e-6
N_C = 10         # nb de valeurs de C

# -----------------------------
# Helpers
# -----------------------------
def cap_tag(c: float) -> str:
    s = f"{c:.2e}"   # ex: 2.76e-06
    s = s.replace(".", "p").replace("-", "m").replace("+", "")
    return s

def r_tag(r: float) -> str:
    rk = int(round(r / 1000.0))
    return f"{rk}k"

def replace_capacitance(text: str, new_c: float) -> str:
    pattern = r"(?m)^\s*Capacitor_1_1\.C\s*=\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$"
    repl = f"Capacitor_1_1.C = {new_c:.12g}"
    out, n = re.subn(pattern, repl, text, count=1)
    if n == 0:
        raise ValueError("Could not find a line 'Capacitor_1_1.C = ...' in the template.")
    return out

def freq_tag(f_hz: float) -> str:
    s = f"{f_hz:.2f}"
    s = s.replace(".", "p").replace("-", "m")
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

def replace_R1(text: str, new_r1: float) -> str:
    pattern = r"(?m)^\s*Resistor_1_1\.R\s*=\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$"
    repl = f"Resistor_1_1.R = {new_r1:.12g}"
    out, n = re.subn(pattern, repl, text, count=1)
    if n == 0:
        raise ValueError("Could not find 'Resistor_1_1.R = ...' in template.")
    return out


def replace_R2(text: str, new_r2: float) -> str:
    pattern = r"(?m)^\s*Resistor_2_1\.R\s*=\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$"
    repl = f"Resistor_2_1.R = {new_r2:.12g}"
    out, n = re.subn(pattern, repl, text, count=1)
    if n == 0:
        raise ValueError("Could not find 'Resistor_2_1.R = ...' in template.")
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

    # log-spaced frequencies and Capa 
    Cs = np.logspace(math.log10(C_MIN), math.log10(C_MAX), N_C)
    R1s = np.logspace(math.log10(R1_MIN), math.log10(R1_MAX), N_R1)
    R2s = np.logspace(math.log10(R2_MIN), math.log10(R2_MAX), N_R2)

    written = 0
    for i_r1, R1 in enumerate(R1s, start=1):
        for i_r2, R2 in enumerate(R2s, start=1):
            for j, C in enumerate(Cs, start=1):
                
                # Calcul de la fréquence 
                fc = compute_fc(R1, R2, C)    # Fréquence de coupure
                F_MIN = fc / 10.0   
                F_MAX = fc * 10.0 
                freqs = np.logspace(math.log10(F_MIN), math.log10(F_MAX), N)

                ctag = cap_tag(float(C))
                r1tag = r_tag(float(R1))
                r2tag = r_tag(float(R2))
                base_folder = OUT_DIR / f"r1_{r1tag}_r2_{r2tag}_c_{ctag}"

                for i, f in enumerate(freqs, start=1):
                    period = 1.0 / float(f)

                    tag = freq_tag(float(f))
                    exp_name = f"e{i:03d}_f{tag}"
                    report_name = f"r{i:03d}_f{tag}.rpt"

                    exp_folder = base_folder / exp_name
                    exp_folder.mkdir(parents=True, exist_ok=True)

                    # EcosimPro convention: <folder>/<folder>.exp
                    exp_path = exp_folder / f"{exp_name}.exp"

                    text = template_text
                    text = replace_experiment_name(text, exp_name)
                    text = replace_low_period(text, period)
                    text = replace_capacitance(text, float(C))
                    text = replace_R1(text, float(R1))
                    text = replace_R2(text, float(R2))

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