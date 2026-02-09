import pandas as pd
from pathlib import Path
import re

# ========= CONFIG =========
ROOT_DIR = Path("/mnt/c/Program Files/EcosimPro/STANDARD/libs/ELECTRICAL_EXAMPLES/experiments/+filter+examples.default_+l+p_+sensor/+l+p_+dataset")   # <-- dossier racine qui contient les sous-dossiers experiments
OUT_DIR  = Path("./LP_Dataset_csv")          # <-- dossier unique où mettre tous les CSV
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = [
    "TIME",
    "Addition_2.s_out.signal[1]",  # Vin
    "SensorVoltage_1.v"            # Vout
]

# ========= HELPERS =========
def normalize_col(c: str) -> str:
    c = c.lstrip("#").strip()
    c = re.sub(r"\([^)]*\)$", "", c)  # supprime un (...) final (unités)
    return c

def find_header_idx(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith("#Group"):
                return i
    raise ValueError("Ligne d'en-tête '#Group' introuvable.")

def safe_name_from_path(file_path: Path) -> str:
    # évite collisions si plusieurs 'results.rpt' dans des sous-dossiers
    # ex: sub1/sub2/results.rpt -> sub1__sub2__results.csv
    rel = file_path.relative_to(ROOT_DIR)
    stem = "__".join(rel.with_suffix("").parts)
    return stem + ".csv"

# ========= MAIN =========
rpt_files = sorted(ROOT_DIR.rglob("*.rpt"))

if not rpt_files:
    raise FileNotFoundError(f"Aucun fichier .rpt trouvé dans {ROOT_DIR}")

print(f"{len(rpt_files)} fichier(s) .rpt trouvé(s)")

ok, failed = 0, 0
for rpt_path in rpt_files:
    try:
        header_idx = find_header_idx(rpt_path)

        df = pd.read_csv(rpt_path, skiprows=header_idx, sep="\t")
        df.columns = [normalize_col(c) for c in df.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        df_filtered = df[REQUIRED_COLUMNS]

        out_csv = OUT_DIR / safe_name_from_path(rpt_path)
        df_filtered.to_csv(out_csv, index=False)

        ok += 1
        print(f"{rpt_path.name} -> {out_csv.name}")

    except Exception as e:
        failed += 1
        print(f"{rpt_path} : {e}")

print(f"\nTerminé: {ok} OK / {failed} échec(s)")
