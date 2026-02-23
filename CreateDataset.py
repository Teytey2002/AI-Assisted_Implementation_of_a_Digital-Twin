import pandas as pd
from pathlib import Path
import re

# ========= CONFIG =========
ROOT_DIR = Path("/mnt/c/Program Files/EcosimPro/STANDARD/libs/ELECTRICAL_EXAMPLES/experiments/+filter+examples.default_+l+p_+sensor")
OUT_ROOT = Path("./ALL_LP_DATASETS_CSV_Deep_learning")  # tout au même endroit
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASET_PREFIX = "dataset_+c_"  # on prend tous les dossiers qui commencent par ça


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
if not ROOT_DIR.exists():
    raise FileNotFoundError(f"Dossier parent introuvable : {ROOT_DIR}")

dataset_dirs = sorted([d for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith(DATASET_PREFIX)])

if not dataset_dirs:
    raise FileNotFoundError(f"Aucun dossier '{DATASET_PREFIX}*' trouvé dans {ROOT_DIR}")

print(f"📁 {len(dataset_dirs)} dataset(s) trouvé(s) :")
for d in dataset_dirs:
    print(" -", d.name)

total_ok, total_failed = 0, 0

for ds_dir in dataset_dirs:
    # dossier de sortie pour CE dataset
    out_dir = OUT_ROOT / ds_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rpt_files = sorted(ds_dir.rglob("*.rpt"))
    print(f"\n=== {ds_dir.name} ===")
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

            out_csv = out_dir / safe_name_from_path(rpt_path)
            df_filtered.to_csv(out_csv, index=False)

            ok += 1
        except Exception as e:
            failed += 1
            print(f"❌ {rpt_path.relative_to(ds_dir)} : {e}")

    total_ok += ok
    total_failed += failed
    print(f"✅ Terminé {ds_dir.name}: {ok} OK / {failed} échec(s)")

print(f"\n===== GLOBAL =====")
print(f"OK total     : {total_ok}")
print(f"Échecs total : {total_failed}")
print(f"📦 Sortie dans: {OUT_ROOT.resolve()}")