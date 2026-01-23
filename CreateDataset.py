import pandas as pd
from pathlib import Path
import re

DATA_DIR = Path("/mnt/c/Program Files/EcosimPro/STANDARD/libs/ELECTRICAL_EXAMPLES/experiments/+filter+examples.default/exp1")
FILE_PATH = DATA_DIR / "result_High.rpt"
# Vérification de sécurité
if not FILE_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {FILE_PATH}")

# Normaliser les noms de colonnes:
#    - enlever le '#' initial
#    - enlever le suffixe d’unité en fin, ex '(V)', '(-)', '(A)', '(Ohm)', '(s)', '(rad)', etc.
def normalize_col(c: str) -> str:
    c = c.lstrip("#").strip()
    c = re.sub(r"\([^)]*\)$", "", c)  # supprime un (...) final
    return c

# 1️⃣ Trouver la ligne d'en-tête (#Group ...)
header_idx = None
with FILE_PATH.open("r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if line.lstrip().startswith("#Group"):
            header_idx = i
            break

if header_idx is None:
    raise ValueError("Ligne d'en-tête '#Group' introuvable.")

# 2️⃣ Lecture du tableau avec parsing par tabulation
df = pd.read_csv(FILE_PATH, skiprows=header_idx, sep="\t")

# 3️⃣ Nettoyage des noms de colonnes
df.columns = [normalize_col(c) for c in df.columns] # Normalise les colonnes

required_columns = [
    "TIME",
    "Addition_2.s_out.signal[1]",   # Vin
    "High_freq.s_out.signal[1]",
    "Mid_freq.s_out.signal[1]",
    "Low_freq.s_out.signal[1]",
    "Resistor_2.e_p.v",             # Vout_HP
    "Capacitor_1_1.e_p.v",          # Vout_LP
    "Capacitor_1_2.e_p.v",          # Vout_BP

]

# Vérification si les colonnes sont présentent dans le fichier
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Colonnes manquantes dans le fichier : {missing_cols}")

# Création du nouveau DataFrame
df_filtered = df[required_columns]

# 6️⃣ Sauvegarde dans un nouveau fichier
df_filtered.to_csv(
    "result_High_filtered.csv",
    index=False
)

print("✅ Fichier sauvegardé : dataset_exp1_filtered.csv")
print("Colonnes conservées :", required_columns)
