import pandas as pd
from pathlib import Path

FOLDER_A = Path("./LP_Dataset_csv_C_Modified")
FOLDER_B = Path("./LP_Dataset_csv_Reference")

# Récupération des fichiers CSV
files_A = {f.name: f for f in FOLDER_A.glob("*.csv")}
files_B = {f.name: f for f in FOLDER_B.glob("*.csv")}

common_files = sorted(set(files_A) & set(files_B))

if not common_files:
    raise ValueError("Aucun fichier CSV commun entre les deux dossiers.")

print(f"🔎 {len(common_files)} fichier(s) commun(s) trouvé(s)\n")

ok, diff, error = 0, 0, 0

for fname in common_files:
    try:
        df_A = pd.read_csv(files_A[fname])
        df_B = pd.read_csv(files_B[fname])

        # Vérifier qu'il y a au moins 2 colonnes
        if df_A.shape[1] < 2 or df_B.shape[1] < 2:
            raise ValueError("Moins de 2 colonnes")

        # Extraire les 2 premières colonnes
        A_cols = df_A.iloc[:, :2]
        B_cols = df_B.iloc[:, :2]

        # Vérifier noms des colonnes
        if list(A_cols.columns) != list(B_cols.columns):
            print(f"❌ {fname} → noms de colonnes différents")
            diff += 1
            continue

        # Vérifier valeurs
        if A_cols.equals(B_cols):
            print(f"✅ {fname} → OK")
            ok += 1
        else:
            print(f"⚠️ {fname} → valeurs différentes")
            diff += 1

    except Exception as e:
        print(f"💥 {fname} → erreur: {e}")
        error += 1

print("\n===== RÉSUMÉ =====")
print(f"OK       : {ok}")
print(f"Différent: {diff}")
print(f"Erreur   : {error}")
