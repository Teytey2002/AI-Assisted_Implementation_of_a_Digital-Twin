from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from dtcalib.simulation import LowPassR1CR2Simulator, ThreeStageRCLadderSimulator, ThreeStageRLCLadderSimulator

def normalize_col(c: str) -> str:
    c = c.lstrip("#").strip()
    c = re.sub(r"\([^)]*\)$", "", c)  # retire un éventuel suffixe unité
    return c


def find_header_idx(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith("#Group"):
                return i
    raise ValueError("Ligne d'en-tête '#Group' introuvable.")

# =========================================================
# 1) Charger le fichier exporté depuis LTspice
# =========================================================
file_path_lt = Path("./threeStageRLC50hz.txt")   # adapte si nécessaire

df_lt = pd.read_csv(file_path_lt, sep=r"\s+", engine="python")

print("Colonnes trouvées :", df_lt.columns.tolist())
print(df_lt.head())

# =========================================================
# 1) Charger CSV généré depuis EcoSimPro
# =========================================================
#file_path_eco = Path("r1_10_r2_10_c22micro.rpt")  # 
#
#header_idx = find_header_idx(file_path_eco)
#
#df_eco = pd.read_csv(file_path_eco, skiprows=header_idx, sep="\t")
#df_eco.columns = [normalize_col(c) for c in df_eco.columns]
#
#print("Colonnes trouvées :", df_eco.columns.tolist())
#print(df_eco.head())


# =========================================================
# 2) Extraire t, Vin, Vout
# =========================================================
# D'après l'export précédent :
# - V(n002) = Vin
# - V(n001) = Vout

# For Ltpice
t_lt = df_lt["time"].to_numpy(dtype=float)
vin_lt = df_lt["V(vin)"].to_numpy(dtype=float)
vout_lt = df_lt["V(vout)"].to_numpy(dtype=float)

# For EcoSimPro 
#t_Eco = df_eco["TIME"].to_numpy(dtype=float)
#vin_Eco = df_eco["Addition_2.s_out.signal[1]"].to_numpy(dtype=float)
#vout_Eco = df_eco["SensorVoltage_1.v"].to_numpy(dtype=float)


# =========================================================
# 3) Définir les paramètres du circuit dans ta classe Python
# =========================================================
# Donc choisis ici les vraies valeurs utilisées dans LTspice.

R1 = 10.0
L1 = 10e-3
R2 = 42.2
C1 = 1e-6

R3 = 22.1
L2 = 22e-3
R4 = 15.0
C2 = 10e-6

R5 = 33.2
L3 = 33e-3
R6 = 68.1
R7 = 100.0
C3 = 15e-6

simulator = ThreeStageRLCLadderSimulator(
    calibrated_params=(
        "R1", "L1", "R2", "C1",
        "R3", "L2", "R4", "C2",
        "R5", "L3", "R6", "C3", "R7",
    ),
    fixed_params={},
    y0_mode="zero",
)

theta = np.array([
    R1, L1, R2, C1,
    R3, L2, R4, C2,
    R5, L3, R6, C3, R7,
], dtype=float)


# =========================================================
# 4) Simuler avec exactement le même temps et le même input
# =========================================================
result = simulator.simulate(t=t_lt, u=vin_lt, theta=theta)
vout_py = result.y

print("\nInfos auxiliaires du simulateur :")
for k, v in result.aux.items():
    print(f"{k}: {v}")


# =========================================================
# 5) Calculer les erreurs
# =========================================================
err = vout_py - vout_lt

mae = np.mean(np.abs(err))
rmse = np.sqrt(np.mean(err**2))
max_abs = np.max(np.abs(err))

print("\n--- Metrics ---")
print(f"MAE   = {mae:.6e}")
print(f"RMSE  = {rmse:.6e}")
print(f"MAXAE = {max_abs:.6e}")


# =========================================================
# 6) Tracés
# =========================================================
plt.figure(figsize=(10, 4))
plt.plot(t_lt, vin_lt, label="Vin LTspice")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Input signal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t_lt, vout_lt, label="Vout LTspice")
plt.plot(t_lt, vout_py, "--", label="Vout Python")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Comparison of output")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t_lt, err, label="Error = Python - LTspice")
plt.xlabel("Time [s]")
plt.ylabel("Error [V]")
plt.title("Pointwise error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#plt.figure(figsize=(10, 4))
#plt.plot(t_Eco, vout_Eco, label="Vout EcoSimPro")
#plt.plot(t_lt, vout_lt, label="Vout LTspice")
#plt.plot(t_lt, vout_py, "--", label="Vout Python")
#plt.xlabel("Time [s]")
#plt.ylabel("Voltage [V]")
#plt.title("Comparison of output")
#plt.grid(True)
#plt.legend()
#plt.tight_layout()
#plt.show()