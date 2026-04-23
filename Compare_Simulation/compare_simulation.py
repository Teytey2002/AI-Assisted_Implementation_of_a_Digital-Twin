from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from dtcalib.simulation import LowPassR1CR2Simulator

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
#file_path = Path("./R1_10_R2_10_C_22micro.txt")   # adapte si nécessaire
#
#df = pd.read_csv(file_path, sep=r"\s+", engine="python")
#
#print("Colonnes trouvées :", df.columns.tolist())
#print(df.head())

# =========================================================
# 1) Charger CSV généré depuis EcoSimPro
# =========================================================
file_path = Path("r1_10_r2_10_c22micro.rpt")  # 

header_idx = find_header_idx(file_path)

df = pd.read_csv(file_path, skiprows=header_idx, sep="\t")
df.columns = [normalize_col(c) for c in df.columns]

print("Colonnes trouvées :", df.columns.tolist())
print(df.head())


# =========================================================
# 2) Extraire t, Vin, Vout
# =========================================================
# D'après l'export précédent :
# - V(n002) = Vin
# - V(n001) = Vout

# For Ltpice
#t = df["time"].to_numpy(dtype=float)
#vin_lt = df["V(n002)"].to_numpy(dtype=float)
#vout_lt = df["V(n001)"].to_numpy(dtype=float)

# For EcoSimPro 
t = df["TIME"].to_numpy(dtype=float)
vin = df["Addition_2.s_out.signal[1]"].to_numpy(dtype=float)
vout_eco = df["SensorVoltage_1.v"].to_numpy(dtype=float)


# =========================================================
# 3) Définir les paramètres du circuit dans ta classe Python
# =========================================================
# Donc choisis ici les vraies valeurs utilisées dans LTspice.

R1 = 10.0       
R2 = 10.0       
C = 22e-6

simulator = LowPassR1CR2Simulator(
    calibrated_params=("C",),
    fixed_params={
        "R1": R1,
        "R2": R2,
    },
    y0_mode="dc_from_u0",
)

theta = np.array([C], dtype=float)


# =========================================================
# 4) Simuler avec exactement le même temps et le même input
# =========================================================
result = simulator.simulate(t=t, u=vin, theta=theta)
vout_py = result.y

print("\nInfos auxiliaires du simulateur :")
for k, v in result.aux.items():
    print(f"{k}: {v}")


# =========================================================
# 5) Calculer les erreurs
# =========================================================
err = vout_py - vout_eco

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
plt.plot(t, vin, label="Vin LTspice")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Input signal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, vout_eco, label="Vout LTspice")
plt.plot(t, vout_py, "--", label="Vout Python")
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Comparison of output")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, err, label="Error = Python - EcoSimPro")
plt.xlabel("Time [s]")
plt.ylabel("Error [V]")
plt.title("Pointwise error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()