from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from dtcalib.data import ExperimentsDataset
from dtcalib.simulation import LowPassR1CR2Simulator
from dtcalib.metrics import Metrics


def evaluate_theta_on_experiment(simulator, experiment, theta):
    y_pred = simulator.simulate(experiment.t, experiment.u, theta).y
    metrics = Metrics.compute(experiment.y, y_pred)
    return metrics.rmse


def main():
    parser = argparse.ArgumentParser()

    # irace passe les paramètres comme ça :
    # --C=1e-6 ou --logC=-13
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--logC", type=float, default=None)

    # instance = path vers CSV
    parser.add_argument("--instance", type=str, required=True)

    args = parser.parse_args()

    # -----------------------------
    # 1. Lire theta
    # -----------------------------
    if args.C is not None:
        C = args.C
    elif args.logC is not None:
        C = float(np.exp(args.logC))
    else:
        raise ValueError("Either --C or --logC must be provided.")

    theta = np.array([C], dtype=float)

    # -----------------------------
    # 2. Charger 1 seul experiment
    # -----------------------------
    csv_path = Path(args.instance)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # On utilise le loader existant mais sur un seul fichier
    ds = ExperimentsDataset.from_csv_folder(csv_path.parent)
    
    # récupérer le bon experiment
    exp = None
    for e in ds.experiments:
        if e.meta["filename"] == csv_path.name:
            exp = e
            break

    if exp is None:
        raise RuntimeError(f"Experiment not found for {csv_path.name}")

    # -----------------------------
    # 3. Simulator 
    # -----------------------------
    simulator = LowPassR1CR2Simulator(R1=10_000.0, R2=10_000.0, use_C=True, y0_mode="dc_from_u0")

    # -----------------------------
    # 4. Evaluation
    # -----------------------------
    cost = evaluate_theta_on_experiment(simulator, exp, theta)

    # -----------------------------
    # 5. OUTPUT (CRITIQUE pour irace)
    # -----------------------------
    print(cost)


if __name__ == "__main__":
    main()