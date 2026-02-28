# src/dtcalib/deep_learning/make_split.py
"""
Command used for create the split : 
(torch_gpu) xxxx@TeyteyCase:/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma2/TFE/AI-Assisted_Implementation_of_a_Digital-Twin/Digital_Twin_Calibration/src/dtcalib/deep_learning$ 
python3 make_split.py --root-dir ../../../data/ALL_LP_DATASETS_CSV_Deep_learning
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np


SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SPLIT_DIR = Path(__file__).resolve().parent / "splits"
SPLIT_PATH = SPLIT_DIR / "rc_split_seed42_70_15_15.json"


def parse_samples(root_dir: Path):
    """
    Reproduit la logique de RCSignalDataset:
    - détecte les dossiers dataset_+c_...
    - extrait C_value
    - récupère tous les CSV 'results*.csv'
    Retourne une liste ordonnée et déterministe:
        samples = [(csv_path_str, C_value_float), ...]
    """
    root_dir = Path(root_dir)
    samples = []

    for c_folder in sorted(root_dir.iterdir()):
        if not c_folder.is_dir():
            continue

        match = re.search(r"\+c_([0-9p]+e[m|p][0-9]+)", c_folder.name)
        if match is None:
            continue

        c_token = match.group(1)
        c_str = c_token.replace("p", ".").replace("em", "e-").replace("ep", "e+")
        C_value = float(c_str)

        for csv_file in sorted(c_folder.rglob("*.csv")):
            if "results" not in csv_file.name.lower():
                continue
            samples.append((str(csv_file), C_value))

    return samples


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=str, required=True)
    args = p.parse_args()

    root_dir = Path(args.root_dir)
    samples = parse_samples(root_dir)

    if len(samples) == 0:
        raise ValueError("0 samples trouvés. Vérifie le root-dir et le regex +c_...")

    # Group by C (float). Ici c'est ok car les valeurs viennent d'un parse unique et exact du nom de dossier.
    groups = defaultdict(list)
    for idx, (_path, cval) in enumerate(samples):
        groups[cval].append(idx)

    rng = np.random.default_rng(SPLIT_SEED)

    train_idx, val_idx, test_idx = [], [], []

    for cval, idxs in groups.items():
        idxs = np.array(idxs, dtype=int)
        rng.shuffle(idxs)

        n = len(idxs)
        n_train = int(round(TRAIN_RATIO * n))
        n_val = int(round(VAL_RATIO * n))
        # assure que tout est utilisé
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        test_idx.extend(idxs[n_train + n_val:].tolist())

    # Optionnel: on reshuffle globalement pour éviter un ordre par C
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Sauvegarde
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": SPLIT_SEED,
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "root_dir": str(root_dir),
        "n_samples": len(samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "samples": [{"csv_path": p, "C_value": c} for (p, c) in samples],
        "indices": {"train": train_idx, "val": val_idx, "test": test_idx},
    }

    with open(SPLIT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Split écrit dans: {SPLIT_PATH}")
    print(f"   train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    print("   (Split stratifié par C : 70/15/15 à l'intérieur de chaque C)")


if __name__ == "__main__":
    main()