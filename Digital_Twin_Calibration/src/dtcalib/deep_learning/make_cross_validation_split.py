# src/dtcalib/deep_learning/make_cv_splits.py
"""
Command to use it : 
python3 make_cross_validation_split.py --root-dir ../../../data/ALL_LP_DATASETS_CSV_Deep_learning --k 5 --seed 42 --out-prefix rc_cv

exemple : 
(torch_gpu) pc_ai@PCAI:/mnt/c/Users/samue/Documents/AI-Assisted_Implementation_of_a_Digital-Twin/Digital_Twin_Calibration/src/dtcalib/deep_learning$ python3 make_cross_validation_split.py --root-dir ../../../data/ALL_LP_DATASETS_CSV_Deep_learning --k 5 --seed 42 --out-prefix rc_cv
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

from dtcalib.deep_learning.make_split import parse_samples  

SPLIT_DIR = Path(__file__).resolve().parent / "splits"

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", type=str, default="rc_cv")
    args = p.parse_args()

    root_dir = Path(args.root_dir)
    samples = parse_samples(root_dir)
    if len(samples) == 0:
        raise ValueError("0 samples trouvés. Vérifie le root-dir et le regex +c_...")

    # group indices by C
    groups: Dict[float, List[int]] = defaultdict(list)
    for idx, (_path, cval) in enumerate(samples):
        groups[cval].append(idx)

    c_values = sorted(groups.keys())
    k = int(args.k)
    if k < 2:
        raise ValueError("k must be >= 2")

    rng = np.random.default_rng(args.seed)
    c_values = np.array(c_values, dtype=float)
    rng.shuffle(c_values)

    # split C-values into k folds
    c_folds = np.array_split(c_values, k)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_id, cval_holdout in enumerate(c_folds):
        holdout_set = set(map(float, cval_holdout.tolist()))

        train_idx: List[int] = []
        val_idx: List[int] = []

        for cval, idxs in groups.items():
            if float(cval) in holdout_set:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)

        # shuffle indices for randomness (optional)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        payload: Dict[str, Any] = {
            "seed": args.seed,
            "cv": {"k": k, "fold": fold_id, "group": "C_value"},
            "root_dir": str(root_dir),
            "n_samples": len(samples),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": 0,
            "samples": [{"csv_path": p, "C_value": c} for (p, c) in samples],
            "indices": {"train": train_idx, "val": val_idx, "test": []},
            "heldout_C_values": sorted(list(holdout_set)),
        }

        out_path = SPLIT_DIR / f"{args.out_prefix}_fold{fold_id}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"✅ Fold {fold_id}: train={len(train_idx)} | val={len(val_idx)} -> {out_path}")

if __name__ == "__main__":
    main()