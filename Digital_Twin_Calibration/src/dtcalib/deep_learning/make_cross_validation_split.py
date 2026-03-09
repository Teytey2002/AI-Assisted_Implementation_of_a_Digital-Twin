# src/dtcalib/deep_learning/make_cv_splits.py
"""
test = le fold externe tenu à l’écart
train/val = split fait sur les autres groupes de C

Donc pour chaque fold_id :
- test_idx = toutes les samples des C du fold

sur les C restants :
- une partie en train
- une partie en val

Conclusion : 
train_idx, val_idx, test_idx n’ont aucun C en commun

Command to use it : 
python3 make_cv_splits.py \
  --root-dir ../../../data/ALL_LP_DATASETS_CSV_Deep_learning \
  --k 5 \
  --seed 42 \
  --out-prefix rc_nested \
  --val-ratio 0.2
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

from dtcalib.deep_learning.splits_utils import parse_samples

SPLIT_DIR = Path(__file__).resolve().parent / "splits"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", type=str, default="rc_cv")
    p.add_argument("--val-ratio", type=float, default=0.2)  # fraction of remaining C groups used for val
    args = p.parse_args()

    root_dir = Path(args.root_dir)
    samples = parse_samples(root_dir)
    if len(samples) == 0:
        raise ValueError("0 samples trouvés. Vérifie le root-dir et le regex +c_...")

    # Group sample indices by C value
    groups: Dict[float, List[int]] = defaultdict(list)
    for idx, (_path, cval) in enumerate(samples):
        groups[cval].append(idx)

    c_values = sorted(groups.keys())
    k = int(args.k)
    if k < 2:
        raise ValueError("k must be >= 2")

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    rng = np.random.default_rng(args.seed)

    c_values = np.array(c_values, dtype=float)
    rng.shuffle(c_values)

    # Outer CV folds: these are the TEST groups
    c_test_folds = np.array_split(c_values, k)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_id, cval_test in enumerate(c_test_folds):
        test_c_set = set(map(float, cval_test.tolist()))

        # Remaining C groups used for train/val split
        remaining_c = [float(c) for c in c_values.tolist() if float(c) not in test_c_set]
        remaining_c = np.array(remaining_c, dtype=float)
        rng_inner = np.random.default_rng(args.seed + fold_id)
        rng_inner.shuffle(remaining_c)

        n_remaining = len(remaining_c)
        n_val_groups = max(1, int(round(args.val_ratio * n_remaining)))
        n_val_groups = min(n_val_groups, n_remaining - 1)  # keep at least 1 group for train

        val_c_set = set(map(float, remaining_c[:n_val_groups].tolist()))
        train_c_set = set(map(float, remaining_c[n_val_groups:].tolist()))

        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for cval, idxs in groups.items():
            cval_f = float(cval)
            if cval_f in test_c_set:
                test_idx.extend(idxs)
            elif cval_f in val_c_set:
                val_idx.extend(idxs)
            elif cval_f in train_c_set:
                train_idx.extend(idxs)
            else:
                raise RuntimeError(f"C value {cval_f} not assigned to train/val/test.")

        rng_inner.shuffle(train_idx)
        rng_inner.shuffle(val_idx)
        rng_inner.shuffle(test_idx)

        payload: Dict[str, Any] = {
            "seed": args.seed,
            "cv": {
                "k": k,
                "fold": fold_id,
                "group": "C_value",
                "val_ratio_within_remaining_groups": args.val_ratio,
            },
            "root_dir": str(root_dir),
            "n_samples": len(samples),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "samples": [{"csv_path": p, "C_value": c} for (p, c) in samples],
            "indices": {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            },
            "train_C_values": sorted(list(train_c_set)),
            "val_C_values": sorted(list(val_c_set)),
            "test_C_values": sorted(list(test_c_set)),
        }

        out_path = SPLIT_DIR / f"{args.out_prefix}_fold{fold_id}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(
            f"✅ Fold {fold_id}: "
            f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)} "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()