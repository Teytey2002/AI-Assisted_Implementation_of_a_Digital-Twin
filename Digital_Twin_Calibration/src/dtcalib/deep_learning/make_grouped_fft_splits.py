"""
python3 make_grouped_fft_splits.py \
  --root-dir ../../../data/LP_DATASET_R1_R2_C \
  --k 5 \
  --seed 42 \
  --out-prefix rc_r1r2c_fft_grouped \
  --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPLIT_DIR = Path(__file__).resolve().parent / "splits"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=str, default="rc_r1r2c_fft_grouped")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--manifest-name", type=str, default="manifest.csv")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    manifest_path = root_dir / args.manifest_name

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = {"group_name", "csv_path", "R1", "R2", "C", "freq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    group_rows: list[dict[str, Any]] = []

    for group_name, g in df.groupby("group_name", sort=True):
        g = g.sort_values("freq")
        first = g.iloc[0]

        group_rows.append(
            {
                "group_name": str(group_name),
                "R1": float(first["R1"]),
                "R2": float(first["R2"]),
                "C": float(first["C"]),
                "n_freq": int(len(g)),
            }
        )

    n_groups = len(group_rows)
    if n_groups < args.k:
        raise ValueError(f"Not enough groups ({n_groups}) for k={args.k}")

    rng = np.random.default_rng(args.seed)
    group_indices = np.arange(n_groups)
    rng.shuffle(group_indices)

    outer_folds = np.array_split(group_indices, args.k)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_id, test_idx_arr in enumerate(outer_folds):
        test_idx = list(map(int, test_idx_arr.tolist()))
        test_set = set(test_idx)

        remaining = [int(i) for i in group_indices.tolist() if int(i) not in test_set]

        rng_inner = np.random.default_rng(args.seed + fold_id)
        rng_inner.shuffle(remaining)

        n_val = max(1, int(round(args.val_ratio * len(remaining))))
        n_val = min(n_val, len(remaining) - 1)

        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]

        payload = {
            "seed": args.seed,
            "domain": "fft_grouped",
            "root_dir": str(root_dir),
            "n_groups": n_groups,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "groups": group_rows,
            "indices": {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            },
            "train_groups": [group_rows[i] for i in train_idx],
            "val_groups": [group_rows[i] for i in val_idx],
            "test_groups": [group_rows[i] for i in test_idx],
        }

        out_path = SPLIT_DIR / f"{args.out_prefix}_fold{fold_id}.json"

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(
            f"Fold {fold_id}: "
            f"groups(train/val/test)=({len(train_idx)}/{len(val_idx)}/{len(test_idx)}) "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()