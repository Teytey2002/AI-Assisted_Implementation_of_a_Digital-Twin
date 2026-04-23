# src/dtcalib/deep_learning/make_cv_splits.py
"""
test = external fold held out (unseen physical systems)
train/val = split performed on the remaining systems

Each system is defined by a triplet (R1, R2, C)

For each fold_id:
- test_idx = all samples belonging to the selected (R1, R2, C) groups

On the remaining groups:
- a subset is used for training
- a subset is used for validation

Conclusion:
train_idx, val_idx, test_idx share no common (R1, R2, C) combinations

This ensures the model is evaluated on completely unseen systems,
which is consistent with a real-world calibration scenario.

Command to use it:
python3 make_cross_validation_split.py \
  --root-dir ../../../data/LP_DATASET_R1_R2_C \
  --k 5 \
  --seed 42 \
  --out-prefix rc_r1r2c_nested \
  --val-ratio 0.2
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

from dtcalib.deep_learning.splits_utils import parse_samples_from_manifest


SPLIT_DIR = Path(__file__).resolve().parent / "splits"


def group_key_from_sample(sample: dict[str, Any], decimals: int = 12) -> Tuple[float, float, float]:
    """
    Build a robust grouping key for one physical system.
    By default, group = unique triplet (R1, R2, C).
    Rounding is used only to avoid tiny float representation noise.
    """
    return (
        round(float(sample["R1"]), decimals),
        round(float(sample["R2"]), decimals),
        round(float(sample["C"]), decimals),
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=str, required=True, help="Dataset root containing manifest.csv")
    p.add_argument("--k", type=int, default=5, help="Number of outer folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", type=str, default="rc_nested")
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of remaining groups used for validation inside each outer fold",
    )
    args = p.parse_args()

    root_dir = Path(args.root_dir)
    samples = parse_samples_from_manifest(root_dir)

    if len(samples) == 0:
        raise ValueError("No samples found in manifest.")

    k = int(args.k)
    if k < 2:
        raise ValueError("k must be >= 2")

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    # ------------------------------------------------------------
    # Group sample indices by physical-system triplet (R1, R2, C)
    # ------------------------------------------------------------
    groups: Dict[Tuple[float, float, float], List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        key = group_key_from_sample(sample)
        groups[key].append(idx)

    group_keys = list(groups.keys())
    if len(group_keys) < k:
        raise ValueError(
            f"Not enough unique groups ({len(group_keys)}) for k={k}. "
            "Reduce k or generate more systems."
        )

    rng = np.random.default_rng(args.seed)
    group_keys = list(group_keys)
    rng.shuffle(group_keys)

    # Outer CV folds = TEST groups (pure Python split, keeps tuples hashable)
    outer_test_folds = []
    n_groups = len(group_keys)

    for fold_id in range(k):
        start = fold_id * n_groups // k
        end = (fold_id + 1) * n_groups // k
        outer_test_folds.append(group_keys[start:end])

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_id, test_group_keys in enumerate(outer_test_folds):
        test_group_keys = list(test_group_keys)
        test_group_set = set(test_group_keys)

        # Remaining groups for train/val
        remaining_group_keys = [g for g in group_keys if g not in test_group_set]

        rng_inner = np.random.default_rng(args.seed + fold_id)
        rng_inner.shuffle(remaining_group_keys)

        n_remaining = len(remaining_group_keys)
        n_val_groups = max(1, int(round(args.val_ratio * n_remaining)))
        n_val_groups = min(n_val_groups, n_remaining - 1)  # keep at least one train group

        val_group_keys = remaining_group_keys[:n_val_groups]
        train_group_keys = remaining_group_keys[n_val_groups:]

        train_group_set = set(train_group_keys)
        val_group_set = set(val_group_keys)

        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for group_key, idxs in groups.items():
            if group_key in test_group_set:
                test_idx.extend(idxs)
            elif group_key in val_group_set:
                val_idx.extend(idxs)
            elif group_key in train_group_set:
                train_idx.extend(idxs)
            else:
                raise RuntimeError(f"Group {group_key} not assigned to train/val/test.")

        rng_inner.shuffle(train_idx)
        rng_inner.shuffle(val_idx)
        rng_inner.shuffle(test_idx)

        payload: Dict[str, Any] = {
            "seed": args.seed,
            "cv": {
                "k": k,
                "fold": fold_id,
                "group": "R1_R2_C_triplet",
                "val_ratio_within_remaining_groups": args.val_ratio,
            },
            "root_dir": str(root_dir),
            "n_samples": len(samples),
            "n_groups": len(groups),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "samples": samples,
            "indices": {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            },
            "train_groups": [
                {"R1": g[0], "R2": g[1], "C": g[2]} for g in sorted(train_group_keys)
            ],
            "val_groups": [
                {"R1": g[0], "R2": g[1], "C": g[2]} for g in sorted(val_group_keys)
            ],
            "test_groups": [
                {"R1": g[0], "R2": g[1], "C": g[2]} for g in sorted(test_group_keys)
            ],
        }

        out_path = SPLIT_DIR / f"{args.out_prefix}_fold{fold_id}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(
            f"✅ Fold {fold_id}: "
            f"groups(train/val/test)=({len(train_group_keys)}/{len(val_group_keys)}/{len(test_group_keys)}) | "
            f"samples(train/val/test)=({len(train_idx)}/{len(val_idx)}/{len(test_idx)}) "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()