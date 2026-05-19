"""
Create grouped FFT splits.

Single split example:
python3 make_grouped_fft_splits.py \
  --root-dir ../../../data/DL_DATASETS/ThreeStageRC_caps_only \
  --mode single \
  --seed 42 \
  --out-prefix ThreeStageRC_caps_only_fft_grouped \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1

K-fold example:
python3 make_grouped_fft_splits.py \
  --root-dir ../../../data/DL_DATASETS/ThreeStageRC_caps_only \
  --mode kfold \
  --k 5 \
  --seed 42 \
  --out-prefix ThreeStageRC_caps_only_fft_grouped \
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


def load_metadata(root_dir: Path) -> dict[str, Any]:
    metadata_path = root_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        return json.load(f)


def build_group_rows(
    df: pd.DataFrame,
    parameter_names: list[str],
) -> list[dict[str, Any]]:
    group_rows: list[dict[str, Any]] = []

    for group_name, g in df.groupby("group_name", sort=True):
        g = g.sort_values("freq")
        first = g.iloc[0]

        row: dict[str, Any] = {
            "group_name": str(group_name),
            "n_freq": int(len(g)),
        }

        for p in parameter_names:
            if p in df.columns:
                row[p] = float(first[p])

        group_rows.append(row)

    return group_rows


def write_split(
    *,
    out_path: Path,
    seed: int,
    root_dir: Path,
    group_rows: list[dict[str, Any]],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    metadata: dict[str, Any],
    mode: str,
) -> None:
    payload = {
        "seed": seed,
        "mode": mode,
        "domain": "fft_grouped",
        "root_dir": str(root_dir),
        "metadata": metadata,
        "n_groups": len(group_rows),
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

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Split saved: {out_path}\n"
        f"groups(train/val/test)=({len(train_idx)}/{len(val_idx)}/{len(test_idx)})"
    )


def make_single_split(
    *,
    group_rows: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[int], list[int], list[int]]:
    total = train_ratio + val_ratio + test_ratio

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1.0, got {total}"
        )

    n_groups = len(group_rows)

    if n_groups < 3:
        raise ValueError("Need at least 3 groups for train/val/test split.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_groups)
    rng.shuffle(indices)

    n_train = int(round(train_ratio * n_groups))
    n_val = int(round(val_ratio * n_groups))

    if n_train <= 0 or n_val <= 0:
        raise ValueError("Train and validation splits must not be empty.")

    if n_train + n_val >= n_groups:
        raise ValueError("Test split would be empty. Adjust ratios.")

    train_idx = indices[:n_train].astype(int).tolist()
    val_idx = indices[n_train:n_train + n_val].astype(int).tolist()
    test_idx = indices[n_train + n_val:].astype(int).tolist()

    return train_idx, val_idx, test_idx


def make_kfold_splits(
    *,
    group_rows: list[dict[str, Any]],
    seed: int,
    k: int,
    val_ratio: float,
) -> list[tuple[list[int], list[int], list[int]]]:
    n_groups = len(group_rows)

    if k < 2:
        raise ValueError("k must be >= 2 for kfold mode.")

    if n_groups < k:
        raise ValueError(f"Not enough groups ({n_groups}) for k={k}")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    group_indices = np.arange(n_groups)
    rng.shuffle(group_indices)

    outer_folds = np.array_split(group_indices, k)

    splits = []

    for fold_id, test_idx_arr in enumerate(outer_folds):
        test_idx = list(map(int, test_idx_arr.tolist()))
        test_set = set(test_idx)

        remaining = [
            int(i)
            for i in group_indices.tolist()
            if int(i) not in test_set
        ]

        rng_inner = np.random.default_rng(seed + fold_id)
        rng_inner.shuffle(remaining)

        n_val = max(1, int(round(val_ratio * len(remaining))))
        n_val = min(n_val, len(remaining) - 1)

        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]

        splits.append((train_idx, val_idx, test_idx))

    return splits


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="single", choices=["single", "kfold"])

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=str, default="fft_grouped")
    parser.add_argument("--manifest-name", type=str, default="manifest.csv")

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    manifest_path = root_dir / args.manifest_name

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    metadata = load_metadata(root_dir)

    all_params = list(metadata.get("all_params", []))
    calibrated_params = list(metadata.get("calibrated_params", []))

    parameter_names = list(dict.fromkeys(all_params + calibrated_params))

    df = pd.read_csv(manifest_path)

    required = {"group_name", "csv_path", "freq"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    group_rows = build_group_rows(
        df=df,
        parameter_names=parameter_names,
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        train_idx, val_idx, test_idx = make_single_split(
            group_rows=group_rows,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        out_path = SPLIT_DIR / f"{args.out_prefix}_split.json"

        write_split(
            out_path=out_path,
            seed=args.seed,
            root_dir=root_dir,
            group_rows=group_rows,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            metadata=metadata,
            mode="single",
        )

    else:
        splits = make_kfold_splits(
            group_rows=group_rows,
            seed=args.seed,
            k=args.k,
            val_ratio=args.val_ratio,
        )

        for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits):
            out_path = SPLIT_DIR / f"{args.out_prefix}_fold{fold_id}.json"

            write_split(
                out_path=out_path,
                seed=args.seed,
                root_dir=root_dir,
                group_rows=group_rows,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                metadata=metadata,
                mode="kfold",
            )


if __name__ == "__main__":
    main()