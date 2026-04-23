# src/dtcalib/deep_learning/splits_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def load_split(split_json_path: str | Path) -> Dict[str, Any]:
    split_json_path = Path(split_json_path)
    with open(split_json_path, "r") as f:
        payload = json.load(f)
    return payload


def get_indices(payload: Dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    idx = payload["indices"]
    return idx["train"], idx["val"], idx["test"]


def parse_samples_from_manifest(
    root_dir: str | Path,
    manifest_name: str = "manifest.csv",
) -> list[dict[str, Any]]:
    """
    Read dataset samples from manifest.csv.

    Returns a deterministic ordered list of dicts:
        [
            {
                "csv_path": "<absolute path>",
                "group_name": "...",
                "experiment_name": "...",
                "R1": ...,
                "R2": ...,
                "C": ...,
                "fc": ...,
                "freq": ...,
            },
            ...
        ]
    """
    root_dir = Path(root_dir)
    manifest_path = root_dir / manifest_name

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_cols = {"csv_path", "R1", "R2", "C"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    samples: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        rel_csv = Path(str(row["csv_path"]))
        abs_csv = root_dir / rel_csv

        sample = {
            "csv_path": str(abs_csv),
            "group_name": str(row["group_name"]) if "group_name" in df.columns else "",
            "experiment_name": str(row["experiment_name"]) if "experiment_name" in df.columns else "",
            "R1": float(row["R1"]),
            "R2": float(row["R2"]),
            "C": float(row["C"]),
        }

        if "fc" in df.columns:
            sample["fc"] = float(row["fc"])
        if "freq" in df.columns:
            sample["freq"] = float(row["freq"])

        samples.append(sample)

    return samples