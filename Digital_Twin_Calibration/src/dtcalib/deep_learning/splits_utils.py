# src/dtcalib/deep_learning/splits_utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any


def load_split(split_json_path: str | Path) -> Dict[str, Any]:
    split_json_path = Path(split_json_path)
    with open(split_json_path, "r") as f:
        payload = json.load(f)
    return payload


def get_indices(payload: Dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    idx = payload["indices"]
    return idx["train"], idx["val"], idx["test"]