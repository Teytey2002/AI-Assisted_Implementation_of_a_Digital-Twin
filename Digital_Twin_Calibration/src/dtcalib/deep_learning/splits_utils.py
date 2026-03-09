# src/dtcalib/deep_learning/splits_utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import re


def load_split(split_json_path: str | Path) -> Dict[str, Any]:
    split_json_path = Path(split_json_path)
    with open(split_json_path, "r") as f:
        payload = json.load(f)
    return payload


def get_indices(payload: Dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    idx = payload["indices"]
    return idx["train"], idx["val"], idx["test"]


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