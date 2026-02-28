# search/logger.py
from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Any, List
import csv
import os
import time

from .evaluator import EvalResult
from .statistical_tests import RaceDecision


class RaceLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.eval_path = os.path.join(out_dir, "racing_evals.csv")
        self.dec_path = os.path.join(out_dir, "racing_decisions.csv")

        if not os.path.exists(self.eval_path):
            with open(self.eval_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "timestamp", "iteration", "config_id", "seed", "epochs",
                    "val_loss", "train_loss", "wall_time_s", "params_json"
                ])
                w.writeheader()

        if not os.path.exists(self.dec_path):
            with open(self.dec_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "timestamp", "iteration", "p_value", "keep_config_ids", "mean_ranks"
                ])
                w.writeheader()

    def log_eval(self, iteration: int, params_json: str, res: EvalResult) -> None:
        row = {
            "timestamp": int(time.time()),
            "iteration": iteration,
            "config_id": res.config_id,
            "seed": res.seed,
            "epochs": res.epochs,
            "val_loss": res.val_loss,
            "train_loss": res.train_loss,
            "wall_time_s": res.wall_time_s,
            "params_json": params_json,
        }
        with open(self.eval_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writerow(row)

    def log_decision(self, iteration: int, decision: RaceDecision, kept_ids: List[int]) -> None:
        row = {
            "timestamp": int(time.time()),
            "iteration": iteration,
            "p_value": decision.p_value,
            "keep_config_ids": kept_ids,
            "mean_ranks": decision.mean_ranks,
        }
        with open(self.dec_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writerow(row)