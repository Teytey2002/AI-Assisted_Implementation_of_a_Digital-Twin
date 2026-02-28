# search/evaluator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import gc
import time

import numpy as np

try:
    import torch
except Exception:
    torch = None

"""
Fixer seed

Lancer un entraînement à epochs fixes

Retourner uniquement la métrique utilisée pour le racing : val_loss

Nettoyage GPU

Option cache (évite de relancer 2 fois la même config/budget)
"""

@dataclass(frozen=True)
class EvalRequest:
    config_id: int
    params: Dict[str, Any]
    seed: int
    epochs: int
    iteration: int


@dataclass
class EvalResult:
    config_id: int
    seed: int
    epochs: int
    iteration: int
    val_loss: float
    train_loss: Optional[float] = None
    wall_time_s: Optional[float] = None


class Evaluator:
    """
    Adapter between Iterated Racing and your existing training pipeline.

    You must implement/plug `run_training_job(params, seed, epochs)` which returns:
        {"val_loss": float, "train_loss": float (optional)}
    """
    def __init__(self, device: str = "cuda", cache: bool = True):
        self.device = device
        self.cache_enabled = cache
        self._cache: Dict[Tuple[frozenset, int, int], EvalResult] = {}

    def evaluate(self, req: EvalRequest) -> EvalResult:
        key = (frozenset(req.params.items()), req.seed, req.epochs)
        if self.cache_enabled and key in self._cache:
            cached = self._cache[key]
            # keep iteration metadata from request (so logs remain consistent)
            return EvalResult(
                config_id=req.config_id,
                seed=req.seed,
                epochs=req.epochs,
                iteration=req.iteration,
                val_loss=cached.val_loss,
                train_loss=cached.train_loss,
                wall_time_s=cached.wall_time_s,
            )

        self._set_seed(req.seed)

        t0 = time.time()
        metrics = self.run_training_job(req.params, seed=req.seed, epochs=req.epochs)
        dt = time.time() - t0

        res = EvalResult(
            config_id=req.config_id,
            seed=req.seed,
            epochs=req.epochs,
            iteration=req.iteration,
            val_loss=float(metrics["val_loss"]),
            train_loss=float(metrics["train_loss"]) if "train_loss" in metrics and metrics["train_loss"] is not None else None,
            wall_time_s=dt,
        )

        if self.cache_enabled:
            # store without iteration (but same class)
            self._cache[key] = res

        self._cleanup()
        return res

    # --------- to be plugged ----------
    def run_training_job(self, params: Dict[str, Any], seed: int, epochs: int) -> Dict[str, float]:
        """
        TODO: Plug your existing training code here.
        Must be deterministic w.r.t (params, seed, epochs).
        """
        raise NotImplementedError("Plug your project training pipeline here (run_training_job).")

    # --------- helpers ----------
    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        if torch is None:
            return
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism (optional; can slow down)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def _cleanup(self) -> None:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()