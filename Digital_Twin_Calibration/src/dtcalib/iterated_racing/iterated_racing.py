"""
utilise Sampler (déjà fait)

utilise Evaluator (ci-dessus)

utilise friedman_race

budget schedule : [50, 75, 100, 150, 200]

initial pop 20

keep_top_k décroissant : ex 10 → 8 → 6 → 5 → 5
"""

# search/iterated_racing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import numpy as np
from tqdm import tqdm

from .config import Config
from .sampler import Sampler
from .evaluator import Evaluator, EvalRequest
from .statistical_tests import friedman_race
from .logger import RaceLogger

@dataclass
class RacingSettings:
    seed: int = 42
    alpha: float = 0.05
    n_init: int = 20
    max_iter: int = 5
    budgets: List[int] = None
    keep_top_k: List[int] = None
    n_new_each_iter: int = 10  # how many new configs to generate after elimination

    def __post_init__(self):
        if self.budgets is None:
            self.budgets = [50, 75, 100, 150, 200]
        if self.keep_top_k is None:
            self.keep_top_k = [10, 8, 6, 5, 5]
        assert len(self.budgets) == self.max_iter
        assert len(self.keep_top_k) == self.max_iter


class IteratedRacing:
    def __init__(
        self,
        sampler: Sampler,
        evaluator: Evaluator,
        logger: RaceLogger,
        settings: RacingSettings,
    ):
        self.sampler = sampler
        self.evaluator = evaluator
        self.logger = logger
        self.settings = settings

        self.population: List[Config] = []

    def run(self) -> List[Config]:
        self.population = self.sampler.initial_population(self.settings.n_init)

        # Each config accumulates history per iteration (val_loss)
        for it in range(self.settings.max_iter):
            epochs = self.settings.budgets[it]

            # ---- evaluate all configs at current budget ----
            for c in tqdm(self.population, desc=f"Racing iter {it+1}/{self.settings.max_iter} (epochs={epochs})"):
                req = EvalRequest(
                    config_id=c.id,
                    params=c.params,
                    seed=self.settings.seed,
                    epochs=epochs,
                    iteration=it,
                )
                res = self.evaluator.evaluate(req)
                c.add_result(res.val_loss)

                self.logger.log_eval(
                    iteration=it,
                    params_json=json.dumps(c.params, sort_keys=True),
                    res=res,
                )

            # ---- build scores matrix: only eligible configs (full history) ----
            t = it + 1
            eligible_idx = [i for i, c in enumerate(self.population) if len(c.history) == t]
            eligible = [self.population[i] for i in eligible_idx]

            # If not enough configs are eligible, keep everyone (no statistical test possible)
            if len(eligible) < 2:
                survivors = self.population
                decision = None
            else:
                scores = np.array([c.history for c in eligible], dtype=float)  # (n_eligible, t)

                decision = friedman_race(
                    scores_matrix=scores,
                    alpha=self.settings.alpha,
                    keep_top_k=min(self.settings.keep_top_k[it], len(eligible)),
                )

                # decision.keep_indices are indices within `eligible`
                keep_in_eligible = decision.keep_indices
                survivors = [eligible[j] for j in keep_in_eligible]

            # ---- statistical elimination ----
            decision = friedman_race(
                scores_matrix=scores,
                alpha=self.settings.alpha,
                keep_top_k=self.settings.keep_top_k[it],
            )
            kept_ids = [c.id for c in survivors]
            if decision is not None:
                self.logger.log_decision(it, decision, kept_ids)
            else:
                # no decision (not enough eligible configs), log p_value=1.0 style decision
                # simplest: skip decision logging or log a dummy one
                pass

            # stop early if already small
            if len(survivors) <= self.settings.keep_top_k[-1]:
                self.population = survivors
                continue

            # ---- resample new configs around survivors ----
            n_new = max(0, self.settings.n_new_each_iter)
            new_configs = self.sampler.adaptive_resample(survivors, n_new=n_new)

            self.population = survivors + new_configs

        # final sort by last val_loss (or average)
        self.population.sort(key=lambda c: c.history[-1])
        return self.population
