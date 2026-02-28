"""
On crée un evaluator mock qui :

retourne une val_loss déterministe basée sur params
permet de simuler une hiérarchie de configs
Cela permet de tester :

élimination correcte

p-value bien calculée

ranking cohérent

resampling bien déclenché

historique bien mis à jour
"""
from __future__ import annotations

import numpy as np

from dtcalib.iterated_racing.search_space import SearchSpace
from dtcalib.iterated_racing.sampler import Sampler
from dtcalib.iterated_racing.iterated_racing import IteratedRacing, RacingSettings
from dtcalib.iterated_racing.evaluator import Evaluator
from dtcalib.iterated_racing.logger import RaceLogger


# ==========================================================
# Fake Evaluator
# ==========================================================

class FakeEvaluator(Evaluator):
    """
    Deterministic evaluator:
    val_loss = lr + weight_decay
    Smaller is better.
    """

    def run_training_job(self, params, seed: int, epochs: int):
        score = params["lr"] + params["weight_decay"]
        return {"val_loss": score, "train_loss": score}


# ==========================================================
# Core algorithm tests
# ==========================================================

def test_iterated_racing_runs_without_crash(tmp_path):
    space = SearchSpace()
    sampler = Sampler(space, seed=123)
    logger = RaceLogger(out_dir=str(tmp_path))

    evaluator = FakeEvaluator(device="cpu", cache=False)

    settings = RacingSettings(
        seed=123,
        alpha=0.05,
        n_init=10,
        max_iter=2,
        budgets=[5, 10],
        keep_top_k=[5, 3],
        n_new_each_iter=5,
    )

    racing = IteratedRacing(
        sampler=sampler,
        evaluator=evaluator,
        logger=logger,
        settings=settings,
    )

    final_population = racing.run()

    assert len(final_population) <= settings.keep_top_k[-1] + settings.n_new_each_iter


def test_best_config_has_lowest_score(tmp_path):
    space = SearchSpace()
    sampler = Sampler(space, seed=123)
    logger = RaceLogger(out_dir=str(tmp_path))
    evaluator = FakeEvaluator(device="cpu", cache=False)

    settings = RacingSettings(
        seed=123,
        alpha=0.05,
        n_init=20,
        max_iter=1,
        budgets=[5],
        keep_top_k=[5],
        n_new_each_iter=0,
    )

    racing = IteratedRacing(
        sampler=sampler,
        evaluator=evaluator,
        logger=logger,
        settings=settings,
    )

    final_population = racing.run()

    scores = [c.history[-1] for c in final_population]
    assert scores == sorted(scores)