"""
Comme validé : SciPy + méthode pragmatique :

On fait Friedman sur la matrice configs x iterations

Si p < alpha : on calcule le rank moyen et on garde les meilleurs

On élimine ceux trop loin du meilleur via une règle simple (top-K) + option “margin”

Ça reste “racing” (élimination progressive) sans passer 2 jours sur un post-hoc complet.
"""


# search/statistical_tests.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.stats import friedmanchisquare, rankdata


@dataclass
class RaceDecision:
    p_value: float
    keep_indices: List[int]     # indices in the input order
    mean_ranks: List[float]     # same order as input


def friedman_race(
    scores_matrix: np.ndarray,
    alpha: float = 0.05,
    keep_top_k: int = 8,
) -> RaceDecision:
    """
    scores_matrix: shape (n_configs, n_blocks)
        lower is better (loss). Each column is a "block" (here: iteration budget stage).
    We run Friedman across configs using blocks as repeated measures.

    Practical elimination:
      - if Friedman not significant -> keep all
      - else keep top-k by mean rank (best=lowest rank)
    """
    if scores_matrix.ndim != 2:
        raise ValueError("scores_matrix must be 2D (n_configs, n_blocks)")
    n_configs, n_blocks = scores_matrix.shape
    if n_configs < 2 or n_blocks < 2:
        # not enough evidence to compare statistically
        keep = list(range(n_configs))
        mean_ranks = [1.0] * n_configs
        return RaceDecision(p_value=1.0, keep_indices=keep, mean_ranks=mean_ranks)

    # SciPy expects each "treatment" as a separate array of blocks
    arrays = [scores_matrix[i, :] for i in range(n_configs)]
    stat, p = friedmanchisquare(*arrays)
    # If all scores are identical across configs for every block,
    # there is no statistical difference.
    if np.allclose(scores_matrix, scores_matrix[0, :]):
        keep = list(range(n_configs))
        mean_ranks = [1.0] * n_configs
        return RaceDecision(p_value=1.0, keep_indices=keep, mean_ranks=mean_ranks)
        
    # ranks per block (column): best loss -> rank 1
    ranks = np.zeros_like(scores_matrix, dtype=float)
    for j in range(n_blocks):
        ranks[:, j] = rankdata(scores_matrix[:, j], method="average")  # smaller loss -> smaller rank
    mean_ranks = ranks.mean(axis=1)

    if p >= alpha:
        keep = list(range(n_configs))
    else:
        k = min(max(2, keep_top_k), n_configs)
        keep = np.argsort(mean_ranks)[:k].tolist()

    return RaceDecision(p_value=float(p), keep_indices=keep, mean_ranks=mean_ranks.tolist())