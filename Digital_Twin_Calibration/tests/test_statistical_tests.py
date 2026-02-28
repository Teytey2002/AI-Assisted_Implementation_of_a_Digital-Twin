"""
On doit vérifier :

Friedman renvoie p-value faible quand une config est clairement meilleure

p-value élevée quand tout est identique
"""
import pytest
import numpy as np
from dtcalib.iterated_racing.statistical_tests import friedman_race


def test_friedman_detects_difference():
    # 5 configs, 12 blocks
    rng = np.random.default_rng(0)
    n_cfg, n_blk = 5, 12

    # base noise
    scores = rng.normal(loc=0.0, scale=0.05, size=(n_cfg, n_blk))

    # make config 0 consistently better (smaller loss)
    scores[0, :] -= 1.0

    decision = friedman_race(scores, alpha=0.05, keep_top_k=2)
    assert decision.p_value < 0.05
    assert len(decision.keep_indices) == 2
    assert 0 in decision.keep_indices

@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
def test_friedman_no_difference_all_equal():
    scores = np.ones((5, 3))
    decision = friedman_race(scores, alpha=0.05, keep_top_k=3)

    # SciPy can yield nan when variance is zero; treat as "no evidence"
    assert decision.p_value >= 0.05
    # In this case, function keeps all
    assert decision.keep_indices == list(range(5))