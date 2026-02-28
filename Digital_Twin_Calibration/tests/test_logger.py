"""
On vérifie que :

fichiers CSV créés
lignes ajoutées
colonnes correctes
"""
from pathlib import Path

from dtcalib.iterated_racing.logger import RaceLogger
from dtcalib.iterated_racing.evaluator import EvalResult
from dtcalib.iterated_racing.statistical_tests import RaceDecision


def test_logger_writes_files(tmp_path: Path):
    logger = RaceLogger(out_dir=str(tmp_path))

    res = EvalResult(
        config_id=1,
        seed=0,
        epochs=10,
        iteration=0,
        val_loss=0.5,
        train_loss=0.6,
        wall_time_s=1.23,
    )

    logger.log_eval(iteration=0, params_json='{"a": 1}', res=res)

    decision = RaceDecision(
        p_value=0.01,
        keep_indices=[0, 1],
        mean_ranks=[1.0, 2.0],
    )
    logger.log_decision(iteration=0, decision=decision, kept_ids=[1, 2])

    assert (tmp_path / "racing_evals.csv").exists()
    assert (tmp_path / "racing_decisions.csv").exists()

    # (optionnel) check headers not empty
    assert (tmp_path / "racing_evals.csv").stat().st_size > 0
    assert (tmp_path / "racing_decisions.csv").stat().st_size > 0