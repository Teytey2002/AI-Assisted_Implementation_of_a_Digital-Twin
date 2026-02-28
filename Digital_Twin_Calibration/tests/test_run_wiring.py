"""
Un smoke test qui vérifie que ProjectEvaluator.run_training_job() appelle bien la fonction de train et renvoie un dict avec val_loss/train_loss, sans lancer un vrai training GPU.

Comme ProjectEvaluator est dans run.py (et appelle une fonction réelle qui chargerait le dataset), on ne va pas exécuter le training : on va monkeypatcher la fonction run_training_job_fixed_epochs pour vérifier le branchement.
"""

from __future__ import annotations

from pathlib import Path
import types

from dtcalib.iterated_racing import run as run_module


def test_project_evaluator_calls_train_api(monkeypatch):
    calls = {"n": 0}

    def fake_train(*, root_dir, split_json_path, params, seed, epochs, device="cuda"):
        calls["n"] += 1
        # check args propagated
        assert Path(root_dir)
        assert Path(split_json_path)
        assert isinstance(params, dict)
        assert isinstance(seed, int)
        assert isinstance(epochs, int)
        return {"val_loss": 0.123, "train_loss": 0.456}

    # monkeypatch the function imported/used by run.py
    monkeypatch.setattr(run_module, "run_training_job_fixed_epochs", fake_train)

    ev = run_module.ProjectEvaluator(
        root_dir="some/root",
        split_json_path="some/split.json",
        device="cpu",
        cache=False,
    )

    out = ev.run_training_job(params={"lr": 1e-3}, seed=0, epochs=10)
    assert calls["n"] == 1
    assert out["val_loss"] == 0.123
    assert out["train_loss"] == 0.456