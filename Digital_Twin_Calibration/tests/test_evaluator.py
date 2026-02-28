"""
On doit tester que :

cache=True → même config évaluée 1 seule fois
"""
from dtcalib.iterated_racing.evaluator import Evaluator, EvalRequest


class DummyEvaluator(Evaluator):
    def __init__(self):
        super().__init__(device="cpu", cache=True)
        self.calls = 0

    def run_training_job(self, params, seed, epochs):
        self.calls += 1
        return {"val_loss": 1.0, "train_loss": 1.0}


def test_cache_avoids_duplicate_calls():
    ev = DummyEvaluator()

    req = EvalRequest(
        config_id=1,
        params={"a": 1},
        seed=0,
        epochs=10,
        iteration=0,
    )

    ev.evaluate(req)
    ev.evaluate(req)

    assert ev.calls == 1