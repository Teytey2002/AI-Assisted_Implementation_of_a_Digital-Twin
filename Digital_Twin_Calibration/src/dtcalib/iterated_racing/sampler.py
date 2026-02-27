import numpy as np
import random
from .config import Config


class Sampler:
    def __init__(self, search_space, seed=42):
        self.space = search_space
        self.rng = np.random.default_rng(seed)     # numpy RNG (local)
        self.py_rng = random.Random(seed)          # python RNG (local)
        self.next_id = 0

    # ---------- LOG-UNIFORM ----------
    def _sample_log_uniform(self, low, high):
        return 10 ** self.rng.uniform(np.log10(low), np.log10(high))

    # ---------- INITIAL POPULATION ----------
    def initial_population(self, n_configs):
        population = []
        for _ in range(n_configs):
            params = {
                "lr": self._sample_log_uniform(*self.space.lr_range),
                "weight_decay": self._sample_log_uniform(*self.space.wd_range),
                "batch_size": self.py_rng.choice(list(self.space.batch_sizes)),
                "optimizer": self.py_rng.choice(list(self.space.optimizers)),
                "scheduler_factor": self.py_rng.choice(list(self.space.scheduler_factors)),
                "scheduler_patience": self.py_rng.choice(list(self.space.scheduler_patiences)),
                "target_transform": self.py_rng.choice(list(self.space.target_transforms)),
            }
            population.append(Config(self.next_id, params))
            self.next_id += 1
        return population

    # ---------- ADAPTIVE RESAMPLING ----------
    def adaptive_resample(self, survivors, n_new):
        new_configs = []

        lrs = np.array([c.params["lr"] for c in survivors])
        wds = np.array([c.params["weight_decay"] for c in survivors])

        lr_min, lr_max = lrs.min(), lrs.max()
        wd_min, wd_max = wds.min(), wds.max()

        lr_low = max(self.space.lr_range[0], lr_min * 0.5)
        lr_high = min(self.space.lr_range[1], lr_max * 2)

        wd_low = max(self.space.wd_range[0], wd_min * 0.5)
        wd_high = min(self.space.wd_range[1], wd_max * 2)

        def weighted_choice(values, key):
            freqs = {}
            for c in survivors:
                v = c.params[key]
                freqs[v] = freqs.get(v, 0) + 1
            total = sum(freqs.values())
            weights = [freqs.get(v, 0) / total for v in values]
            return self.py_rng.choices(list(values), weights=weights, k=1)[0]

        for _ in range(n_new):
            params = {
                "lr": 10 ** self.rng.uniform(np.log10(lr_low), np.log10(lr_high)),
                "weight_decay": 10 ** self.rng.uniform(np.log10(wd_low), np.log10(wd_high)),
                "batch_size": weighted_choice(self.space.batch_sizes, "batch_size"),
                "optimizer": weighted_choice(self.space.optimizers, "optimizer"),
                "scheduler_factor": weighted_choice(self.space.scheduler_factors, "scheduler_factor"),
                "scheduler_patience": weighted_choice(self.space.scheduler_patiences, "scheduler_patience"),
                "target_transform": weighted_choice(self.space.target_transforms, "target_transform"),
            }
            new_configs.append(Config(self.next_id, params))
            self.next_id += 1

        return new_configs