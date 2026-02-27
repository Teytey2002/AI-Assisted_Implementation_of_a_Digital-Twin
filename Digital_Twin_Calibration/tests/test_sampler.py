import numpy as np
from dtcalib.iterated_racing.search_space import SearchSpace
from dtcalib.iterated_racing.sampler import Sampler


def test_initial_population_size():
    space = SearchSpace()
    sampler = Sampler(space, seed=123)

    pop = sampler.initial_population(10)

    assert len(pop) == 10


def test_initial_population_ranges():
    space = SearchSpace()
    sampler = Sampler(space, seed=123)

    pop = sampler.initial_population(50)

    for config in pop:
        p = config.params

        assert space.lr_range[0] <= p["lr"] <= space.lr_range[1]
        assert space.wd_range[0] <= p["weight_decay"] <= space.wd_range[1]
        assert p["batch_size"] in space.batch_sizes
        assert p["optimizer"] in space.optimizers
        assert p["scheduler_factor"] in space.scheduler_factors
        assert p["scheduler_patience"] in space.scheduler_patiences
        assert p["target_transform"] in space.target_transforms


def test_log_uniform_sampling_positive():
    space = SearchSpace()
    sampler = Sampler(space, seed=123)

    pop = sampler.initial_population(20)

    for config in pop:
        assert config.params["lr"] > 0
        assert config.params["weight_decay"] > 0


def test_unique_ids():
    space = SearchSpace()
    sampler = Sampler(space, seed=123)

    pop = sampler.initial_population(30)

    ids = [c.id for c in pop]
    assert len(ids) == len(set(ids))


def test_adaptive_resample_restricts_range():
    space = SearchSpace()
    sampler = Sampler(space, seed=123)

    initial_pop = sampler.initial_population(10)

    survivors = initial_pop[:5]
    new_configs = sampler.adaptive_resample(survivors, 10)

    # Compute survivor ranges
    lr_values = [c.params["lr"] for c in survivors]
    wd_values = [c.params["weight_decay"] for c in survivors]

    lr_min, lr_max = min(lr_values), max(lr_values)
    wd_min, wd_max = min(wd_values), max(wd_values)

    for config in new_configs:
        assert config.params["lr"] >= space.lr_range[0]
        assert config.params["lr"] <= space.lr_range[1]
        assert config.params["weight_decay"] >= space.wd_range[0]
        assert config.params["weight_decay"] <= space.wd_range[1]


def test_reproducibility_with_seed():
    space = SearchSpace()

    sampler1 = Sampler(space, seed=42)
    sampler2 = Sampler(space, seed=42)

    pop1 = sampler1.initial_population(10)
    pop2 = sampler2.initial_population(10)

    for c1, c2 in zip(pop1, pop2):
        assert c1.params == c2.params