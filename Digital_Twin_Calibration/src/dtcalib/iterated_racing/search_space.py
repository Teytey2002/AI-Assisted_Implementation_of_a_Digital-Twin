from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import random


@dataclass
class SearchSpace:
    lr_range: Tuple[float, float] = (1e-5, 3e-3)
    wd_range: Tuple[float, float] = (1e-8, 1e-3)

    batch_sizes: List[int] = (64, 128)
    optimizers: List[str] = ("adam", "adamw")
    scheduler_factors: List[float] = (0.5, 0.8)
    scheduler_patiences: List[int] = (5, 10, 20)
    target_transforms: List[str] = ("C", "logC")