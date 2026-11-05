from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    success: bool
    best_step: int
    best_prob: float
    best_image: np.ndarray
    best_property_value: object
