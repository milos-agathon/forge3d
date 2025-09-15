#!/usr/bin/env python3
"""A25: Object Importance Sampling"""

import numpy as np
from typing import Dict, Optional, Tuple

class ObjectImportanceSampler:
    """A25: Per-object importance hints."""

    def __init__(self):
        self.object_weights: Dict[int, float] = {}
        self.total_weight: float = 0.0

    def set_object_importance(self, object_id: int, weight: float) -> None:
        """A25: MIS weighting tweaks; tags."""
        old_weight = self.object_weights.get(object_id, 0.0)
        self.total_weight = self.total_weight - old_weight + weight
        self.object_weights[object_id] = weight

    def sample_object(self, u: float) -> Optional[Tuple[int, float]]:
        """Sample object based on importance weights."""
        if self.total_weight <= 0.0:
            return None

        target = u * self.total_weight
        cumulative = 0.0

        for object_id, weight in self.object_weights.items():
            cumulative += weight
            if cumulative >= target:
                mis_weight = weight / self.total_weight
                return object_id, mis_weight

        return None

    def calculate_variance_reduction(self, baseline_mse: float, optimized_mse: float) -> float:
        """A25: ≥15% MSE ↓ on tagged objects w/o bias."""
        if baseline_mse <= 0.0:
            return 0.0
        return (baseline_mse - optimized_mse) / baseline_mse

    def meets_performance_target(self, variance_reduction: float) -> bool:
        """Check if meets A25 requirement: ≥15% MSE reduction."""
        return variance_reduction >= 0.15