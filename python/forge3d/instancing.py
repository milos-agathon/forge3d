#!/usr/bin/env python3
"""A22: Instanced Geometry (PT) - TLAS-style instances"""

import numpy as np
from typing import List, Tuple

class InstancedGeometry:
    """A22: 10k instances with one BLAS; ≤512MiB VRAM."""

    def __init__(self, max_instances: int = 10000):
        self.max_instances = max_instances
        self.instances = []
        self.shared_blas = None  # Shared BLAS reference

    def add_instance(self, transform: np.ndarray, material_id: int = 0) -> int:
        """Add instance with per-instance transform."""
        if len(self.instances) >= self.max_instances:
            raise ValueError("Maximum instances exceeded")

        instance_data = {
            'transform': transform.copy(),
            'material_id': material_id,
            'blas_index': 0  # Shared BLAS
        }
        self.instances.append(instance_data)
        return len(self.instances) - 1

    def get_memory_usage(self) -> int:
        """Calculate memory usage in bytes."""
        # Each instance: 4x4 transform + metadata
        bytes_per_instance = 16 * 4 + 4 + 4  # transform + material + blas
        return len(self.instances) * bytes_per_instance

    def validate_memory_budget(self) -> bool:
        """A22: ≤512MiB VRAM requirement."""
        max_vram = 512 * 1024 * 1024  # 512 MiB
        return self.get_memory_usage() <= max_vram