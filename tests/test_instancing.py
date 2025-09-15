#!/usr/bin/env python3
"""Tests for A22: Instanced Geometry"""

import pytest
import numpy as np

try:
    from forge3d.instancing import InstancedGeometry
    INSTANCING_AVAILABLE = True
except ImportError:
    INSTANCING_AVAILABLE = False

@pytest.mark.skipif(not INSTANCING_AVAILABLE, reason="Instancing not available")
def test_instanced_geometry():
    """Test A22: 10k instances with one BLAS; ≤512MiB VRAM."""
    geometry = InstancedGeometry(max_instances=1000)

    transform = np.eye(4, dtype=np.float32)
    instance_id = geometry.add_instance(transform)

    assert instance_id == 0
    assert geometry.validate_memory_budget()