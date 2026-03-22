"""TV13 — Terrain Population LOD Pipeline tests.

Covers:
  - TV13.1: QEM mesh simplification (Rust) and Python wrapper
  - TV13.2: LOD chain generation and auto_lod_levels
  - TV13.3: HLOD clustering, rendering, stats, and memory tracking
"""
from __future__ import annotations

import numpy as np
import pytest

import forge3d as f3d
from forge3d._native import NATIVE_AVAILABLE

if not NATIVE_AVAILABLE:
    pytest.skip(
        "TV13 tests require the compiled _forge3d extension",
        allow_module_level=True,
    )

ts = f3d.terrain_scatter
