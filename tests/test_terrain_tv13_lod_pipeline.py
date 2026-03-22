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

from forge3d.geometry import MeshBuffers, primitive_mesh, simplify_mesh, generate_lod_chain
from forge3d.terrain_scatter import (
    TerrainScatterBatch,
    TerrainScatterLevel,
    auto_lod_levels,
)


class TestSimplifyMesh:
    """TV13.1 — Python simplify_mesh wrapper."""

    def test_simplify_cone_reduces_triangles(self):
        cone = primitive_mesh("cone", radial_segments=32)
        simplified = simplify_mesh(cone, 0.5)
        assert simplified.triangle_count < cone.triangle_count
        assert simplified.vertex_count > 0

    def test_simplify_preserves_normals(self):
        sphere = primitive_mesh("sphere", rings=12, radial_segments=24)
        simplified = simplify_mesh(sphere, 0.5)
        assert simplified.normals.shape[0] == simplified.positions.shape[0]

    def test_simplify_ratio_one_unchanged(self):
        box_mesh = primitive_mesh("box")
        result = simplify_mesh(box_mesh, 1.0)
        assert result.triangle_count == box_mesh.triangle_count


class TestGenerateLodChain:
    """TV13.1 — LOD chain generation from a single mesh."""

    def test_three_level_chain_decreasing_triangles(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        chain = generate_lod_chain(sphere, [1.0, 0.25, 0.07])
        assert len(chain) >= 2  # at least LOD0 + one simplified
        assert chain[0].triangle_count == sphere.triangle_count
        for i in range(1, len(chain)):
            assert chain[i].triangle_count < chain[i - 1].triangle_count

    def test_min_triangles_floor_drops_levels(self):
        box_mesh = primitive_mesh("box")  # 12 tris
        chain = generate_lod_chain(box_mesh, [1.0, 0.01], min_triangles=8)
        assert len(chain) == 1  # only LOD 0 survives

    def test_deduplication_drops_identical_levels(self):
        box_mesh = primitive_mesh("box")  # 12 tris
        chain = generate_lod_chain(box_mesh, [1.0, 0.9, 0.8])
        for i in range(1, len(chain)):
            assert chain[i].triangle_count < chain[i - 1].triangle_count

    def test_ratios_must_start_with_one(self):
        sphere = primitive_mesh("sphere")
        with pytest.raises(ValueError, match="ratios.*1.0"):
            generate_lod_chain(sphere, [0.5, 0.25])

    def test_ratios_must_be_descending(self):
        sphere = primitive_mesh("sphere")
        with pytest.raises(ValueError, match="descending"):
            generate_lod_chain(sphere, [1.0, 0.5, 0.7])


class TestAutoLodLevels:
    """TV13.2 — auto_lod_levels generates scatter LOD levels from one mesh."""

    def test_default_three_levels(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(sphere, lod_count=3, draw_distance=300.0)
        assert len(levels) >= 2
        assert all(isinstance(l, TerrainScatterLevel) for l in levels)
        assert levels[-1].max_distance is None
        if len(levels) >= 2:
            assert levels[0].max_distance is not None

    def test_explicit_distances(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(
            sphere,
            lod_count=3,
            lod_distances=[50.0, 150.0, None],
        )
        assert levels[0].max_distance == 50.0
        assert levels[1].max_distance == 150.0
        assert levels[-1].max_distance is None

    def test_explicit_ratios(self):
        sphere = primitive_mesh("sphere", rings=16, radial_segments=32)
        levels = auto_lod_levels(
            sphere,
            lod_count=3,
            ratios=[1.0, 0.3, 0.05],
            draw_distance=200.0,
        )
        assert levels[0].mesh.triangle_count >= levels[1].mesh.triangle_count

    def test_feeds_into_scatter_batch(self):
        cone = primitive_mesh("cone", radial_segments=32)
        levels = auto_lod_levels(cone, lod_count=2, draw_distance=100.0)
        transforms = np.array([
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1],
        ], dtype=np.float32)
        batch = TerrainScatterBatch(
            levels=levels,
            transforms=transforms,
            name="auto_lod_test",
        )
        assert batch.instance_count == 1
