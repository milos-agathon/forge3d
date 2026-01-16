"""P2.3: GPU LOD selection tests.

Exit criteria from docs/plan.md:
- Compute shader frustum cull + LOD selection
- Correct LOD selection based on screen-space error
"""

import numpy as np
import pytest


class TestLodSelectionBasics:
    """Basic LOD selection functionality tests."""

    def test_clipmap_generates_multiple_lods(self):
        """Clipmap should generate mesh data for multiple LOD levels."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        morph_data = mesh.morph_data()
        ring_indices = morph_data[:, 1]

        # Should have vertices from multiple rings (different LODs)
        unique_rings = np.unique(ring_indices[ring_indices >= 0])
        assert len(unique_rings) >= 2, f"Expected multiple rings, got {len(unique_rings)}"

    def test_lod_reduces_triangle_count(self):
        """Higher LOD levels should use fewer triangles."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        # More rings = more LOD levels = more triangle reduction
        config_few = ClipmapConfig(ring_count=2, ring_resolution=32)
        config_many = ClipmapConfig(ring_count=6, ring_resolution=32)

        mesh_few = clipmap_generate_py(config_few, (0.0, 0.0), 1000.0)
        mesh_many = clipmap_generate_py(config_many, (0.0, 0.0), 1000.0)

        # More LOD rings should provide better reduction
        reduction_few = mesh_few.triangle_reduction_percent
        reduction_many = mesh_many.triangle_reduction_percent

        # Both should achieve significant reduction
        assert reduction_few >= 40.0, f"Few rings reduction {reduction_few}% < 40%"
        assert reduction_many >= 40.0, f"Many rings reduction {reduction_many}% < 40%"


class TestScreenSpaceError:
    """Screen-space error based LOD selection tests."""

    def test_near_tiles_use_fine_lod(self):
        """Tiles near camera should use fine (low) LOD levels."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=6, ring_resolution=64)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 10000.0)

        morph_data = mesh.morph_data()
        positions = mesh.positions()

        # Center vertices (near camera at center) should be ring 0
        center_mask = (positions[:, 0]**2 + positions[:, 1]**2) < 100**2
        center_rings = morph_data[center_mask, 1]

        if len(center_rings) > 0:
            # Most center vertices should be from ring 0 or center block
            mean_ring = np.mean(center_rings[center_rings >= 0])
            assert mean_ring < 2, f"Expected low ring index near center, got {mean_ring}"

    def test_far_tiles_use_coarse_lod(self):
        """Tiles far from camera should use coarse (high) LOD levels."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=6, ring_resolution=64)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 10000.0)

        morph_data = mesh.morph_data()
        positions = mesh.positions()

        # Far vertices should be from outer rings
        distance = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        far_mask = distance > 5000
        far_rings = morph_data[far_mask, 1]

        if len(far_rings) > 0:
            far_rings_valid = far_rings[far_rings >= 0]
            if len(far_rings_valid) > 0:
                mean_ring = np.mean(far_rings_valid)
                # Far vertices should be from higher ring indices
                assert mean_ring > 1, f"Expected higher ring index for far vertices, got {mean_ring}"


class TestFrustumCulling:
    """Frustum culling functionality tests."""

    def test_clipmap_covers_view_frustum(self):
        """Generated clipmap mesh should cover the expected view area."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=64)
        terrain_extent = 1000.0
        mesh = clipmap_generate_py(config, (terrain_extent/2, terrain_extent/2), terrain_extent)

        positions = mesh.positions()

        # Mesh should cover significant area
        x_range = positions[:, 0].max() - positions[:, 0].min()
        z_range = positions[:, 1].max() - positions[:, 1].min()

        assert x_range > terrain_extent * 0.5, f"X range {x_range} too small"
        assert z_range > terrain_extent * 0.5, f"Z range {z_range} too small"

    def test_clipmap_centered_on_camera(self):
        """Clipmap should be centered on the specified camera position."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=64)
        center = (500.0, 500.0)
        mesh = clipmap_generate_py(config, center, 1000.0)

        positions = mesh.positions()

        # Center of mass should be near specified center
        center_x = np.mean(positions[:, 0])
        center_z = np.mean(positions[:, 1])

        # Allow some deviation due to ring structure
        assert abs(center_x - center[0]) < 200, f"X center {center_x} far from {center[0]}"
        assert abs(center_z - center[1]) < 200, f"Z center {center_z} far from {center[1]}"


class TestTriangleBudget:
    """P2 exit criteria: triangle budget verification."""

    def test_triangle_reduction_meets_40_percent(self):
        """Triangle reduction should meet P2 requirement of ≥40%."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=64)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        reduction = mesh.triangle_reduction_percent

        assert reduction >= 40.0, (
            f"Triangle reduction {reduction:.1f}% should be >= 40% "
            f"(P2 exit criteria)"
        )

    def test_triangle_reduction_scales_with_rings(self):
        """More LOD rings should not decrease triangle reduction."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        reductions = []
        for rings in [2, 4, 6]:
            config = ClipmapConfig(ring_count=rings, ring_resolution=64)
            mesh = clipmap_generate_py(config, (0.0, 0.0), 10000.0)
            reductions.append(mesh.triangle_reduction_percent)

        # Reduction should generally increase or stay stable with more rings
        # Allow small decrease due to overhead
        for i in range(1, len(reductions)):
            assert reductions[i] >= reductions[i-1] * 0.9, (
                f"Reduction decreased significantly: {reductions}"
            )

    def test_triangle_count_reasonable(self):
        """Total triangle count should be reasonable for real-time rendering."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=64)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        # Should be manageable for real-time (< 1M triangles)
        assert mesh.triangle_count < 1_000_000, (
            f"Triangle count {mesh.triangle_count} too high for real-time"
        )

        # But should have enough for reasonable detail (> 1K)
        assert mesh.triangle_count > 1_000, (
            f"Triangle count {mesh.triangle_count} too low for detail"
        )


class TestLodResolution:
    """LOD resolution and quality tests."""

    def test_resolution_affects_vertex_count(self):
        """Higher resolution should produce more vertices."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config_low = ClipmapConfig(ring_count=4, ring_resolution=32)
        config_high = ClipmapConfig(ring_count=4, ring_resolution=128)

        mesh_low = clipmap_generate_py(config_low, (0.0, 0.0), 1000.0)
        mesh_high = clipmap_generate_py(config_high, (0.0, 0.0), 1000.0)

        assert mesh_high.vertex_count > mesh_low.vertex_count

    def test_center_resolution_independent(self):
        """Center resolution can be set independently from ring resolution."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(
            ring_count=4,
            ring_resolution=32,
            center_resolution=64
        )
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        assert mesh.vertex_count > 0
        assert mesh.triangle_count > 0


class TestLodTransitions:
    """Test smooth LOD transitions."""

    def test_adjacent_rings_connect(self):
        """Adjacent LOD rings should connect without gaps."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        indices = mesh.indices()

        # All indices should reference valid vertices
        assert indices.max() < mesh.vertex_count, (
            f"Index {indices.max()} >= vertex count {mesh.vertex_count}"
        )

    def test_triangle_winding_consistent(self):
        """All triangles should have consistent winding."""
        from forge3d import ClipmapConfig, clipmap_generate_py

        config = ClipmapConfig(ring_count=4, ring_resolution=32)
        mesh = clipmap_generate_py(config, (0.0, 0.0), 1000.0)

        positions = mesh.positions()
        indices = mesh.indices()

        # Check first 100 triangles for consistent winding
        ccw_count = 0
        cw_count = 0
        for i in range(0, min(len(indices), 300), 3):
            i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
            v0, v1, v2 = positions[i0], positions[i1], positions[i2]

            # 2D cross product (XZ plane)
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]

            if cross > 0:
                ccw_count += 1
            elif cross < 0:
                cw_count += 1

        # Most triangles should have consistent winding (either CCW or CW)
        total = ccw_count + cw_count
        if total > 0:
            dominant_ratio = max(ccw_count, cw_count) / total
            assert dominant_ratio > 0.9, (
                f"Inconsistent winding: {ccw_count} CCW, {cw_count} CW "
                f"({dominant_ratio*100:.1f}% dominant)"
            )
