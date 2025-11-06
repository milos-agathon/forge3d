#!/usr/bin/env python3
"""
Milestone 5: Spatial variance and monotonicity tests

- Spatial variance (catch "flat disks"): stddev over the disk mask > 0.02 for G-only and DFG-only
- Monotonicity (existing goals): FWHM increases with roughness (covered in M3),
  and peak intensity decreases with roughness (added here for full BRDF)
"""
import numpy as np
import pytest

try:
    import forge3d as f3d
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False

try:
    import forge3d._forge3d as f3d_native
    NATIVE_AVAILABLE = hasattr(f3d_native, 'render_brdf_tile') if FORGE3D_AVAILABLE else False
except (ImportError, AttributeError):
    NATIVE_AVAILABLE = False

skip_if_no_forge3d = pytest.mark.skipif(
    not FORGE3D_AVAILABLE,
    reason="forge3d not available (build with: maturin develop --release)",
)
skip_if_no_native = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason="Native module with GPU support not available (expected on CPU-only CI)",
)


def _luminance(tile: np.ndarray) -> np.ndarray:
    return (0.299 * tile[:, :, 0] + 0.587 * tile[:, :, 1] + 0.114 * tile[:, :, 2]).astype(np.float32) / 255.0


def _disk_mask(lum: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Approximate sphere disk by non-zero luminance pixels."""
    return lum > eps


@skip_if_no_forge3d
@skip_if_no_native
class TestM5SpatialVariance:
    def test_g_only_stddev_over_disk(self):
        width = height = 256
        tile = f3d.render_brdf_tile(
            "ggx", 0.5, width, height,
            ndf_only=False, g_only=True, dfg_only=False,
            roughness_visualize=False, exposure=1.0, light_intensity=0.8,
        )
        lum = _luminance(tile)
        mask = _disk_mask(lum)
        values = lum[mask]
        std = float(np.std(values)) if values.size else 0.0
        assert std > 0.02, f"G-only stddev too low (flat disk?): {std:.4f}"

    def test_dfg_only_stddev_over_disk(self):
        width = height = 256
        tile = f3d.render_brdf_tile(
            "ggx", 0.5, width, height,
            ndf_only=False, g_only=False, dfg_only=True,
            roughness_visualize=False, exposure=1.0, light_intensity=0.8,
        )
        lum = _luminance(tile)
        mask = _disk_mask(lum)
        values = lum[mask]
        std = float(np.std(values)) if values.size else 0.0
        assert std > 0.02, f"DFG-only stddev too low (flat disk?): {std:.4f}"


@skip_if_no_forge3d
@skip_if_no_native
class TestM5PeakMonotonicity:
    def test_full_brdf_peak_decreases_with_roughness(self):
        """Peaks should decrease as roughness increases (highlight broadens)."""
        width = height = 256
        roughness_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        peaks = []
        for r in roughness_values:
            tile = f3d.render_brdf_tile(
                "ggx", r, width, height,
                ndf_only=False, g_only=False, dfg_only=False,
                roughness_visualize=False, exposure=1.0, light_intensity=0.8,
            )
            lum = _luminance(tile)
            peaks.append(float(lum.max()))
        # Debug print for investigation
        print(f"\nM5 peak sequence (should be decreasing): {['{:.3f}'.format(p) for p in peaks]}")
        # Non-increasing sequence with small tolerance (allow 1% absolute wiggle)
        for i in range(len(peaks) - 1):
            assert peaks[i] >= peaks[i + 1] - 0.01, (
                f"Peak not non-increasing at r={roughness_values[i]}->{roughness_values[i+1]}: "
                f"{peaks[i]:.3f} -> {peaks[i+1]:.3f}"
            )
        # Ensure significant drop from lowest to highest roughness
        assert peaks[-1] <= peaks[0] * 0.8, (
            f"Peak drop insufficient: first={peaks[0]:.3f}, last={peaks[-1]:.3f}")
