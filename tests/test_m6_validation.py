import numpy as np
import pytest

from examples import m6_generate as m6


def _gpu_available() -> bool:
    try:
        import forge3d as f3d
    except Exception:
        return False
    try:
        tile = f3d.render_brdf_tile
    except AttributeError:
        return False
    return callable(tile)


@pytest.mark.skipif(not _gpu_available(), reason="render_brdf_tile not available")
def test_m6_cpu_gpu_rms_within_threshold() -> None:
    result = m6.run_validation(
        tile_size=256,
        roughness=0.5,
        samples_per_axis=32,
        eval_scale=2,
    )
    assert result.sample_count == result.sample_rows * result.sample_cols
    assert np.all(result.rms <= 1e-3 + 1e-6), f"RMS exceeded: {result.rms}"
    assert result.percentile_999 <= 5e-3 + 1e-6, f"99.9th percentile exceeded: {result.percentile_999}"


@pytest.mark.skipif(not _gpu_available(), reason="render_brdf_tile not available")
def test_render_brdf_tile_overrides_and_debug_terms() -> None:
    import forge3d as f3d

    base = f3d.render_brdf_tile("ggx", 0.5, 64, 64, False)
    full = f3d.render_brdf_tile_full("ggx", 0.5, 64, 64)
    assert np.array_equal(base, full), "Full wrapper must match legacy path"

    light_override = (0.2, 0.8, 0.556776)
    alt = f3d.render_brdf_tile_full("ggx", 0.5, 64, 64, light_dir=light_override)
    assert not np.array_equal(base, alt), "Changing light_dir should change the image"

    for kind in (1, 2, 3):
        dbg = f3d.render_brdf_tile_debug("ggx", 0.5, 64, 64, light_dir=light_override, debug_kind=kind)
        assert dbg.shape == base.shape
        assert dbg.dtype == base.dtype
        assert np.isfinite(dbg).all()
        assert np.any(dbg[..., :3] > 0), f"Debug kind {kind} should produce visible signal"
