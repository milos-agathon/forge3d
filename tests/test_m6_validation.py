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
