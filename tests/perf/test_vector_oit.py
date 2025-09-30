import pytest
import numpy as np


def _has_native():
    try:
        import forge3d._forge3d  # noqa: F401
        return True
    except Exception:
        return False


def _oit_available():
    import forge3d as f3d
    try:
        return bool(f3d.is_weighted_oit_available())
    except Exception:
        return False


@pytest.mark.skipif(not _has_native(), reason="native module not available in this environment")
@pytest.mark.skipif(not _oit_available(), reason="weighted OIT not available on this platform/build")
def test_vector_oit_and_pick_demo_runs_fast_and_valid():
    import time
    import forge3d as f3d

    # Configure impostors and LOD (won't raise on CPU-only fallback)
    f3d.set_point_shape_mode(5)
    f3d.set_point_lod_threshold(16.0)

    t0 = time.time()
    rgba, pick_id = f3d.vector_oit_and_pick_demo(256, 144)
    dt = time.time() - t0

    assert isinstance(rgba, np.ndarray)
    assert rgba.dtype == np.uint8
    assert rgba.shape == (144, 256, 4)
    assert pick_id >= 0

    # Basic sanity on alpha coverage (some non-zero alpha expected)
    assert int(rgba[..., 3].sum()) > 0

    # Perf sanity (not strict; CI variance friendly)
    assert dt < 5.0  # should easily fit on CI runners with GPU
