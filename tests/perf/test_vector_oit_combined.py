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
def test_vector_render_oit_and_pick_py_shapes_and_perf():
    import time
    import forge3d as f3d

    # Configure impostors and LOD (no-op in CPU-only builds)
    f3d.set_point_shape_mode(5)
    f3d.set_point_lod_threshold(16.0)

    points = [(-0.5, -0.5), (0.4, 0.2)]
    colors = [(1.0, 0.2, 0.2, 0.9), (0.2, 0.8, 1.0, 0.7)]
    sizes  = [24.0, 32.0]
    polylines = [[(-0.8, -0.8), (0.8, 0.5), (0.4, 0.8)]]
    poly_colors = [(0.1, 0.9, 0.3, 0.6)]
    poly_widths = [8.0]

    t0 = time.time()
    rgba, picks = f3d.vector_render_oit_and_pick_py(
        256, 144,
        points_xy=points,
        point_rgba=colors,
        point_size=sizes,
        polylines=polylines,
        polyline_rgba=poly_colors,
        stroke_width=poly_widths,
        base_pick_id=1,
    )
    dt = time.time() - t0

    assert isinstance(rgba, np.ndarray) and rgba.dtype == np.uint8 and rgba.shape == (144, 256, 4)
    assert isinstance(picks, np.ndarray) and picks.dtype == np.uint32 and picks.shape == (144, 256)

    # Non-zero alpha coverage expected
    assert int(rgba[..., 3].sum()) > 0
    # Some non-zero IDs expected in pick map
    assert int((picks > 0).sum()) > 0

    # Loose perf bound suitable for CI GPU runners
    assert dt < 5.0


@pytest.mark.skipif(not _has_native(), reason="native module not available in this environment")
@pytest.mark.skipif(not _oit_available(), reason="weighted OIT not available on this platform/build")
def test_vector_scene_render_oit_and_pick():
    import forge3d as f3d
    vs = f3d.VectorScene()
    vs.add_point(-0.5, -0.5, (1.0, 0.2, 0.2, 0.9), 24.0)
    vs.add_point(0.4, 0.2, (0.2, 0.8, 1.0, 0.7), 32.0)
    vs.add_polyline([(-0.8, -0.8), (0.8, 0.5), (0.4, 0.8)], (0.1, 0.9, 0.3, 0.6), 8.0)

    rgba, picks = vs.render_oit_and_pick(128, 72, base_pick_id=1)

    assert isinstance(rgba, np.ndarray) and rgba.dtype == np.uint8 and rgba.shape == (72, 128, 4)
    assert isinstance(picks, np.ndarray) and picks.dtype == np.uint32 and picks.shape == (72, 128)
    assert int(rgba[..., 3].sum()) > 0
    assert int((picks > 0).sum()) > 0
