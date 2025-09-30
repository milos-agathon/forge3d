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
def test_transparency_heavy_oit_perf_and_output():
    import time
    import forge3d as f3d

    # Many semi-transparent points and lines
    n_pts = 10_000
    n_lines = 400

    # Points in a circle
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False, dtype=np.float32)
    r = 0.8
    points = list(zip((r*np.cos(theta)).tolist(), (r*np.sin(theta)).tolist()))
    colors = [(0.9, 0.2, 0.3, 0.25)] * n_pts
    sizes = [6.0] * n_pts

    # Lines as random polylines
    rng = np.random.default_rng(123)
    polylines = []
    poly_colors = []
    poly_widths = []
    for _ in range(n_lines):
        m = 16
        xs = rng.uniform(-1.0, 1.0, size=m)
        ys = rng.uniform(-1.0, 1.0, size=m)
        polylines.append(list(zip(xs.tolist(), ys.tolist())))
        poly_colors.append((0.2, 0.8, 1.0, 0.25))
        poly_widths.append(2.0)

    f3d.set_point_shape_mode(5)
    f3d.set_point_lod_threshold(12.0)

    t0 = time.time()
    rgba = f3d.vector_render_oit_py(
        512, 288,
        points_xy=points,
        point_rgba=colors,
        point_size=sizes,
        polylines=polylines,
        polyline_rgba=poly_colors,
        stroke_width=poly_widths,
    )
    dt = time.time() - t0

    assert isinstance(rgba, np.ndarray) and rgba.dtype == np.uint8 and rgba.shape == (288, 512, 4)
    assert int(rgba[..., 3].sum()) > 0

    # Relaxed but bounded performance
    assert dt < 8.0
