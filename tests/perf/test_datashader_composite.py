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


def _datashader_available():
    try:
        from forge3d.adapters import is_datashader_available
        return bool(is_datashader_available())
    except Exception:
        return False


@pytest.mark.skipif(not _has_native(), reason="native module not available in this environment")
@pytest.mark.skipif(not _oit_available(), reason="weighted OIT not available on this platform/build")
@pytest.mark.skipif(not _datashader_available(), reason="datashader not available")
def test_datashader_and_vector_oit_composite():
    import time
    import pandas as pd
    import datashader as ds
    import forge3d as f3d
    from forge3d.adapters import shade_to_overlay, premultiply_rgba

    # Build synthetic dataset
    n = 300_000
    rng = np.random.default_rng(123)
    df = pd.DataFrame({
        "x": rng.uniform(-1.0, 1.0, size=n),
        "y": rng.uniform(-1.0, 1.0, size=n),
        "value": rng.normal(0.0, 1.0, size=n),
    })

    width, height = 512, 288
    extent = (-1.0, -1.0, 1.0, 1.0)
    canvas = ds.Canvas(plot_width=width, plot_height=height,
                       x_range=(extent[0], extent[2]), y_range=(extent[1], extent[3]))
    agg = canvas.points(df, "x", "y", ds.mean("value"))

    # Datashader overlay (premultiplied)
    overlay = shade_to_overlay(agg, extent, cmap="magma", how="linear", premultiply=True)
    rgba_ds = overlay["rgba"]

    # Small vector scene
    f3d.set_point_shape_mode(5)
    f3d.set_point_lod_threshold(24.0)

    points = [(-0.5, -0.5), (0.4, 0.2)]
    colors = [(1.0, 0.2, 0.2, 0.9), (0.2, 0.8, 1.0, 0.7)]
    sizes  = [20.0, 28.0]
    polylines = [[(-0.8, -0.8), (0.8, 0.5), (0.4, 0.8)]]
    poly_colors = [(0.1, 0.9, 0.3, 0.6)]
    poly_widths = [6.0]

    t0 = time.time()
    rgba_vec = f3d.vector_render_oit_py(
        width, height,
        points_xy=points,
        point_rgba=colors,
        point_size=sizes,
        polylines=polylines,
        polyline_rgba=poly_colors,
        stroke_width=poly_widths,
    )
    # Premultiply vector result and composite
    rgba_vec_pm = premultiply_rgba(rgba_vec)
    out = f3d.composite_rgba_over(rgba_ds, rgba_vec_pm, premultiplied=True)
    dt = time.time() - t0

    # Basic checks
    assert rgba_ds.shape == (height, width, 4) and rgba_ds.dtype == np.uint8
    assert rgba_vec.shape == (height, width, 4) and rgba_vec.dtype == np.uint8
    assert out.shape == (height, width, 4) and out.dtype == np.uint8

    # Non-zero coverage
    assert int(out[..., 3].sum()) > 0

    # Loose perf bound suitable for CI GPU runners (aggregation + render + composite)
    assert dt < 10.0
