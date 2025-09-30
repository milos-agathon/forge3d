#!/usr/bin/env python3
"""
Example: Compositing Datashader overlays with vector OIT results.

- Generates a simple Datashader aggregation and converts it to an RGBA overlay
  using forge3d.adapters.shade_to_overlay (premultiplied preferred).
- Renders points/lines with Weighted OIT via vector_render_oit_py.
- Composites the vector result over the Datashader overlay.

Run:
    python examples/vector_datashader_oit_composite.py
Optional args:
    --width 800 --height 600
"""
import argparse
import numpy as np
import forge3d as f3d


def main(width: int = 800, height: int = 600) -> int:
    try:
        from forge3d.adapters import is_datashader_available, shade_to_overlay
        import datashader as ds
        import datashader.transfer_functions as tf  # noqa: F401  # for completeness
    except Exception as e:
        print("Datashader not available; install with: pip install datashader")
        print(f"Reason: {e}")
        return 0

    if not is_datashader_available():
        print("Datashader adapter not available")
        return 0

    # Create a small synthetic dataset (lon/lat-ish in [-1,1])
    rng = np.random.default_rng(42)
    n = 100_000
    x = rng.uniform(-1.0, 1.0, size=n)
    y = rng.uniform(-1.0, 1.0, size=n)
    val = rng.normal(0.0, 1.0, size=n)

    # Define extent to match our data domain
    extent = (-1.0, -1.0, 1.0, 1.0)
    canvas = ds.Canvas(plot_width=width, plot_height=height, x_range=(extent[0], extent[2]), y_range=(extent[1], extent[3]))

    import pandas as pd
    df = pd.DataFrame({"x": x, "y": y, "value": val})
    agg = canvas.points(df, "x", "y", ds.mean("value"))

    # Convert to overlay (premultiply recommended for compositing)
    overlay = shade_to_overlay(agg, extent, cmap="viridis", how="linear", premultiply=True)
    rgba_overlay = overlay["rgba"]  # (H,W,4) uint8

    # Prepare a tiny vector scene: two points and a polyline
    f3d.set_point_shape_mode(5)
    f3d.set_point_lod_threshold(24.0)

    points = [(-0.5, -0.5), (0.4, 0.2)]
    colors = [(1.0, 0.2, 0.2, 0.9), (0.2, 0.8, 1.0, 0.7)]
    sizes  = [24.0, 32.0]
    polylines = [[(-0.8, -0.8), (0.8, 0.5), (0.4, 0.8)]]
    poly_colors = [(0.1, 0.9, 0.3, 0.6)]
    poly_widths = [8.0]

    if not f3d.is_weighted_oit_available():
        print("Weighted OIT unavailable; cannot render vector OIT")
        return 0

    rgba_vec = f3d.vector_render_oit_py(
        int(width), int(height),
        points_xy=points,
        point_rgba=colors,
        point_size=sizes,
        polylines=polylines,
        polyline_rgba=poly_colors,
        stroke_width=poly_widths,
    )

    # Premultiply vector RGBA to match premultiplied bottom overlay
    from forge3d.adapters import premultiply_rgba
    rgba_vec_pm = premultiply_rgba(rgba_vec)

    # Composite vectors on top of datashader overlay (premultiplied compositing)
    composite = f3d.composite_rgba_over(rgba_overlay, rgba_vec_pm, premultiplied=True)

    out = "datashader_vector_composite.png"
    f3d.numpy_to_png(out, composite)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()
    raise SystemExit(main(args.width, args.height))
