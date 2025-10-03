# examples/m2_mpl_data_demo.py
# Workstream M2: Matplotlib Adapter (Data)
# - Extract lines/polygons/text outlines from Matplotlib artists
# - Convert to meshes via forge3d.geometry (if native extension available)
# - Save OBJ optionally; always save a rasterized PNG for visualization

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from forge3d.adapters import (
    extract_lines_from_axes,
    extract_polygons_from_axes,
    text_to_polygons,
    extrude_polygons_to_meshes,
    thicken_lines_to_meshes,
    line_width_world_from_pixels,
    is_matplotlib_available_data,
)
from forge3d.helpers.offscreen import save_png_deterministic
from _m_3d_preview import render_meshes_preview


def build_demo_axes():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Line demo
    x = np.linspace(0, 10, 200)
    y = np.sin(x) * 0.5 + 0.5
    ax.plot(x, y, 'b-', lw=2, label='curve')

    # Polygon demo (rectangle)
    ax.add_patch(Rectangle((2, 0.2), 3, 0.5, facecolor='orange', edgecolor='k', alpha=0.5, label='rect'))

    # Text demo
    ax.text(7.0, 0.8, "Forge3D", fontsize=14, fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("M2 Matplotlib Data → Meshes Demo")
    return fig, ax


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 Matplotlib Data → Meshes demo")
    parser.add_argument("--outdir", type=Path, default=Path("reports/m2_mpl_data_demo"))
    parser.add_argument("--save-obj", action="store_true", help="Attempt to save generated meshes as OBJ (requires native)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--preview-size", type=int, nargs=2, default=(800, 600), metavar=("W","H"), help="3D preview image size")
    parser.add_argument("--preview-dpi", type=int, default=150, help="3D preview DPI for rasterization")
    args = parser.parse_args()

    if not is_matplotlib_available_data():
        print("Matplotlib not available. Install with: pip install matplotlib")
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = build_demo_axes()

    # Always save a rasterized PNG for visual reference
    try:
        from forge3d.adapters import mpl_rasterize_axes
        rgba = mpl_rasterize_axes(ax, dpi=args.dpi, bbox_inches='tight', pad_inches=0.0)
        save_png_deterministic(str(args.outdir / "axes_raster.png"), rgba)
        print(f"Wrote {args.outdir / 'axes_raster.png'}")
    except Exception as exc:
        print(f"Could not rasterize axes: {exc}")

    # Extract artists
    lines = extract_lines_from_axes(ax)
    polys = extract_polygons_from_axes(ax)

    print(f"Extracted: {len(lines)} line(s), {len(polys)} polygon(s)")

    # Convert to meshes if native geometry is available
    meshes = []
    try:
        # Thicken lines using approximate world width (assume camera FOV 45°, z=5, H=600)
        px_width = 6.0
        world_w = line_width_world_from_pixels(px_width, z=5.0, fov_y_deg=45.0, height_px=600)
        meshes.extend(thicken_lines_to_meshes(lines, width_world=float(world_w)))
        # Extrude polygons
        meshes.extend(extrude_polygons_to_meshes(polys, height=0.2))
        # Text outlines → polygons → meshes (optional)
        txt_polys = text_to_polygons("Forge3D", size=12.0)
        meshes.extend(extrude_polygons_to_meshes(txt_polys, height=0.1))
        print(f"Generated {len(meshes)} mesh(es)")
    except Exception as exc:
        print(f"Mesh generation skipped or partial (native geometry may be unavailable): {exc}")

    # 3D preview if we have meshes
    if meshes:
        prev = args.outdir / "preview_3d.png"
        out3d = render_meshes_preview(
            meshes,
            str(prev),
            width=int(args.preview_size[0]),
            height=int(args.preview_size[1]),
            dpi=int(args.preview_dpi),
        )
        if out3d:
            print(f"Wrote {out3d}")

    if args.save_obj and meshes:
        try:
            from forge3d.io import save_obj
            for i, m in enumerate(meshes[:5]):  # limit to a few
                path = args.outdir / f"mesh_{i:02d}.obj"
                save_obj(m, str(path))
                print(f"Wrote {path}")
        except Exception as exc:
            print(f"Could not save OBJ (requires native extension): {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
