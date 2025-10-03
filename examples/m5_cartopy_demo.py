# examples/m5_cartopy_demo.py
# Workstream M5: Cartopy Integration Demo
# - Rasterize a Cartopy GeoAxes to RGBA and save a deterministic PNG

from __future__ import annotations

import argparse
from pathlib import Path

from forge3d.adapters import (
    is_cartopy_available,
    rasterize_geoaxes,
    get_axes_crs,
    get_extent_in_crs,
)
from forge3d.helpers.offscreen import save_png_deterministic
from _m_3d_preview import (
    render_meshes_preview,
    make_displaced_plane_from_heightmap,
)
try:
    from forge3d.io import save_obj as _save_obj
except Exception:
    _save_obj = None


def build_geoaxes():
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(6, 4), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.stock_img()
        ax.coastlines()
        ax.set_title("M5 Cartopy → RGBA Demo")
        return fig, ax
    except Exception as exc:
        raise ImportError("Cartopy + Matplotlib are required. pip install cartopy matplotlib") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 Cartopy → RGBA + 3D preview demo")
    parser.add_argument("--out", type=Path, default=Path("reports/m5_cartopy.png"))
    parser.add_argument("--dpi", type=int, default=150)
    # 3D options
    parser.add_argument("--make-3d", action="store_true", default=True, help="Displace a plane from luminance and save 3D preview")
    parser.add_argument("--height-scale", type=float, default=20.0, help="Displacement scale for heightmap")
    parser.add_argument("--plane-res", type=int, nargs=2, default=(192, 192), metavar=("W","H"), help="Plane resolution for displacement")
    parser.add_argument("--preview-size", type=int, nargs=2, default=(800, 600), metavar=("W","H"), help="3D preview image size")
    parser.add_argument("--preview-dpi", type=int, default=150, help="3D preview DPI")
    parser.add_argument("--save-obj", action="store_true", help="Also save the displaced mesh as OBJ (requires native)")
    args = parser.parse_args()

    if not is_cartopy_available():
        print("Cartopy not available. Install with: pip install cartopy")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = build_geoaxes()

    # Query metadata
    try:
        crs = get_axes_crs(ax)
        ex = get_extent_in_crs(ax, crs)
        print(f"GeoAxes CRS: {crs}, extent: {ex}")
    except Exception as exc:
        print(f"CRS/extent query failed: {exc}")

    # Rasterize
    try:
        rgba = rasterize_geoaxes(ax, dpi=args.dpi, facecolor='white')
        save_png_deterministic(str(args.out), rgba)
        print(f"Wrote {args.out}")

        # Optional 3D displaced preview
        if args.make_3d:
            arr = rgba.astype(np.float32)
            if arr.dtype == np.uint8:
                arr = arr / 255.0
            hm = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            mesh = make_displaced_plane_from_heightmap(
                hm, scale=float(args.height_scale), resolution=(int(args.plane_res[0]), int(args.plane_res[1]))
            )
            if mesh is not None:
                if args.save_obj and _save_obj is not None:
                    obj_path = args.out.with_suffix("")
                    obj_path = obj_path.with_name(obj_path.name + "_mesh.obj")
                    try:
                        _save_obj(mesh, str(obj_path))
                        print(f"Wrote {obj_path}")
                    except Exception as exc:
                        print(f"OBJ export failed: {exc}")
                prev = args.out.with_suffix("")
                prev = prev.with_name(prev.name + "_3d.png")
                out3d = render_meshes_preview(
                    [mesh],
                    str(prev),
                    width=int(args.preview_size[0]),
                    height=int(args.preview_size[1]),
                    dpi=int(args.preview_dpi),
                )
                if out3d:
                    print(f"Wrote {out3d}")
            else:
                print("3D displacement skipped (geometry unavailable)")
    except Exception as exc:
        print(f"Rasterization failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
