# examples/m4_raster_xarray_demo.py
# Workstream M4: Rasterio/Xarray Adapter Demo
# - Convert a rasterio dataset or xarray DataArray to RGBA and save a deterministic PNG

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forge3d.adapters import (
    rasterio_to_rgba,
    dataarray_to_rgba,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 Rasterio/Xarray → RGBA + 3D preview demo")
    parser.add_argument("--geotiff", type=Path, default=None, help="Path to a GeoTIFF to rasterize")
    parser.add_argument("--out", type=Path, default=Path("reports/m4_raster_xarray.png"))
    parser.add_argument("--bands", type=int, nargs=3, default=None, help="RGB band indices for rasterio (1-based)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic xarray DataArray if rasterio not provided")
    # 3D options
    parser.add_argument("--make-3d", action="store_true", default=True, help="Displace a plane from luminance and save 3D preview")
    parser.add_argument("--height-scale", type=float, default=25.0, help="Displacement scale for heightmap")
    parser.add_argument("--plane-res", type=int, nargs=2, default=(256, 256), metavar=("W","H"), help="Plane resolution for displacement")
    parser.add_argument("--preview-size", type=int, nargs=2, default=(800, 600), metavar=("W","H"), help="3D preview image size")
    parser.add_argument("--preview-dpi", type=int, default=150, help="3D preview DPI")
    parser.add_argument("--save-obj", action="store_true", help="Also save the displaced mesh as OBJ (requires native)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rgba = None

    if args.geotiff is not None and args.geotiff.exists():
        try:
            import rasterio
            with rasterio.open(str(args.geotiff)) as ds:
                bands = tuple(args.bands) if args.bands is not None else None
                rgba = rasterio_to_rgba(ds, bands=bands)
                print(f"Loaded GeoTIFF {args.geotiff} → RGBA {rgba.shape}")
        except Exception as exc:
            print(f"Rasterio path failed ({exc}); falling back to xarray synthetic if possible")

    if rgba is None and (args.synthetic or args.geotiff is None):
        try:
            from forge3d.ingest.xarray_adapter import create_synthetic_dataarray
            da = create_synthetic_dataarray((256, 384), crs="EPSG:4326", seed=42)
            rgba = dataarray_to_rgba(da)
            print(f"Created synthetic DataArray → RGBA {rgba.shape}")
        except Exception as exc:
            print(f"Xarray synthetic fallback unavailable: {exc}")

    if rgba is None:
        print("No input produced RGBA. Provide --geotiff (requires rasterio) or use --synthetic with xarray installed.")
        return 0

    save_png_deterministic(str(args.out), rgba)
    print(f"Wrote {args.out}")

    # Optional 3D displaced preview
    if args.make_3d and rgba is not None:
        try:
            arr = rgba.astype(np.float32)
            if arr.dtype == np.uint8:
                arr = arr / 255.0
            # Compute luminance from RGB
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
            print(f"3D preview failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
