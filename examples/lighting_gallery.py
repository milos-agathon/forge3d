#!/usr/bin/env python3
"""
Lighting Gallery Example (P7)

Render a grid that sweeps lighting setups using the high-level mesh tracer
(`python/forge3d/render.py::render_raytrace_mesh`). The script degrades
gracefully to the CPU fallback if GPU is unavailable.

Usage:
    python examples/lighting_gallery.py
    python examples/lighting_gallery.py --output gallery.png --outdir examples/out
    python examples/lighting_gallery.py --tile-size 320 --frames 4 --mesh assets/cornell_sphere.obj
"""

# Ensure the in-repo package is importable when running from the repo root
from _import_shim import ensure_repo_import
ensure_repo_import()

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import forge3d
    from forge3d.render import render_raytrace_mesh
except Exception as exc:  # pragma: no cover
    # Print detailed failure to help diagnose environment issues (e.g., missing numpy)
    try:
        import sys, traceback
        print("[lighting_gallery] forge3d import failure:", repr(exc))
        traceback.print_exc()
        print("[lighting_gallery] sys.executable:", sys.executable)
        print("[lighting_gallery] first sys.path entry:", sys.path[0] if sys.path else "<empty>")
    except Exception:
        pass
    raise SystemExit("forge3d import failed. Build with `maturin develop --release`.") from exc


def _default_mesh_path() -> Path:
    # Prefer repo asset Cornell sphere for lighting tests; fallback to bunny
    repo = Path(__file__).resolve().parents[1]
    sphere = repo / "assets" / "cornell_sphere.obj"
    if sphere.exists():
        return sphere
    bunny = repo / "assets" / "bunny.obj"
    if bunny.exists():
        return bunny
    # Fallback: pick the first OBJ under assets/
    for p in (repo / "assets").glob("*.obj"):
        return p
    return sphere  # may not exist; render call will raise


def render_lighting_gallery(
    *,
    mesh_path: Path,
    output_path: Path,
    outdir: Path,
    tile_size: int = 320,
    frames: int = 4,
    save_tiles: bool = False,
    zoom: float = 0.75,
) -> None:
    """Render a simple lighting gallery on a mesh using the mesh path tracer.

    Tiles showcase a mix of direct lighting models and IBL rotations.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Use repository HDR if present
    repo = Path(__file__).resolve().parents[1]
    default_hdr = repo / "assets" / "snow_field_4k.hdr"
    hdr_ok = default_hdr.exists()

    variants: List[Tuple[str, dict]] = [
        ("Lambert x1.0", {"lighting_type": "lambertian", "lighting_intensity": 1.0, "shadows": True, "shadow_intensity": 0.6}),
        ("Lambert x1.3", {"lighting_type": "lambertian", "lighting_intensity": 1.3, "shadows": True, "shadow_intensity": 0.6}),
        ("Phong", {"lighting_type": "phong", "lighting_intensity": 1.0, "shadows": True, "shadow_intensity": 0.5}),
        ("Blinn-Phong", {"lighting_type": "blinn-phong", "lighting_intensity": 1.0, "shadows": True, "shadow_intensity": 0.5}),
    ]

    if hdr_ok:
        variants.extend(
            [
                ("IBL rot 0°", {"hdri_path": str(default_hdr), "hdri_rotation_deg": 0.0, "hdri_intensity": 0.9, "lighting_type": "flat", "shadows": False}),
                ("IBL rot 90°", {"hdri_path": str(default_hdr), "hdri_rotation_deg": 90.0, "hdri_intensity": 0.9, "lighting_type": "flat", "shadows": False}),
                ("IBL rot 180°", {"hdri_path": str(default_hdr), "hdri_rotation_deg": 180.0, "hdri_intensity": 0.9, "lighting_type": "flat", "shadows": False}),
                ("IBL rot 270°", {"hdri_path": str(default_hdr), "hdri_rotation_deg": 270.0, "hdri_intensity": 0.9, "lighting_type": "flat", "shadows": False}),
            ]
        )

    cols = 4
    rows = int(np.ceil(len(variants) / cols))
    W = cols * tile_size
    H = rows * tile_size
    mosaic = np.zeros((H, W, 4), dtype=np.uint8)

    for idx, (label, params) in enumerate(variants):
        r = idx // cols
        c = idx % cols
        x0, y0 = c * tile_size, r * tile_size
        print(f"[lighting_gallery] Rendering {label} → tile ({r},{c})")
        try:
            rgba, meta = render_raytrace_mesh(
                mesh=str(mesh_path),
                size=(tile_size, tile_size),
                frames=int(frames),
                seed=7 + idx,
                denoiser="off",
                lighting_type=params.get("lighting_type", "lambertian"),
                lighting_intensity=params.get("lighting_intensity", 1.0),
                lighting_azimuth=315.0,
                lighting_elevation=45.0,
                shadows=params.get("shadows", True),
                shadow_intensity=params.get("shadow_intensity", 0.6),
                hdri_path=params.get("hdri_path"),
                hdri_rotation_deg=params.get("hdri_rotation_deg", 0.0),
                hdri_intensity=params.get("hdri_intensity", 0.0),
                background_color=(1.0, 1.0, 1.0),
                zoom=float(zoom),  # Zoom < 1.0 moves camera closer
                outfile=None,
                verbose=False,
            )
        except Exception as exc:
            print(f"  Warning: variant '{label}' failed ({exc}); using placeholder")
            rgba = _placeholder_tile(tile_size, label)

        rgba = _label_tile(rgba, label)
        mosaic[y0 : y0 + tile_size, x0 : x0 + tile_size] = rgba

        if save_tiles:
            tile_path = outdir / f"tile_{idx:02d}.png"
            _save_image(rgba, tile_path)

    _save_image(mosaic, output_path)
    print(f"[lighting_gallery] Saved mosaic → {output_path}")


def _placeholder_tile(size: int, label: str) -> np.ndarray:
    # Simple gradient + border placeholder
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, size, dtype=np.float32)[None, :]
    base = np.clip(0.25 + 0.75 * 0.5 * (x + y), 0.0, 1.0)
    rgb = (np.stack([base, base * 0.9, base * 0.7], axis=-1) * 255.0 + 0.5).astype(np.uint8)
    tile = np.concatenate([rgb, 255 * np.ones((size, size, 1), dtype=np.uint8)], axis=-1)
    tile[:4, :] = [255, 255, 255, 255]
    tile[-4:, :] = [255, 255, 255, 255]
    tile[:, :4] = [255, 255, 255, 255]
    tile[:, -4:] = [255, 255, 255, 255]
    return _label_tile(tile, label)


def _label_tile(img: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img, mode="RGBA")
        draw = ImageDraw.Draw(im)
        # Semi-transparent band at top
        w, h = im.size
        band_h = max(18, h // 16)
        draw.rectangle([0, 0, w, band_h], fill=(0, 0, 0, 96))
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((8, 2), text, fill=(255, 255, 255, 255), font=font)
        return np.array(im, dtype=np.uint8)
    except Exception:
        # No PIL: just add a solid top band
        out = np.array(img, copy=True)
        out[:6, :] = [0, 0, 0, 255]
        return out


def _save_image(img: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(img, mode="RGBA").save(str(path))
    except Exception:
        # Fallback via forge3d helper if available
        try:
            forge3d.numpy_to_png(str(path), img)
        except Exception:
            np.save(str(path).replace(".png", ".npy"), img)
            print("  Warning: Saved as .npy (no PNG writer available)")


def main():
    parser = argparse.ArgumentParser(description="Render Lighting/IBL gallery using mesh path tracer")
    parser.add_argument("--output", "-o", default="lighting_gallery.png", help="Mosaic output image path")
    parser.add_argument("--outdir", default="examples/out", help="Directory to save per-tile images (optional)")
    parser.add_argument("--mesh", default=str(_default_mesh_path()), help="OBJ mesh path (default: assets/cornell_sphere.obj)")
    parser.add_argument("--tile-size", "-s", type=int, default=320, help="Tile size in pixels (default: 320)")
    parser.add_argument("--frames", type=int, default=4, help="Samples/frames for tracer accumulation (default: 4)")
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.75,
        help="Camera zoom factor (<1 closer, >1 farther; default: 0.75)",
    )
    parser.add_argument("--save-tiles", action="store_true", help="Also save individual tile PNGs")
    args = parser.parse_args()

    output_path = Path(args.output)
    outdir = Path(args.outdir)
    mesh_path = Path(args.mesh)

    render_lighting_gallery(
        mesh_path=mesh_path,
        output_path=output_path,
        outdir=outdir,
        tile_size=int(args.tile_size),
        frames=int(args.frames),
        save_tiles=bool(args.save_tiles),
        zoom=float(args.zoom),
    )


if __name__ == '__main__':
    main()
