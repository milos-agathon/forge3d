#!/usr/bin/env python3
"""
IBL Gallery Example (P7)

Render IBL galleries using the native terrain pipeline (IBL.from_hdr) where
available, with a CPU-safe fallback via the high-level mesh tracer. No
placeholders are used; images are rendered in all modes.

Modes:
- rotation: rotate the HDR environment across tiles (native terrain when available)
- roughness: sweep a material roughness parameter (mesh tracer HDR tinting)
- metallic: compare metallic vs dielectric across roughness values (mesh tracer)

Usage:
    python examples/ibl_gallery.py --hdr assets/snow_field_4k.hdr --mode rotation
    python examples/ibl_gallery.py --hdr assets/snow_field_4k.hdr --mode roughness
    python examples/ibl_gallery.py --hdr assets/snow_field_4k.hdr --mode metallic
    python examples/ibl_gallery.py --mode rotation  # auto-picks repo HDR if present
"""

from _import_shim import ensure_repo_import
ensure_repo_import()

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    import forge3d as f3d
except Exception as exc:  # pragma: no cover
    raise SystemExit("forge3d import failed. Build with `maturin develop --release`.") from exc

try:
    from forge3d.render import render_raytrace_mesh
except Exception:
    render_raytrace_mesh = None  # type: ignore


def _repo_hdr() -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    cand = repo / "assets" / "snow_field_4k.hdr"
    return cand if cand.exists() else None


def _have_native() -> bool:
    return bool(f3d.has_gpu()) and all(
        hasattr(f3d, name)
        for name in ("Session", "TerrainRenderer", "MaterialSet", "IBL", "TerrainRenderParams")
    )


def _synthetic_dem(w: int = 256, h: int = 256) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    peak = 400.0 * np.exp(-(xx ** 2 + yy ** 2) / 0.18)
    ridges = 120.0 * np.sin(9.0 * np.arctan2(yy, xx)) * np.exp(-(xx ** 2 + yy ** 2) / 0.5)
    np.random.seed(7)
    noise = 35.0 * np.random.randn(h, w).astype(np.float32)
    heightmap = np.clip(peak + ridges + noise, 0.0, 1000.0).astype(np.float32)
    return heightmap


def _label_tile(img: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img, mode="RGBA")
        draw = ImageDraw.Draw(im)
        w, h = im.size
        band_h = max(18, h // 16)
        draw.rectangle([0, 0, w, band_h], fill=(0, 0, 0, 110))
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((8, 2), text, fill=(255, 255, 255, 255), font=font)
        return np.array(im, dtype=np.uint8)
    except Exception:
        out = np.array(img, copy=True)
        out[:6, :] = [0, 0, 0, 255]
        return out


def _save_image(img: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(img, mode="RGBA").save(str(path))
    except Exception:
        try:
            f3d.numpy_to_png(str(path), img)
        except Exception:
            np.save(str(path).replace(".png", ".npy"), img)
            print("  Warning: Saved as .npy (no PNG writer available)")


def render_rotation_sweep_native_or_fallback(
    hdr_path: Path,
    output_path: Path,
    tile_size: int = 384,
    rotation_steps: int = 8,
    outdir: Path | None = None,
) -> None:
    print(f"[ibl_gallery] Rotation sweep • HDR={hdr_path}")
    cols = min(rotation_steps, 4)
    rows = (rotation_steps + cols - 1) // cols
    mosaic = np.zeros((rows * tile_size, cols * tile_size, 4), dtype=np.uint8)

    native = _have_native()
    if native:
        sess = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(sess)
        materials = f3d.MaterialSet.terrain_default()
        params_cfg = f3d.TerrainRenderParamsConfig(
            size_px=(tile_size, tile_size),
            render_scale=1.0,
            msaa_samples=1,
            z_scale=1.0,
            cam_target=[0.0, 0.0, 0.0],
            cam_radius=6.0,
            cam_phi_deg=135.0,
            cam_theta_deg=42.0,
            cam_gamma_deg=0.0,
            fov_y_deg=55.0,
            clip=(0.1, 2000.0),
            light=f3d.LightSettings("Directional", 135.0, 40.0, 3.0, [1.0, 0.97, 0.92]),
            ibl=f3d.IblSettings(True, 1.0, 0.0),
            shadows=f3d.ShadowSettings(True, "PCF", 1024, 1, 500.0, 1.0, 0.8, 0.5, 0.002, 0.5, 1e-4, 0.5, 40.0, 1.0),
            triplanar=f3d.TriplanarSettings(6.0, 4.0, 1.0),
            pom=f3d.PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
            lod=f3d.LodSettings(0, 0.0, -0.5),
            sampling=f3d.SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
            clamp=f3d.ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            overlays=[],
            exposure=1.0,
            gamma=2.2,
            albedo_mode="mix",
            colormap_strength=0.5,
        )
        params = f3d.TerrainRenderParams(params_cfg)
        hm = _synthetic_dem(256, 256)

    for i in range(rotation_steps):
        rot = (i * 360.0) / rotation_steps
        r, c = i // cols, i % cols
        x0, y0 = c * tile_size, r * tile_size

        try:
            if native:
                ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0, rotate_deg=float(rot))
                ibl.set_base_resolution(256)
                frame = renderer.render_terrain_pbr_pom(material_set=materials, env_maps=ibl, params=params, heightmap=hm, target=None)
                rgba = frame.to_numpy()
            else:
                if render_raytrace_mesh is None:
                    raise RuntimeError("mesh tracer unavailable")
                repo = Path(__file__).resolve().parents[1]
                mesh = repo / "assets" / "bunny.obj"
                rgba, _ = render_raytrace_mesh(
                    mesh=str(mesh),
                    size=(tile_size, tile_size),
                    frames=4,
                    seed=7 + i,
                    lighting_type="flat",
                    shadows=False,
                    hdri_path=str(hdr_path),
                    hdri_rotation_deg=float(rot),
                    hdri_intensity=0.8,
                    background_color=(1.0, 1.0, 1.0),
                    verbose=False,
                )
        except Exception as exc:
            raise SystemExit(f"Rotation sweep failed at {rot:.1f}°: {exc}")

        mosaic[y0 : y0 + tile_size, x0 : x0 + tile_size] = _label_tile(rgba, f"{rot:.0f}°")
        if outdir is not None:
            _save_image(rgba, outdir / f"ibl_rot_{i:02d}.png")

    _save_image(mosaic, output_path)


def render_roughness_sweep_mesh(hdr_path: Path, output_path: Path, tile_size: int = 384, roughness_steps: int = 10, outdir: Path | None = None) -> None:
    print(f"[ibl_gallery] Roughness sweep • HDR={hdr_path}")
    if render_raytrace_mesh is None:
        raise SystemExit("mesh tracer unavailable; cannot render roughness sweep")
    cols = min(roughness_steps, 5)
    rows = (roughness_steps + cols - 1) // cols
    mosaic = np.zeros((rows * tile_size, cols * tile_size, 4), dtype=np.uint8)
    repo = Path(__file__).resolve().parents[1]
    mesh = repo / "assets" / "bunny.obj"
    for i in range(roughness_steps):
        rough = i / max(1, roughness_steps - 1)
        r, c = i // cols, i % cols
        x0, y0 = c * tile_size, r * tile_size
        # Simulate roughness by varying specular-style lighting a bit and blending HDR tint
        rgba, _ = render_raytrace_mesh(
            mesh=str(mesh),
            size=(tile_size, tile_size),
            frames=4,
            seed=11 + i,
            lighting_type="blinn-phong",
            lighting_intensity=max(0.6, 1.2 - 0.5 * rough),
            shadows=True,
            shadow_intensity=0.4 + 0.4 * (1.0 - rough),
            hdri_path=str(hdr_path),
            hdri_rotation_deg=0.0,
            hdri_intensity=0.3 + 0.6 * (1.0 - rough),
            background_color=(1.0, 1.0, 1.0),
            verbose=False,
        )
        mosaic[y0 : y0 + tile_size, x0 : x0 + tile_size] = _label_tile(rgba, f"R={rough:.2f}")
        if outdir is not None:
            _save_image(rgba, outdir / f"ibl_rough_{i:02d}.png")
    _save_image(mosaic, output_path)


def render_metallic_comparison_mesh(hdr_path: Path, output_path: Path, tile_size: int = 384, outdir: Path | None = None) -> None:
    print(f"[ibl_gallery] Metallic vs dielectric • HDR={hdr_path}")
    if render_raytrace_mesh is None:
        raise SystemExit("mesh tracer unavailable; cannot render metallic comparison")
    rows, cols = 2, 5
    rough_vals = [0.0, 0.2, 0.4, 0.6, 0.8]
    mosaic = np.zeros((rows * tile_size, cols * tile_size, 4), dtype=np.uint8)
    repo = Path(__file__).resolve().parents[1]
    mesh = repo / "assets" / "bunny.obj"
    for r_idx, metal in enumerate([0.0, 1.0]):
        for c_idx, rough in enumerate(rough_vals):
            label = ("Dielectric" if metal < 0.5 else "Metallic") + f" R={rough:.1f}"
            x0, y0 = c_idx * tile_size, r_idx * tile_size
            # Simulate metallic vs dielectric via lighting/spec parameters; not exact BRDF but illustrative
            rgba, _ = render_raytrace_mesh(
                mesh=str(mesh),
                size=(tile_size, tile_size),
                frames=4,
                seed=101 + r_idx * 10 + c_idx,
                lighting_type="blinn-phong",
                lighting_intensity=1.1 if metal < 0.5 else 1.0,
                shadows=True,
                shadow_intensity=0.45 if metal < 0.5 else 0.55,
                hdri_path=str(hdr_path),
                hdri_rotation_deg=0.0,
                hdri_intensity=0.6 if metal < 0.5 else 0.8,
                background_color=(1.0, 1.0, 1.0),
                verbose=False,
            )
            mosaic[y0 : y0 + tile_size, x0 : x0 + tile_size] = _label_tile(rgba, label)
            if outdir is not None:
                _save_image(rgba, outdir / f"ibl_metal_{r_idx}_{c_idx}.png")
    _save_image(mosaic, output_path)


def add_label_border(tile: np.ndarray, label: str):  # Back-compat no-op; labels handled by _label_tile
    pass


def save_image(img: np.ndarray, path: str):  # Back-compat helper
    _save_image(img, Path(path))


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrate IBL with environment rotation and roughness sweeps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  rotation  - Show effect of HDR environment rotation (8 angles)
  roughness - Show effect of material roughness (0.0 to 1.0)
  metallic  - Compare metallic vs. dielectric at various roughness

Examples:
  # Rotation sweep
  python examples/ibl_gallery.py --hdr env.hdr --mode rotation

  # Roughness sweep for dielectrics
  python examples/ibl_gallery.py --hdr env.hdr --mode roughness

  # Metallic vs. dielectric comparison
  python examples/ibl_gallery.py --hdr env.hdr --mode metallic
        """
    )

    parser.add_argument('--hdr', '-i', required=False,
                       help='Path to HDR environment map (.hdr or .exr). Defaults to assets/snow_field_4k.hdr if present.')
    parser.add_argument('--outdir', default='examples/out',
                       help='Directory to save per-tile images (optional)')
    parser.add_argument('--output', '-o',
                       help='Output image path (auto-generated if not specified)')
    parser.add_argument('--mode', '-m',
                       choices=['rotation', 'roughness', 'metallic', 'all'],
                       default='all',
                       help='Gallery mode (default: all)')
    parser.add_argument('--tile-size', '-s', type=int, default=400,
                       help='Size of each tile in pixels (default: 400)')
    parser.add_argument('--rotation-steps', type=int, default=8,
                       help='Number of rotation angles (default: 8)')
    parser.add_argument('--roughness-steps', type=int, default=10,
                       help='Number of roughness steps (default: 10)')

    args = parser.parse_args()

    # Resolve HDR path (prefer user; else repo asset)
    if args.hdr is not None:
        hdr_path = Path(args.hdr)
    else:
        cand = _repo_hdr()
        if cand is None:
            raise SystemExit("No HDR map provided and assets/snow_field_4k.hdr not found.")
        hdr_path = cand
    if not hdr_path.exists():
        raise SystemExit(f"HDR file not found: {hdr_path}")

    # Render requested modes
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode in ['rotation', 'all']:
        output = Path(args.output or 'ibl_rotation.png')
        render_rotation_sweep_native_or_fallback(
            hdr_path=hdr_path,
            output_path=output,
            tile_size=int(args.tile_size),
            rotation_steps=int(args.rotation_steps),
            outdir=outdir,
        )

    if args.mode in ['roughness', 'all']:
        output = Path(args.output or 'ibl_roughness.png')
        render_roughness_sweep_mesh(
            hdr_path=hdr_path,
            output_path=output,
            tile_size=int(args.tile_size),
            roughness_steps=int(args.roughness_steps),
            outdir=outdir,
        )

    if args.mode in ['metallic', 'all']:
        output = Path(args.output or 'ibl_metallic.png')
        render_metallic_comparison_mesh(
            hdr_path=hdr_path,
            output_path=output,
            tile_size=int(args.tile_size),
            outdir=outdir,
        )


if __name__ == '__main__':
    main()
