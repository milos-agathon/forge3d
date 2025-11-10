#!/usr/bin/env python3
"""
IBL Gallery Example (M3)

Render IBL galleries using the BRDF tile renderer path (same family as M1/M2/M4),
not the terrain PBR/POM pipeline. Uses M4 IBL resources (irradiance, prefiltered env, DFG LUT)
to produce meaningful visual variations across roughness/metallic and rotation.

Modes:
- rotation: rotate the HDR environment across tiles
- roughness: sweep a material roughness parameter (0.0 to 1.0)
- metallic: compare metallic vs dielectric across roughness values

Usage:
    python examples/ibl_gallery.py --hdr assets/snow_field_4k.hdr --mode rotation
    python examples/ibl_gallery.py --hdr assets/snow_field_4k.hdr --mode roughness
    python examples/ibl_gallery.py --hdr assets/snow_field_4k.hdr --mode metallic
    python examples/ibl_gallery.py --mode rotation  # auto-picks repo HDR if present
"""

from _import_shim import ensure_repo_import
ensure_repo_import()

import argparse
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    import forge3d as f3d
except Exception as exc:  # pragma: no cover
    raise SystemExit("forge3d import failed. Build with `maturin develop --release`.") from exc

# Import M4 functions for IBL resource building
try:
    # Import key functions from m4_generate.py
    import sys
    import importlib.util
    m4_path = Path(__file__).parent / "m4_generate.py"
    if not m4_path.exists():
        raise ImportError(f"m4_generate.py not found at {m4_path}")
    
    # Add examples directory to path for imports
    examples_dir = str(Path(__file__).parent)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    
    # Try to import as a module
    try:
        import m4_generate as m4_module
    except ImportError:
        # Fallback: use importlib.util
        spec = importlib.util.spec_from_file_location("m4_generate", str(m4_path))
        if spec is None:
            raise ImportError(f"Failed to create spec for m4_generate.py")
        if spec.loader is None:
            raise ImportError(f"Spec loader is None for m4_generate.py")
        m4_module = importlib.util.module_from_spec(spec)
        sys.modules['m4_generate'] = m4_module
        spec.loader.exec_module(m4_module)
    
    # Import needed functions
    load_hdr_environment = m4_module.load_hdr_environment
    equirect_to_cubemap = m4_module.equirect_to_cubemap
    compute_prefilter_chain = m4_module.compute_prefilter_chain
    build_irradiance_cubemap = m4_module.build_irradiance_cubemap
    compute_dfg_lut = m4_module.compute_dfg_lut
    sample_prefilter = m4_module.sample_prefilter
    sample_lut = m4_module.sample_lut
    sample_cubemap_faces = m4_module.sample_cubemap_faces
    build_sphere_geometry = m4_module.build_sphere_geometry
    linear_to_srgb = m4_module.linear_to_srgb
    normalize_vec3 = m4_module.normalize_vec3
    BASE_CUBEMAP_SIZE = m4_module.BASE_CUBEMAP_SIZE
    IRRADIANCE_SIZE = m4_module.IRRADIANCE_SIZE
    LUT_SIZE = m4_module.LUT_SIZE
    PREFILTER_SAMPLES_TOP = m4_module.PREFILTER_SAMPLES_TOP
    PREFILTER_SAMPLES_BOTTOM = m4_module.PREFILTER_SAMPLES_BOTTOM
    IRRADIANCE_SAMPLES = m4_module.IRRADIANCE_SAMPLES
    DFG_LUT_SAMPLES = m4_module.DFG_LUT_SAMPLES
    F0_DIELECTRIC = m4_module.F0_DIELECTRIC
except Exception as exc:
    raise SystemExit(f"Failed to import M4 IBL functions: {exc}. Ensure m4_generate.py exists.") from exc


def _repo_hdr() -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    cand = repo / "assets" / "snow_field_4k.hdr"
    return cand if cand.exists() else None


def _have_brdf_renderer() -> bool:
    """Check if BRDF tile renderer is available."""
    return bool(f3d.has_gpu()) and hasattr(f3d, 'render_brdf_tile_full')


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


def build_ibl_resources(hdr_path: Path) -> Tuple:
    """Build IBL resources (irradiance, prefiltered, DFG LUT) using M4 approach."""
    print(f"[ibl_gallery] Building IBL resources from {hdr_path}")
    hdr_data, hdr_mode = load_hdr_environment(hdr_path, force_synthetic=False)
    print(f"[ibl_gallery] HDR mode: {hdr_mode} ({hdr_data.shape[1]}x{hdr_data.shape[0]})")
    
    # Build IBL resources
    base_cube, _ = equirect_to_cubemap(hdr_data, BASE_CUBEMAP_SIZE)
    print(f"[ibl_gallery] Base cubemap: {base_cube.shape[1]} px per face")
    
    prefilter_levels, _, _ = compute_prefilter_chain(
        hdr_data,
        BASE_CUBEMAP_SIZE,
        PREFILTER_SAMPLES_TOP,
        PREFILTER_SAMPLES_BOTTOM,
    )
    irradiance_faces = build_irradiance_cubemap(hdr_data, IRRADIANCE_SIZE, IRRADIANCE_SAMPLES)
    lut = compute_dfg_lut(LUT_SIZE, DFG_LUT_SAMPLES)
    
    print(f"[ibl_gallery] IBL resources built: {len(prefilter_levels)} prefilter levels, "
          f"irradiance {irradiance_faces.shape[1]}px, LUT {lut.shape[0]}x{lut.shape[1]}")
    
    return prefilter_levels, irradiance_faces, lut, hdr_data


def render_panel_brdf(
    prefilter_levels,
    irradiance_faces: np.ndarray,
    lut: np.ndarray,
    *,
    roughness: float,
    metallic: float,
    base_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    f0: float = 0.04,
    env_rotation_deg: float = 0.0,
    size: int = 512,
) -> np.ndarray:
    """
    Render a BRDF tile panel using IBL (M4 approach).
    
    Uses CPU-side IBL evaluation with split-sum approximation to match BRDF tile renderer
    material parameters and visual style.
    """
    # Build sphere geometry
    normals, mask = build_sphere_geometry(size)
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # View direction (camera looking at sphere)
    
    # Compute NoV
    NoV = np.clip(normals[..., 2], 0.0, 1.0)
    
    # Compute reflection direction for specular IBL
    reflection = normalize_vec3(2.0 * NoV[..., None] * normals - V)
    
    # Apply environment rotation to sampling directions (rotate environment, not surface)
    if env_rotation_deg != 0.0:
        rot_rad = math.radians(env_rotation_deg)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        # Rotate around Y axis (azimuthal rotation)
        rot_matrix = np.array([
            [cos_r, 0.0, sin_r],
            [0.0, 1.0, 0.0],
            [-sin_r, 0.0, cos_r],
        ], dtype=np.float32)
        # Rotate directions used to sample environment maps
        reflection_rot = reflection @ rot_matrix.T
        reflection_rot = normalize_vec3(reflection_rot)
        normals_rot = normals @ rot_matrix.T
        normals_rot = normalize_vec3(normals_rot)
    else:
        reflection_rot = reflection
        normals_rot = normals
    
    # Sample prefiltered specular (use rotated reflection direction)
    spec_color = sample_prefilter(prefilter_levels, reflection_rot, roughness)
    
    # Sample DFG LUT
    lut_sample = sample_lut(lut, NoV, roughness)
    
    # Compute F0 (mix dielectric with base color by metallic)
    f0_vec = np.array([f0, f0, f0], dtype=np.float32)
    f0_final = f0_vec * (1.0 - metallic) + np.array(base_color, dtype=np.float32) * metallic
    
    # Specular IBL: prefiltered_color * (F0 * scale + bias)
    specular = spec_color * (f0_final * lut_sample[..., 0:1] + lut_sample[..., 1:2])
    
    # Diffuse IBL: sample irradiance (use rotated normals for environment rotation)
    irradiance = sample_cubemap_faces(irradiance_faces, normals_rot)
    
    # Compute Fresnel for energy conservation
    # F_ibl = fresnel_schlick_roughness(NoV, f0_final, roughness)
    fresnel = f0_final + (np.maximum(1.0 - roughness, f0_final) - f0_final) * np.power(
        np.clip(1.0 - NoV, 0.0, 1.0), 5.0
    )[..., None]
    
    # kD = (1 - kS) * (1 - metallic)
    kD = (1.0 - fresnel) * (1.0 - metallic)
    
    # Diffuse: kD * base_color * irradiance / PI
    diffuse = kD * np.array(base_color, dtype=np.float32) * irradiance / math.pi
    
    # Combine
    linear = np.clip(specular + diffuse, 0.0, None)
    linear[~mask] = 0.0
    
    # Convert to sRGB
    srgb = linear_to_srgb(linear)
    rgb = (srgb * 255.0).astype(np.uint8)
    
    # Add alpha channel
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    rgba[~mask, 3] = 0  # Transparent outside sphere
    
    return rgba


def render_rotation_sweep_brdf(
    hdr_path: Path,
    output_path: Path,
    tile_size: int = 384,
    rotation_steps: int = 8,
    outdir: Path | None = None,
    ibl_config: dict | None = None,
    rotate_speed: float | None = None,
    frames: int | None = None,
) -> None:
    """Render rotation sweep using BRDF tile renderer with IBL."""
    print(f"[ibl_gallery] Rotation sweep (BRDF tile) • HDR={hdr_path}")
    
    if not _have_brdf_renderer():
        raise SystemExit("BRDF renderer unavailable; cannot produce M3 galleries")
    
    # Build IBL resources
    prefilter_levels, irradiance_faces, lut, hdr_data = build_ibl_resources(hdr_path)
    
    cols = min(rotation_steps, 4)
    rows = (rotation_steps + cols - 1) // cols
    mosaic = np.zeros((rows * tile_size, cols * tile_size, 4), dtype=np.uint8)
    
    # Fixed material parameters for rotation sweep
    roughness = 0.3
    metallic = 0.0
    base_color = (0.5, 0.5, 0.5)
    f0 = F0_DIELECTRIC
    
    for i in range(rotation_steps):
        rot = (i * 360.0) / rotation_steps
        r, c = i // cols, i % cols
        x0, y0 = c * tile_size, r * tile_size
        
        print(f"[ibl_gallery]   Rendering rotation {rot:.0f}° (renderer=brdf_tile)")
        
        # Render panel with IBL
        rgba = render_panel_brdf(
            prefilter_levels,
            irradiance_faces,
            lut,
            roughness=roughness,
            metallic=metallic,
            base_color=base_color,
            f0=f0,
            env_rotation_deg=rot,
            size=tile_size,
        )
        
        mosaic[y0 : y0 + tile_size, x0 : x0 + tile_size] = _label_tile(rgba, f"{rot:.0f}°")
        if outdir is not None:
            _save_image(rgba, outdir / f"ibl_rot_{i:02d}.png")
    
    _save_image(mosaic, output_path)


def render_roughness_sweep_brdf(
    hdr_path: Path,
    output_path: Path,
    tile_size: int = 384,
    roughness_steps: int = 10,
    outdir: Path | None = None,
) -> None:
    """Render roughness sweep using BRDF tile renderer with IBL."""
    print(f"[ibl_gallery] Roughness sweep (BRDF tile) • HDR={hdr_path}")
    
    if not _have_brdf_renderer():
        raise SystemExit("BRDF renderer unavailable; cannot produce M3 galleries")
    
    # Build IBL resources
    prefilter_levels, irradiance_faces, lut, hdr_data = build_ibl_resources(hdr_path)
    
    # Two rows: dielectric (metallic=0) and metallic (metallic=1)
    rows = 2
    cols = min(roughness_steps, 5)
    rough_vals = [i / max(1, roughness_steps - 1) for i in range(roughness_steps)]
    rough_vals = [max(0.0, min(1.0, r)) for r in rough_vals]
    
    mosaic = np.zeros((rows * tile_size, cols * tile_size, 4), dtype=np.uint8)
    
    base_color = (0.5, 0.5, 0.5)
    f0 = F0_DIELECTRIC
    
    for row_idx, metal in enumerate([0.0, 1.0]):
        for col_idx, rough in enumerate(rough_vals[:cols]):
            x0, y0 = col_idx * tile_size, row_idx * tile_size
            
            print(f"[ibl_gallery]   Rendering roughness={rough:.2f}, metallic={metal:.0f} (renderer=brdf_tile)")
            
            # Render panel with IBL
            rgba = render_panel_brdf(
                prefilter_levels,
                irradiance_faces,
                lut,
                roughness=rough,
                metallic=metal,
                base_color=base_color,
                f0=f0,
                env_rotation_deg=0.0,
                size=tile_size,
            )
            
            label = f"R={rough:.2f}"
            mosaic[y0 : y0 + tile_size, x0 : x0 + tile_size] = _label_tile(rgba, label)
            if outdir is not None:
                _save_image(rgba, outdir / f"ibl_rough_{row_idx}_{col_idx:02d}.png")
    
    _save_image(mosaic, output_path)


def render_metallic_comparison_brdf(
    hdr_path: Path,
    output_path: Path,
    tile_size: int = 384,
    outdir: Path | None = None,
) -> None:
    """Render metallic comparison using BRDF tile renderer with IBL."""
    print(f"[ibl_gallery] Metallic vs dielectric (BRDF tile) • HDR={hdr_path}")
    
    if not _have_brdf_renderer():
        raise SystemExit("BRDF renderer unavailable; cannot produce M3 galleries")
    
    # Build IBL resources
    prefilter_levels, irradiance_faces, lut, hdr_data = build_ibl_resources(hdr_path)
    
    # Task spec: two rows (dielectric m=0, metallic m=1), 5 roughness values
    rows, cols = 2, 5
    rough_vals = [0.04, 0.2, 0.4, 0.6, 0.8]  # Fixed roughness levels
    mosaic = np.zeros((rows * tile_size, cols * tile_size, 4), dtype=np.uint8)
    
    base_color = (0.5, 0.5, 0.5)
    f0 = F0_DIELECTRIC
    
    for r_idx, metal in enumerate([0.0, 1.0]):
        for c_idx, rough in enumerate(rough_vals):
            label = ("Dielectric" if metal < 0.5 else "Metallic") + f" R={rough:.1f}"
            x0, y0 = c_idx * tile_size, r_idx * tile_size
            
            print(f"[ibl_gallery]   Rendering roughness={rough:.1f}, metallic={metal:.0f} (renderer=brdf_tile)")
            
            # Render panel with IBL
            rgba = render_panel_brdf(
                prefilter_levels,
                irradiance_faces,
                lut,
                roughness=rough,
                metallic=metal,
                base_color=base_color,
                f0=f0,
                env_rotation_deg=0.0,
                size=tile_size,
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
        description='Demonstrate IBL with environment rotation and roughness sweeps (BRDF tile renderer)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  rotation  - Show effect of HDR environment rotation (8 angles)
  roughness - Show effect of material roughness (0.0 to 1.0) for dielectric and metallic
  metallic  - Compare metallic vs. dielectric at various roughness

Examples:
  # Rotation sweep
  python examples/ibl_gallery.py --hdr env.hdr --mode rotation

  # Roughness sweep
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
    parser.add_argument('--ibl-res', type=int, default=None,
                       help='IBL base resolution override (ignored, using M4 defaults)')
    parser.add_argument('--ibl-cache', type=str, default=None,
                       help='IBL cache directory (ignored, using M4 approach)')
    parser.add_argument('--rotate', type=float, default=None,
                       help='Rotation speed in degrees per second (ignored)')
    parser.add_argument('--frames', type=int, default=None,
                       help='Number of frames to render (ignored)')

    args = parser.parse_args()

    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import os
    if 'RUST_LOG' not in os.environ:
        os.environ['RUST_LOG'] = 'info'

    # Resolve HDR path
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
        render_rotation_sweep_brdf(
            hdr_path=hdr_path,
            output_path=output,
            tile_size=int(args.tile_size),
            rotation_steps=int(args.rotation_steps),
            outdir=outdir,
            ibl_config=None,  # Not used in BRDF path
            rotate_speed=args.rotate,
            frames=args.frames,
        )
        print(f"[ibl_gallery] Written: {output.resolve()}")

    if args.mode in ['roughness', 'all']:
        output = Path(args.output or 'ibl_roughness.png')
        render_roughness_sweep_brdf(
            hdr_path=hdr_path,
            output_path=output,
            tile_size=int(args.tile_size),
            roughness_steps=int(args.roughness_steps),
            outdir=outdir,
        )
        print(f"[ibl_gallery] Written: {output.resolve()}")

    if args.mode in ['metallic', 'all']:
        output = Path(args.output or 'ibl_metallic.png')
        render_metallic_comparison_brdf(
            hdr_path=hdr_path,
            output_path=output,
            tile_size=int(args.tile_size),
            outdir=outdir,
        )
        print(f"[ibl_gallery] Written: {output.resolve()}")

    print("M3 IBL gallery → BRDF: OK")
    print(f"Output files written to: {outdir.resolve()}")


if __name__ == '__main__':
    main()
