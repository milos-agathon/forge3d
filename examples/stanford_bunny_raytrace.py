#!/usr/bin/env python3
"""Ray trace the Stanford Bunny via :func:`forge3d.render_raytrace_mesh`."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import forge3d as f3d


def ensure_bunny_obj() -> Path:
    """Return the path to the bunny OBJ within this repo (assets/bunny.obj)."""
    repo_root = Path(__file__).resolve().parents[1]
    local_path = repo_root / "assets" / "bunny.obj"
    if not local_path.exists():
        raise FileNotFoundError(
            f"assets/bunny.obj not found at {local_path}. Please place the OBJ there."
        )
    return local_path


def _parse_vec3(values: Optional[Sequence[float]], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if values is None:
        return default
    return tuple(float(v) for v in values)


def _parse_optional_vec3(values: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
    if values is None:
        return None
    return tuple(float(v) for v in values)


def _parse_color(values: Optional[Sequence[str]]) -> Optional[Tuple[float, float, float]]:
    if values is None:
        return None
    tokens = list(values)
    if len(tokens) == 1:
        token = tokens[0].strip()
        if token.startswith("#"):
            token = token[1:]
        if len(token) != 6:
            raise ValueError("Hex color must be RRGGBB")
        r = int(token[0:2], 16) / 255.0
        g = int(token[2:4], 16) / 255.0
        b = int(token[4:6], 16) / 255.0
        return (r, g, b)
    if len(tokens) == 3:
        floats = tuple(float(v) for v in tokens)
        if max(floats) > 1.0 or min(floats) < 0.0:
            return tuple(max(0.0, min(1.0, v / 255.0)) for v in floats)
        return floats
    raise ValueError("--background-color expects either a single hex value or three components")


def main() -> int:
    p = argparse.ArgumentParser(description="Ray trace Stanford Bunny with Forge3D")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--outfile", type=str, default="out/stanford_bunny_rt.png")
    # Camera controls
    p.add_argument("--fovy", type=float, default=34.0, help="Vertical field of view in degrees")
    p.add_argument("--theta", type=float, default=None, help="Yaw angle in degrees around +Y (0=+Z, 90=+X)")
    p.add_argument("--phi", type=float, default=None, help="Pitch angle in degrees (-90..90), 0=horizon, +=up")
    p.add_argument("--radius", type=float, default=None, help="Camera distance from target (overrides --zoom)")
    p.add_argument("--zoom", type=float, default=None, help="Scale auto-framed distance; 1.0=default, <1.0 closer, >1.0 farther")
    p.add_argument("--target", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Look-at target position")
    p.add_argument("--eye", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Camera eye position (overrides theta/phi/radius)")
    p.add_argument("--up", type=float, nargs=3, metavar=("X","Y","Z"), default=None, help="Up vector")
    # Rendering
    p.add_argument("--frames", type=int, default=8, help="Accumulation frames (>=1)")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU fallback even if native GPU tracer is available")
    # Preview overlay (CPU point cloud) to visualize mesh when GPU is unavailable
    p.add_argument("--preview", action="store_true", help="Draw CPU point-cloud overlay of the bunny onto the output")
    p.add_argument("--preview-size", type=int, default=1, help="Point size (pixels radius)")
    p.add_argument("--preview-color", type=int, nargs=3, metavar=("R","G","B"), default=(255, 160, 40), help="Point color")
    p.add_argument(
        "--background-color",
        type=str,
        nargs="+",
        metavar="COLOR",
        default=None,
        help="Background color as hex (#RRGGBB) or three numeric components",
    )
    # Lighting & palette controls
    p.add_argument("--palette", type=str, nargs="*", default=None, help="Palette specification (hex stops or colormap key)")
    p.add_argument("--invert-palette", action="store_true", help="Invert palette direction")
    p.add_argument("--lighting-type", type=str, default="lambertian", help="Lighting model (lambertian, none, flat)")
    p.add_argument("--lighting-intensity", type=float, default=1.0, help="Lighting intensity multiplier")
    p.add_argument("--lighting-azimuth", type=float, default=315.0, help="Lighting azimuth in degrees")
    p.add_argument("--lighting-elevation", type=float, default=45.0, help="Lighting elevation in degrees")
    shadow_group = p.add_mutually_exclusive_group()
    shadow_group.add_argument("--shadows", dest="shadows", action="store_true", help="Enable analytic shadows")
    shadow_group.add_argument("--no-shadows", dest="shadows", action="store_false", help="Disable analytic shadows")
    p.set_defaults(shadows=True)
    p.add_argument("--shadow-intensity", type=float, default=1.0, help="Shadow strength [0..1]")
    # HDRI environment controls
    p.add_argument("--hdri", type=str, default=None, help="Path to HDR environment map")
    p.add_argument("--hdri-rotation", type=float, default=0.0, help="HDRI rotation in degrees")
    p.add_argument("--hdri-intensity", type=float, default=1.0, help="HDRI blend intensity")
    # Denoiser & firefly controls
    p.add_argument("--denoiser", type=str, choices=["off", "svgf"], default="off", help="Apply denoiser to beauty")
    p.add_argument("--svgf-iters", type=int, default=5, help="Iterations for SVGF denoiser")
    p.add_argument("--luminance-clamp", dest="lum_clamp", type=float, default=None, help="Optional luminance clamp to suppress fireflies")
    # Ray tracing controls
    p.add_argument("--render-mode", default="raster", choices=["raster", "raytrace"], 
                        help="Rendering mode: raster (fast GPU rasterization) or raytrace (high quality path tracing)")
    p.add_argument("--rt-spp", type=int, default=64, help="Ray tracing samples per pixel (raytrace mode only)")
    p.add_argument("--rt-seed", type=int, default=0, help="Random seed for ray tracing (raytrace mode only)")
    p.add_argument("--rt-batch-spp", type=int, default=8, help="Ray tracing batch size for progressive rendering (raytrace mode only)")
    p.add_argument("--rt-sampling-mode", default="sobol", choices=["rng", "sobol", "cmj"], 
                        help="Sampling mode: rng (fast), sobol (best quality, recommended), cmj (alternative)")
    p.add_argument("--max-rt-triangles", type=int, default=2000000, help="Max triangles for ray tracing mesh (will decimate if needed)")
    # AOV export
    p.add_argument("--save-aovs", action="store_true", help="Render and save AOVs (EXR where available, PNG for visibility)")
    p.add_argument("--aovs", type=str, default="albedo,normal,depth,visibility", help="Comma-separated AOV list to render")
    p.add_argument("--aov-dir", type=str, default=None, help="Directory for AOV outputs (defaults to output file directory)")
    p.add_argument("--basename", type=str, default=None, help="Basename for outputs (defaults to output filename stem)")
    args = p.parse_args()

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target = _parse_vec3(args.target, (0.0, 0.05, 0.0))
    up = _parse_vec3(args.up, (0.0, 1.0, 0.0))
    eye = _parse_optional_vec3(args.eye)
    preview_color = tuple(int(c) for c in args.preview_color)
    aov_list = [s.strip() for s in str(args.aovs).split(',') if s.strip()]

    palette: Optional[Union[str, Sequence[str]]] = None
    if args.palette:
        palette = args.palette if len(args.palette) > 1 else args.palette[0]
    hdri_path = Path(args.hdri) if args.hdri is not None else None
    background_color = _parse_color(args.background_color)

    _, meta = f3d.render_raytrace_mesh(
        ensure_bunny_obj(),
        size=(int(args.width), int(args.height)),
        frames=max(1, int(args.frames)),
        seed=int(args.seed),
        fov_y=float(args.fovy),
        target=target,
        up=up,
        orbit_theta=args.theta,
        orbit_phi=args.phi,
        radius=args.radius,
        zoom=args.zoom,
        eye=eye,
        prefer_gpu=not bool(args.no_gpu),
        denoiser=str(args.denoiser),
        svgf_iters=int(args.svgf_iters),
        luminance_clamp=(float(args.lum_clamp) if args.lum_clamp is not None else None),
        preview=bool(args.preview),
        preview_size=int(args.preview_size),
        preview_color=preview_color,
        background_color=background_color,
        palette=palette,
        invert_palette=bool(args.invert_palette),
        lighting_type=str(args.lighting_type),
        lighting_intensity=float(args.lighting_intensity),
        lighting_azimuth=float(args.lighting_azimuth),
        lighting_elevation=float(args.lighting_elevation),
        shadows=bool(args.shadows),
        shadow_intensity=float(args.shadow_intensity),
        hdri_path=hdri_path,
        hdri_rotation_deg=float(args.hdri_rotation),
        hdri_intensity=float(args.hdri_intensity),
        save_aovs=bool(args.save_aovs),
        aovs=aov_list,
        aov_dir=args.aov_dir,
        basename=args.basename,
        outfile=out_path,
        verbose=False,
    )

    gpu_used = bool(meta.get("gpu_used", False))
    probe_status = meta.get("probe_status", "n/a")

    if gpu_used:
        print("[stanford_bunny_raytrace] used_native_gpu_mesh=True")
    elif args.preview:
        print("[stanford_bunny_raytrace] GPU render unavailable; preview points overlay applied.")

    print(
        f"[stanford_bunny_raytrace] prefer_gpu={not bool(args.no_gpu)} "
        f"gpu_used={gpu_used} probe={probe_status} tris={meta.get('triangles', 'n/a')}"
    )

    outfile_record = meta.get("outfile")
    if outfile_record is not None:
        print(f"Saved: {outfile_record}")

    aov_outputs = meta.get("aov_outputs")
    if aov_outputs:
        print("Saved AOVs:")
        for name, path in aov_outputs.items():
            print(f"  {name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
