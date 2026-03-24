from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np


def _import_forge3d():
    try:
        import forge3d as f3d

        return f3d
    except ModuleNotFoundError:
        from _import_shim import ensure_repo_import

        ensure_repo_import()
        import forge3d as f3d

        return f3d


f3d = _import_forge3d()
f3dio = f3d.io
from forge3d.terrain_params import (
    MaterialLayerSettings,
    PomSettings,
    make_terrain_params_config,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv10_subsurface"
SCENES = (
    dict(
        name="mount_rainier",
        dem=ROOT / "assets" / "tif" / "dem_rainier.tif",
        light_azimuth_deg=138.0,
        light_elevation_deg=8.0,
        sun_intensity=3.0,
        cam_radius_scale=0.18,
        cam_phi_deg=176.0,
        cam_theta_deg=38.0,
        fov_y_deg=26.0,
        snow_start=0.18,
        snow_blend=0.42,
        rock_slope_min=14.0,
        rock_slope_blend=28.0,
    ),
    dict(
        name="gore_range",
        dem=ROOT / "assets" / "tif" / "Gore_Range_Albers_1m.tif",
        light_azimuth_deg=214.0,
        light_elevation_deg=9.0,
        sun_intensity=2.9,
        cam_radius_scale=0.22,
        cam_phi_deg=224.0,
        cam_theta_deg=38.0,
        fov_y_deg=28.0,
        snow_start=0.44,
        snow_blend=0.30,
        rock_slope_min=14.0,
        rock_slope_blend=30.0,
    ),
)


def _write_preview_hdr(path: Path, width: int = 8, height: int = 4) -> None:
    with path.open("wb") as handle:
        handle.write(b"#?RADIANCE\n")
        handle.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        handle.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                r = int((x / max(width - 1, 1)) * 255)
                g = int((y / max(height - 1, 1)) * 255)
                handle.write(bytes([r, g, 190, 128]))


def _downsample_heightmap(heightmap: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return np.ascontiguousarray(heightmap)
    longest = max(int(heightmap.shape[0]), int(heightmap.shape[1]))
    if longest <= max_dim:
        return np.ascontiguousarray(heightmap)
    step = int(math.ceil(longest / max_dim))
    return np.ascontiguousarray(heightmap[::step, ::step])


def _load_dem(path: Path, max_dim: int) -> tuple[object, np.ndarray]:
    try:
        dem = f3dio.load_dem(str(path), fill_nodata_values=True)
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:
            raise SystemExit(
                "DEM loading requires rasterio or Pillow. "
                "Install with `pip install rasterio` or `pip install pillow`."
            ) from exc

        with Image.open(path) as image:
            data = np.asarray(image, dtype=np.float32)
        if data.ndim == 3:
            data = data[..., 0]
        dem = f3dio.load_dem_from_array(data)

    heightmap = np.asarray(dem.data, dtype=np.float32).copy()
    return dem, _downsample_heightmap(heightmap, max_dim)


def _terrain_span(dem: object, heightmap: np.ndarray) -> float:
    resolution = getattr(dem, "resolution", None)
    if resolution is not None:
        try:
            dx = float(resolution[0] or 1.0)
            dy = float(resolution[1] or 1.0)
            if dx > 0.0 and dy > 0.0:
                return max(dx * heightmap.shape[1], dy * heightmap.shape[0])
        except Exception:
            pass
    return float(max(heightmap.shape))


def _relief_scale(domain: tuple[float, float], terrain_span: float) -> float:
    relief = max(float(domain[1]) - float(domain[0]), 1e-6)
    return float(np.clip((terrain_span / relief) * 0.18, 0.12, 10.0))


def _make_overlay(domain: tuple[float, float]):
    lo, hi = map(float, domain)
    span = max(hi - lo, 1e-6)
    cmap = f3d.Colormap1D.from_stops(
        stops=[
            (lo + 0.00 * span, "#17371c"),
            (lo + 0.20 * span, "#40692e"),
            (lo + 0.46 * span, "#7d7c4b"),
            (lo + 0.72 * span, "#b7ab92"),
            (lo + 1.00 * span, "#f5f8fb"),
        ],
        domain=(lo, hi),
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _build_materials(domain: tuple[float, float], z_scale: float, scene: dict) -> tuple[MaterialLayerSettings, MaterialLayerSettings]:
    scaled_min = float(domain[0]) * z_scale
    scaled_max = float(domain[1]) * z_scale
    scaled_span = max(scaled_max - scaled_min, 1e-6)
    snow_altitude_min = scaled_min + scaled_span * float(scene["snow_start"])
    snow_altitude_blend = scaled_span * float(scene["snow_blend"])

    baseline = MaterialLayerSettings(
        snow_enabled=True,
        snow_altitude_min=snow_altitude_min,
        snow_altitude_blend=snow_altitude_blend,
        snow_slope_max=58.0,
        snow_slope_blend=22.0,
        rock_enabled=True,
        rock_slope_min=float(scene["rock_slope_min"]),
        rock_slope_blend=float(scene["rock_slope_blend"]),
        wetness_enabled=True,
        wetness_strength=0.16,
        wetness_slope_influence=0.42,
        snow_subsurface_strength=0.0,
        rock_subsurface_strength=0.0,
        wetness_subsurface_strength=0.0,
    )
    subsurface = replace(
        baseline,
        snow_subsurface_strength=0.58,
        snow_subsurface_color=(0.72, 0.85, 0.98),
        rock_subsurface_strength=0.04,
        rock_subsurface_color=(0.44, 0.37, 0.30),
        wetness_subsurface_strength=0.14,
        wetness_subsurface_color=(0.38, 0.27, 0.18),
    )
    return baseline, subsurface


def _build_params(
    *,
    width: int,
    height: int,
    terrain_span: float,
    domain: tuple[float, float],
    overlay,
    z_scale: float,
    materials: MaterialLayerSettings,
    scene: dict,
):
    clip_far = max(terrain_span * 6.0, 4000.0)
    config = make_terrain_params_config(
        size_px=(width, height),
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=4,
        z_scale=z_scale,
        exposure=1.0,
        domain=domain,
        albedo_mode="mix",
        colormap_strength=0.22,
        ibl_enabled=True,
        light_azimuth_deg=float(scene["light_azimuth_deg"]),
        light_elevation_deg=float(scene["light_elevation_deg"]),
        sun_intensity=float(scene["sun_intensity"]),
        cam_radius=max(terrain_span * float(scene["cam_radius_scale"]), 5.0),
        cam_phi_deg=float(scene["cam_phi_deg"]),
        cam_theta_deg=float(scene["cam_theta_deg"]),
        fov_y_deg=float(scene["fov_y_deg"]),
        camera_mode="mesh",
        clip=(0.1, clip_far),
        overlays=[overlay],
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        materials=materials,
    )
    return f3d.TerrainRenderParams(config)


def _compose_comparison(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h or right.shape[0] != h:
        raise ValueError("comparison images must share the same height")
    divider = np.full((h, 12, 4), 255, dtype=np.uint8)
    divider[..., :3] = 12
    return np.concatenate([left, divider, right], axis=1)


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        raise ValueError("rows must not be empty")
    width = rows[0].shape[1]
    for row in rows:
        if row.shape[1] != width:
            raise ValueError("all rows must share the same width")
    divider = np.full((12, width, 4), 255, dtype=np.uint8)
    divider[..., :3] = 12
    stacked: list[np.ndarray] = []
    for index, row in enumerate(rows):
        if index:
            stacked.append(divider)
        stacked.append(row)
    return np.concatenate(stacked, axis=0)


def render_demo(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    width: int = 1440,
    height: int = 900,
    max_dem_size: int = 1024,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()

    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        hdr_path = Path(tmp.name)
    try:
        _write_preview_hdr(hdr_path)
        ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    finally:
        hdr_path.unlink(missing_ok=True)

    scene_rows: list[np.ndarray] = []
    scene_results: list[dict[str, object]] = []
    for scene in SCENES:
        dem_path = Path(scene["dem"]).resolve()
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM not found: {dem_path}")

        dem, heightmap = _load_dem(dem_path, max_dem_size)
        domain = getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))
        domain = (float(domain[0]), float(domain[1]))
        terrain_span = _terrain_span(dem, heightmap)
        z_scale = _relief_scale(domain, terrain_span)
        overlay = _make_overlay(domain)
        baseline_materials, subsurface_materials = _build_materials(domain, z_scale, scene)

        baseline_params = _build_params(
            width=width,
            height=height,
            terrain_span=float(terrain_span),
            domain=domain,
            overlay=overlay,
            z_scale=float(z_scale),
            materials=baseline_materials,
            scene=scene,
        )
        subsurface_params = _build_params(
            width=width,
            height=height,
            terrain_span=float(terrain_span),
            domain=domain,
            overlay=overlay,
            z_scale=float(z_scale),
            materials=subsurface_materials,
            scene=scene,
        )

        baseline_frame = renderer.render_terrain_pbr_pom(
            material_set=material_set,
            env_maps=ibl,
            params=baseline_params,
            heightmap=heightmap,
        )
        subsurface_frame = renderer.render_terrain_pbr_pom(
            material_set=material_set,
            env_maps=ibl,
            params=subsurface_params,
            heightmap=heightmap,
        )

        baseline_rgba = baseline_frame.to_numpy()
        subsurface_rgba = subsurface_frame.to_numpy()
        comparison = _compose_comparison(baseline_rgba, subsurface_rgba)
        diff = np.abs(baseline_rgba[..., :3].astype(np.float32) - subsurface_rgba[..., :3].astype(np.float32))
        mean_abs_diff = float(np.mean(diff))
        peak_p99_diff = float(np.percentile(diff, 99.0))

        baseline_path = output_dir / f"{scene['name']}_baseline.png"
        subsurface_path = output_dir / f"{scene['name']}_subsurface.png"
        comparison_path = output_dir / f"{scene['name']}_comparison.png"
        baseline_frame.save(str(baseline_path))
        subsurface_frame.save(str(subsurface_path))
        f3d.numpy_to_png(comparison_path, comparison)

        scene_rows.append(comparison)
        scene_results.append(
            {
                "name": scene["name"],
                "dem_path": str(dem_path),
                "heightmap_shape": (int(heightmap.shape[0]), int(heightmap.shape[1])),
                "domain": domain,
                "terrain_span": float(terrain_span),
                "z_scale": float(z_scale),
                "baseline_path": str(baseline_path),
                "subsurface_path": str(subsurface_path),
                "comparison_path": str(comparison_path),
                "mean_abs_diff": mean_abs_diff,
                "peak_p99_diff": peak_p99_diff,
            }
        )

    summary = _stack_rows(scene_rows)
    summary_path = output_dir / "terrain_tv10_summary.png"
    f3d.numpy_to_png(summary_path, summary)

    return {
        "summary_path": str(summary_path),
        "scenes": scene_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Epic TV10 demo: terrain subsurface materials on two real repo DEMs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=1440, help="Render width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Render height in pixels")
    parser.add_argument(
        "--max-dem-size",
        type=int,
        default=1024,
        help="Clamp the longest DEM dimension before upload (0 disables downsampling)",
    )
    args = parser.parse_args()

    result = render_demo(
        output_dir=args.output_dir,
        width=int(args.width),
        height=int(args.height),
        max_dem_size=int(args.max_dem_size),
    )

    print(f"Wrote {result['summary_path']}")
    for scene in result["scenes"]:
        print(f"{scene['name']}: {scene['dem_path']}")
        print(f"  Mean RGB difference: {scene['mean_abs_diff']:.3f}")
        print(f"  99th percentile RGB difference: {scene['peak_p99_diff']:.3f}")
        print(f"  Baseline: {scene['baseline_path']}")
        print(f"  Subsurface: {scene['subsurface_path']}")
        print(f"  Comparison: {scene['comparison_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
