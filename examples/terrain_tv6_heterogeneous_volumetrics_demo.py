from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np


def _import_forge3d():
    try:
        from _import_shim import ensure_repo_import

        ensure_repo_import()
    except ModuleNotFoundError:
        pass

    try:
        import forge3d as f3d

        return f3d
    except ModuleNotFoundError:
        from _import_shim import ensure_repo_import

        ensure_repo_import()
        import forge3d as f3d

        return f3d


f3d = _import_forge3d()
from forge3d.terrain_params import (
    VolumetricsSettings,
    localized_haze_volume,
    plume_volume,
    valley_fog_volume,
)


DEFAULT_DEM = Path(__file__).resolve().parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "out" / "terrain_tv6_heterogeneous_volumetrics"


def _write_heightmap_tiff(path: Path, heightmap: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Writing a temporary viewer DEM requires Pillow (`pip install pillow`).") from exc

    Image.fromarray(np.ascontiguousarray(heightmap, dtype=np.float32)).save(path, format="TIFF")


def _downsample_heightmap(heightmap: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return np.ascontiguousarray(heightmap)
    longest = max(int(heightmap.shape[0]), int(heightmap.shape[1]))
    if longest <= max_dim:
        return np.ascontiguousarray(heightmap)
    step = int(math.ceil(longest / max_dim))
    return np.ascontiguousarray(heightmap[::step, ::step])


def _load_dem(path: Path, max_dim: int) -> tuple[np.ndarray, tuple[float, float]]:
    try:
        dem = f3d.io.load_dem(str(path), fill_nodata_values=True)
        heightmap = np.asarray(dem.data, dtype=np.float32).copy()
        domain = getattr(dem, "domain", (float(np.min(heightmap)), float(np.max(heightmap))))
        return _downsample_heightmap(heightmap, max_dim), (float(domain[0]), float(domain[1]))
    except Exception:
        try:
            from PIL import Image
        except ImportError as exc:
            raise SystemExit(
                "DEM loading requires rasterio or Pillow. "
                "Install with `pip install rasterio` or `pip install pillow`."
            ) from exc

        with Image.open(path) as image:
            heightmap = np.asarray(image, dtype=np.float32)
        if heightmap.ndim == 3:
            heightmap = heightmap[..., 0]
        heightmap = _downsample_heightmap(heightmap, max_dim)
        return heightmap, (float(np.min(heightmap)), float(np.max(heightmap)))


def _viewer_z_scale(domain: tuple[float, float], terrain_width: float) -> float:
    relief = max(float(domain[1]) - float(domain[0]), 1e-6)
    return float(np.clip((terrain_width / relief) * 0.35, 0.08, 0.22))


def _peak_world_position(
    heightmap: np.ndarray,
    domain: tuple[float, float],
    terrain_width: float,
    z_scale: float,
) -> tuple[float, float, float]:
    row, col = np.unravel_index(int(np.argmax(heightmap)), heightmap.shape)
    height_range = max(float(domain[1]) - float(domain[0]), 1e-6)
    peak_raw = float(heightmap[row, col])
    peak_x = float(col / max(heightmap.shape[1] - 1, 1) * terrain_width)
    peak_z = float(row / max(heightmap.shape[0] - 1, 1) * terrain_width)
    peak_y = float((peak_raw - domain[0]) * z_scale)
    return peak_x, peak_y, peak_z


def _build_contact_sheet(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("images must be non-empty")
    height = images[0].shape[0]
    divider = np.full((height, 10, 4), 255, dtype=np.uint8)
    divider[..., :3] = 18
    sheet = images[0]
    for image in images[1:]:
        if image.shape[0] != height:
            raise ValueError("all images must share the same height")
        sheet = np.concatenate([sheet, divider, image], axis=1)
    return sheet


def _diff_metrics(baseline: np.ndarray, variant: np.ndarray) -> dict[str, float]:
    delta = np.abs(baseline[..., :3].astype(np.float32) - variant[..., :3].astype(np.float32))
    return {
        "mean_abs_diff": float(np.mean(delta)),
        "changed_pixels": float(np.count_nonzero(np.any(delta > 0.0, axis=-1))),
    }


def _build_volume_presets(
    terrain_width: float,
    domain: tuple[float, float],
    z_scale: float,
    peak: tuple[float, float, float],
) -> dict[str, VolumetricsSettings]:
    peak_x, peak_y, peak_z = peak
    scaled_height = max(float(domain[1]) - float(domain[0]), 1e-6) * z_scale
    valley = valley_fog_volume(
        center=(peak_x, scaled_height * 0.10, peak_z),
        size=(terrain_width * 0.90, scaled_height * 0.30, terrain_width * 0.90),
        resolution=(72, 32, 72),
        density_scale=1.35,
        edge_softness=0.28,
        noise_strength=0.42,
        floor_offset=10.0,
        ceiling=0.34,
        seed=11,
    )
    plume = plume_volume(
        center=(peak_x, peak_y + scaled_height * 0.10, peak_z),
        size=(terrain_width * 0.16, scaled_height * 0.70, terrain_width * 0.16),
        resolution=(40, 88, 40),
        density_scale=1.6,
        edge_softness=0.16,
        noise_strength=0.55,
        plume_spread=0.44,
        wind=(0.28, 1.0, -0.16),
        seed=19,
    )
    haze = localized_haze_volume(
        center=(
            float(np.clip(peak_x + terrain_width * 0.18, terrain_width * 0.20, terrain_width * 0.82)),
            scaled_height * 0.26,
            float(np.clip(peak_z - terrain_width * 0.10, terrain_width * 0.18, terrain_width * 0.82)),
        ),
        size=(terrain_width * 0.34, scaled_height * 0.18, terrain_width * 0.28),
        resolution=(48, 24, 48),
        density_scale=0.90,
        edge_softness=0.42,
        noise_strength=0.30,
        ceiling=0.58,
        seed=5,
    )

    return {
        "valley_fog": VolumetricsSettings(
            enabled=True,
            mode="height",
            density=0.0025,
            height_falloff=0.18,
            scattering=0.76,
            absorption=0.08,
            light_shafts=True,
            shaft_intensity=1.3,
            shaft_samples=48,
            density_volumes=(valley,),
        ),
        "plume": VolumetricsSettings(
            enabled=True,
            mode="height",
            density=0.0015,
            height_falloff=0.14,
            scattering=0.82,
            absorption=0.09,
            light_shafts=True,
            shaft_intensity=1.55,
            shaft_samples=56,
            density_volumes=(plume,),
        ),
        "localized_haze": VolumetricsSettings(
            enabled=True,
            mode="height",
            density=0.0020,
            height_falloff=0.16,
            scattering=0.68,
            absorption=0.10,
            light_shafts=True,
            shaft_intensity=1.1,
            shaft_samples=40,
            density_volumes=(haze,),
        ),
    }


def render_demo(
    *,
    dem_path: Path = DEFAULT_DEM,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    width: int = 1400,
    height: int = 900,
    max_dem_size: int = 1024,
    timeout: float = 90.0,
) -> dict[str, object]:
    dem_path = dem_path.resolve()
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    heightmap, domain = _load_dem(dem_path, max_dem_size)
    terrain_width = float(max(heightmap.shape[0], heightmap.shape[1]))
    z_scale = _viewer_z_scale(domain, terrain_width)
    peak = _peak_world_position(heightmap, domain, terrain_width, z_scale)
    presets = _build_volume_presets(terrain_width, domain, z_scale, peak)

    temp_dem: Path | None = None
    viewer_dem_path = dem_path
    if max_dem_size > 0:
        longest = max(int(heightmap.shape[0]), int(heightmap.shape[1]))
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_dem = Path(tmp.name)
        _write_heightmap_tiff(temp_dem, heightmap)
        viewer_dem_path = temp_dem

    outputs = {
        "baseline": output_dir / "terrain_tv6_baseline.png",
        "valley_fog": output_dir / "terrain_tv6_valley_fog.png",
        "plume": output_dir / "terrain_tv6_plume.png",
        "localized_haze": output_dir / "terrain_tv6_localized_haze.png",
        "contact_sheet": output_dir / "terrain_tv6_contact_sheet.png",
        "manifest": output_dir / "terrain_tv6_manifest.json",
    }

    start_total = time.perf_counter()
    try:
        with f3d.open_viewer_async(
            width=width,
            height=height,
            title="Forge3D TV6 Heterogeneous Volumetrics Demo",
            terrain_path=str(viewer_dem_path),
            timeout=timeout,
        ) as viewer:
            viewer.set_z_scale(z_scale)
            viewer.set_orbit_camera(
                phi_deg=168.0,
                theta_deg=60.0,
                radius=terrain_width * 1.22,
                fov_deg=42.0,
            )
            viewer.send_ipc(
                {
                    "cmd": "set_terrain_sun",
                    "azimuth_deg": 118.0,
                    "elevation_deg": 14.0,
                    "intensity": 2.8,
                }
            )

            baseline_start = time.perf_counter()
            viewer.send_ipc(
                {
                    "cmd": "set_terrain_pbr",
                    "enabled": True,
                    "exposure": 1.05,
                    "normal_strength": 1.05,
                    "shadow_map_res": 2048,
                    "volumetrics": {"enabled": False},
                }
            )
            time.sleep(0.2)
            viewer.snapshot(outputs["baseline"], width=width, height=height)
            baseline_report = viewer.get_terrain_volumetrics_report()
            baseline_seconds = time.perf_counter() - baseline_start
            baseline_image = f3d.png_to_numpy(outputs["baseline"])

            scene_results: dict[str, dict[str, object]] = {}
            contact_images = [baseline_image]
            for name, settings in presets.items():
                shot_start = time.perf_counter()
                viewer.send_ipc(
                    {
                        "cmd": "set_terrain_pbr",
                        "enabled": True,
                        "exposure": 1.05,
                        "normal_strength": 1.05,
                        "shadow_map_res": 2048,
                        "volumetrics": settings.to_viewer_dict(),
                    }
                )
                time.sleep(0.25)
                viewer.snapshot(outputs[name], width=width, height=height)
                report = viewer.get_terrain_volumetrics_report()
                image = f3d.png_to_numpy(outputs[name])
                metrics = _diff_metrics(baseline_image, image)
                scene_results[name] = {
                    "path": str(outputs[name]),
                    "report": report,
                    "render_seconds": float(time.perf_counter() - shot_start),
                    **metrics,
                }
                contact_images.append(image)
    finally:
        if temp_dem is not None:
            temp_dem.unlink(missing_ok=True)

    contact_sheet = _build_contact_sheet(contact_images)
    f3d.numpy_to_png(outputs["contact_sheet"], contact_sheet)

    total_seconds = float(time.perf_counter() - start_total)
    result = {
        "dem_path": str(dem_path),
        "effective_shape": (int(heightmap.shape[0]), int(heightmap.shape[1])),
        "domain": domain,
        "terrain_width": terrain_width,
        "z_scale": z_scale,
        "peak_world": peak,
        "baseline_path": str(outputs["baseline"]),
        "baseline_report": baseline_report,
        "baseline_render_seconds": float(baseline_seconds),
        "contact_sheet_path": str(outputs["contact_sheet"]),
        "manifest_path": str(outputs["manifest"]),
        "total_render_seconds": total_seconds,
        "scenes": scene_results,
    }

    outputs["manifest"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Epic TV6 demo: heterogeneous terrain volumetrics on a real repo DEM."
    )
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM, help="Path to a DEM GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=1400, help="Snapshot width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Snapshot height in pixels")
    parser.add_argument(
        "--max-dem-size",
        type=int,
        default=1024,
        help="Clamp the longest DEM dimension before viewer upload (0 keeps the source raster)",
    )
    parser.add_argument("--timeout", type=float, default=90.0, help="Viewer startup timeout in seconds")
    args = parser.parse_args()

    result = render_demo(
        dem_path=args.dem,
        output_dir=args.output_dir,
        width=int(args.width),
        height=int(args.height),
        max_dem_size=int(args.max_dem_size),
        timeout=float(args.timeout),
    )

    print(f"DEM: {result['dem_path']}")
    print(f"Effective DEM: {result['effective_shape'][1]}x{result['effective_shape'][0]}")
    print(f"Domain: {result['domain'][0]:.3f} .. {result['domain'][1]:.3f}")
    print(f"Terrain width: {result['terrain_width']:.1f}")
    print(f"Viewer z-scale: {result['z_scale']:.4f}")
    print(
        f"Peak world position: x={result['peak_world'][0]:.1f}, "
        f"y={result['peak_world'][1]:.1f}, z={result['peak_world'][2]:.1f}"
    )
    print(f"Baseline report: {result['baseline_report']}")
    for name, scene in result["scenes"].items():
        report = scene["report"]
        print(
            f"{name}: diff={scene['mean_abs_diff']:.3f}, "
            f"changed_pixels={int(scene['changed_pixels'])}, "
            f"active_volumes={report['active_volume_count']}, "
            f"atlas={report['atlas_dimensions']}, "
            f"bytes={report['texture_bytes']}"
        )
        print(f"Wrote {scene['path']}")
    print(f"Wrote {result['contact_sheet_path']}")
    print(f"Wrote {result['manifest_path']}")
    print(f"Total render time: {result['total_render_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
