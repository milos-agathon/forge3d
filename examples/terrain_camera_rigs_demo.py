#!/usr/bin/env python3
"""TV17 terrain camera rig demo with three preset shots.

Usage:
    python examples/terrain_camera_rigs_demo.py --preset orbit_rainier
    python examples/terrain_camera_rigs_demo.py --preset rail_luxembourg --export-dir frames/rail
    python examples/terrain_camera_rigs_demo.py --preset follow_rainier --loop
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import forge3d as f3d
from forge3d import io as f3dio
from forge3d.animation import CameraKeyframe
from forge3d.camera_rigs import (
    TerrainClearance,
    TerrainOrbitRig,
    TerrainRailRig,
    TerrainTargetFollowRig,
    _orbit_eye,
    _orbit_from_eye_target,
    _unwrap_phi,
)
from forge3d.terrain_scatter import TerrainScatterSource


PRESET_DATASETS = {
    "orbit_rainier": "rainier",
    "rail_luxembourg": "luxembourg",
    "follow_rainier": "rainier",
}


def _contract_point(width: float, x: float, z: float) -> tuple[float, float]:
    return (float(width * x), float(width * z))


def _load_rig_terrain(path: Path, z_scale: float):
    if path.suffix.lower() == ".npy":
        heightmap = np.asarray(np.load(path), dtype=np.float32)
        height, width = heightmap.shape
        origin = (0.0, 0.0)
        span = (float(width), float(height))
    else:
        dem = f3dio.load_dem(str(path), fill_nodata_values=True)
        heightmap = dem.data
        origin, span = _viewer_footprint(dem)

    source = TerrainScatterSource(heightmap, z_scale=z_scale, terrain_width=max(span))
    return source, origin, span


def _viewer_footprint(dem: f3dio.DEMData):
    from rasterio.crs import CRS

    if dem.transform is None or dem.transform.is_identity:
        height, width = dem.data.shape
        return (0.0, 0.0), (float(width), float(height))
    west, south, east, north = dem.bounds
    crs = CRS.from_string(dem.crs) if dem.crs else None
    authority = crs.to_authority() if crs is not None else None
    planetary = authority is not None and authority[0].startswith("IAU")
    if crs is not None and crs.is_geographic and not planetary:
        if crs.units_factor[0] != "degree":
            raise ValueError("The camera rig example requires geographic coordinates in degrees")
        # Match the viewer's WGS84 scale at the footprint centre latitude.
        latitude = math.radians((south + north) / 2.0)
        eccentricity_squared = 6.694_379_990_141_316e-3
        w = math.sqrt(1.0 - eccentricity_squared * math.sin(latitude) ** 2)
        prime_vertical = 6_378_137.0 / w
        meridional = 6_378_137.0 * (1.0 - eccentricity_squared) / w**3
        span_x = math.radians(east - west) * prime_vertical * math.cos(latitude)
        span_z = math.radians(north - south) * meridional
        return (0.0, 0.0), (span_x, span_z)
    return (west, -north), (east - west, north - south)


def _place_animation(animation, width: float, origin, span) -> None:
    """Map the rig's square heightfield to the viewer's rectangular footprint."""
    scale_x, scale_z = span[0] / width, span[1] / width

    def world_point(point):
        x, y, z = point
        return (origin[0] + x * scale_x, y, origin[1] + z * scale_z)

    keyframes = []
    for key in animation.get_keyframes():
        eye = _orbit_eye(key.target, key.radius, key.phi_deg, key.theta_deg)
        target = world_point(key.target)
        phi, theta, radius = _orbit_from_eye_target(world_point(eye), target)
        phi = _unwrap_phi(keyframes[-1].phi_deg if keyframes else key.phi_deg, phi)
        keyframes.append(CameraKeyframe(key.time, phi, theta, radius, key.fov_deg, target))
    animation.replace_keyframes(keyframes)


def _build_rig(preset: str, source: TerrainScatterSource):
    width = float(source.terrain_width)

    if preset == "orbit_rainier":
        return TerrainOrbitRig(
            target_xz=_contract_point(width, 0.52, 0.48),
            duration=8.0,
            radius=0.22 * width,
            phi_start_deg=25.0,
            phi_end_deg=335.0,
            theta_start_deg=58.0,
            theta_end_deg=44.0,
            fov_start_deg=52.0,
            fov_end_deg=48.0,
            target_height_offset=0.02 * width,
            clearance=TerrainClearance(minimum_height=0.015 * width),
        )

    if preset == "rail_luxembourg":
        return TerrainRailRig(
            path_xz=[
                _contract_point(width, 0.22, 0.65),
                _contract_point(width, 0.34, 0.62),
                _contract_point(width, 0.50, 0.57),
                _contract_point(width, 0.70, 0.55),
                _contract_point(width, 0.83, 0.61),
            ],
            duration=10.0,
            camera_height_offset=0.03 * width,
            look_ahead_distance=0.08 * width,
            lateral_offset=0.01 * width,
            target_height_offset=0.01 * width,
            fov_deg=50.0,
            clearance=TerrainClearance(minimum_height=0.01 * width),
        )

    if preset == "follow_rainier":
        return TerrainTargetFollowRig(
            target_path_xz=[
                _contract_point(width, 0.30, 0.70),
                _contract_point(width, 0.42, 0.60),
                _contract_point(width, 0.54, 0.52),
                _contract_point(width, 0.66, 0.44),
                _contract_point(width, 0.76, 0.36),
            ],
            duration=9.0,
            radius=0.15 * width,
            theta_deg=62.0,
            heading_offset_deg=205.0,
            target_height_offset=0.015 * width,
            fov_deg=47.0,
            clearance=TerrainClearance(minimum_height=0.012 * width),
        )

    raise KeyError(f"Unknown preset: {preset}")


def _play_animation(viewer: "f3d.ViewerHandle", animation, fps: int, loop: bool) -> None:
    total_frames = animation.get_frame_count(fps)
    if total_frames <= 0:
        return

    while True:
        start_time = time.perf_counter()
        for frame in range(total_frames):
            state = animation.evaluate(frame / fps)
            if state is None:
                continue
            viewer.set_orbit_camera(
                phi_deg=state.phi_deg,
                theta_deg=state.theta_deg,
                radius=state.radius,
                fov_deg=state.fov_deg,
                target=state.target,
            )
            target_time = start_time + ((frame + 1) / fps)
            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        if not loop:
            break


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terrain camera rig toolkit demo.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_DATASETS),
        default="orbit_rainier",
        help="Rig preset to bake and preview.",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        help="Optional DEM override. Defaults to the dataset paired with the selected preset.",
    )
    parser.add_argument("--z-scale", type=float, default=1.0, help="Terrain z-scale used for viewer and bake.")
    parser.add_argument("--fps", type=int, default=30, help="Playback or export frame rate.")
    parser.add_argument(
        "--samples-per-second",
        type=int,
        default=60,
        help="Bake density before evaluator-based refinement.",
    )
    parser.add_argument("--width", type=int, default=1440, help="Viewer or snapshot width.")
    parser.add_argument("--height", type=int, default=900, help="Viewer or snapshot height.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="If set, render PNG frames to this directory instead of playing a live preview.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop the live preview until interrupted.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dem_path = args.dem or f3d.fetch_dem(PRESET_DATASETS[args.preset])
    scatter_source, origin, span = _load_rig_terrain(Path(dem_path), z_scale=float(args.z_scale))
    rig = _build_rig(args.preset, scatter_source)
    animation = rig.bake(scatter_source, samples_per_second=int(args.samples_per_second))
    _place_animation(animation, scatter_source.terrain_width, origin, span)

    print(
        f"[tv17] preset={args.preset} dem={Path(dem_path).name} "
        f"keyframes={animation.keyframe_count} duration={animation.duration:.2f}s"
    )

    with f3d.open_viewer_async(
        terrain_path=dem_path,
        width=int(args.width),
        height=int(args.height),
        fov_deg=55.0,
    ) as viewer:
        viewer.set_z_scale(float(args.z_scale))
        if args.export_dir is not None:
            print(f"[tv17] rendering frames to {args.export_dir}")
            viewer.render_animation(
                animation,
                args.export_dir,
                fps=int(args.fps),
                width=int(args.width),
                height=int(args.height),
            )
        else:
            print("[tv17] playing live preview; press Ctrl+C to stop")
            try:
                _play_animation(viewer, animation, int(args.fps), bool(args.loop))
            except KeyboardInterrupt:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
