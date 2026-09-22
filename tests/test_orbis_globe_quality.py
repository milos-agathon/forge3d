"""GPU acceptance for ORBIS streamed detail, oblique cameras, and globe context."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d


ROOT = Path(__file__).parents[1]
SWISS_DEM = ROOT / "assets" / "tif" / "switzerland_dem.tif"
JUNGFRAU_LON = 7.910
JUNGFRAU_LAT = 46.494
ORBIS_SELECTED = os.environ.get("FORGE3D_RUN_ORBIS_GPU") == "1"

sys.path.insert(0, str(ROOT / "examples"))
from orbis_swiss_alps_descent import descent_waypoints


def _require_orbis_gpu() -> None:
    if not ORBIS_SELECTED:
        pytest.skip("set FORGE3D_RUN_ORBIS_GPU=1 for physical ORBIS acceptance")
    assert SWISS_DEM.stat().st_size > 1024, "Swiss DEM is an unrestored LFS pointer"


def _luma(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame[..., :3], dtype=np.float64)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _phase_shift(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, int]:
    a = reference - float(reference.mean())
    b = candidate - float(candidate.mean())
    correlation = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    y, x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    if y > reference.shape[0] // 2:
        y -= reference.shape[0]
    if x > reference.shape[1] // 2:
        x -= reference.shape[1]
    return int(y), int(x)


def _detail_energy(frame: np.ndarray) -> float:
    image = _luma(frame)
    center = image[1:-1, 1:-1]
    laplacian = (
        image[:-2, 1:-1]
        + image[2:, 1:-1]
        + image[1:-1, :-2]
        + image[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.mean(np.abs(laplacian)))


def _surface_mask(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame[..., :3], dtype=np.uint8)
    return np.any(rgb != rgb[0, 0], axis=2)


def _assert_curved_limb_without_holes(frame: np.ndarray) -> None:
    mask = _surface_mask(frame)
    columns = []
    boundary = []
    for x in range(mask.shape[1]):
        covered = np.flatnonzero(mask[:, x])
        if covered.size == 0 or covered[0] == 0:
            continue
        assert bool(mask[covered[0] :, x].all()), f"clip/background hole in column {x}"
        columns.append(float(x))
        boundary.append(float(covered[0]))
    assert len(columns) >= 3, "render contains no measurable Earth limb"
    x = np.asarray(columns, dtype=np.float64)
    y = np.asarray(boundary, dtype=np.float64)
    linear = np.column_stack((x, np.ones_like(x)))
    curved = np.column_stack((x * x, x, np.ones_like(x)))
    linear_residual = float(np.square(y - linear @ np.linalg.lstsq(linear, y, rcond=None)[0]).sum())
    curved_residual = float(np.square(y - curved @ np.linalg.lstsq(curved, y, rcond=None)[0]).sum())
    assert float(np.ptp(y)) > 0.0
    assert curved_residual < linear_residual


def _mean_rgb_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                first[..., :3].astype(np.float64) - second[..., :3].astype(np.float64)
            )
        )
    )


@pytest.mark.offscreen
def test_streamed_swiss_detail_is_registered_and_reaches_source_resolution() -> None:
    _require_orbis_gpu()
    waypoints = descent_waypoints(240, end_altitude_m=120.0)
    streamed_scene = f3d.GlobeScene(SWISS_DEM, JUNGFRAU_LON, JUNGFRAU_LAT, "Jungfrau")
    streamed_frame = None
    for index, waypoint in enumerate(waypoints):
        frame = streamed_scene.fly_to(*waypoint[:3])
        if index == 180:
            streamed_frame = frame.to_numpy()
    assert streamed_frame is not None

    fresh_scene = f3d.GlobeScene(SWISS_DEM, JUNGFRAU_LON, JUNGFRAU_LAT, "Jungfrau")
    fresh_frame = fresh_scene.fly_to(*waypoints[180][:3]).to_numpy()
    assert _phase_shift(_luma(fresh_frame), _luma(streamed_frame)) == (0, 0)
    assert _detail_energy(streamed_frame) > _detail_energy(fresh_frame)

    stats = streamed_scene.streaming_stats()
    assert stats["source_sample_spacing_m"] <= 60.0
    assert stats["resident_fine_tiles"] >= stats["required_leaf_tiles"]
    assert stats["loader_pending"] == 0


@pytest.mark.offscreen
def test_oblique_waypoints_show_relief_without_clip_jitter_or_cracks() -> None:
    _require_orbis_gpu()
    scene = f3d.GlobeScene(SWISS_DEM, JUNGFRAU_LON, JUNGFRAU_LAT, "Jungfrau")
    metrics = scene.scripted_descent(
        [
            (JUNGFRAU_LON, JUNGFRAU_LAT, 20_000.0, 225.0, 70.0),
            (JUNGFRAU_LON, JUNGFRAU_LAT, 10_000.0, 225.0, 80.0),
            (JUNGFRAU_LON, JUNGFRAU_LAT, 5_000.0, 225.0, 85.0),
        ]
    )
    _assert_curved_limb_without_holes(scene.snapshot().to_numpy())
    assert metrics.max_vertex_jitter_px < 0.5
    assert metrics.lod_crack_pixels == 0


@pytest.mark.offscreen
def test_earth_context_blends_dem_and_crosses_overview_boundary_without_pop() -> None:
    _require_orbis_gpu()
    limb_scene = f3d.GlobeScene(SWISS_DEM, JUNGFRAU_LON, JUNGFRAU_LAT, "Jungfrau")
    limb = limb_scene.fly_to(
        JUNGFRAU_LON, JUNGFRAU_LAT, 408_000.0, 225.0, 55.0
    ).to_numpy()
    _assert_curved_limb_without_holes(limb)

    axis = 1 << 10
    target_x = math.floor((JUNGFRAU_LON + 180.0) / 360.0 * axis)
    boundary_lon = (target_x + 1) / axis * 360.0 - 180.0
    scene = f3d.GlobeScene(SWISS_DEM, JUNGFRAU_LON, JUNGFRAU_LAT, "Jungfrau")
    bounds = scene.source_bounds
    source_width = scene.source_dimensions[0]
    step = (bounds[2] - bounds[0]) / source_width
    frames = []
    for lon, altitude in zip(
        [
            boundary_lon - 1.5 * step,
            boundary_lon - 0.5 * step,
            boundary_lon + 0.5 * step,
            boundary_lon + 1.5 * step,
        ],
        [20_000.0, 15_000.0, 10_000.0, 5_000.0],
    ):
        frames.append(scene.fly_to(lon, JUNGFRAU_LAT, altitude, 225.0, 80.0).to_numpy())
    differences = [
        _mean_rgb_difference(first, second)
        for first, second in zip(frames, frames[1:])
    ]
    assert differences[1] <= max(differences[0], differences[2])
    _assert_curved_limb_without_holes(frames[-1])
