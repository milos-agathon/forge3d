from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "california_fire_smoke_effect.py"


def load_module():
    spec = importlib.util.spec_from_file_location("california_fire_smoke_effect", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _single_northern_hotspot(module):
    return [hotspot for hotspot in module.default_hotspots() if hotspot.name == "northern_sierra_cluster"][:1]


def _weighted_centroid(density: np.ndarray) -> tuple[float, float]:
    yy, xx = np.mgrid[0 : density.shape[0], 0 : density.shape[1]].astype(np.float64)
    weights = np.asarray(density, dtype=np.float64)
    mass = float(weights.sum())
    assert mass > 0.0
    return float((xx * weights).sum() / mass), float((yy * weights).sum() / mass)


def test_lonlat_to_pixel_bounds() -> None:
    module = load_module()
    bounds = module.DEFAULT_BOUNDS

    west_x, _ = module.lonlat_to_pixel(bounds.lon_min, 37.0, bounds, 128, 96)
    east_x, _ = module.lonlat_to_pixel(bounds.lon_max, 37.0, bounds, 128, 96)
    _, north_y = module.lonlat_to_pixel(-120.0, bounds.lat_max, bounds, 128, 96)
    _, south_y = module.lonlat_to_pixel(-120.0, bounds.lat_min, bounds, 128, 96)

    assert west_x == pytest.approx(0.0)
    assert east_x == pytest.approx(127.0)
    assert north_y == pytest.approx(0.0)
    assert south_y == pytest.approx(95.0)
    for hotspot in module.default_hotspots():
        x, y = module.lonlat_to_pixel(hotspot.lon, hotspot.lat, bounds, 128, 96)
        assert 0.0 < x < 127.0
        assert 0.0 < y < 95.0


def test_default_smoke_density_evolves_and_is_finite() -> None:
    module = load_module()
    density_0 = module.simulate_smoke_frame(
        module.default_hotspots(), frame=0, frames=120, size=128, bounds=module.DEFAULT_BOUNDS, seed=7
    )
    density_24 = module.simulate_smoke_frame(
        module.default_hotspots(), frame=24, frames=120, size=128, bounds=module.DEFAULT_BOUNDS, seed=7
    )

    assert np.isfinite(density_0).all()
    assert np.isfinite(density_24).all()
    assert float(density_0.max()) > 0.0
    assert float(density_24.max()) > 0.0
    assert not np.allclose(density_0, density_24)


def test_smoke_centroid_moves_east_northeast() -> None:
    module = load_module()
    hotspot = _single_northern_hotspot(module)
    density_2 = module.simulate_smoke_frame(hotspot, frame=2, frames=120, size=128, bounds=module.DEFAULT_BOUNDS, seed=11)
    density_36 = module.simulate_smoke_frame(hotspot, frame=36, frames=120, size=128, bounds=module.DEFAULT_BOUNDS, seed=11)

    x2, y2 = _weighted_centroid(density_2)
    x36, y36 = _weighted_centroid(density_36)
    assert x36 - x2 > 4.0
    assert y36 - y2 < -1.0


def test_smoke_is_elongated_not_round_blob() -> None:
    module = load_module()
    density = module.simulate_smoke_frame(
        _single_northern_hotspot(module), frame=55, frames=120, size=128, bounds=module.DEFAULT_BOUNDS, seed=13
    )
    mask = density > np.percentile(density[density > 0.0], 35.0)
    yy, xx = np.mgrid[0 : density.shape[0], 0 : density.shape[1]].astype(np.float64)
    weights = density[mask].astype(np.float64)
    coords = np.column_stack((xx[mask], yy[mask]))
    center = np.average(coords, axis=0, weights=weights)
    centered = coords - center
    covariance = np.array(
        [
            [
                np.sum(weights * centered[:, 0] * centered[:, 0]),
                np.sum(weights * centered[:, 0] * centered[:, 1]),
            ],
            [
                np.sum(weights * centered[:, 0] * centered[:, 1]),
                np.sum(weights * centered[:, 1] * centered[:, 1]),
            ],
        ],
        dtype=np.float64,
    ) / weights.sum()
    eigenvalues = np.linalg.eigvalsh(covariance)

    assert float(eigenvalues[-1] / max(eigenvalues[0], 1.0e-6)) > 1.8


def test_smoke_has_downwind_tail() -> None:
    module = load_module()
    hotspot = _single_northern_hotspot(module)[0]
    density = module.simulate_smoke_frame(
        [hotspot], frame=55, frames=120, size=128, bounds=module.DEFAULT_BOUNDS, seed=17
    )
    source_x, source_y = module.lonlat_to_pixel(hotspot.lon, hotspot.lat, module.DEFAULT_BOUNDS, 128, 128)
    tail = density[:, int(source_x + 20) :]
    y_idx, x_idx = np.nonzero(tail > max(float(density.max()) * 0.018, 0.001))

    assert y_idx.size > 0
    assert np.any((y_idx + 0.0) < source_y + 14.0)
    assert x_idx.size > 0


def test_smoke_rgba_is_translucent_not_opaque() -> None:
    module = load_module()
    density = module.simulate_smoke_frame(
        module.default_hotspots(), frame=36, frames=120, size=128, bounds=module.DEFAULT_BOUNDS, seed=19
    )
    rgba = module.density_to_smoke_rgba(density, frame=36, seed=19)
    alpha = rgba[..., 3]
    positive = alpha > 0

    assert rgba.shape == (128, 128, 4)
    assert rgba.dtype == np.uint8
    assert int(alpha.max()) < 220
    assert np.any(alpha == 0)
    assert np.any(positive)
    smoke_rgb = rgba[..., :3][positive].astype(np.int16)
    assert np.percentile(np.abs(smoke_rgb[:, 0] - smoke_rgb[:, 1]), 99.0) <= 8
    assert np.percentile(np.abs(smoke_rgb[:, 1] - smoke_rgb[:, 2]), 99.0) <= 8
    assert float(np.mean(smoke_rgb[:, 0])) >= 125.0
    assert float(np.mean(smoke_rgb[:, 1])) >= 125.0
    assert float(np.mean(smoke_rgb[:, 2])) >= 120.0
    assert float(np.mean(smoke_rgb[:, 0] - smoke_rgb[:, 2])) < 10.0


def test_smoke_has_wispy_alpha_distribution() -> None:
    module = load_module()
    density = module.simulate_smoke_frame(
        module.default_hotspots(), frame=72, frames=120, size=128, bounds=module.DEFAULT_BOUNDS, seed=23
    )
    alpha = module.density_to_smoke_rgba(density, frame=72, seed=23)[..., 3]
    bands = [
        np.any((1 <= alpha) & (alpha <= 30)),
        np.any((31 <= alpha) & (alpha <= 80)),
        np.any((81 <= alpha) & (alpha <= 140)),
        np.any((141 <= alpha) & (alpha <= 220)),
    ]

    assert sum(bool(item) for item in bands) >= 3


def test_fire_hotspots_rgba_has_small_orange_cores() -> None:
    module = load_module()
    rgba = module.fire_hotspots_rgba(
        module.default_hotspots(), frame=36, size=128, bounds=module.DEFAULT_BOUNDS, seed=29
    )
    positive = rgba[..., 3] > 0

    assert rgba.shape == (128, 128, 4)
    assert rgba.dtype == np.uint8
    assert np.any(positive)
    assert np.count_nonzero(positive) < 128 * 128 * 0.03
    glow = rgba[..., :3][positive].astype(np.int16)
    assert np.all(glow[:, 0] >= glow[:, 1])
    assert np.all(glow[:, 1] >= glow[:, 2])
    assert int(rgba[..., 3].max()) > 120


def test_write_overlay_pngs(tmp_path: Path) -> None:
    module = load_module()
    paths = module.write_overlay_pngs(
        tmp_path,
        frame=24,
        frames=120,
        size=96,
        bounds=module.DEFAULT_BOUNDS,
        hotspots=module.default_hotspots(),
        seed=31,
    )

    for key in ("base", "smoke", "fire"):
        path = paths[key]
        assert path.exists()
        assert path.stat().st_size > 0
        image = Image.open(path).convert("RGBA")
        assert image.size == (96, 96)


def test_firms_csv_parser(tmp_path: Path) -> None:
    module = load_module()
    csv_path = tmp_path / "firms.csv"
    csv_path.write_text(
        "longitude,latitude,frp,extra\n"
        "-121.2,40.1,8.0,a\n"
        "-118.7,34.3,42.0,b\n"
        "-80.0,25.0,999.0,outside\n",
        encoding="utf-8",
    )

    hotspots = module.load_firms_hotspots(csv_path, module.DEFAULT_BOUNDS)

    assert len(hotspots) == 2
    assert all(module.DEFAULT_BOUNDS.lon_min <= item.lon <= module.DEFAULT_BOUNDS.lon_max for item in hotspots)
    assert all(module.DEFAULT_BOUNDS.lat_min <= item.lat <= module.DEFAULT_BOUNDS.lat_max for item in hotspots)
    strengths = np.array([item.strength for item in hotspots], dtype=np.float32)
    assert np.isfinite(strengths).all()
    assert float(strengths.min()) >= 0.2
    assert float(strengths.max()) <= 1.25
