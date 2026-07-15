from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / "examples"
    module_path = example_dir / "california_wildfire_smoke_video.py"
    if not module_path.exists():
        pytest.skip(
            "example 'california_wildfire_smoke_video.py' is untracked/local-only",
            allow_module_level=True,
        )
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    spec = importlib.util.spec_from_file_location("california_wildfire_smoke_video", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_example_module()


class _FlatProjector:
    screen_mask = None

    def lonlat_to_screen(self, lon: float, lat: float, width: int, height: int) -> tuple[float, float]:
        x = (lon - smoke.LON_MIN) / (smoke.LON_MAX - smoke.LON_MIN) * width
        y = (smoke.LAT_MAX - lat) / (smoke.LAT_MAX - smoke.LAT_MIN) * height
        return float(x), float(y)


def test_move_lonlat_by_meters_preserves_expected_directions() -> None:
    lon, lat = -120.0, 37.0

    east_lon, east_lat = smoke.move_lonlat_by_meters(lon, lat, 1_000.0, 0.0)
    north_lon, north_lat = smoke.move_lonlat_by_meters(lon, lat, 0.0, 1_000.0)

    assert east_lon > lon
    assert east_lat == pytest.approx(lat)
    assert north_lon == pytest.approx(lon)
    assert north_lat > lat


def test_windfield_json_samples_space_time_and_height(tmp_path: Path) -> None:
    times = ["2020-08-16T00:00:00", "2020-08-16T12:00:00"]
    heights = [10.0, 1000.0]
    latitudes = [36.0, 38.0]
    longitudes = [-122.0, -120.0]

    u = np.zeros((2, 2, 2, 2), dtype=np.float32)
    for ti in range(2):
        for zi in range(2):
            for yi in range(2):
                for xi in range(2):
                    u[ti, zi, yi, xi] = ti * 8.0 + zi * 4.0 + yi * 2.0 + xi

    wind_path = tmp_path / "wind.json"
    wind_path.write_text(
        json.dumps(
            {
                "source": "unit-test",
                "times": times,
                "heights_m": heights,
                "latitudes": latitudes,
                "longitudes": longitudes,
                "u_mps": u.tolist(),
                "v_mps": (-u).tolist(),
            }
        ),
        encoding="utf-8",
    )

    field = smoke.WindField.from_file(wind_path)
    u_mps, v_mps = field.sample(
        -121.0,
        37.0,
        np.datetime64("2020-08-16T06:00:00"),
        height_m=505.0,
    )

    assert u_mps == pytest.approx(7.5)
    assert v_mps == pytest.approx(-7.5)


def test_smoke_particle_age_does_not_backfill_new_fire() -> None:
    assert smoke._smoke_particle_age_hours(0.0, 0.0) is None
    assert smoke._smoke_particle_age_hours(1.0, 2.0, max_age_hours=14.0) is None
    assert smoke._smoke_particle_age_hours(3.5, 1.25, max_age_hours=14.0) == pytest.approx(
        2.25
    )


def test_smoke_particle_age_cycles_after_max_age() -> None:
    assert smoke._smoke_particle_age_hours(17.0, 2.0, max_age_hours=14.0) == pytest.approx(1.0)


def test_smoke_lifecycle_alpha_fades_before_old_smoke_expires() -> None:
    young = smoke._smoke_lifecycle_alpha(0.25, max_age_hours=14.0)
    mature = smoke._smoke_lifecycle_alpha(4.0, max_age_hours=14.0)
    old = smoke._smoke_lifecycle_alpha(13.2, max_age_hours=14.0)

    assert young < mature
    assert old < mature * 0.2


def test_smoke_veil_parameters_stay_filamentary() -> None:
    assert smoke.SMOKE_DENSITY_RESOLUTION >= 384
    assert smoke.SMOKE_DENSITY_MAX_ALPHA <= 180
    assert smoke.SMOKE_SOURCE_MAX_RADIUS_M <= 24_000


def test_smoke_density_raster_advects_east_northeast() -> None:
    early = smoke.simulate_smoke_density_frame(10, 60, 96)
    later = smoke.simulate_smoke_density_frame(32, 60, 96)

    assert np.isfinite(early).all()
    assert np.isfinite(later).all()
    assert float(early.max()) > 0.0
    assert float(later.max()) > 0.0
    assert not np.allclose(early, later)

    yy, xx = np.mgrid[0:96, 0:96].astype(np.float64)
    early_mass = float(np.sum(early))
    later_mass = float(np.sum(later))
    early_x = float(np.sum(xx * early) / early_mass)
    early_y = float(np.sum(yy * early) / early_mass)
    later_x = float(np.sum(xx * later) / later_mass)
    later_y = float(np.sum(yy * later) / later_mass)

    assert later_x > early_x
    assert later_y < early_y


def test_draw_smoke_layer_returns_requested_rgba_size() -> None:
    image = smoke.draw_smoke_layer(
        96,
        54,
        24,
        60,
        _FlatProjector(),
        wind_field=None,
    )

    assert image.mode == "RGBA"
    assert image.size == (96, 54)
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    assert int(alpha.max()) <= smoke.SMOKE_DENSITY_MAX_ALPHA
    assert np.count_nonzero(alpha) > 0
