from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d.terrain import viewshed


pytestmark = pytest.mark.skipif(not f3d.has_gpu(), reason="viewshed requires a GPU")


def test_viewshed_returns_deterministic_visibility_and_physics_arrays() -> None:
    dem = np.zeros((33, 33), dtype=np.float32)
    dem[:, 16] = 600.0
    kwargs = dict(
        observer=(0.5, 0.1),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        observer_height=100.0,
        target_height=0.0,
        max_distance=120_000.0,
        earth_model="ellipsoid",
        refraction_model="bennett",
        return_diagnostics=True,
    )

    first = viewshed(dem, **kwargs)
    second = viewshed(dem, **kwargs)

    assert first["visibility"].dtype == np.bool_
    assert first["visibility"].shape == dem.shape
    assert np.array_equal(first["visibility"], second["visibility"])
    assert np.array_equal(first["curvature_drop_m"], second["curvature_drop_m"])
    assert first["visibility"][16, 10]
    assert not first["visibility"][16, 24]
    assert first["curvature_drop_m"][16, 32] > 0.0
    assert first["refraction_gain_m"][16, 32] > 0.0
    assert first["horizon_distance_m"][16, 32] > 0.0


def test_viewshed_default_return_is_boolean_array() -> None:
    result = viewshed(
        np.zeros((8, 8), dtype=np.float32),
        observer=(0.5, 0.5),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        earth_model="flat",
        refraction_model="none",
    )
    assert result.dtype == np.bool_
    assert result.all()


def test_viewshed_rejects_unsupported_models() -> None:
    with pytest.raises(ValueError, match="unsupported earth_model"):
        viewshed(
            np.zeros((4, 4), dtype=np.float32),
            observer=(0.5, 0.5),
            bounds=(0.0, 0.0, 1.0, 1.0),
            height_system="ellipsoidal",
            earth_model="cube",
        )


def test_viewshed_rejects_nonlocal_global_bounds() -> None:
    with pytest.raises(ValueError, match="spanning less than 180 degrees"):
        viewshed(
            np.zeros((2, 2), dtype=np.float32),
            observer=(0.0, 0.0),
            bounds=(-180.0, -1.0, 180.0, 1.0),
            height_system="ellipsoidal",
            earth_model="flat",
            refraction_model="none",
        )


def test_viewshed_accepts_local_antimeridian_bounds() -> None:
    result = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        observer=(-9.5, 180.0),
        bounds=(179.8, -10.0, -179.8, -9.0),
        height_system="ellipsoidal",
        earth_model="flat",
        refraction_model="none",
    )
    assert result.shape == (3, 3)


def test_viewshed_uses_geodesic_distance_at_high_latitude() -> None:
    diagnostics = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        observer=(75.0, 0.0),
        bounds=(0.0, 70.0, 10.0, 80.0),
        height_system="ellipsoidal",
        earth_model="ellipsoid",
        refraction_model="none",
        return_diagnostics=True,
    )
    inverse = f3d.geodesic_inverse(75.0, 0.0, 78.33333333333333, 8.333333333333334)
    latitude = np.deg2rad(75.0)
    azimuth = np.deg2rad(inverse["azi1"])
    eccentricity_squared = 6.694_379_990_141_316_5e-3
    w = np.sqrt(1.0 - eccentricity_squared * np.sin(latitude) ** 2)
    meridional = 6_378_137.0 * (1.0 - eccentricity_squared) / w**3
    prime_vertical = 6_378_137.0 / w
    radius = 1.0 / (
        np.cos(azimuth) ** 2 / meridional + np.sin(azimuth) ** 2 / prime_vertical
    )
    expected_drop = inverse["s12"] ** 2 / (2.0 * radius)
    assert diagnostics["curvature_drop_m"][0, 2] == pytest.approx(
        expected_drop, rel=2e-5
    )


def test_viewshed_marches_blockers_on_the_geodesic() -> None:
    dem = np.zeros((11, 11), dtype=np.float32)
    dem[2, 4] = 1_500.0
    visibility = viewshed(
        dem,
        observer=(75.0, 0.0),
        bounds=(0.0, 70.0, 10.0, 80.0),
        height_system="ellipsoidal",
        observer_height=1_000.0,
        earth_model="flat",
        refraction_model="none",
    )
    assert not visibility[0, 10]


def test_horizon_diagnostic_includes_terrain_elevation() -> None:
    dem = np.zeros((3, 3), dtype=np.float32)
    dem[1, 2] = 1_000.0
    diagnostics = viewshed(
        dem,
        observer=(0.5, 0.5),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        observer_height=2.0,
        earth_model="sphere",
        refraction_model="none",
        return_diagnostics=True,
    )
    assert (
        diagnostics["horizon_distance_m"][1, 2]
        > diagnostics["horizon_distance_m"][0, 2]
    )


def test_spherical_viewshed_scales_with_configured_radius() -> None:
    kwargs = dict(
        observer=(0.5, 0.5),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        earth_model="sphere",
        refraction_model="none",
        return_diagnostics=True,
    )
    small = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        sphere_radius_m=3_000_000.0,
        **kwargs,
    )
    large = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        sphere_radius_m=6_000_000.0,
        **kwargs,
    )
    assert large["curvature_drop_m"][0, 2] == pytest.approx(
        2.0 * small["curvature_drop_m"][0, 2], rel=2e-5
    )


def test_viewshed_preserves_observer_at_raster_edge() -> None:
    result = viewshed(
        np.zeros((3, 3), dtype=np.float32),
        observer=(0.5, 0.0),
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_system="ellipsoidal",
        earth_model="flat",
        refraction_model="none",
    )
    assert result.shape == (3, 3)


def test_viewshed_rejects_geodesics_outside_dem_footprint() -> None:
    with pytest.raises(RuntimeError, match="geodesic leaves the DEM footprint"):
        viewshed(
            np.zeros((3, 3), dtype=np.float32),
            observer=(74.1666666667, -26.6666666667),
            bounds=(-40.0, 70.0, 40.0, 75.0),
            height_system="ellipsoidal",
            earth_model="flat",
            refraction_model="none",
        )


def test_viewshed_shader_has_no_fixed_los_step_cap() -> None:
    shader = (
        Path(__file__).parents[1] / "src" / "shaders" / "terrain_viewshed.wgsl"
    ).read_text(encoding="utf-8")
    assert "8192u" not in shader
    assert "let steps = max(half_cell_steps, 1u);" in shader


def test_viewshed_reports_missing_native_lazily(monkeypatch) -> None:
    import forge3d.terrain as terrain

    monkeypatch.setattr(terrain, "get_native_module", lambda: None)
    monkeypatch.setattr(terrain, "native_import_error", lambda: ImportError("missing test module"))
    with pytest.raises(RuntimeError, match="missing test module"):
        terrain.viewshed(
            np.zeros((2, 2), dtype=np.float32),
            (0.0, 0.0),
            bounds=(-1.0, -1.0, 1.0, 1.0),
            height_system="ellipsoidal",
        )


def test_viewshed_reports_stale_native_extension(monkeypatch) -> None:
    import forge3d.terrain as terrain

    monkeypatch.setattr(terrain, "get_native_module", object)
    with pytest.raises(RuntimeError, match="does not provide terrain_viewshed"):
        terrain.viewshed(
            np.zeros((2, 2), dtype=np.float32),
            (0.0, 0.0),
            bounds=(-1.0, -1.0, 1.0, 1.0),
            height_system="ellipsoidal",
        )


def test_viewshed_height_system_is_explicit_and_strict() -> None:
    with pytest.raises(TypeError, match="height_system"):
        viewshed(
            np.zeros((2, 2), dtype=np.float32),
            (0.0, 0.0),
            bounds=(-1.0, -1.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="unsupported height_system"):
        viewshed(
            np.zeros((2, 2), dtype=np.float32),
            (0.0, 0.0),
            bounds=(-1.0, -1.0, 1.0, 1.0),
            height_system="unknown",
        )
    result = viewshed(
        np.zeros((2, 2), dtype=np.float32),
        (0.0, 0.0),
        bounds=(-1.0, -1.0, 1.0, 1.0),
        height_system="orthometric_egm96",
        earth_model="flat",
        refraction_model="none",
    )
    assert result.shape == (2, 2)
