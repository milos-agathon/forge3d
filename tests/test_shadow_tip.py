from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import forge3d
from forge3d.geo import SolarTime, solar_position
from forge3d.terrain import shadow_mask, shadow_tip
from forge3d.viewer import ViewerHandle


BOUNDS = (-105.25, 39.70, -105.10, 39.80)
TIME = SolarTime(
    utc=(2003, 10, 17, 17, 0, 0),
    observer_lat=39.742476,
    observer_lon=-105.1786,
    observer_elev_m=1830.14,
    tz_offset_hours=-7.0,
    delta_t_seconds=67.0,
    pressure_mbar=820.0,
    temperature_c=11.0,
)


def _peak_dem() -> np.ndarray:
    yy, xx = np.mgrid[-1.0:1.0:65j, -1.0:1.0:65j]
    return (1800.0 + 2500.0 * np.exp(-24.0 * (xx * xx + yy * yy))).astype(np.float32)


def test_shadow_tip_bearing_and_curved_length_contract() -> None:
    dem = _peak_dem()
    result = shadow_tip(
        dem,
        39.75,
        -105.175,
        TIME,
        bounds=BOUNDS,
        height_system="ellipsoidal",
        earth_model="ellipsoid",
        refraction_model="effective_radius",
        refraction_k=0.13,
    )
    peak_h = float(dem[32, 32])
    solar = solar_position(
        TIME.utc,
        39.75,
        -105.175,
        peak_h,
        tz_offset_hours=TIME.tz_offset_hours,
        delta_t_seconds=TIME.delta_t_seconds,
        pressure_mbar=TIME.pressure_mbar,
        temperature_c=TIME.temperature_c,
    )
    expected_bearing = (solar["azimuth_deg"] + 180.0) % 360.0
    bearing_error = abs((result["bearing_deg"] - expected_bearing + 180.0) % 360.0 - 180.0)

    tan_alpha = math.tan(math.radians(solar["apparent_elevation_deg"]))
    radius = result["effective_radius_m"]
    prediction = 2.0 * peak_h / (
        tan_alpha + math.sqrt(tan_alpha * tan_alpha - 2.0 * peak_h / radius)
    )
    flat = peak_h / tan_alpha

    assert bearing_error <= 0.05
    assert abs(result["length_m"] - prediction) / prediction <= 0.005
    assert abs(result["length_m"] - flat) / result["length_m"] > 0.02


def test_no_refraction_shadow_tip_uses_pressure_independent_true_elevation() -> None:
    low_pressure = SolarTime(**{**TIME.__dict__, "pressure_mbar": 500.0})
    high_pressure = SolarTime(**{**TIME.__dict__, "pressure_mbar": 1050.0})
    kwargs = dict(
        bounds=BOUNDS,
        height_system="ellipsoidal",
        earth_model="ellipsoid",
        refraction_model="none",
    )
    low = shadow_tip(_peak_dem(), 39.75, -105.175, low_pressure, **kwargs)
    high = shadow_tip(_peak_dem(), 39.75, -105.175, high_pressure, **kwargs)
    expected = solar_position(
        low_pressure.utc,
        39.75,
        -105.175,
        low["peak_height_m"],
        tz_offset_hours=low_pressure.tz_offset_hours,
        delta_t_seconds=low_pressure.delta_t_seconds,
        pressure_mbar=low_pressure.pressure_mbar,
        temperature_c=low_pressure.temperature_c,
    )
    assert low["length_m"] == pytest.approx(high["length_m"], abs=1e-9)
    assert low["solar_launch_elevation_deg"] == pytest.approx(
        expected["true_elevation_deg"], abs=1e-10
    )


def test_shadow_mask_is_boolean_and_bitwise_deterministic() -> None:
    kwargs = dict(
        bounds=BOUNDS,
        height_system="ellipsoidal",
        earth_model="ellipsoid",
        refraction_model="effective_radius",
        refraction_k=0.13,
    )
    first = shadow_mask(_peak_dem(), TIME, **kwargs)
    second = shadow_mask(_peak_dem(), TIME, **kwargs)
    assert first.dtype == np.bool_
    assert first.shape == (65, 65)
    assert np.array_equal(first, second)
    assert np.any(first)
    assert np.any(~first)


def _shadow_mask_rgba() -> np.ndarray:
    mask = shadow_mask(
        _peak_dem(),
        TIME,
        bounds=BOUNDS,
        height_system="ellipsoidal",
        earth_model="ellipsoid",
        refraction_model="effective_radius",
        refraction_k=0.13,
    )
    gray = mask.astype(np.uint8) * 255
    return np.dstack((gray, gray, gray, np.full_like(gray, 255)))


def _assert_shadow_mask_golden(actual: np.ndarray) -> float:
    golden = Path(__file__).parent / "golden" / "helios" / "shadow_mask.png"
    if os.environ.get("FORGE3D_UPDATE_HELIOS_GOLDENS") == "1":
        forge3d.numpy_to_png(golden, actual)
    expected = forge3d.png_to_numpy(golden)
    mae = float(np.mean(np.abs(actual.astype(np.int16) - expected.astype(np.int16))))
    assert np.array_equal(actual, expected)
    return mae


def test_shadow_mask_golden() -> None:
    if os.environ.get("FORGE3D_RUN_TERRAIN_GOLDENS") != "1":
        pytest.skip("set FORGE3D_RUN_TERRAIN_GOLDENS=1 to run GPU goldens")
    mae = _assert_shadow_mask_golden(_shadow_mask_rgba())
    print(f"HELIOS shadow golden: MAE={mae:.6f}")


def test_shadow_mask_golden_negative_control(monkeypatch) -> None:
    golden = Path(__file__).parent / "golden" / "helios" / "shadow_mask.png"
    before = golden.read_bytes()
    corrupted = forge3d.png_to_numpy(golden)
    corrupted[0, 0, 0] ^= 255
    monkeypatch.setenv("FORGE3D_UPDATE_HELIOS_GOLDENS", "1")
    monkeypatch.delenv("FORGE3D_UPDATE_HELIOS_GOLDENS")
    with pytest.raises(AssertionError):
        _assert_shadow_mask_golden(corrupted)
    assert golden.read_bytes() == before


@pytest.mark.skipif(sys.platform != "win32", reason="DX12/Vulkan comparison is Windows-only")
def test_shadow_mask_is_identical_on_dx12_and_vulkan() -> None:
    if os.environ.get("FORGE3D_RUN_TERRAIN_GOLDENS") != "1":
        pytest.skip("set FORGE3D_RUN_TERRAIN_GOLDENS=1 to run GPU goldens")
    script = """
import hashlib, json, numpy as np, forge3d
from forge3d.geo import SolarTime
from forge3d.terrain import shadow_mask
y, x = np.mgrid[-1.0:1.0:65j, -1.0:1.0:65j]
dem = (1800.0 + 2500.0 * np.exp(-24.0 * (x*x + y*y))).astype(np.float32)
time = SolarTime(utc=(2003,10,17,17,0,0), observer_lat=39.742476,
    observer_lon=-105.1786, observer_elev_m=1830.14, tz_offset_hours=-7.0,
    delta_t_seconds=67.0, pressure_mbar=820.0, temperature_c=11.0)
mask = shadow_mask(dem, time, bounds=(-105.25,39.70,-105.10,39.80),
    height_system="ellipsoidal", earth_model="ellipsoid",
    refraction_model="effective_radius", refraction_k=0.13)
print(json.dumps({"sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                  "adapter": forge3d.device_probe()}))
"""
    results = {}
    for backend in ("dx12", "vulkan"):
        env = dict(os.environ)
        env.update(
            FORGE3D_DETERMINISTIC="1",
            WGPU_BACKEND=backend,
            WGPU_BACKENDS=backend,
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        results[backend] = json.loads(completed.stdout)
        assert not results[backend]["adapter"]["software_fallback"]
    print(f"HELIOS cross-backend: {results}")
    assert results["dx12"]["adapter"]["backend"].lower() == "dx12"
    assert results["vulkan"]["adapter"]["backend"].lower() == "vulkan"
    assert results["dx12"]["sha256"] == results["vulkan"]["sha256"]


def test_curved_shadow_memory_matches_flat_baseline() -> None:
    if not forge3d.has_gpu():
        pytest.skip("terrain path-tracer memory gate requires a GPU")
    from forge3d.datasets import mini_dem
    from forge3d.path_tracing import hybrid_render_terrain_reference

    dem = mini_dem()[::4, ::4].astype(np.float32)
    dem -= dem.min()
    dem /= max(float(dem.max()), 1e-6)
    spacing = 100.0 / (dem.shape[1] - 1)
    common = dict(
        width=64,
        height=64,
        camera={
            "origin": (0.0, 35.0, 90.0),
            "look_at": (0.0, 5.0, 0.0),
            "up": (0.0, 1.0, 0.0),
            "fov_y": 45.0,
            "exposure": 1.0,
        },
        spacing=(spacing, spacing),
        exaggeration=20.0,
        albedo=(0.55, 0.52, 0.48),
        sun_azimuth_deg=225.0,
        sun_elevation_deg=35.0,
        observer_latitude_deg=46.5,
        observer_longitude_deg=7.5,
        spp=1,
        min_frames=32,
        max_frames=32,
        variance_threshold=1e9,
        seed=7,
    )
    baseline = hybrid_render_terrain_reference(
        dem, earth_model="flat", refraction_model="none", **common
    )
    helios = hybrid_render_terrain_reference(
        dem,
        earth_model="ellipsoid",
        refraction_model="effective_radius",
        **common,
    )
    metrics = {
        name: {
            key: int(result[key])
            for key in (
                "peak_host_visible_bytes",
                "gpu_resource_bytes",
                "minmax_pyramid_bytes",
            )
        }
        for name, result in (("flat_baseline", baseline), ("helios", helios))
    }
    print(f"HELIOS memory: {metrics}")
    for key in ("peak_host_visible_bytes", "gpu_resource_bytes"):
        assert metrics["helios"][key] <= math.ceil(
            metrics["flat_baseline"][key] * 1.05
        )
    assert (
        metrics["helios"]["minmax_pyramid_bytes"]
        == metrics["flat_baseline"]["minmax_pyramid_bytes"]
    )


def test_shadow_mapping_uses_solar_time_defaults() -> None:
    result = shadow_tip(
        _peak_dem(),
        39.75,
        -105.175,
        {
            "utc": (2003, 10, 17, 17, 0, 0),
            "observer_lat": 39.742476,
            "observer_lon": -105.1786,
        },
        bounds=BOUNDS,
        height_system="ellipsoidal",
    )
    assert result["length_m"] > 0.0


def test_shadow_mask_custom_sphere_marches_to_its_own_footprint_edge() -> None:
    mask = shadow_mask(
        _peak_dem(),
        TIME,
        bounds=BOUNDS,
        height_system="ellipsoidal",
        earth_model="sphere",
        sphere_radius_m=63_710_088.0,
        refraction_model="none",
    )
    assert np.any(~mask)


def test_shadow_apis_require_explicit_height_datum() -> None:
    with pytest.raises(TypeError):
        shadow_mask(_peak_dem(), TIME, bounds=BOUNDS)
    with pytest.raises(TypeError):
        shadow_tip(_peak_dem(), 39.75, -105.175, TIME, bounds=BOUNDS)


def test_shadow_mask_solves_solar_direction_per_dem_cell() -> None:
    equinox_noon = SolarTime(
        utc=(2024, 3, 20, 12, 0, 0),
        observer_lat=0.0,
        observer_lon=0.0,
        pressure_mbar=1013.25,
        temperature_c=15.0,
    )
    mask = shadow_mask(
        np.zeros((2, 2), dtype=np.float32),
        equinox_noon,
        bounds=(-179.0, -1.0, -1.0, 1.0),
        height_system="ellipsoidal",
        earth_model="flat",
        refraction_model="none",
    )
    assert not mask[:, 0].any()  # Local night near 135° W.
    assert mask[:, 1].all()  # Local morning near 45° W.


def test_viewer_set_sun_time_sends_spa_direction_and_source() -> None:
    handle = object.__new__(ViewerHandle)
    sent: list[dict[str, object]] = []
    handle._send_command = lambda command: sent.append(command) or {"ok": True}  # type: ignore[method-assign]

    handle.set_sun_time(TIME)

    solar = TIME.position()
    assert sent == [
        {
            "cmd": "lit_sun",
            "azimuth_deg": pytest.approx(solar["azimuth_deg"]),
            "elevation_deg": pytest.approx(solar["apparent_elevation_deg"]),
        },
        {
            "cmd": "set_terrain_sun",
            "azimuth_deg": pytest.approx(solar["azimuth_deg"]),
            "elevation_deg": pytest.approx(solar["apparent_elevation_deg"]),
            "intensity": 1.0,
            "source": "solar_time",
        }
    ]


def test_viewer_set_sun_updates_generic_and_terrain_lighting() -> None:
    handle = object.__new__(ViewerHandle)
    sent: list[dict[str, object]] = []
    handle._send_command = lambda command: sent.append(command) or {"ok": True}  # type: ignore[method-assign]
    handle.set_sun(123.0, 45.0)
    assert sent == [
        {"cmd": "lit_sun", "azimuth_deg": 123.0, "elevation_deg": 45.0},
        {
            "cmd": "set_terrain_sun",
            "azimuth_deg": 123.0,
            "elevation_deg": 45.0,
            "intensity": 1.0,
            "source": "manual_angles",
        },
    ]


def test_set_terrain_angles_reset_solar_time_provenance() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "viewer" / "cmd" / "terrain_command.rs"
    ).read_text(encoding="utf-8")
    assert "if sun_azimuth.is_some() || sun_elevation.is_some()" in source
    assert 'terrain.sun_source = "manual_angles".to_string();' in source


def test_native_shadow_symbols_are_registered() -> None:
    from forge3d import _forge3d

    assert hasattr(_forge3d, "terrain_shadow_mask")
    assert hasattr(_forge3d, "terrain_shadow_tip")
    mask = _forge3d.terrain_shadow_mask(
        _peak_dem(), TIME, BOUNDS, "ellipsoidal"
    )
    tip = _forge3d.terrain_shadow_tip(
        _peak_dem(),
        39.75,
        -105.175,
        {
            "utc": (2003, 10, 17, 17, 0, 0),
            "observer_lat": 39.75,
            "observer_lon": -105.175,
        },
        BOUNDS,
        "ellipsoidal",
    )
    assert np.asarray(mask).shape == (65, 65)
    assert tip["length_m"] > 0.0
    alias_time = SolarTime(
        utc=TIME.utc,
        observer_lat=TIME.observer_lat,
        observer_lon=TIME.observer_lon,
        delta_t_seconds=69.0,
        delta_t=10.0,
    )
    alias_tip = _forge3d.terrain_shadow_tip(
        _peak_dem(),
        39.75,
        -105.175,
        alias_time,
        BOUNDS,
        "ellipsoidal",
    )
    expected = solar_position(alias_time.utc, 39.75, -105.175, alias_tip["peak_height_m"], delta_t_seconds=10.0)
    assert alias_tip["solar_apparent_elevation_deg"] == pytest.approx(
        expected["apparent_elevation_deg"], abs=1e-10
    )
