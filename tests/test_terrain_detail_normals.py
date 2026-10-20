"""DEM detail normals: native boundary, terrain UV orientation and default parity."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d.terrain_params import (
    DetailSettings, PomSettings, TriplanarSettings,
    make_terrain_params_config,
)
from _terrain_runtime import terrain_rendering_available


def build_config(detail=None, *, azimuth=90.0, camera_mode="screen", cam_radius=4.0, cam_theta_deg=55.0):
    white = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#ffffff"), (1.0, "#ffffff")], domain=(0.0, 1.0),
    )
    config = make_terrain_params_config(
        size_px=(64, 64), render_scale=1.0, terrain_span=6.0,
        msaa_samples=1, z_scale=1.0, exposure=1.0, domain=(0.0, 1.0),
        albedo_mode="colormap", colormap_strength=1.0,
        overlays=[f3d.OverlayLayer.from_colormap1d(white, strength=1.0)],
        ibl_enabled=False, ibl_intensity=0.0,
        light_azimuth_deg=azimuth, light_elevation_deg=30.0, sun_intensity=1.0,
        camera_mode=camera_mode, cam_radius=cam_radius, cam_phi_deg=140.0,
        cam_theta_deg=cam_theta_deg, fov_y_deg=50.0,
        triplanar=TriplanarSettings(6.0, 4.0, 0.0),
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        detail=detail, aa_seed=7,
    )
    config.shadows.enabled = False
    return config


def write_dem_map(path: Path, residual: np.ndarray) -> None:
    # Exercise the production DEM-residual encoder, including channel signs.
    source = Path(__file__).resolve().parents[1] / "scripts" / "detail_normals.py"
    spec = importlib.util.spec_from_file_location("detail_normals_encoder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.save_normal_map(module.detail_to_normal_map(residual), path)


def write_ridge_map(path: Path) -> None:
    rows, cols = np.mgrid[:32, :32].astype(np.float32)
    write_dem_map(path, -np.abs(cols - 15.5) - 0.5 * np.abs(rows - 15.5))


def write_ramp_map(path: Path, axis: str) -> None:
    rows, cols = np.mgrid[:32, :32].astype(np.float32)
    write_dem_map(path, -cols if axis == "east" else rows)


def active_detail(path: Path, *, strength=1.0, enabled=True, fade_start=100.0, fade_end=200.0):
    return DetailSettings(
        enabled=enabled, normal_strength=0.0, albedo_noise=0.0,
        fade_start=fade_start, fade_end=fade_end,
        detail_normal_path=str(path), detail_strength=strength,
    )


def test_detail_settings_public_and_serialized_contract():
    settings = active_detail(Path("ridge.png"))
    assert f3d.DetailSettings is DetailSettings
    assert "DetailSettings" in f3d.__all__
    values = asdict(settings)
    assert values["detail_normal_path"] == "ridge.png"
    assert values["detail_strength"] == 1.0
    assert values["detail_sigma_px"] == 3.0


@pytest.mark.parametrize("field,value", [
    ("detail_normal_path", ""), ("detail_normal_path", 42),
    ("detail_strength", float("nan")), ("detail_strength", float("inf")),
    ("detail_strength", -0.1), ("detail_strength", 1.1),
    ("detail_sigma_px", float("nan")), ("detail_sigma_px", float("inf")),
    ("detail_sigma_px", 0.0),
])
def test_detail_settings_validate_python_and_native(field, value):
    with pytest.raises((ValueError, TypeError), match=field):
        DetailSettings(**{field: value})
    config = build_config()
    setattr(config.detail, field, value)  # Bypass dataclass validation.
    with pytest.raises(ValueError, match=field):
        f3d.TerrainRenderParams(config)


def test_active_dem_detail_requires_a_path():
    with pytest.raises(ValueError, match="detail_normal_path"):
        DetailSettings(enabled=True, detail_strength=1.0)
    config = build_config()
    config.detail.enabled = True
    config.detail.detail_strength = 1.0
    with pytest.raises(ValueError, match="detail_normal_path"):
        f3d.TerrainRenderParams(config)


@pytest.fixture(scope="module")
def gpu_render(tmp_path_factory):
    if not terrain_rendering_available():
        pytest.skip("DEM detail-normal render requires a physical terrain-safe GPU")
    probe = f3d.device_probe()
    assert str(probe["device_type"]).lower() in {"discretegpu", "integratedgpu"}, probe
    print(f"DEM detail-normal physical adapter: {probe}")
    root = tmp_path_factory.mktemp("detail_normal_gpu")
    hdr = root / "env.hdr"
    with hdr.open("wb") as handle:
        handle.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n")
        handle.write(bytes([128, 128, 128, 128]) * 4)
    ibl = f3d.IBL.from_hdr(str(hdr), intensity=0.0)
    renderer = f3d.TerrainRenderer(f3d.Session(window=False))
    materials = f3d.MaterialSet.terrain_default(normal_strength=0.0)
    heightmap = np.full((32, 32), 0.5, dtype=np.float32)

    def render(detail=None, *, azimuth=90.0, camera_mode="screen", entry_point="beauty",
               cam_radius=4.0, cam_theta_deg=55.0):
        entry = (renderer.render_with_aov if entry_point == "aov"
                 else renderer.render_terrain_pbr_pom)
        result = entry(
            material_set=materials, env_maps=ibl,
            params=f3d.TerrainRenderParams(build_config(
                detail, azimuth=azimuth, camera_mode=camera_mode,
                cam_radius=cam_radius, cam_theta_deg=cam_theta_deg,
            )), heightmap=heightmap,
        )
        frame = result[0] if entry_point == "aov" else result
        return np.asarray(frame.to_numpy()).copy()

    return render


@pytest.mark.parametrize("axis,facing,opposite,orthogonal", [
    ("east", 0.0, 180.0, (90.0, 270.0)),
    ("north", 270.0, 90.0, (0.0, 180.0)),
])
@pytest.mark.parametrize("entry_point", ["beauty", "aov"])
def test_dem_ramp_orientation_under_four_sun_directions(tmp_path, gpu_render, axis, facing, opposite, orthogonal, entry_point):
    # Screen mode covers only UV [0, .5]^2. A constant tilt is invariant to
    # that crop; comparing all four suns detects axis swaps and sign flips.
    path = tmp_path / f"{axis}.png"
    write_ramp_map(path, axis)
    brightness = {
        azimuth: gpu_render(active_detail(path), azimuth=azimuth, entry_point=entry_point)[..., :3].mean()
        for azimuth in (0.0, 90.0, 180.0, 270.0)
    }
    print(f"{entry_point} {axis} ramp brightness: {brightness}")
    for azimuth in orthogonal:
        # One RGB8 output step is the quantization floor. Sub-byte mean-only
        # differences (such as the former 0.02 north margin) cannot pass.
        assert brightness[facing] > brightness[azimuth] + 1.0, brightness
        assert brightness[azimuth] > brightness[opposite] + 1.0, brightness


@pytest.mark.parametrize("azimuth,axis", [(0.0, "east"), (180.0, "west"), (270.0, "north"), (90.0, "south")])
@pytest.mark.parametrize("entry_point", ["beauty", "aov"])
def test_asymmetric_dem_ridge_lights_the_sun_facing_side(tmp_path, gpu_render, azimuth, axis, entry_point):
    path = tmp_path / "ridge.png"
    write_ridge_map(path)
    # Nadir Z-up mesh mode shows the entire map, with east at the right and
    # north at the bottom. Radius 8 puts the whole 6-unit terrain in view.
    camera = dict(camera_mode="mesh:zup", cam_radius=8.0, cam_theta_deg=0.0,
                  azimuth=azimuth, entry_point=entry_point)
    baseline = gpu_render(**camera)
    mapped = gpu_render(active_detail(path), **camera)
    assert not np.array_equal(mapped, baseline)
    # Compare terrain interiors, excluding the sky border and ridge seam.
    brightness = mapped[..., :3].astype(np.float32).mean(axis=2)
    sides = {"east": brightness[16:48, 40:48].mean(),
             "west": brightness[16:48, 16:24].mean(),
             "north": brightness[40:48, 16:48].mean(),
             "south": brightness[16:24, 16:48].mean()}
    opposite = {"east": "west", "west": "east", "north": "south", "south": "north"}[axis]
    print(f"{entry_point} ridge sun={axis}: {sides}")
    assert sides[axis] > sides[opposite] + 1.0, sides


@pytest.mark.parametrize("entry_point", ["beauty", "aov"])
def test_dem_detail_distance_fade(tmp_path, gpu_render, entry_point):
    path = tmp_path / "east.png"
    write_ramp_map(path, "east")
    camera = dict(azimuth=0.0, camera_mode="mesh:zup", cam_radius=8.0,
                  cam_theta_deg=0.0, entry_point=entry_point)
    baseline = gpu_render(**camera)
    full = gpu_render(active_detail(path), **camera)
    # The flat terrain is at Z=.5 and the nadir camera at Z=8: every terrain
    # fragment is beyond 4, and the central patch lies within (0, 16).
    faded = gpu_render(active_detail(path, fade_start=0.0, fade_end=4.0), **camera)
    partial = gpu_render(active_detail(path, fade_start=0.0, fade_end=16.0), **camera)
    np.testing.assert_array_equal(faded, baseline)
    mean = lambda frame: frame[24:40, 24:40, :3].mean()
    print(f"{entry_point} fade: baseline={mean(baseline)}, partial={mean(partial)}, full={mean(full)}")
    assert mean(baseline) + 1.0 < mean(partial) < mean(full) - 1.0


@pytest.mark.parametrize("camera_mode", ["screen", "mesh:zup"])
def test_zero_strength_and_disabled_detail_are_byte_identical(tmp_path, gpu_render, camera_mode):
    path = tmp_path / "ridge.png"
    write_ridge_map(path)
    baseline = gpu_render(camera_mode=camera_mode)
    zero = gpu_render(active_detail(path, strength=0.0), camera_mode=camera_mode)
    disabled = gpu_render(active_detail(path, enabled=False), camera_mode=camera_mode)
    np.testing.assert_array_equal(zero, baseline)
    np.testing.assert_array_equal(disabled, baseline)


@pytest.mark.parametrize("entry_point", ["beauty", "aov"])
def test_dem_detail_changes_mesh_shading(tmp_path, gpu_render, entry_point):
    path = tmp_path / "ridge.png"
    write_ridge_map(path)
    baseline = gpu_render(camera_mode="mesh:zup", entry_point=entry_point)
    mapped = gpu_render(active_detail(path), camera_mode="mesh:zup", entry_point=entry_point)
    assert not np.array_equal(mapped, baseline)


def test_unreadable_active_detail_map_raises(tmp_path, gpu_render):
    missing = tmp_path / "missing.png"
    np.testing.assert_array_equal(
        gpu_render(active_detail(missing, strength=0.0)), gpu_render(),
    )
    with pytest.raises(RuntimeError, match="detail_normal_path"):
        gpu_render(active_detail(missing))
