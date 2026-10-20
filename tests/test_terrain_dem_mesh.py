"""Opt-in DEM mesh density, shared by native beauty/AOV/offline rendering."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import forge3d as f3d
from forge3d.terrain_params import (
    AovSettings, DetailSettings, HeightAoSettings, PomSettings,
    ReflectionSettings, make_terrain_params_config,
)
from tests._terrain_runtime import _build_heightmap, _write_test_hdr, terrain_rendering_available


pytestmark = pytest.mark.skipif(
    not terrain_rendering_available(), reason="requires GPU-backed terrain runtime",
)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    hdr = tmp_path_factory.mktemp("dem-mesh") / "environment.hdr"
    _write_test_hdr(hdr)
    session = f3d.Session(window=False)
    return session, f3d.MaterialSet.terrain_default(), f3d.IBL.from_hdr(str(hdr), intensity=1.0)


@pytest.fixture(scope="module")
def renderer(assets):
    return f3d.TerrainRenderer(assets[0])


def config(mode, *, shadows=False, reflection=False):
    settings = make_terrain_params_config(
        size_px=(192, 128), render_scale=1.0, terrain_span=2.8,
        z_scale=1.0, exposure=1.0, domain=(0.0, 1.0),
        camera_mode=mode, msaa_samples=1, ao_weight=0.0,
        cam_radius=5.0, cam_phi_deg=0.0, cam_theta_deg=63.0, clip=(0.1, 100.0),
        aa_samples=1, aa_seed=77,
        pom=PomSettings(False, "Occlusion", 0.0, 1, 1, 0, False, False),
        detail=DetailSettings(enabled=False), height_ao=HeightAoSettings(enabled=False),
        aov=AovSettings(enabled=True, albedo=True, normal=True, depth=True),
        reflection=ReflectionSettings(enabled=reflection, water_plane_height=-0.2),
    )
    settings.shadows = replace(settings.shadows, enabled=shadows, technique="PCF" if shadows else "NONE")
    return settings


def render(renderer, material, ibl, settings, heights, entry_point, water_mask=None):
    params = f3d.TerrainRenderParams(settings)
    assert params.camera_mode == settings.camera_mode
    if entry_point == "beauty":
        return {"beauty": renderer.render_terrain_pbr_pom(
            material, ibl, params, heights, water_mask=water_mask,
        ).to_numpy()}
    if entry_point == "aov":
        frame, aov = renderer.render_with_aov(material, ibl, params, heights, water_mask=water_mask)
        result = {"beauty": frame.to_numpy()}
    else:
        offline = f3d.render_offline(
            renderer, material, ibl, params, heights,
            settings=f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=1),
            water_mask=water_mask,
        )
        aov = offline.aov_frame
        result = {"beauty": offline.frame.to_numpy(), "hdr": offline.hdr_frame.to_numpy_f32()}
    return dict(result, depth=aov.depth(), normal=aov.normal(), albedo=aov.albedo())


@pytest.mark.parametrize("entry_point", ["beauty", "aov", "offline"])
def test_square_512_dem_matches_legacy_with_shadows_disabled(assets, renderer, entry_point):
    _, material, ibl = assets
    heights = _build_heightmap(512)
    old = render(renderer, material, ibl, config("mesh:zup", reflection=True), heights, entry_point)
    dem = render(renderer, material, ibl, config("mesh:zup:dem", reflection=True), heights, entry_point)
    for name in old:
        np.testing.assert_array_equal(dem[name], old[name])


@pytest.mark.parametrize("entry_point", ["beauty", "aov", "offline"])
@pytest.mark.parametrize("shape,axis,ridge,phi", [
    ((1025, 1025), 1, 512, 0.0),
    ((1025, 700), 1, 343, 0.0),
    ((700, 1025), 0, 343, 90.0),
])
def test_dem_mesh_resolves_ridge_between_legacy_vertices(
    assets, renderer, entry_point, shape, axis, ridge, phi,
):
    _, material, ibl = assets
    # On rectangular inputs, a square max-axis grid also misses this ridge.
    heights = np.zeros(shape, dtype=np.float32)
    if axis == 1:
        heights[:, ridge] = 1.0
    else:
        heights[ridge, :] = 1.0
    legacy_settings = replace(config("mesh:zup"), cam_phi_deg=phi)
    dem_settings = replace(legacy_settings, camera_mode="mesh:dem:zup")
    old = render(renderer, material, ibl, legacy_settings, heights, entry_point)
    dem = render(renderer, material, ibl, dem_settings, heights, entry_point)
    assert not np.array_equal(dem["beauty"], old["beauty"])
    if "depth" in dem:
        # More than shading changes: the ridge reaches pixels outside the old silhouette.
        assert np.any((old["depth"] == 0) & (dem["depth"] > 0))
        # An independent CPU rasterizer projects every full-resolution DEM triangle.
        reference = f3d.TerrainRenderParams(legacy_settings).project_terrain_depth(
            heights, (192, 128), None,
        ) < 1.0
        old_error = np.count_nonzero((old["depth"] > 0) != reference)
        dem_error = np.count_nonzero((dem["depth"] > 0) != reference)
        assert dem_error < old_error
        assert dem_error == 0
    # Changing density must not leak into the next legacy render.
    again = render(renderer, material, ibl, legacy_settings, heights, entry_point)
    for name in old:
        np.testing.assert_array_equal(again[name], old[name])


@pytest.mark.parametrize("shape", [(65, 65), (700, 1025), (1025, 700)])
def test_dem_mesh_shadow_and_reflection_paths(assets, renderer, shape):
    session, material, ibl = assets
    heights = np.ascontiguousarray(_build_heightmap(max(shape))[:shape[0], :shape[1]])
    settings = config("mesh:zup:dem", shadows=True, reflection=True)
    water = np.zeros(shape, dtype=np.float32)
    water[shape[0] // 3:2 * shape[0] // 3, shape[1] // 3:2 * shape[1] // 3] = 1.0
    first = render(renderer, material, ibl, settings, heights, "aov", water)
    fresh = render(f3d.TerrainRenderer(session), material, ibl, settings, heights, "aov", water)
    for name, values in first.items():
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values, fresh[name])


def test_dem_suffix_preserves_legacy_orbit(assets, renderer):
    _, material, ibl = assets
    heights = _build_heightmap(512)
    old = render(renderer, material, ibl, config("mesh"), heights, "beauty")
    dem = render(renderer, material, ibl, config("mesh:dem"), heights, "beauty")
    np.testing.assert_array_equal(dem["beauty"], old["beauty"])


@pytest.mark.parametrize("shape", [(1, 65), (65, 1)])
def test_dem_mesh_singleton_axis_has_finite_geometry(assets, renderer, shape):
    _, material, ibl = assets
    heights = np.ascontiguousarray(_build_heightmap(65)[:shape[0], :shape[1]])
    arrays = render(renderer, material, ibl, config("mesh:zup:dem"), heights, "aov")
    assert np.any(arrays["depth"] > 0)
    for values in arrays.values():
        assert np.isfinite(values).all()


def test_shadow_cache_separates_dem_density(assets, renderer, tmp_path):
    _, material, ibl = assets
    heights = _build_heightmap(65)
    cache = tmp_path / "store"

    def draw(mode, use_cache=True):
        params = f3d.TerrainRenderParams(config(mode, shadows=True))
        frame = renderer.render_terrain_pbr_pom(
            material, ibl, params, heights, cache=cache if use_cache else None,
        )
        return frame.to_numpy(), dict(renderer.last_anamnesis_cache_report)

    legacy, _ = draw("mesh:zup")
    repeated, report = draw("mesh:zup")
    np.testing.assert_array_equal(repeated, legacy)
    assert "terrain.shadow" in report["hits"]
    dem, report = draw("mesh:zup:dem")
    assert "terrain.shadow" in report["misses"]
    fresh, _ = draw("mesh:zup:dem", use_cache=False)
    np.testing.assert_array_equal(dem, fresh)
    repeated, report = draw("mesh:zup:dem")
    np.testing.assert_array_equal(repeated, dem)
    assert "terrain.shadow" in report["hits"]
    restored, report = draw("mesh:zup")
    np.testing.assert_array_equal(restored, legacy)
    assert "terrain.shadow" in report["hits"]
