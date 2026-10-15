"""Coarse AO is prepared only when the terrain shader consumes binding 12."""
from __future__ import annotations

import numpy as np
import pytest

import forge3d as f3d
from forge3d.diagnostics import render_certificate
from forge3d.terrain_params import HeightAoSettings, make_terrain_params_config
from tests._terrain_runtime import _build_heightmap, _write_test_hdr, terrain_rendering_available


pytestmark = pytest.mark.skipif(
    not terrain_rendering_available(), reason="requires GPU-backed terrain runtime"
)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    hdr = tmp_path_factory.mktemp("coarse-ao") / "environment.hdr"
    _write_test_hdr(hdr)
    session = f3d.Session(window=False)
    return session, f3d.MaterialSet.terrain_default(), f3d.IBL.from_hdr(str(hdr), intensity=1.0), _build_heightmap(64)


def params(ao_weight=0.0, debug_mode=0):
    return f3d.TerrainRenderParams(make_terrain_params_config(
        size_px=(96, 64), render_scale=1.0, terrain_span=2.8,
        z_scale=1.0, exposure=1.0, domain=(0.0, 1.0),
        camera_mode="mesh:zup", msaa_samples=1, ao_weight=ao_weight,
        cam_radius=5.0, cam_theta_deg=63.0, clip=(0.1, 100.0),
        debug_mode=debug_mode, aa_samples=1, aa_seed=77,
        height_ao=HeightAoSettings(enabled=False),
    ))


def render(renderer, material, ibl, config, heights, entry_point):
    if entry_point == "beauty":
        return renderer.render_terrain_pbr_pom(material, ibl, config, heights)
    if entry_point == "aov":
        return renderer.render_with_aov(material, ibl, config, heights)[0]
    return f3d.render_offline(
        renderer, material, ibl, config, heights,
        settings=f3d.OfflineQualitySettings(enabled=True, adaptive=False, batch_size=1),
    ).frame


@pytest.mark.parametrize("entry_point", ["beauty", "aov", "offline"])
@pytest.mark.parametrize("ao_weight,debug_mode,env_debug,required", [
    (0.0, 0, None, False),
    (0.7, 0, None, True),
    (0.0, 28, None, True),
    (0.0, 0, "28", True),
    (0.0, 6, "28", False),  # Explicit parameter takes precedence over the environment.
])
def test_coarse_ao_allocation_follows_shader_consumers(
    assets, monkeypatch, entry_point, ao_weight, debug_mode, env_debug, required,
):
    if env_debug is None:
        monkeypatch.delenv("VF_COLOR_DEBUG_MODE", raising=False)
    else:
        monkeypatch.setenv("VF_COLOR_DEBUG_MODE", env_debug)
    session, material, ibl, heights = assets
    renderer = f3d.TerrainRenderer(session)
    frame = render(renderer, material, ibl, params(ao_weight, debug_mode), heights, entry_point)
    assert np.isfinite(frame.to_numpy()).all()
    labels = render_certificate(sign=False)["allocations"]["by_label"]
    assert ("terrain.coarse_ao" in labels) is required


@pytest.mark.parametrize("entry_point", ["beauty", "aov", "offline"])
def test_coarse_ao_refreshes_after_unused_render(assets, monkeypatch, entry_point):
    monkeypatch.delenv("VF_COLOR_DEBUG_MODE", raising=False)
    session, material, ibl, heights = assets
    renderer = f3d.TerrainRenderer(session)
    debug = params(debug_mode=28)
    # A displaced valley has non-white AO and differs from the first hill.
    # Flat heights cannot distinguish a recomputation from the white fallback.
    changed = np.ascontiguousarray(1.0 - np.roll(heights, 17, axis=1))
    first = render(renderer, material, ibl, debug, heights, entry_point).to_numpy()
    render(renderer, material, ibl, params(), changed, entry_point)
    reused = render(renderer, material, ibl, debug, changed, entry_point).to_numpy()
    fresh = render(f3d.TerrainRenderer(session), material, ibl, debug, changed, entry_point).to_numpy()
    fallback = render(f3d.TerrainRenderer(session), material, ibl, debug, np.zeros_like(changed), entry_point).to_numpy()
    rgb = fresh[..., :3]
    achromatic = (rgb[..., 0] == rgb[..., 1]) & (rgb[..., 1] == rgb[..., 2])
    assert np.any(achromatic & (rgb[..., 0] > 0) & (rgb[..., 0] < 255))
    assert fresh.tobytes() != fallback.tobytes()
    assert fresh.tobytes() != first.tobytes()
    assert reused.tobytes() == fresh.tobytes()
