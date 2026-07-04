"""P0 API truth-pass contracts."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_path_tracer_synthetic_output_is_opt_in() -> None:
    from forge3d.path_tracing import ExperimentalSyntheticOutput, PathTracer

    tracer = PathTracer(4, 3)
    with pytest.raises(ExperimentalSyntheticOutput, match="synthetic_ok=True"):
        tracer.render_rgba()

    rgba = tracer.render_rgba(synthetic_ok=True)
    assert rgba.shape == (3, 4, 4)


def test_render_aovs_synthetic_output_is_opt_in() -> None:
    from forge3d.path_tracing import ExperimentalSyntheticOutput, render_aovs

    with pytest.raises(ExperimentalSyntheticOutput, match="synthetic_ok=True"):
        render_aovs(4, 3, scene=None, camera=None)

    aovs = render_aovs(
        4,
        3,
        scene=None,
        camera=None,
        aovs=("albedo", "normal", "depth"),
        synthetic_ok=True,
    )
    assert set(aovs) == {"albedo", "normal", "depth"}


def test_restir_di_warns_that_native_rendering_is_experimental() -> None:
    from forge3d.lighting import RestirDI

    with pytest.warns(UserWarning, match="experimental"):
        restir = RestirDI()

    assert restir.num_lights == 0


def test_terrain_public_docs_drop_stale_sky_and_shadow_claims() -> None:
    from forge3d.terrain_params import ShadowSettings, SkySettings

    sky_doc = SkySettings.__doc__ or ""
    shadow_doc = ShadowSettings.validate_for_terrain.__doc__ or ""

    assert "Rayleigh" not in sky_doc
    assert "Mie" not in sky_doc
    assert "Hosek-Wilkie RGB coefficient-table" in sky_doc
    assert "NOT implemented" not in shadow_doc
    assert "moment_maps binding exists" not in shadow_doc

    repo = Path(__file__).resolve().parents[1]
    for rel in ("python/forge3d/terrain_params.py", "python/forge3d/terrain_demo.py"):
        text = (repo / rel).read_text(encoding="utf-8")
        assert "moment_maps binding exists but is never sampled" not in text
    shader = (repo / "src/shaders/sky.wgsl").read_text(encoding="utf-8")
    assert "full implementation would use the paper's datasets" not in shader
    assert "hosek_coeffs_a_d" in shader


def test_material_texture_modules_state_gpu_boundary() -> None:
    import forge3d.materials as materials
    import forge3d.textures as textures

    assert "does not upload" in (materials.__doc__ or "")
    assert "does not upload" in (textures.__doc__ or "")
