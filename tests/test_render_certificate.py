# tests/test_render_certificate.py
# CENSOR Task 11: end-to-end RenderCertificate coverage over a real native
# render. Exercises the live-pass ledger, the merged degradation list, the
# Ed25519 seal, and the determinism guarantee of the signed payload.
# RELEVANT FILES: python/forge3d/diagnostics.py, python/forge3d/certificate.py,
# src/core/certificate.rs, tests/test_recipe_goldens.py

from __future__ import annotations

import pytest

import forge3d as f3d
from forge3d.diagnostics import render_certificate

from _terrain_runtime import terrain_rendering_available
from test_recipe_goldens import RECIPE_GOLDENS


def _skip_without_terrain() -> None:
    if not terrain_rendering_available():
        pytest.skip("RenderCertificate tests require a terrain-capable forge3d runtime")


def _clear_sinks() -> None:
    from forge3d._forge3d import clear_native_degradations
    from forge3d import _degradation

    clear_native_degradations()
    _degradation.clear()


def _build_and_render(tmp_path):
    """Build RECIPE_GOLDENS[0] and drive one native render, returning the scene."""
    spec = RECIPE_GOLDENS[0]
    scene = spec.build(tmp_path)
    scene.render()
    assert scene.last_render_backend == "gpu_terrain"
    return scene


def test_certificate_has_live_passes_and_empty_degradations(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    _build_and_render(tmp_path)

    cert = render_certificate()
    assert cert["schema"] == "forge3d.render_certificate/1"

    granted = set(cert["capabilities"]["granted"])
    timed = [p for p in cert["passes"] if p["gpu_ms"] > 0]
    if "timestamp_query" in granted:
        assert len(timed) >= 5, cert["passes"]
    else:
        assert any(d["name"] == "timestamp_query" for d in cert["degradations"])

    # A healthy render records no degradations except the (expected) absence of
    # a negotiated capability such as timestamp_query.
    assert cert["degradations"] == [] or "timestamp_query" not in granted

    assert cert["signature"]["alg"] == "ed25519"


def test_signed_payload_deterministic_across_two_renders(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    from forge3d.certificate import payload_sha256

    spec = RECIPE_GOLDENS[0]
    scene = spec.build(tmp_path)

    hashes = []
    for _ in range(2):
        scene.render()
        hashes.append(payload_sha256(render_certificate()))

    assert hashes[0] == hashes[1], (
        "signed certificate payload must be byte-identical across two renders "
        f"of the same scene on the same adapter: {hashes}"
    )


def test_certificate_excludes_shaders_owned_by_another_renderer(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    # Compile Scene-owned pipelines first. A terrain certificate must not copy
    # these labels merely because they exist in the same process-wide cache.
    f3d.Scene(2, 2)
    _build_and_render(tmp_path)

    hashes = render_certificate(sign=False)["engine"]["wgsl_module_hashes"]
    assert hashes, "terrain render must report its renderer-owned WGSL modules"
    unrelated = {
        "mesh_basic_shader",
        "overlays_shader",
        "ssao-compute",
        "text_overlay_shader",
    }
    assert unrelated.isdisjoint(hashes), hashes


def test_scene_certificate_reports_only_scene_owned_shaders(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    _build_and_render(tmp_path)
    scene = f3d.Scene(2, 2)
    scene.render_rgba()

    hashes = render_certificate(sign=False)["engine"]["wgsl_module_hashes"]
    assert hashes, "Scene render must report its renderer-owned WGSL modules"
    assert "terrain_pbr_pom.shader" not in hashes, hashes


def test_certificate_kwarg_writes_signed_file(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    from forge3d import certificate as _certificate

    spec = RECIPE_GOLDENS[0]
    scene = spec.build(tmp_path)
    cert_path = tmp_path / "render_certificate.json"
    scene.render(certificate=cert_path)

    assert cert_path.exists(), "certificate= path must write the signed certificate JSON"

    # The stashed payload sha matches the written certificate, and the written
    # certificate verifies against its embedded dev public key.
    metadata = scene.last_render_metadata or {}
    assert "certificate_payload_sha256" in metadata

    import json

    written = json.loads(cert_path.read_text(encoding="utf-8"))
    assert _certificate.payload_sha256(written) == metadata["certificate_payload_sha256"]

    pubkey = written["signature"]["pubkey"]
    assert _certificate.verify(cert_path, pubkey) is True


def test_certificate_kwarg_false_leaves_metadata_clean(tmp_path):
    _skip_without_terrain()
    _clear_sinks()

    spec = RECIPE_GOLDENS[0]
    scene = spec.build(tmp_path)
    scene.render()  # certificate defaults to False

    metadata = scene.last_render_metadata or {}
    assert "certificate_payload_sha256" not in metadata
