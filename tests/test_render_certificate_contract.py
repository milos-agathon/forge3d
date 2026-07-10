from __future__ import annotations

import inspect

import forge3d as f3d
from forge3d import path_tracing
from forge3d.helpers.offscreen import render_offscreen_rgba
from forge3d.sdf import HybridRenderer
from forge3d.vector import VectorScene


def test_public_render_entrypoints_expose_certificate_keyword() -> None:
    entrypoints = {
        "TerrainRenderer.render_terrain_pbr_pom": f3d.TerrainRenderer.render_terrain_pbr_pom,
        "TerrainRenderer.render_with_aov": f3d.TerrainRenderer.render_with_aov,
        "render_adjudication_pair": f3d.render_adjudication_pair,
        "hybrid_render_terrain_reference": f3d.hybrid_render_terrain_reference,
        "render_offscreen_rgba": render_offscreen_rgba,
        "PathTracer.render_rgba": path_tracing.PathTracer.render_rgba,
        "PathTracer.render_progressive": path_tracing.PathTracer.render_progressive,
        "path_tracing.render_aovs": path_tracing.render_aovs,
        "path_tracing.render_rgba": path_tracing.render_rgba,
        "path_tracing.hybrid_render_terrain_reference": path_tracing.hybrid_render_terrain_reference,
        "HybridRenderer.render_sdf_scene": HybridRenderer.render_sdf_scene,
        "VectorScene.render_oit": VectorScene.render_oit,
        "VectorScene.render_pick_map": VectorScene.render_pick_map,
        "VectorScene.render_oit_and_pick": VectorScene.render_oit_and_pick,
        "vector_render_oit_py": f3d.vector_render_oit_py,
        "vector_render_oit_edl_py": f3d.vector_render_oit_edl_py,
        "vector_render_pick_map_py": f3d.vector_render_pick_map_py,
        "vector_render_oit_and_pick_py": f3d.vector_render_oit_and_pick_py,
    }
    missing = [
        name
        for name, entrypoint in entrypoints.items()
        if "certificate" not in inspect.signature(entrypoint).parameters
    ]
    assert not missing, f"render entrypoints missing certificate= contract: {missing}"


def test_vector_render_certificate_is_fresh_and_exact():
    if not f3d.has_gpu() or not f3d.is_weighted_oit_available():
        import pytest

        pytest.skip("weighted OIT GPU path unavailable")

    from forge3d.diagnostics import render_certificate

    scene = VectorScene()
    scene.add_point(0.0, 0.0)
    rgba = scene.render_oit(16, 16, certificate=True)
    cert = render_certificate(sign=False)

    assert rgba.shape == (16, 16, 4)
    assert [entry["label"] for entry in cert["passes"]] == [
        "vector.oit",
        "vector.oit.compose",
    ]
    assert set(cert["engine"]["wgsl_module_hashes"]) == {
        "point_instanced.wgsl",
        "vf.Vector.OIT.Compose",
    }
