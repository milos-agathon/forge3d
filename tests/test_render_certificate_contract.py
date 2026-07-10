from __future__ import annotations

import inspect

import numpy as np

import forge3d as f3d
from forge3d import determinism, geometry, path_tracing, terrain_demo
from forge3d.helpers.offscreen import render_offscreen_rgba
from forge3d import sdf
from forge3d.legend import Legend
from forge3d.lighting import RestirDI
from forge3d.map_scene import MapScene
from forge3d.north_arrow import NorthArrow
from forge3d.offline import render_offline
from forge3d.scale_bar import ScaleBar
from forge3d.sdf import HybridRenderer, SdfPrimitive, SdfScene
from forge3d.smoke import SmokeDomain
from forge3d.vector import VectorScene


def test_public_render_entrypoints_expose_certificate_keyword() -> None:
    entrypoints = {
        "TerrainRenderer.render_terrain_pbr_pom": f3d.TerrainRenderer.render_terrain_pbr_pom,
        "TerrainRenderer.render_with_aov": f3d.TerrainRenderer.render_with_aov,
        "render_adjudication_pair": f3d.render_adjudication_pair,
        "hybrid_render_terrain_reference": f3d.hybrid_render_terrain_reference,
        "render_offscreen_rgba": render_offscreen_rgba,
        "Renderer.render_triangle_rgba": f3d.Renderer.render_triangle_rgba,
        "Renderer.render_triangle_png": f3d.Renderer.render_triangle_png,
        "MapScene.render": MapScene.render,
        "render_offline": render_offline,
        "determinism.render_reference": determinism.render_reference,
        "PathTracer.render_rgba": path_tracing.PathTracer.render_rgba,
        "PathTracer.render_progressive": path_tracing.PathTracer.render_progressive,
        "path_tracing.render_aovs": path_tracing.render_aovs,
        "path_tracing.render_rgba": path_tracing.render_rgba,
        "path_tracing.hybrid_render_terrain_reference": path_tracing.hybrid_render_terrain_reference,
        "HybridRenderer.render_sdf_scene": HybridRenderer.render_sdf_scene,
        "sdf.render_simple_scene": sdf.render_simple_scene,
        "Legend.render": Legend.render,
        "NorthArrow.render": NorthArrow.render,
        "ScaleBar.render": ScaleBar.render,
        "RestirDI.render_frame": RestirDI.render_frame,
        "geometry.instance_mesh_gpu_render": geometry.instance_mesh_gpu_render,
        "SmokeDomain.render_rgba": SmokeDomain.render_rgba,
        "SmokeDomain.render_projection_rgba": SmokeDomain.render_projection_rgba,
        "VectorScene.render_oit": VectorScene.render_oit,
        "VectorScene.render_pick_map": VectorScene.render_pick_map,
        "VectorScene.render_oit_and_pick": VectorScene.render_oit_and_pick,
        "vector_render_oit_py": f3d.vector_render_oit_py,
        "vector_render_oit_edl_py": f3d.vector_render_oit_edl_py,
        "vector_render_pick_map_py": f3d.vector_render_pick_map_py,
        "vector_render_oit_and_pick_py": f3d.vector_render_oit_and_pick_py,
        "vector_render_polygons_fill_py": f3d.vector_render_polygons_fill_py,
        "vector_oit_and_pick_demo": f3d.vector_oit_and_pick_demo,
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


def test_cloud_render_keeps_following_vector_allocation_usable():
    if not f3d.has_gpu() or not f3d.is_weighted_oit_available():
        import pytest

        pytest.skip("cloud/vector GPU path unavailable")

    baseline = f3d.Scene(32, 32).render_rgba()
    scene = f3d.Scene(32, 32)
    scene.enable_clouds("low")
    scene.set_realtime_cloud_density(1.0)
    scene.set_realtime_cloud_coverage(1.0)
    scene.set_cloud_render_mode("volumetric")
    cloud_rgba = scene.render_rgba()

    changed = np.any(cloud_rgba[:, :, :3] != baseline[:, :, :3], axis=2)
    assert np.count_nonzero(changed) >= 100

    vector = VectorScene()
    vector.add_point(0.0, 0.0)
    rgba = vector.render_oit(16, 16)

    assert rgba.shape == (16, 16, 4)


def test_native_sdf_render_uses_the_supplied_scene_and_emits_certificate(monkeypatch):
    if not sdf.NATIVE_AVAILABLE or not hasattr(sdf.forge3d_native, "hybrid_render"):
        import pytest

        pytest.skip("native SDF renderer unavailable")

    from forge3d.diagnostics import render_certificate

    monkeypatch.setattr(sdf, "USE_NATIVE_SDF", True)
    scene = SdfScene()
    scene.add_primitive(SdfPrimitive.sphere((0.0, 0.0, 0.0), 1.0, 1))

    rgba = HybridRenderer(17, 17).render_sdf_scene(scene, certificate=True)
    cert = render_certificate(sign=False)

    assert rgba.shape == (17, 17, 4)
    assert rgba[8, 8].tolist() == [204, 51, 51, 255]
    assert [entry["label"] for entry in cert["passes"]] == ["sdf.native_cpu"]
    assert cert["adapter"]["backend"] == "cpu"
    assert cert["engine"]["wgsl_module_hashes"] == {}
    assert cert["degradations"] == []


def test_polygon_fill_certificate_names_only_the_polygon_shader():
    if not f3d.has_gpu():
        import pytest

        pytest.skip("GPU path unavailable")

    from forge3d.diagnostics import render_certificate

    exterior = np.asarray(
        [[-0.8, -0.8], [0.8, -0.8], [0.0, 0.8]], dtype=np.float64
    )
    rgba = f3d.vector_render_polygons_fill_py(
        16,
        16,
        [exterior],
        coordinates_are_ndc=True,
        certificate=True,
    )
    cert = render_certificate(sign=False)

    assert rgba.shape == (16, 16, 4)
    assert [entry["label"] for entry in cert["passes"]] == ["vector.polygon_fill"]
    assert set(cert["engine"]["wgsl_module_hashes"]) == {"polygon_fill.wgsl"}


def test_native_smoke_certificate_uses_cpu_identity():
    from forge3d.diagnostics import render_certificate

    density = np.zeros((3, 3, 3), dtype=np.float32)
    density[1, 1, 1] = 1.0
    domain = SmokeDomain.from_density(density)
    rgba = domain.render_projection_rgba(8, 8, certificate=True)
    cert = render_certificate(sign=False)

    assert rgba.shape == (8, 8, 4)
    assert [entry["label"] for entry in cert["passes"]] == ["smoke.cpu_projection"]
    assert cert["adapter"]["backend"] == "cpu"
    assert cert["engine"]["wgsl_module_hashes"] == {}
    assert cert["degradations"] == []


def test_fallback_renderer_discloses_cpu_degradation():
    from forge3d.diagnostics import render_certificate

    rgba = f3d.Renderer(8, 8).render_triangle_rgba(certificate=True)
    cert = render_certificate(sign=False)

    assert rgba.shape == (8, 8, 4)
    assert [entry["label"] for entry in cert["passes"]] == ["renderer.cpu_triangle"]
    assert cert["adapter"]["backend"] == "cpu"
    assert {(entry["kind"], entry["name"]) for entry in cert["degradations"]} == {
        ("cpu_fallback", "renderer.triangle")
    }


def test_instanced_mesh_certificate_names_only_the_instancing_shader():
    if not f3d.has_gpu() or not geometry.gpu_instancing_available():
        import pytest

        pytest.skip("GPU instancing path unavailable")

    from forge3d.diagnostics import render_certificate

    mesh = geometry.MeshBuffers(
        positions=np.asarray(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float32,
        ),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        indices=np.asarray([[0, 1, 2]], dtype=np.uint32),
    )
    transforms = np.eye(4, dtype=np.float32).reshape(1, 16)

    rgba = geometry.instance_mesh_gpu_render(
        mesh, transforms, 16, 16, certificate=True
    )
    cert = render_certificate(sign=False)

    assert rgba.shape == (16, 16, 4)
    assert [entry["label"] for entry in cert["passes"]] == [
        "geometry.instanced_mesh"
    ]
    assert set(cert["engine"]["wgsl_module_hashes"]) == {"mesh_instanced_shader"}


def test_terrain_sequence_never_substitutes_triangle_placeholder(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(terrain_demo.f3d, "has_gpu", lambda: False)
    with pytest.raises(RuntimeError, match="fallback triangle placeholder has been removed"):
        terrain_demo.render_sunrise_to_noon_sequence(
            dem_path=tmp_path / "missing-dem.tif",
            hdr_path=tmp_path / "missing-env.hdr",
            output_dir=tmp_path / "frames",
        )
