from __future__ import annotations

import json
from types import SimpleNamespace


def _recorded_pairs() -> set[tuple[str, str]]:
    from forge3d import _degradation

    return {(entry["kind"], entry["name"]) for entry in _degradation.snapshot()}


def test_all_lighting_no_effect_paths_record_degradation(monkeypatch):
    from forge3d import _degradation, lighting

    _degradation.clear()
    monkeypatch.setattr(lighting, "_forge3d", SimpleNamespace())

    lighting.set_exposure_stops(1.0)
    controller = lighting.CsmController()
    controller.enable_shadows()
    controller.set_light_direction((1.0, -1.0, 0.0))
    controller.configure_pcf(3)
    controller.set_bias_parameters(0.01, 0.02, 0.001)
    controller.set_debug_mode(1)
    controller.get_cascade_info()
    controller.validate_peter_panning_prevention()

    names = {name for kind, name in _recorded_pairs() if kind == "native_setter_unavailable"}
    assert names == {
        "lighting.configure_csm",
        "lighting.get_csm_cascade_info",
        "lighting.set_csm_bias_params",
        "lighting.set_csm_debug_mode",
        "lighting.set_csm_enabled",
        "lighting.set_csm_light_direction",
        "lighting.set_csm_pcf_kernel",
        "lighting.set_exposure_scale",
        "lighting.validate_csm_peter_panning",
    }


def test_sdf_cpu_render_records_render_local_fallback():
    from forge3d import _degradation
    from forge3d.diagnostics import render_certificate
    from forge3d.sdf import HybridRenderer, SdfScene, SdfPrimitive

    _degradation.clear()
    scene = SdfScene()
    scene.add_primitive(SdfPrimitive.sphere((0.0, 0.0, 0.0), 0.5))
    image = HybridRenderer(8, 8).render_sdf_scene(scene, certificate=True)

    assert image.shape == (8, 8, 4)
    cert = render_certificate(sign=False)
    assert cert["adapter"]["backend"] == "cpu"
    assert any(
        entry["kind"] == "cpu_fallback" and entry["name"] == "sdf.render"
        for entry in cert["degradations"]
    )


def test_building_placeholder_records_empty_geometry(monkeypatch, tmp_path):
    from forge3d import _degradation, buildings

    _degradation.clear()
    monkeypatch.setattr(buildings, "_NATIVE", None)
    monkeypatch.setattr(buildings, "_check_pro_access", lambda _feature: None)
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": "one"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
            }
        ],
    }
    path = tmp_path / "buildings.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")
    layer = buildings.add_buildings(path)

    assert layer.buildings[0].vertex_count == 0
    assert ("empty_geometry", "buildings.extrusion") in _recorded_pairs()
