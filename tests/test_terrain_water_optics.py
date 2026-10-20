"""Public water inputs and physical-GPU water optics regression checks."""
from dataclasses import asdict

import numpy as np
import pytest

import forge3d as f3d
from forge3d._canonical_json import canonical_json_bytes
from forge3d.terrain_params import WaterOpticsSettings, WaterSettings


def test_public_water_optics_contract():
    assert f3d.WaterOpticsSettings is WaterOpticsSettings
    assert "WaterOpticsSettings" in f3d.__all__
    assert WaterSettings().optics is None
    assert WaterOpticsSettings().mode == "physical"
    assert WaterOpticsSettings(mode="procedural").mode == "procedural"


def test_water_surface_snapshot_is_serializable_and_owns_input():
    height = np.ones((3, 4), dtype=np.float32)
    normals = np.zeros((3, 4, 3), dtype=np.float32)
    normals[..., 2] = 1
    settings = WaterOpticsSettings(surface_height=height, surface_normals=normals)
    before = canonical_json_bytes(asdict(settings), error_context="water")
    expected = dict(mode="physical", surface_height=height.tolist(),
                    surface_normals=normals.tolist(), color=None,
                    roughness=float(np.float32(0.02)))
    assert before == canonical_json_bytes(expected, error_context="water")
    height[:] = 2
    normals[:] = 0
    assert canonical_json_bytes(asdict(settings), error_context="water") == before
    assert asdict(WaterSettings(optics=asdict(settings)).optics) == asdict(settings)
    assert np.asarray(settings.surface_height).nbytes == height.nbytes
    assert np.asarray(settings.surface_normals).nbytes == normals.nbytes
    with pytest.raises(ValueError):
        np.asarray(settings.surface_normals).flags.writeable = True


def _gpu_module():
    from _terrain_runtime import terrain_rendering_available
    if not terrain_rendering_available():
        pytest.skip("water optics requires the existing terrain-safe GPU capability gate")
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "examples/terrain_water_optics.py"
    spec = importlib.util.spec_from_file_location("water_optics_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu_output(tmp_path, name):
    import os
    from pathlib import Path
    root = os.environ.get("FORGE3D_VISUAL_ARTIFACT_DIR")
    output = Path(root) / "water-optics" / name if root else tmp_path
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.mark.offscreen
def test_gpu_water_optics_analytic_regression(tmp_path):
    module = _gpu_module()
    report = module.main(_gpu_output(tmp_path, "analytic"))
    assert report["measurements"]["flat"]["reflection_mask_error_px"] == 0
    assert report["measurements"]["terrain_shadow"]["unshadowed_glint_delta_rgb"] != [0, 0, 0]


@pytest.mark.offscreen
def test_gpu_water_lighting_and_proof_exclusion(tmp_path):
    import json
    from forge3d import verify
    module = _gpu_module()
    output = _gpu_output(tmp_path, "lighting")
    fixture = module.Fixture(output)
    color = (0.24, 0.13, 0.055)
    params = fixture.params(sun=0, color=color, roughness=0.8)
    params.ibl.enabled = False
    dark, layers = fixture.render(params, aov=True)
    foreground = np.linalg.norm(layers.normal(), axis=-1) > 0
    assert np.all(dark[foreground, :3] == 0), "water must have no emissive body colour"
    verification = verify.shader_report()
    assert verification["status"] == "ok", "the preserved static proof domain must still pass"
    runtime = verification["runtime_assert"]
    assert runtime["feature_enabled"], "regression lane requires shader-contract-asserts"
    assert runtime["status"] == "excluded" and runtime["observed_inputs"]
    entry = next(e for e in runtime["checked_entries"] if e["module"] == "terrain_pbr_pom")
    assert entry["status"] == "excluded" and "physical water" in entry["exclusion_reason"]
    checks = {c["name"]: c for c in entry["checked_bindings"]}
    assert checks["water_reflection_uniforms.optics.x"]["observed_max"] == 1
    assert checks["water_reflection_uniforms.optics.x"]["status"] == "excluded"
    params.ibl.enabled = True
    sky = fixture.render(params)
    params.water.optics.color = (0, 0, 0)
    sky_black = fixture.render(params)
    sky_body = sky[..., :3].astype(int) - sky_black[..., :3].astype(int)
    assert np.any(sky_body[foreground] > 0), "sky irradiance must light the body colour"
    params.ibl.enabled = False
    params.light.intensity = 2.0
    params.water.optics.color = color
    sun = fixture.render(params)
    params.water.optics.color = (0, 0, 0)
    sun_black = fixture.render(params)
    sun_body = sun[..., :3].astype(int) - sun_black[..., :3].astype(int)
    assert np.any(sun_body[foreground] > 0), "sunlight must light the body colour"

    light = module.normalize([0, np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))])
    x, y = np.round(module.project(module.analytic_glint("flat", light))).astype(int)
    assert np.any(sun_body[y, x] > 0), "body-shadow check needs a lit positive control"
    bed = np.zeros_like(fixture.bed)
    bed[32:40] = 1.9
    mask = np.ones_like(bed)
    mask[32:40] = 0
    params.clamp.height_range = (0.0, 2.0)
    params.cam_target = (0, 0, -0.5)
    shadow_black = fixture.render(params, bed=bed, mask=mask)
    params.water.optics.color = color
    shadow = fixture.render(params, bed=bed, mask=mask)
    shadow_body = shadow[y, x, :3].astype(int) - shadow_black[y, x, :3].astype(int)
    assert np.all(shadow_body == 0), "terrain shadow must suppress the sun-lit body term"
    report = dict(static_proof_status=verification["status"], runtime_assert=runtime,
                  dark_body_max_rgb=int(dark[foreground, :3].max()),
                  sky_body_max_delta=int(sky_body[foreground].max()),
                  sun_body_control_delta_rgb=sun_body[y, x].tolist(),
                  shadowed_body_delta_rgb=shadow_body.tolist())
    (output / "lighting-proof.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


@pytest.mark.offscreen
def test_gpu_water_y_tilt_and_vertical_scale(tmp_path):
    import json
    module = _gpu_module()
    output = _gpu_output(tmp_path, "y-tilt")
    fixture = module.Fixture(output)
    light = module.normalize([0, np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60))])
    normal = module.normalize([0, -0.08, 1])
    reflected = -light + 2 * np.dot(light, normal) * normal
    eye = module.camera()[0]
    distance = (np.dot(normal, eye) - normal[2] * module.LEVEL) / np.dot(normal, reflected)
    expected = module.project(eye - reflected * distance)
    measurements = []
    for scale, normals_only in ((1.0, False), (2.0, False), (2.0, True)):
        height = ((module.LEVEL + 0.08 * fixture.y) / scale).astype(np.float32)
        params = fixture.params(sun=0, color=(0, 0, 0))
        params.z_scale = scale
        params.cam_target = (0, 0, module.LEVEL - 0.5 * scale)
        params.water.optics = WaterOpticsSettings(
            surface_height=None if normals_only else height,
            surface_normals=np.broadcast_to(normal, (module.GRID, module.GRID, 3)) if normals_only else None,
            color=(0, 0, 0), roughness=0.09,
        )
        bed = height if normals_only else fixture.bed
        dark = fixture.render(params, bed=bed)
        params.light.intensity = 0.005
        bright = fixture.render(params, bed=bed)
        delta = bright[..., :3].astype(float).sum(axis=-1) - dark[..., :3].astype(float).sum(axis=-1)
        rows, columns = np.where(delta == delta.max())
        assert delta.max() > 0
        # Exact pixel sampling-cell coverage, rather than an invented error cap.
        assert np.any((np.abs(columns - expected[0]) <= 0.5) & (np.abs(rows - expected[1]) <= 0.5))
        measurements.append(dict(z_scale=scale, normals_only=normals_only,
                                 analytic_glint_px=expected.tolist(),
                                 rendered_peak_centroid_px=[float(columns.mean()), float(rows.mean())],
                                 glint_error_px=float(np.hypot(columns.mean() - expected[0], rows.mean() - expected[1])),
                                 nearest_peak_pixel_error_px=float(np.min(np.hypot(columns - expected[0], rows - expected[1])))))
    (output / "y-tilt.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"roughness": float("nan")}, {"roughness": -0.1},
    {"roughness": 1.1}, {"color": (1, 2, 0)}, {"color": (0, 0)},
    {"surface_height": [[1, 2], [3, float("inf")]]},
    {"surface_height": [[1, 2]]},
    {"surface_normals": np.zeros((2, 2, 3))},
    {"surface_normals": np.ones((2, 2, 4))},
    {"surface_normals": np.tile([1, 0, 0], (2, 2, 1))},
    {"surface_normals": np.tile([0, 0, -1], (2, 2, 1))},
    {"surface_height": np.zeros((2, 3)), "surface_normals": np.ones((2, 2, 3))},
])
def test_invalid_water_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        WaterOpticsSettings(**kwargs)
