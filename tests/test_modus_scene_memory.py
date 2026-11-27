"""Whole-process admission at the MapScene and flythrough public boundaries."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import forge3d as f3d
from forge3d import mem
from forge3d.chronos import render_flythrough
from _obliqua_gpu_guard import require_qualifying_gpu


def scene(size):
    return f3d.MapScene(
        terrain=f3d.TerrainSource(data=np.zeros((size, size), dtype="f4"), crs="EPSG:32610"),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64),
    )


def test_preflight_precedes_compile_and_frame_output(monkeypatch, tmp_path):
    small, large = scene(64), scene(256)
    output = tmp_path / "render"
    options = dict(camera_path={3: small, 7: large}, base_seed=0, samples=1, out_dir=output)
    single = mem.estimate(dict(entry_point="MapScene.render", scene=large))
    flight = mem.estimate(dict(entry_point="render_flythrough", **options))
    assert flight["contributors"]["compiled_frames"] > 0
    assert flight["contributors"]["terrain_normalize_hash_upload"] == single["contributors"]["terrain_normalize_hash_upload"]
    assert sum(flight["contributors"].values()) == flight["predicted_peak_bytes"]

    def unexpected(*args, **kwargs):
        pytest.fail("refused call reached scene compilation or native work")

    monkeypatch.setattr(f3d.MapScene, "compile_plan", unexpected)
    monkeypatch.setattr("forge3d.chronos._native", unexpected)
    monkeypatch.setattr("forge3d._job_memory.Measurement", unexpected)
    for call in (lambda: large.render(str(output), process_limit_bytes=1),
                 lambda: render_flythrough(**options, process_limit_bytes=1)):
        with pytest.raises(mem.MemoryAdmissionRefused) as exc:
            call()
        assert exc.value.limit_bytes == 1
        assert exc.value.estimate["predicted_peak_bytes"] > 1
        assert not output.exists()


def test_file_backed_float64_geotiff_preflight(tmp_path, monkeypatch):
    from forge3d import gis, io

    path = tmp_path / "height.tif"
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    gis.write_raster(path, values, crs="EPSG:32610", transform=(1, 0, 0, 0, -1, 8))
    recipe = scene(8).recipe
    file_scene = f3d.MapScene(replace(recipe, terrain=replace(recipe.terrain, path=path, data=None)))
    array_scene = f3d.MapScene(replace(recipe, terrain=replace(recipe.terrain, data=values)))
    assert file_scene.recipe.terrain.data is None

    def unexpected(*args, **kwargs):
        pytest.fail("GeoTIFF preflight reached a pixel read or scene compilation")

    monkeypatch.setattr(io, "load_dem", unexpected)
    monkeypatch.setattr(gis, "read_raster", unexpected)
    monkeypatch.setattr(f3d.MapScene, "compile_plan", unexpected)
    term = "terrain_normalize_hash_upload"

    def terrain_bytes(s):
        return mem.estimate(dict(entry_point="MapScene.render", scene=s))["contributors"][term]

    assert terrain_bytes(file_scene) == terrain_bytes(array_scene)
    output = tmp_path / "refused.png"
    with pytest.raises(mem.MemoryAdmissionRefused) as caught:
        file_scene.render(str(output), process_limit_bytes=1)
    assert caught.value.limit_bytes == 1 < caught.value.estimate["predicted_peak_bytes"]
    assert not output.exists()


def test_decoded_vector_features_with_source_path(tmp_path, monkeypatch):
    from test_mapscene_render_png import _public_label_vector_scene

    output = tmp_path / "map.png"
    scene = _public_label_vector_scene(output)
    vector, labels = scene.recipe.layers
    assert len(vector.features) == 2
    vector = replace(vector, path=str(tmp_path / "roads.geojson"))
    scene = f3d.MapScene(replace(scene.recipe, layers=[vector, labels]))
    assert not scene.validate().render_blocked(scene.render_policy)
    job = dict(entry_point="MapScene.render", scene=scene)
    estimate = mem.estimate(job)
    assert estimate["contributors"]["recipe_copies"] > 0
    # Independent Windows peak growth of this decoded-vector fixture.
    assert estimate["predicted_peak_bytes"] - estimate["process_baseline_bytes"] >= 511_381_504

    def reached_renderer(*args, **kwargs):
        raise RuntimeError("reached render implementation")

    monkeypatch.setattr(f3d.MapScene, "_render_impl", reached_renderer)
    with pytest.raises(mem.MemoryAdmissionRefused) as refused:
        scene.render(str(output), process_limit_bytes=1)
    assert refused.value.limit_bytes == 1 < refused.value.estimate["predicted_peak_bytes"]
    with pytest.raises(RuntimeError, match="^reached render implementation$"):
        scene.render(str(output), process_limit_bytes=sys.maxsize)

    path_only = f3d.MapScene(replace(scene.recipe, layers=[replace(vector, features=[]), labels]))
    for call in (lambda: mem.estimate(dict(job, scene=path_only)),
                 lambda: path_only.render(str(output), process_limit_bytes=sys.maxsize)):
        with pytest.raises(TypeError) as unknown:
            call()
        assert unknown.value.code == "memory_estimate_unavailable"
    assert not output.exists()


def test_resolved_shadow_atlas_and_offline_targets():
    base = scene(64).recipe

    def estimate(**changes):
        return mem.estimate(dict(entry_point="MapScene.render", scene=f3d.MapScene(replace(base, **changes))))

    def shadow(technique, resolution, cascades):
        lighting = replace(base.lighting, settings={"renderer_config": {"shadows": {
            "technique": technique, "map_size": resolution, "cascades": cascades,
        }}})
        return estimate(lighting=lighting)

    one, four = shadow("pcf", 4096, 1), shadow("pcf", 4096, 4)
    for result, atlas in [(one, 4096**2*4), (four, 4096**2*4*4),
                          (shadow("vsm", 1024, 2), 1024**2*2*20)]:
        gpu = result["device_local_contributors"]
        assert gpu["shadow_atlas"] == atlas
        assert sum(gpu.values()) == result["device_local_peak_bytes"]
    assert four["device_local_peak_bytes"] - one["device_local_peak_bytes"] == 4096**2*3*4
    assert four["contributors"]["runtime_shadow"] > one["contributors"]["runtime_shadow"]

    grid = replace(base.terrain, metadata={"terrain_geometry": {"mode": "grid", "enabled": False}})
    assert estimate(terrain=grid)["device_local_contributors"] == estimate()["device_local_contributors"]
    offline = estimate(output=replace(base.output, samples=4))
    # An already warm system can hide the cold four-sample regression.
    assert offline["predicted_peak_bytes"] - offline["process_baseline_bytes"] >= 2596114432 - 866471936
    assert offline["device_local_contributors"]["frame_targets"] == 64**2 * 172 + 16**2 * 4
    for output in [replace(base.output, samples=8), replace(base.output, format="exr")]:
        assert estimate(output=output)["device_local_contributors"] == offline["device_local_contributors"]


def test_streaming_estimate_uses_metadata_without_constructing_resources(monkeypatch):
    base = scene(32).recipe

    def estimate(**metadata):
        terrain = replace(base.terrain, metadata=metadata)
        return mem.estimate(dict(entry_point="MapScene.render", scene=f3d.MapScene(replace(base, terrain=terrain))))

    def unexpected(*args, **kwargs):
        pytest.fail("preflight built a mesh or decoded virtual texture pixels")

    monkeypatch.setattr(f3d, "clipmap_generate_py", unexpected)
    monkeypatch.setattr("forge3d.map_scene._procedural_vt_source", unexpected)
    clip = estimate(clipmap={"ring_count": 3, "ring_resolution": 16})
    larger = estimate(clipmap={"ring_count": 4, "ring_resolution": 32})
    assert 0 < clip["contributors"]["clipmap_mesh_and_upload"] < larger["contributors"]["clipmap_mesh_and_upload"]
    assert clip["contributors"]["clipmap_selection_readback"] == 504
    options = dict(procedural_sources=True, source_size=(256, 512), atlas_size=1024)
    vt = estimate(vt=options)
    assert vt["contributors"]["vt_feedback_readback"] == (2**16 + 2) * 4
    assert estimate(vt=dict(options, use_feedback=False))["contributors"]["vt_feedback_readback"] == 0
    assert estimate(vt=dict(options, source_count=8))["contributors"]["vt_sources_and_paging"] > vt["contributors"]["vt_sources_and_paging"]
    atlas_growth = estimate(vt=dict(options, atlas_size=2048))["device_local_peak_bytes"] - vt["device_local_peak_bytes"]
    assert atlas_growth == (2048**2 - 1024**2) * 4
    assert "vt_sources_and_paging" not in estimate(vt={"enabled": False})["contributors"]


def test_qualification_rejects_cpu_overrun_away_from_process_peak(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from measure_modus_poster import assert_tracked_cpu_peak

    baseline = dict(rust_heap=900_000, python_arrays=100_000)
    at_cpu_peak = dict(rust_heap=32_900_000, python_arrays=1_334_919)
    report = dict(ledger_samples=dict(baseline=baseline, at_peak=baseline, at_cpu_peak=at_cpu_peak),
                  sampled_cpu_peak_growth_bytes=33_234_919, cpu_peak_bytes=6_772_034)
    with pytest.raises(AssertionError, match="33,234,919 exceeds estimate 6,772,034"):
        assert_tracked_cpu_peak(report)
    report["cpu_peak_bytes"] = report["sampled_cpu_peak_growth_bytes"]
    assert_tracked_cpu_peak(report)
    del report["ledger_samples"]["at_cpu_peak"]
    with pytest.raises(KeyError, match="at_cpu_peak"):
        assert_tracked_cpu_peak(report)


def test_return_peak_includes_publication(monkeypatch):
    from forge3d import _job_memory, certificate

    peak = 100
    exported = []
    result = {}

    class RenderResource:
        def __del__(self):
            nonlocal peak
            peak = 125

    def export(_):
        nonlocal peak
        exported.append(result["memory"].copy())
        peak = 150  # the first certificate export raises the OS high-water mark
        return str(exported[-1]["measured_peak_bytes"])

    monkeypatch.setattr(_job_memory, "get_native_symbol", lambda _: None)
    monkeypatch.setattr(_job_memory._poster_memory, "process_bytes", lambda: (peak, peak))
    monkeypatch.setattr(_job_memory, "predict", lambda *_: dict(
        process_baseline_bytes=50, prior_peak_bytes=50, predicted_peak_bytes=200, known_growth_bytes=25
    ))
    monkeypatch.setattr(certificate, "_emit_current_render_certificate", export)

    @_job_memory.measured_render("MapScene.render")
    def render():
        resource = RenderResource()
        result["memory"] = _job_memory.job_report(True, digest_target=result, digest_key="digest")
        assert resource is not None and peak == 100
        return result

    assert render() is result
    assert [item["measured_peak_bytes"] for item in exported] == [125, 150]
    assert exported[-1] == result["memory"]
    assert result["memory"]["prediction_ratio"] == 200 / 150
    assert result["digest"] == "150"


@pytest.mark.parametrize(("route", "width", "dem", "frames", "options"), [
    ("map", 96, 8, 1, ("--decoded-vector",)),
    pytest.param("map", 64, 64, 1, ("--float64-geotiff",), id="float64-geotiff"),
    ("map", 64, 64, 1, ()), ("map", 256, 256, 1, ()), ("map", 512, 512, 1, ()),
    ("map", 64, 1024, 1, ()), ("map", 64, 2048, 1, ()),
    ("fly", 64, 64, 1, ()), ("fly", 128, 128, 3, ()), ("fly", 256, 256, 4, ()),
    ("fly", 64, 2048, 3, ()),
    ("map", 64, 64, 1, ("--samples", "2")), ("map", 64, 64, 1, ("--samples", "4")),
    ("map", 64, 64, 1, ("--samples", "8")), ("map", 1024, 64, 1, ("--samples", "4")),
    ("map", 256, 64, 1, ("--samples", "4")), ("fly", 64, 64, 2, ("--samples", "4")),
    ("map", 64, 64, 1, ("--pcf-cascades", "1")), ("map", 64, 64, 1, ("--pcf-cascades", "4")),
    ("map", 64, 64, 1, ("--disabled-grid",)),
    ("map", 64, 32, 1, ("--streaming", "clipmap")),
    ("map", 64, 32, 1, ("--streaming", "clipmap", "--clipmap-resolution", "4")),
    ("map", 64, 32, 1, ("--streaming", "vt")),
    ("map", 64, 32, 1, ("--streaming", "both")),
    ("map", 64, 32, 1, ("--streaming", "vt", "--vt-atlas-size", "4096")),
    ("fly", 64, 32, 2, ("--streaming", "both", "--vt-atlas-size", "4096")),
])
@pytest.mark.parametrize("mode", ("admit", "unlimited"))
def test_scene_peak_in_fresh_process(route, width, dem, frames, options, mode, tmp_path):
    require_qualifying_gpu("MODUS scene process peak")
    runner = Path(__file__).resolve().parents[1] / "tools/measure_modus_scene.py"
    command = [sys.executable, str(runner), route, mode, str(tmp_path),
               "--width", str(width), "--dem", str(dem), "--frames", str(frames), *options]
    if mode == "admit" or "--decoded-vector" in options or "--clipmap-resolution" in options:
        command.append("--certificate")
    subprocess.run(command, check=True)


@pytest.mark.parametrize(("route", "options"), [
    ("map", ()), ("fly", ()),
    ("map", ("--decoded-vector",)),
    pytest.param("map", ("--float64-geotiff",), id="float64-geotiff"),
    ("map", ("--width", "64", "--samples", "4", "--refusal-limit", "1349347938")),
    ("map", ("--dem", "32", "--width", "64", "--streaming", "clipmap")),
    ("map", ("--dem", "32", "--width", "64", "--streaming", "vt")),
])
def test_refused_scene_growth_within_empty_call_noise(route, options, tmp_path):
    runner = Path(__file__).resolve().parents[1] / "tools/measure_modus_scene.py"
    empty = tmp_path / "empty"
    subprocess.run([sys.executable, str(runner), route, "empty", str(empty), *options], check=True)
    subprocess.run([sys.executable, str(runner), route, "refuse", str(tmp_path / "refused"),
                    "--reference", str(empty), *options], check=True)
