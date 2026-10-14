"""SUTURA integrity gate: zero-placeholder MapScene, bit-exact bundles.

Covers the definition-of-done for docs/prompts/fable5-moonshots/10-sutura.md:

- no ``allow_placeholder`` escape hatch remains anywhere in the package,
- every DoD recipe either renders natively or raises a structured
  ``MapSceneNativeUnavailable`` diagnostic block (no silent placeholders),
- ``PointCloudLayer`` carries a ``support_level`` classification,
- depth-occlusion label culling runs only in ``MapScene.compile_plan()`` and
  the render phase is pure (guard raises on an uncompiled plan),
- compiled ``RecipeManifest`` JSON round-trips byte-identically,
- the measurable win: render -> save_bundle -> load_bundle -> re-render
  reproduces pixels with SSIM >= 0.99 and a byte-identical validation report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
import forge3d.map_scene as map_scene
import forge3d._map_scene_compiler as scene_compiler
from forge3d import recipe_manifest as rm
from forge3d._map_scene_validation import (
    classify_layer,
    diagnostic_block,
    probe_native_capability,
)

from _sutura_recipes import RECIPE_NAMES, _write_raster_overlay, build_scene
from _terrain_runtime import terrain_rendering_available

PACKAGE_ROOT = Path(f3d.__file__).resolve().parent


def _native_terrain_available() -> bool:
    return terrain_rendering_available()


def _ssim(image_a: np.ndarray, image_b: np.ndarray, *, block: int = 8) -> float:
    """Global mean SSIM over non-overlapping blocks (numpy only)."""
    a = np.asarray(image_a, dtype=np.float64)[..., :3].mean(axis=2)
    b = np.asarray(image_b, dtype=np.float64)[..., :3].mean(axis=2)
    assert a.shape == b.shape, "SSIM inputs must share a shape"
    height = (a.shape[0] // block) * block
    width = (a.shape[1] // block) * block

    def _blocks(img: np.ndarray) -> np.ndarray:
        img = img[:height, :width]
        img = img.reshape(height // block, block, width // block, block)
        return img.transpose(0, 2, 1, 3).reshape(-1, block * block)

    pa = _blocks(a)
    pb = _blocks(b)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = pa.mean(axis=1)
    mu_b = pb.mean(axis=1)
    var_a = pa.var(axis=1)
    var_b = pb.var(axis=1)
    cov = ((pa - mu_a[:, None]) * (pb - mu_b[:, None])).mean(axis=1)
    ssim = ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(ssim.mean())


def _load_png_rgba(path: Path) -> np.ndarray:
    from forge3d._png import load_png_rgba

    return np.asarray(load_png_rgba(path), dtype=np.uint8)


def _report_bytes(report) -> bytes:
    return json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _assert_structured_block(exc: "f3d.MapSceneNativeUnavailable") -> None:
    blocks = exc.diagnostics
    assert blocks, "MapSceneNativeUnavailable must carry diagnostic blocks"
    for block in blocks:
        assert block["status"] == "diagnostic_block"
        assert isinstance(block["layer"], str) and block["layer"]
        assert isinstance(block["reason"], str) and block["reason"]
        assert isinstance(block["required_native"], str) and block["required_native"]


def test_no_allow_placeholder_symbol():
    hits = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if path.suffix not in {".py", ".pyi"} or not path.is_file():
            continue
        if "allow_placeholder" in path.read_text(encoding="utf-8", errors="ignore"):
            hits.append(str(path.relative_to(PACKAGE_ROOT)))
    assert hits == [], f"allow_placeholder must not exist in python/forge3d: {hits}"


def test_diagnostic_block_shape_and_capability_probe():
    block = diagnostic_block(
        layer="points",
        reason="native point-cloud compositing unavailable",
        required_native="vector_render_oit_py",
    )
    assert block == {
        "status": "diagnostic_block",
        "layer": "points",
        "reason": "native point-cloud compositing unavailable",
        "required_native": "vector_render_oit_py",
    }
    for kind in ("terrain", "raster", "vector", "labels", "buildings", "point_cloud", "tiles3d"):
        assert isinstance(probe_native_capability(kind), bool)
    assert probe_native_capability("definitely-not-a-kind") is False


def test_classify_layer_returns_native_or_diagnostic_block(tmp_path):
    scene = build_scene("all_layers", tmp_path)
    for layer in scene.recipe.layers:
        assert classify_layer(layer) in {"native", "diagnostic_block"}
    assert classify_layer(object()) == "diagnostic_block"


def test_pointcloud_layer_support_level():
    layer = f3d.PointCloudLayer(layer_id="pc")
    assert layer.support_level == "native-required"
    assert layer.to_dict()["support_level"] == "native-required"
    # round-trips through the recipe layer decoder
    decoded = map_scene.MapScene._layer_from_dict(layer.to_dict())
    assert decoded.support_level == "native-required"


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_each_recipe_renders_or_blocks(recipe_name, tmp_path):
    scene = build_scene(recipe_name, tmp_path)
    compiled = scene.compile_plan()
    assert compiled is not None
    output_path = tmp_path / f"{recipe_name}.png"
    try:
        report = scene.render(str(output_path))
    except f3d.MapSceneNativeUnavailable as exc:
        _assert_structured_block(exc)
        assert not output_path.exists(), "a blocked render must not write pixels"
    else:
        assert output_path.exists()
        assert report.status in {"ok", "warning"}
        assert scene.last_render_backend == "gpu_terrain"
        rgba = _load_png_rgba(output_path)
        assert rgba.shape == (64, 96, 4)


def test_depth_occlusion_only_in_compile(tmp_path):
    # Depth-occlusion re-planning must not exist in the render phase anymore.
    assert not hasattr(map_scene, "_plans_with_depth_occlusion")

    scene = build_scene("terrain_raster_labels_buildings", tmp_path)

    # The render phase is pure: it refuses anything but a compiled plan.
    with pytest.raises(RuntimeError, match="compile"):
        map_scene._render_native_offscreen_result(scene.recipe, {})

    # Compiling twice on identical inputs freezes the identical label set.
    first = scene.compile_plan()
    second = scene.compile_plan()
    assert rm.manifest_to_json(first.manifest) == rm.manifest_to_json(second.manifest)

    # The frozen manifest carries the compiled label plans and cull decisions.
    payload = rm.manifest_to_dict(first.manifest)
    assert payload["compiled_label_plans"], "compiled label plans must be frozen in the manifest"
    assert payload["depth_cull"]["camera_terrain_key"]
    layers = payload["depth_cull"]["layers"]
    assert "labels" in layers
    visibility = layers["labels"]["visibility"]
    assert visibility, "per-label visibility flags must be frozen"
    assert all(key.startswith("label-instance:") for key in visibility)
    assert all(entry["instance_id"] == key for key, entry in visibility.items())
    assert {entry["label_id"] for entry in visibility.values()} == {
        "summit",
        "valley",
    }
    assert all(entry["source_id"] for entry in visibility.values())


def test_bundle_roundtrip_bitexact(tmp_path):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    compiled = scene.compile_plan()
    text = rm.manifest_to_json(compiled.manifest)
    rehydrated = rm.manifest_from_json(text)
    assert rm.manifest_to_json(rehydrated) == text
    # and once more through plain dicts
    assert rm.manifest_to_json(rm.manifest_from_dict(rm.manifest_to_dict(compiled.manifest))) == text


@pytest.mark.parametrize("validate_first", [False, True])
def test_validation_preserves_compiled_decisions_and_attached_frame(
    tmp_path, monkeypatch, validate_first
):
    from dataclasses import replace

    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    if validate_first:
        scene.validate()
    compiled = replace(scene.compile_plan(), frame=object())
    scene.compiled_plan = compiled
    frozen = rm.manifest_to_json(compiled.manifest)

    def unexpected_planning(*_args, **_kwargs):
        pytest.fail("validation of an unchanged compiled recipe must reuse frozen labels")

    monkeypatch.setattr(scene_compiler, "_label_plan_from_layer", unexpected_planning)
    report = scene.validate()

    assert scene.compiled_plan is compiled
    assert scene._compiled_plan_for_current_recipe() is compiled
    assert scene.last_validation_report is report
    assert _report_bytes(report) == _report_bytes(compiled.validation_report)
    assert rm.manifest_to_json(scene.compiled_plan.manifest) == frozen


def test_restored_plan_validation_and_save_never_replan(tmp_path, monkeypatch):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    scene.save_bundle(tmp_path / "original")
    original_path = Path(scene.last_bundle_path) / "scene" / "compiled_plan.json"
    frozen_bytes = original_path.read_bytes()
    frozen_labels = scene.compiled_label_plans["labels"].to_dict()

    def unexpected_planning(*_args, **_kwargs):
        pytest.fail("persisted-plan restoration and validation must not freshly plan labels")

    monkeypatch.setattr(scene_compiler, "_label_plan_from_layer", unexpected_planning)
    loaded = f3d.MapScene.load_bundle(Path(scene.last_bundle_path))
    restored = loaded.compiled_plan
    report = loaded.validate()
    assert loaded.compiled_plan is restored
    assert loaded.compiled_label_plans["labels"].to_dict() == frozen_labels
    assert _report_bytes(report) == _report_bytes(restored.validation_report)
    loaded.save_bundle(tmp_path / "restored")
    saved_path = Path(loaded.last_bundle_path) / "scene" / "compiled_plan.json"
    assert saved_path.read_bytes() == frozen_bytes


def test_validate_invalidates_plan_after_recipe_mutation(tmp_path):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    original = scene.compile_plan()
    scene.recipe.layers = ()
    report = scene.validate()

    assert scene.compiled_plan is None
    assert scene.compiled_label_plans == {}
    current = scene._compiled_plan_for_current_recipe()
    assert current.recipe_hash != original.recipe_hash
    assert current.recipe_hash == map_scene._stable_hash(scene.recipe.to_dict())
    assert current.label_plans == {}
    assert _report_bytes(current.validation_report) == _report_bytes(report)


def test_restoration_rejects_missing_frozen_labels_without_publishing(tmp_path):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    compiled = scene.compile_plan()
    manifest = rm.manifest_from_json(rm.manifest_to_json(compiled.manifest))
    manifest.compiled_label_plans = {}

    with pytest.raises(ValueError, match="Compiled plan missing label layer 'labels'"):
        scene._rehydrate_compiled_plan(manifest)
    assert scene.compiled_plan is compiled


@pytest.mark.parametrize("restore", [False, True])
def test_mutable_label_inspection_view_cannot_change_frozen_plan(tmp_path, restore):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    compiled = scene.compile_plan()
    if restore:
        compiled = scene._rehydrate_compiled_plan(
            rm.manifest_from_json(rm.manifest_to_json(compiled.manifest))
        )
    frozen = compiled.label_plans["labels"].to_dict()
    scene.compiled_label_plans["labels"].accepted.clear()
    scene.compiled_label_plans = {}

    assert scene._compiled_plan_for_current_recipe() is compiled
    assert compiled.label_plans["labels"].to_dict() == frozen
    scene.validate()
    assert scene.compiled_label_plans["labels"].to_dict() == frozen


def test_failed_compilation_does_not_publish_partial_state(tmp_path, monkeypatch):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    compiled = scene.compile_plan()
    report = scene.last_validation_report
    labels = scene.compiled_label_plans
    scene.recipe.camera.azimuth_deg += 1.0

    def failed_validation(*_args, **_kwargs):
        raise ValueError("label validation failed")

    monkeypatch.setattr(scene_compiler, "_atlas_glyph_set", failed_validation)
    with pytest.raises(ValueError, match="label validation failed"):
        scene.compile_plan()
    assert scene.compiled_plan is compiled
    assert scene.last_validation_report is report
    assert scene.compiled_label_plans is labels
    with pytest.raises(ValueError, match="label validation failed"):
        scene._compiled_plan_for_current_recipe()


def test_compilation_detaches_a_supplied_mutable_label_plan(tmp_path):
    from dataclasses import replace

    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    supplied = scene.compile_plan().label_plans["labels"]
    scene.recipe.layers = tuple(
        replace(layer, plan=supplied) if isinstance(layer, f3d.LabelLayer) else layer
        for layer in scene.recipe.layers
    )
    compiled = scene.compile_plan()
    frozen = compiled.label_plans["labels"].to_dict()
    supplied.accepted.clear()

    assert compiled.label_plans["labels"].to_dict() == frozen
    assert compiled.manifest.compiled_label_plans["labels"] == frozen


@pytest.mark.parametrize("operation", ["compile", "validate", "restore"])
def test_duplicate_label_ids_fail_without_publishing(tmp_path, operation):
    from dataclasses import replace

    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    original = scene.compile_plan()
    original_report = scene.last_validation_report
    original_labels = scene.compiled_label_plans
    layer = next(layer for layer in scene.recipe.layers if isinstance(layer, f3d.LabelLayer))
    scene.recipe.layers += (replace(layer, labels=tuple(layer.labels[:1])),)
    manifest = rm.manifest_from_json(rm.manifest_to_json(original.manifest))
    # An older compiler could persist ambiguous IDs. Match its recipe identity
    # so restoration must reject the duplicate itself, rather than a stale ID.
    manifest.recipe_id = f"mapscene-compiled-{map_scene._stable_hash(scene.recipe.to_dict())[:16]}"

    with pytest.raises(ValueError, match="Duplicate label layer id 'labels'"):
        if operation == "compile":
            scene.compile_plan()
        elif operation == "validate":
            scene.validate()
        else:
            scene._rehydrate_compiled_plan(manifest)
    assert scene.compiled_plan is original
    assert scene.last_validation_report is original_report
    assert scene.compiled_label_plans is original_labels


@pytest.mark.parametrize("tampered_file", ["mapscene_recipe.json", "compiled_plan.json"])
def test_bundle_rejects_mismatched_compiled_recipe_identity(tmp_path, tampered_file):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    scene.save_bundle(tmp_path / "identity")
    path = Path(scene.last_bundle_path) / "scene" / tampered_file
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tampered_file == "mapscene_recipe.json":
        payload["camera"]["azimuth_deg"] += 1.0
    else:
        payload["recipe_id"] = "mapscene-compiled-tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Compiled plan recipe identity"):
        f3d.MapScene.load_bundle(Path(scene.last_bundle_path))


def test_restoration_identity_failure_preserves_previous_state(tmp_path):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    original = scene.compile_plan()
    original_report = scene.last_validation_report
    original_labels = scene.compiled_label_plans
    manifest = rm.manifest_from_json(rm.manifest_to_json(original.manifest))
    scene.recipe.camera.azimuth_deg += 1.0

    with pytest.raises(ValueError, match="Compiled plan recipe identity"):
        scene._rehydrate_compiled_plan(manifest)
    assert scene.compiled_plan is original
    assert scene.last_validation_report is original_report
    assert scene.compiled_label_plans is original_labels


@pytest.mark.parametrize("restore", [False, True])
def test_public_compiled_labels_cannot_mutate_frozen_decisions(tmp_path, restore):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    compiled = scene.compile_plan()
    if restore:
        compiled = scene._rehydrate_compiled_plan(
            rm.manifest_from_json(rm.manifest_to_json(compiled.manifest))
        )
    frozen = compiled.label_plans["labels"].to_dict()
    accessed = compiled.label_plans["labels"]
    accessed.accepted[0].candidate.anchor = (999.0, 999.0, 999.0)
    accessed.accepted[0].typography["font_size"] = 999.0
    accessed.accepted.clear()
    accessed.rejected.clear()
    accessed.diagnostics.clear()
    accessed.bounds.clear()

    assert compiled.label_plans["labels"].to_dict() == frozen
    assert compiled.manifest.compiled_label_plans["labels"] == frozen
    assert scene._compiled_plan_for_current_recipe() is compiled


def test_failed_validation_preserves_all_previous_state(tmp_path, monkeypatch):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    original = scene.compile_plan()
    original_report = scene.last_validation_report
    original_labels = scene.compiled_label_plans
    scene.recipe.camera.azimuth_deg += 1.0

    def failed_validation(*_args, **_kwargs):
        raise ValueError("label validation failed")

    monkeypatch.setattr(scene_compiler, "_atlas_glyph_set", failed_validation)
    with pytest.raises(ValueError, match="label validation failed"):
        scene.validate()
    assert scene.compiled_plan is original
    assert scene.last_validation_report is original_report
    assert scene.compiled_label_plans is original_labels
    with pytest.raises(ValueError, match="label validation failed"):
        scene._compiled_plan_for_current_recipe()


def test_last_validation_report_has_one_compiler_owner(tmp_path):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    assert scene.last_validation_report is None
    report = scene.validate()
    assert scene._compiler.report is report
    assert scene.last_validation_report is report
    assert "last_validation_report" not in vars(scene)
    saved_report = scene.save_bundle(tmp_path / "report_owner")
    assert scene._compiler.report is saved_report
    loaded = f3d.MapScene.load_bundle(Path(scene.last_bundle_path))
    assert loaded.last_validation_report is loaded._compiler.report
    assert _report_bytes(loaded.last_validation_report) == _report_bytes(saved_report)


def test_every_bundle_file_roundtrips_byte_identically(tmp_path):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    scene.save_bundle(tmp_path / "all_bundle_bytes")
    bundle_path = Path(scene.last_bundle_path)

    def bundle_bytes():
        return {
            path.relative_to(bundle_path).as_posix(): path.read_bytes()
            for path in bundle_path.rglob("*") if path.is_file()
        }

    before = bundle_bytes()
    loaded = f3d.MapScene.load_bundle(bundle_path)
    loaded.save_bundle(bundle_path)
    assert bundle_bytes() == before


def test_nodata_candidates_compile_and_render_without_relaxing_manifest_json(tmp_path):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    terrain_path = Path(scene.recipe.terrain.path)
    heightmap = np.load(terrain_path)
    # Reproduce the reviewed clipped-DEM case: its top quarter is nodata.
    heightmap[:heightmap.shape[0] // 4] = np.nan
    np.save(terrain_path, heightmap)

    compiled = scene.compile_plan()
    labels = compiled.label_plans["labels"]
    assert any(
        np.isnan(candidate.terrain_sample.get("elevation", 0.0))
        for label in labels.accepted
        for candidate in label.candidates
    ), "the regression requires an unselected candidate sampling nodata"
    output_path = tmp_path / "nodata_labels.png"
    report = scene.render(str(output_path))
    assert report.status == "ok"
    assert scene.last_render_backend == "gpu_terrain"
    assert _load_png_rgba(output_path).shape == (64, 96, 4)
    assert scene.compiled_plan is compiled
    # In-memory candidate metadata and persisted finite JSON are different
    # boundaries. Preserve the existing manifest rejection contract.
    with pytest.raises(ValueError, match="NaN|infinite"):
        rm.manifest_to_json(compiled.manifest)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_restore_rejects_nonfinite_saved_metadata_without_publishing(tmp_path, nonfinite):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    report = scene.save_bundle(tmp_path / "nonfinite_saved")
    compiled = scene.compiled_plan
    inspected = scene.compiled_label_plans
    path = Path(scene.last_bundle_path) / "scene" / "compiled_plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload["compiled_label_plans"]["labels"]["accepted"][0]["candidates"][0]["terrain_sample"]
    sample["elevation"] = nonfinite
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="NaN|infinite"):
        f3d.MapScene.load_bundle(Path(scene.last_bundle_path))
    with pytest.raises(ValueError, match="NaN|infinite"):
        scene._rehydrate_compiled_plan(rm.manifest_from_dict(payload))
    assert scene.compiled_plan is compiled
    assert scene.last_validation_report is report
    assert scene.compiled_label_plans is inspected


def test_frame_attachment_preserves_latest_report(tmp_path):
    from dataclasses import replace

    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    plan = scene.compile_plan()
    report = scene.save_bundle(tmp_path / "frame_report")
    assert report.supported_features["mapscene.save_bundle"] == "supported"
    attached = replace(plan, frame=object())
    scene.compiled_plan = attached
    assert scene.compiled_plan is attached
    assert scene.last_validation_report is report
    assert scene._compiler.report is report
    replacement_report = f3d.ValidationReport(supported_features={"replacement.plan": "supported"})
    scene.compiled_plan = replace(attached, validation_report=replacement_report)
    assert scene.last_validation_report is replacement_report


def test_bundle_version_v2_read_path_recompiles(tmp_path):
    from forge3d.bundle import BUNDLE_VERSION

    assert BUNDLE_VERSION == 3
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    scene.save_bundle(tmp_path / "bundle_v2")
    bundle_path = Path(scene.last_bundle_path)

    compiled_path = bundle_path / "scene" / "compiled_plan.json"
    assert compiled_path.exists(), "v3 bundles must persist the compiled plan"

    manifest_path = bundle_path / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["version"] == 3

    # Simulate a v2 bundle: no compiled plan, version 2 -> load recompiles once.
    compiled_path.unlink()
    manifest_payload["version"] = 2
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = f3d.MapScene.load_bundle(bundle_path)
    assert loaded.compiled_plan is not None
    assert loaded.compiled_plan.manifest.compiled_label_plans


def test_stale_compiled_plan_not_saved_after_recipe_mutation(tmp_path):
    """save_bundle must serialize a plan compiled from the CURRENT recipe."""
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    scene.compile_plan()
    assert scene.compiled_plan.manifest.compiled_label_plans, "precondition: labels compiled"

    scene.recipe.layers = ()
    scene.save_bundle(tmp_path / "bundle_mutated")
    bundle_path = Path(scene.last_bundle_path)

    compiled_payload = json.loads(
        (bundle_path / "scene" / "compiled_plan.json").read_text(encoding="utf-8")
    )
    assert not compiled_payload.get("compiled_label_plans"), (
        "a stale compiled plan (with label plans from the pre-mutation recipe) "
        "must not be serialized after the recipe was mutated"
    )
    review_payload = json.loads(
        (bundle_path / "scene" / "mapscene_review.json").read_text(encoding="utf-8")
    )
    assert review_payload["compiled_label_plan_ids"] == []
    assert scene.compiled_plan.recipe_hash == map_scene._stable_hash(scene.recipe.to_dict())


def test_render_recompiles_stale_plan_after_recipe_mutation(tmp_path, monkeypatch):
    """render must pass a plan matching the current recipe into the renderer."""
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    scene.compile_plan()
    stale_hash = scene.compiled_plan.recipe_hash

    scene.recipe.layers = ()
    seen: dict[str, object] = {}

    def fake_native(recipe, compiled, **_kwargs):
        seen["compiled"] = compiled
        rgba = np.zeros((64, 96, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    monkeypatch.setattr(map_scene, "_render_native_offscreen_result", fake_native)
    scene.render(str(tmp_path / "mutated.png"))

    current_hash = map_scene._stable_hash(scene.recipe.to_dict())
    compiled = seen["compiled"]
    assert compiled.recipe_hash == current_hash
    assert compiled.recipe_hash != stale_hash


def test_raster_overlay_requires_native_drape_or_blocks(tmp_path, monkeypatch):
    """Missing sources and renderers that omit the drape must block before pixels."""
    def fake_terrain(_recipe, _heightmap, **_kwargs):
        rgba = np.zeros((64, 96, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(rgba=rgba)

    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain)

    missing_overlay_scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "flat-dem"},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=96, height=64),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                crs="EPSG:32610",
                opacity=0.8,
                metadata={"width": 8, "height": 8, "source_id": "sutura-overlay"},
            )
        ],
    )
    # An unloadable overlay must block the whole render with a structured
    # diagnostic instead of writing a placeholder PNG.
    missing_png = tmp_path / "raster_missing.png"
    with pytest.raises(map_scene.MapSceneNativeUnavailable) as excinfo:
        missing_overlay_scene.render(str(missing_png))
    ortho_blocks = [
        block for block in excinfo.value.diagnostics if block.get("layer") == "ortho"
    ]
    assert ortho_blocks, f"expected a diagnostic block for layer 'ortho': {excinfo.value.diagnostics}"
    assert ortho_blocks[0]["required_native"] == "readable RasterOverlay source"
    assert not missing_png.exists(), "a blocked render must not write pixels"

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "flat-dem"},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=96, height=64),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path=str(_write_raster_overlay(tmp_path)),
                crs="EPSG:32610",
                opacity=0.8,
                metadata={"width": 8, "height": 8, "source_id": "sutura-overlay"},
            )
        ],
    )
    output = tmp_path / "raster.png"
    with pytest.raises(f3d.MapSceneNativeUnavailable) as excinfo:
        scene.render(str(output))
    _assert_structured_block(excinfo.value)
    assert "terrain-UV raster drape" in str(excinfo.value)
    assert not output.exists()


def test_load_bundle_rejects_future_bundle_version(tmp_path):
    from forge3d.bundle import BUNDLE_VERSION

    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    scene.save_bundle(tmp_path / "bundle_future")
    bundle_path = Path(scene.last_bundle_path)

    manifest_path = bundle_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["version"] = 999
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from forge3d.bundle import COMPOSITION_BUNDLE_VERSION

    with pytest.raises(ValueError, match=rf"999 > supported version {COMPOSITION_BUNDLE_VERSION}"):
        f3d.MapScene.load_bundle(bundle_path)


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_save_load_rerender_ssim(recipe_name, tmp_path):
    """THE MEASURABLE WIN: bundle round-trip reproduces pixels and report."""
    if not _native_terrain_available():
        pytest.skip(
            "native terrain backend unavailable on this host; the diagnostic-block "
            "contract is enforced by test_each_recipe_renders_or_blocks"
        )

    scene = build_scene(recipe_name, tmp_path)
    first_png = tmp_path / "first.png"
    first_report = scene.render(str(first_png))
    scene.save_bundle(tmp_path / "bundle")
    bundle_path = Path(scene.last_bundle_path)

    loaded = f3d.MapScene.load_bundle(bundle_path)
    assert loaded.compiled_plan is not None, "load_bundle must rehydrate the compiled plan verbatim"
    second_png = tmp_path / "second.png"
    second_report = loaded.render(str(second_png))

    first_rgba = _load_png_rgba(first_png)
    second_rgba = _load_png_rgba(second_png)
    ssim = _ssim(first_rgba, second_rgba)
    assert ssim >= 0.99, f"{recipe_name}: round-trip SSIM {ssim:.6f} < 0.99"

    assert _report_bytes(first_report) == _report_bytes(second_report), (
        f"{recipe_name}: validation reports must serialize byte-identically"
    )
