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
from forge3d import recipe_manifest as rm
from forge3d._map_scene_validation import (
    classify_layer,
    diagnostic_block,
    probe_native_capability,
)

from _sutura_recipes import RECIPE_NAMES, build_scene

PACKAGE_ROOT = Path(f3d.__file__).resolve().parent


def _native_terrain_available() -> bool:
    try:
        return bool(f3d.has_gpu()) and probe_native_capability("terrain")
    except Exception:
        return False


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
    assert set(visibility) == {"summit", "valley"}


def test_bundle_roundtrip_bitexact(tmp_path):
    scene = build_scene("terrain_raster_labels_buildings", tmp_path)
    compiled = scene.compile_plan()
    text = rm.manifest_to_json(compiled.manifest)
    rehydrated = rm.manifest_from_json(text)
    assert rm.manifest_to_json(rehydrated) == text
    # and once more through plain dicts
    assert rm.manifest_to_json(rm.manifest_from_dict(rm.manifest_to_dict(compiled.manifest))) == text


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
