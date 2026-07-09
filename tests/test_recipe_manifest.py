import ast
import json
import socket
from pathlib import Path

import pytest
import numpy as np

import forge3d as f3d
from forge3d import recipe_manifest as rm


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "recipe_manifests"
FIRST_BATCH_FAMILIES = [
    "terrain_demo",
    "terrain_label",
    "landcover_esri_terrain_viewer",
    "climate_bivariate",
    "hydrology_river",
    "mapscene_showcases",
]


def _load_fixture_dict(family: str) -> dict:
    with (FIXTURE_DIR / f"{family}.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _base_manifest_dict() -> dict:
    return _load_fixture_dict("terrain_demo")


def _diagnostics(data: dict) -> list[str]:
    return rm.validate_manifest(data, repo_root=REPO_ROOT)


def test_mapscene_recipe_manifest_shorthand_is_callable_and_deterministic() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 4, "height": 4, "source_id": "inline-dem"},
        ),
        lighting=f3d.LightingPreset(name="rainier_showcase"),
        output=f3d.OutputSpec(width=64, height=64, samples=4, aovs=("albedo",), hdr=True),
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[{"id": "r1", "geometry": {"type": "LineString", "coordinates": [(0, 0), (1, 1)]}}],
            )
        ],
    )

    first = f3d.recipe_manifest(scene)
    second = f3d.recipe_manifest(scene.recipe)

    assert first == second
    assert first["schema"] == "forge3d.mapscene.recipe_manifest.v1"
    assert first["terrain"]["source_id"] == "inline-dem"
    assert first["output"]["samples"] == 4
    assert first["output"]["aovs"] == ["albedo"]
    assert first["layers"][0]["layer_id"] == "roads"
    assert "recipe_manifest" in f3d.__all__


def test_mapscene_recipe_manifest_records_golden_fixture_intent() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:32610",
            metadata={"width": 4, "height": 4, "source_id": "inline-dem"},
        ),
        camera=f3d.OrbitCamera(),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=32, height=32),
    )

    manifest = f3d.recipe_manifest(
        scene,
        golden_fixture_intent={
            "scene_id": "mapscene_fixture",
            "family": "terrain_raster",
            "golden_path": "tests/golden/recipes/mapscene_fixture.png",
            "command": "pytest tests/test_recipe_goldens.py -k mapscene_fixture",
            "backend": "placeholder",
        },
    )

    intent = manifest["golden_fixture_intent"]
    assert intent["schema"] == "forge3d.mapscene.golden_fixture_intent.v1"
    assert intent["status"] == "active"
    assert intent["family"] == "terrain_raster"
    assert intent["tolerance"] == {"ssim_min": 0.995, "mean_abs_max": 2.0}


def test_recipe_manifest_public_surface_is_module_local():
    assert set(rm.__all__) == {
        "RecipeManifest",
        "RecipeInput",
        "RecipeOutput",
        "RecipeLayer",
        "SourceEvidence",
        "GoldenFixtureIntent",
        "manifest_from_dict",
        "manifest_to_dict",
        "manifest_to_json",
        "manifest_from_json",
        "validate_manifest",
        "load_manifest",
        "save_manifest",
    }

    import forge3d as f3d

    assert "RecipeManifest" not in f3d.__all__


@pytest.mark.parametrize("family", FIRST_BATCH_FAMILIES)
def test_first_batch_fixture_manifests_construct_and_validate(family):
    manifest = rm.load_manifest(FIXTURE_DIR / f"{family}.json")

    assert isinstance(manifest, rm.RecipeManifest)
    assert manifest.recipe_family == family
    assert rm.manifest_to_dict(manifest)["recipe_family"] == family
    assert rm.validate_manifest(manifest, repo_root=REPO_ROOT) == manifest.diagnostics


def test_dataclass_construction_for_terrain_demo_manifest():
    manifest = rm.RecipeManifest(
        recipe_family="terrain_demo",
        recipe_id="terrain_demo_recipe_manifest",
        status="proven_in_forge3d",
        source_examples=["python/forge3d/terrain_demo.py"],
        source_evidence=[
            rm.SourceEvidence(
                path="docs/3d-map-rendering-quality-blender-outmatch-plan.md",
                line_start=121,
                line_end=170,
                note="terrain demo recipe evidence",
            )
        ],
        required_inputs=[rm.RecipeInput(name="dem", kind="heightfield", role="terrain")],
        produced_outputs=[
            rm.RecipeOutput(kind="render", format="png", path="example_defined", deterministic=True)
        ],
        layers=[rm.RecipeLayer(layer_id="terrain", layer_type="terrain_dem", role="base", required=True)],
        alignment={"notes": "single DEM input; no cross-layer alignment"},
        render_export_defaults={"path_policy": "example_defined"},
        support_status={"terrain": "proven_in_forge3d"},
        golden_fixture_intent=rm.GoldenFixtureIntent(status="deferred"),
        diagnostics=["recipe_manifest_golden_not_selected"],
    )

    assert _diagnostics(rm.manifest_to_dict(manifest)) == ["recipe_manifest_golden_not_selected"]


def test_deterministic_json_round_trip_and_file_save(tmp_path):
    manifest = rm.load_manifest(FIXTURE_DIR / "climate_bivariate.json")

    first = rm.manifest_to_json(manifest)
    second = rm.manifest_to_json(rm.manifest_from_json(first))
    out_path = tmp_path / "manifest.json"
    rm.save_manifest(manifest, out_path)

    assert first == second
    assert first.endswith("\n")
    assert out_path.read_text(encoding="utf-8") == first
    assert rm.manifest_to_dict(rm.manifest_from_json(first)) == rm.manifest_to_dict(manifest)


@pytest.mark.parametrize("family", FIRST_BATCH_FAMILIES)
def test_fixture_json_files_are_in_deterministic_manifest_format(family):
    fixture_path = FIXTURE_DIR / f"{family}.json"
    text = fixture_path.read_text(encoding="utf-8")
    manifest = rm.manifest_from_json(text)

    assert text == rm.manifest_to_json(manifest)


@pytest.mark.parametrize(
    ("token", "mutate"),
    [
        ("recipe_manifest_missing_field", lambda data: data.pop("recipe_id")),
        ("recipe_manifest_invalid_field", lambda data: data.update({"layers": "not-a-list"})),
        ("recipe_manifest_invalid_status", lambda data: data.update({"status": "almost_done"})),
        ("recipe_manifest_unknown_family", lambda data: data.update({"recipe_family": "unknown_recipe"})),
        ("recipe_manifest_missing_source", lambda data: data.update({"source_examples": ["missing/example.py"]})),
        (
            "recipe_manifest_unsupported_layer",
            lambda data: data["layers"][0].update({"layer_type": "unknown_layer"}),
        ),
        (
            "recipe_manifest_alignment_unspecified",
            lambda data: (
                data["required_inputs"].append({"name": "overlay", "kind": "raster", "role": "overlay"}),
                data.update({"alignment": {}}),
            ),
        ),
        (
            "recipe_manifest_render_path_unspecified",
            lambda data: (
                data["produced_outputs"][0].pop("path", None),
                data.update({"render_export_defaults": {}}),
            ),
        ),
        (
            "recipe_manifest_golden_not_selected",
            lambda data: data.update({"golden_fixture_intent": {"status": "missing"}}),
        ),
        (
            "recipe_manifest_mapscene_partial",
            lambda data: data.update({"support_status": {"mapscene_compatibility": "partially_proven"}}),
        ),
        (
            "recipe_manifest_example_only",
            lambda data: data.update({"status": "exists_only_as_example_or_script_logic"}),
        ),
        ("recipe_manifest_schema_version_unsupported", lambda data: data.update({"schema_version": "2"})),
    ],
)
def test_validation_can_emit_every_stable_diagnostic_token(token, mutate):
    data = _base_manifest_dict()
    mutate(data)

    assert token in _diagnostics(data)


def test_missing_source_path_detection_is_local_to_repo():
    data = _base_manifest_dict()
    data["source_evidence"] = [
        {"path": "docs/3d-map-rendering-quality-blender-outmatch-plan.md"},
        {"path": "docs/carto-engine/does-not-exist.md"},
    ]

    assert "recipe_manifest_missing_source" in _diagnostics(data)


def test_absolute_source_paths_are_rejected_without_existence_probe(monkeypatch):
    data = _base_manifest_dict()
    source_path = REPO_ROOT / "docs" / "3d-map-rendering-quality-blender-outmatch-plan.md"
    data["source_examples"] = [str(source_path)]
    data["source_evidence"] = []

    original_exists = Path.exists

    def fail_on_absolute_probe(path):
        if path == source_path:
            raise AssertionError("absolute source paths must not be probed")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fail_on_absolute_probe)

    assert "recipe_manifest_invalid_field" in _diagnostics(data)


def test_unsupported_layer_and_invalid_status_are_separate_diagnostics():
    data = _base_manifest_dict()
    data["status"] = "not_a_status"
    data["layers"][0]["layer_type"] = "not_a_layer"

    diagnostics = _diagnostics(data)

    assert "recipe_manifest_invalid_status" in diagnostics
    assert "recipe_manifest_unsupported_layer" in diagnostics


def test_validation_does_not_use_network(monkeypatch):
    data = _base_manifest_dict()

    def fail_network(*args, **kwargs):
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    assert rm.validate_manifest(data, repo_root=REPO_ROOT) == data["diagnostics"]


def test_recipe_manifest_module_imports_no_forbidden_runtime_backends():
    source_path = REPO_ROOT / "python" / "forge3d" / "recipe_manifest.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_roots = {
        "rasterio",
        "geopandas",
        "shapely",
        "rioxarray",
        "xarray",
        "terra",
        "requests",
        "urllib",
        "httpx",
        "forge3d.gis",
        "forge3d.map_scene",
        "forge3d._native",
    }
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if node.level:
                module = f"forge3d.{module}" if module else "forge3d"
            imports.add(module)

    assert imports.isdisjoint(forbidden_roots)
