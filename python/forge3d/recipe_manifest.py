"""Recipe-family manifest metadata and provenance helpers."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._map_scene_common import _layer_id, _stable_hash

__all__ = [
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
]

_SCHEMA_VERSION = "1"

_ALLOWED_FAMILIES = frozenset(
    {
        "terrain_demo",
        "terrain_label",
        "landcover_esri_terrain_viewer",
        "climate_bivariate",
        "hydrology_river",
        "mapscene_showcases",
        "terrain_relief_rem",
        "population_spike_worldpop",
        "population_ghsl_3d",
        "builtup_cover_3d",
        "pointcloud_cog",
        "urban_osm_city",
        "luxembourg_rail_overlay",
    }
)

_ALLOWED_STATUSES = frozenset(
    {
        "proven_in_forge3d",
        "partially_proven",
        "exists_only_as_example_or_script_logic",
        "exists_but_not_exposed_as_public_api",
        "exists_but_not_tested",
        "not_found",
        "evidence_missing",
        "unclear_requires_human_confirmation",
    }
)

_ALLOWED_LAYER_TYPES = frozenset(
    {
        "terrain_dem",
        "raster_continuous",
        "raster_categorical",
        "raster_bivariate",
        "vector_polygon",
        "vector_line",
        "label_annotation",
        "map_furniture",
        "mapscene_recipe",
        "pointcloud",
        "cog_raster",
        "building_footprint",
        "temporal_sequence",
        "globe_raster",
    }
)

_DIAGNOSTIC_TOKENS = (
    "recipe_manifest_missing_field",
    "recipe_manifest_invalid_field",
    "recipe_manifest_invalid_status",
    "recipe_manifest_unknown_family",
    "recipe_manifest_missing_source",
    "recipe_manifest_unsupported_layer",
    "recipe_manifest_alignment_unspecified",
    "recipe_manifest_render_path_unspecified",
    "recipe_manifest_golden_not_selected",
    "recipe_manifest_mapscene_partial",
    "recipe_manifest_example_only",
    "recipe_manifest_schema_version_unsupported",
)
_DIAGNOSTIC_SET = frozenset(_DIAGNOSTIC_TOKENS)
_GOLDEN_STATUSES = frozenset({"exists", "missing", "deferred"})

_REQUIRED_FIELDS = (
    "schema_version",
    "recipe_family",
    "recipe_id",
    "status",
    "source_examples",
    "source_evidence",
    "required_inputs",
    "optional_inputs",
    "produced_outputs",
    "layers",
    "alignment",
    "preprocessing",
    "camera_defaults",
    "lighting_defaults",
    "styling_defaults",
    "annotations_defaults",
    "render_export_defaults",
    "support_status",
    "diagnostics",
    "tests",
    "golden_fixture_intent",
    "non_goals",
    "open_questions",
)

_SPATIAL_INPUT_HINTS = (
    "dem",
    "heightfield",
    "raster",
    "vector",
    "pointcloud",
    "point_cloud",
    "cog",
    "terrain",
    "overlay",
    "label",
)


@dataclass
class RecipeInput:
    name: str
    kind: str
    crs: str | None = None
    shape: Any | None = None
    format: str | None = None
    role: str | None = None


@dataclass
class RecipeOutput:
    kind: str
    format: str
    path: str | None = None
    deterministic: bool | None = None


@dataclass
class RecipeLayer:
    layer_id: str
    layer_type: str
    role: str
    required: bool
    style_ref: str | None = None


@dataclass
class SourceEvidence:
    path: str
    line_start: int | None = None
    line_end: int | None = None
    note: str | None = None


@dataclass
class GoldenFixtureIntent:
    status: str
    source_test: str | None = None
    source_file: str | None = None
    note: str | None = None


@dataclass
class RecipeManifest:
    schema_version: str = _SCHEMA_VERSION
    recipe_family: str = ""
    recipe_id: str = ""
    status: str = ""
    source_examples: Sequence[str] = field(default_factory=tuple)
    source_evidence: Sequence[SourceEvidence | Mapping[str, Any]] = field(default_factory=tuple)
    required_inputs: Sequence[RecipeInput | Mapping[str, Any]] = field(default_factory=tuple)
    optional_inputs: Sequence[RecipeInput | Mapping[str, Any]] = field(default_factory=tuple)
    produced_outputs: Sequence[RecipeOutput | Mapping[str, Any]] = field(default_factory=tuple)
    layers: Sequence[RecipeLayer | Mapping[str, Any]] = field(default_factory=tuple)
    alignment: Mapping[str, Any] = field(default_factory=dict)
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    camera_defaults: Mapping[str, Any] = field(default_factory=dict)
    lighting_defaults: Mapping[str, Any] = field(default_factory=dict)
    styling_defaults: Mapping[str, Any] = field(default_factory=dict)
    annotations_defaults: Mapping[str, Any] = field(default_factory=dict)
    render_export_defaults: Mapping[str, Any] = field(default_factory=dict)
    support_status: Mapping[str, str] = field(default_factory=dict)
    diagnostics: Sequence[str] = field(default_factory=tuple)
    tests: Sequence[str] = field(default_factory=tuple)
    golden_fixture_intent: GoldenFixtureIntent | Mapping[str, Any] = field(
        default_factory=lambda: GoldenFixtureIntent(status="missing")
    )
    non_goals: Sequence[str] = field(default_factory=tuple)
    open_questions: Sequence[str] = field(default_factory=tuple)
    # SUTURA: frozen compile-phase state. ``compiled_label_plans`` maps
    # layer_id -> LabelPlan payload; ``depth_cull`` records the deterministic
    # depth-occlusion decisions (per-label visibility flags keyed by the
    # camera+terrain hash) so a reloaded bundle reproduces the identical cull.
    compiled_label_plans: Mapping[str, Any] = field(default_factory=dict)
    depth_cull: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_examples = [str(path) for path in self.source_examples]
        self.source_evidence = [_source_evidence(item) for item in self.source_evidence]
        self.required_inputs = [_recipe_input(item) for item in self.required_inputs]
        self.optional_inputs = [_recipe_input(item) for item in self.optional_inputs]
        self.produced_outputs = [_recipe_output(item) for item in self.produced_outputs]
        self.layers = [_recipe_layer(item) for item in self.layers]
        self.alignment = _dict(self.alignment)
        self.preprocessing = _dict(self.preprocessing)
        self.camera_defaults = _dict(self.camera_defaults)
        self.lighting_defaults = _dict(self.lighting_defaults)
        self.styling_defaults = _dict(self.styling_defaults)
        self.annotations_defaults = _dict(self.annotations_defaults)
        self.render_export_defaults = _dict(self.render_export_defaults)
        self.support_status = {str(key): str(value) for key, value in _dict(self.support_status).items()}
        self.diagnostics = [str(token) for token in self.diagnostics]
        self.tests = [str(test) for test in self.tests]
        self.golden_fixture_intent = _golden_fixture_intent(self.golden_fixture_intent)
        self.non_goals = [str(goal) for goal in self.non_goals]
        self.open_questions = [str(question) for question in self.open_questions]
        self.compiled_label_plans = _dict(self.compiled_label_plans)
        self.depth_cull = _dict(self.depth_cull)


def manifest_from_dict(data: Mapping[str, Any]) -> RecipeManifest:
    return RecipeManifest(
        schema_version=str(data.get("schema_version", _SCHEMA_VERSION)),
        recipe_family=str(data.get("recipe_family", "")),
        recipe_id=str(data.get("recipe_id", "")),
        status=str(data.get("status", "")),
        source_examples=_list(data.get("source_examples")),
        source_evidence=_list(data.get("source_evidence")),
        required_inputs=_list(data.get("required_inputs")),
        optional_inputs=_list(data.get("optional_inputs")),
        produced_outputs=_list(data.get("produced_outputs")),
        layers=_list(data.get("layers")),
        alignment=_dict(data.get("alignment")),
        preprocessing=_dict(data.get("preprocessing")),
        camera_defaults=_dict(data.get("camera_defaults")),
        lighting_defaults=_dict(data.get("lighting_defaults")),
        styling_defaults=_dict(data.get("styling_defaults")),
        annotations_defaults=_dict(data.get("annotations_defaults")),
        render_export_defaults=_dict(data.get("render_export_defaults")),
        support_status=_dict(data.get("support_status")),
        diagnostics=_list(data.get("diagnostics")),
        tests=_list(data.get("tests")),
        golden_fixture_intent=_dict(data.get("golden_fixture_intent")),
        non_goals=_list(data.get("non_goals")),
        open_questions=_list(data.get("open_questions")),
        compiled_label_plans=_dict(data.get("compiled_label_plans")),
        depth_cull=_dict(data.get("depth_cull")),
    )


def manifest_to_dict(manifest: RecipeManifest) -> dict[str, Any]:
    payload = {
        "schema_version": str(manifest.schema_version),
        "recipe_family": str(manifest.recipe_family),
        "recipe_id": str(manifest.recipe_id),
        "status": str(manifest.status),
        "source_examples": [str(path) for path in manifest.source_examples],
        "source_evidence": [_source_evidence_to_dict(item) for item in manifest.source_evidence],
        "required_inputs": [_recipe_input_to_dict(item) for item in manifest.required_inputs],
        "optional_inputs": [_recipe_input_to_dict(item) for item in manifest.optional_inputs],
        "produced_outputs": [_recipe_output_to_dict(item) for item in manifest.produced_outputs],
        "layers": [_recipe_layer_to_dict(item) for item in manifest.layers],
        "alignment": _json_value(manifest.alignment),
        "preprocessing": _json_value(manifest.preprocessing),
        "camera_defaults": _json_value(manifest.camera_defaults),
        "lighting_defaults": _json_value(manifest.lighting_defaults),
        "styling_defaults": _json_value(manifest.styling_defaults),
        "annotations_defaults": _json_value(manifest.annotations_defaults),
        "render_export_defaults": _json_value(manifest.render_export_defaults),
        "support_status": _json_value(manifest.support_status),
        "diagnostics": [str(token) for token in manifest.diagnostics],
        "tests": [str(test) for test in manifest.tests],
        "golden_fixture_intent": _golden_fixture_intent_to_dict(manifest.golden_fixture_intent),
        "non_goals": [str(goal) for goal in manifest.non_goals],
        "open_questions": [str(question) for question in manifest.open_questions],
    }
    # SUTURA compiled-plan fields are serialized only when present so legacy
    # (non-compiled) manifests keep their existing byte format; a compiled
    # manifest always carries both keys and round-trips byte-identically.
    if manifest.compiled_label_plans or manifest.depth_cull:
        payload["compiled_label_plans"] = _json_value(manifest.compiled_label_plans)
        payload["depth_cull"] = _json_value(manifest.depth_cull)
    return payload


def _canonical_json_value(value: Any) -> Any:
    """Canonicalize a JSON value for byte-stable serialization.

    Floats are normalized (``-0.0`` -> ``0.0``; non-finite values are
    rejected) so that ``manifest_to_json`` -> ``manifest_from_json`` ->
    ``manifest_to_json`` round-trips byte-identically: ``json`` serializes
    floats via ``repr``, which is exact for every finite normalized float.
    """
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("RecipeManifest serialization does not permit NaN or infinite floats")
        return 0.0 if value == 0.0 else value
    return value


def manifest_to_json(manifest: RecipeManifest) -> str:
    payload = _canonical_json_value(manifest_to_dict(manifest))
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def manifest_from_json(text: str) -> RecipeManifest:
    return manifest_from_dict(json.loads(text))


def load_manifest(path: str | Path) -> RecipeManifest:
    return manifest_from_json(Path(path).read_text(encoding="utf-8"))


def save_manifest(manifest: RecipeManifest, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest_to_json(manifest), encoding="utf-8")


def recipe_manifest(value: Any, *, golden_fixture_intent: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a deterministic, JSON-safe summary for a MapScene recipe."""

    recipe = dict(_recipe_payload(value))
    terrain = dict(recipe.get("terrain") or {})
    output = dict(recipe.get("output") or {})
    layers = []
    for index, layer in enumerate(recipe.get("layers") or ()):
        if not isinstance(layer, Mapping):
            continue
        layer_id = str(layer.get("layer_id") or _layer_id(layer, f"layer_{index}"))
        metadata = layer.get("metadata")
        layers.append(
            {
                "layer_id": layer_id,
                "kind": str(layer.get("kind") or type(layer).__name__),
                "crs": layer.get("crs"),
                "path": layer.get("path") or layer.get("source"),
                "source_id": metadata.get("source_id") if isinstance(metadata, Mapping) else None,
                "hash": _stable_hash(layer),
            }
        )
    terrain_metadata = terrain.get("metadata")
    manifest = {
        "kind": "mapscene_recipe_manifest",
        "schema": "forge3d.mapscene.recipe_manifest.v1",
        "recipe_hash": _stable_hash(recipe),
        "terrain": {
            "path": terrain.get("path"),
            "crs": terrain.get("crs"),
            "source_id": terrain_metadata.get("source_id") if isinstance(terrain_metadata, Mapping) else None,
            "hash": _stable_hash(terrain),
        },
        "output": {
            "width": int(output.get("width", 0) or 0),
            "height": int(output.get("height", 0) or 0),
            "format": str(output.get("format", "png")),
            "samples": int(output.get("samples", 1) or 1),
            "aovs": list(output.get("aovs") or ()),
            "hdr": bool(output.get("hdr", False)),
        },
        "layers": layers,
    }
    golden_payload = _golden_fixture_payload(golden_fixture_intent)
    if golden_payload is not None:
        manifest["golden_fixture_intent"] = golden_payload
    return manifest


def validate_manifest(manifest: RecipeManifest | Mapping[str, Any], *, repo_root: str | Path | None = None) -> list[str]:
    data = manifest_to_dict(manifest) if isinstance(manifest, RecipeManifest) else dict(manifest)
    emitted: set[str] = set()

    def emit(token: str) -> None:
        emitted.add(token)

    for field_name in _REQUIRED_FIELDS:
        if field_name not in data:
            emit("recipe_manifest_missing_field")

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, str):
        emit("recipe_manifest_invalid_field")
    elif schema_version != _SCHEMA_VERSION:
        emit("recipe_manifest_schema_version_unsupported")

    family = data.get("recipe_family")
    if not isinstance(family, str) or not family:
        emit("recipe_manifest_invalid_field")
    elif family not in _ALLOWED_FAMILIES:
        emit("recipe_manifest_unknown_family")

    status = data.get("status")
    if not isinstance(status, str) or not status:
        emit("recipe_manifest_invalid_field")
    elif status not in _ALLOWED_STATUSES:
        emit("recipe_manifest_invalid_status")
    elif status == "exists_only_as_example_or_script_logic":
        emit("recipe_manifest_example_only")

    _validate_string(data.get("recipe_id"), emit)
    _validate_string_list(data.get("source_examples"), emit)
    _validate_evidence(data.get("source_evidence"), emit)
    required_inputs = _validate_inputs(data.get("required_inputs"), emit)
    optional_inputs = _validate_inputs(data.get("optional_inputs"), emit)
    _validate_outputs(data.get("produced_outputs"), data.get("render_export_defaults"), emit)
    _validate_layers(data.get("layers"), emit)
    _validate_mapping(data.get("alignment"), emit)
    _validate_mapping(data.get("preprocessing"), emit)
    _validate_mapping(data.get("camera_defaults"), emit)
    _validate_mapping(data.get("lighting_defaults"), emit)
    _validate_mapping(data.get("styling_defaults"), emit)
    _validate_mapping(data.get("annotations_defaults"), emit)
    _validate_mapping(data.get("render_export_defaults"), emit)
    _validate_support_status(data.get("support_status"), emit)
    _validate_diagnostics(data.get("diagnostics"), emit)
    _validate_string_list(data.get("tests"), emit)
    _validate_golden(data.get("golden_fixture_intent"), emit)
    _validate_string_list(data.get("non_goals"), emit)
    _validate_string_list(data.get("open_questions"), emit)
    _validate_sources(data, Path(repo_root) if repo_root is not None else Path.cwd(), emit)

    if _spatial_input_count(required_inputs + optional_inputs) > 1 and not _has_alignment(data.get("alignment")):
        emit("recipe_manifest_alignment_unspecified")

    support_status = data.get("support_status")
    if isinstance(support_status, Mapping):
        if support_status.get("mapscene_compatibility") == "partially_proven":
            emit("recipe_manifest_mapscene_partial")
        if any(value == "exists_only_as_example_or_script_logic" for value in support_status.values()):
            emit("recipe_manifest_example_only")

    return [token for token in _DIAGNOSTIC_TOKENS if token in emitted]


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _recipe_payload(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "recipe"):
        value = value.recipe
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("recipe_manifest expects a MapScene, SceneRecipe, or recipe mapping")
    recipe = payload.get("recipe") if isinstance(payload.get("recipe"), Mapping) else payload
    if not isinstance(recipe, Mapping):
        raise TypeError("recipe_manifest could not find a recipe mapping")
    return recipe


def _golden_fixture_payload(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = dict(value)
    tolerance = dict(payload.get("tolerance") or {})
    if "ssim_min" not in tolerance:
        tolerance["ssim_min"] = 0.995
    if "mean_abs_max" not in tolerance:
        tolerance["mean_abs_max"] = 2.0
    return {
        "schema": "forge3d.mapscene.golden_fixture_intent.v1",
        "scene_id": str(payload.get("scene_id") or payload.get("id") or ""),
        "family": str(payload.get("family") or ""),
        "status": str(payload.get("status") or "active"),
        "golden_path": str(payload.get("golden_path") or ""),
        "command": str(payload.get("command") or ""),
        "backend": str(payload.get("backend") or "mapscene"),
        "tolerance": {
            "ssim_min": float(tolerance["ssim_min"]),
            "mean_abs_max": float(tolerance["mean_abs_max"]),
        },
    }


def _source_evidence(value: SourceEvidence | Mapping[str, Any]) -> SourceEvidence:
    if isinstance(value, SourceEvidence):
        return value
    return SourceEvidence(
        path=str(value.get("path", "")),
        line_start=value.get("line_start"),
        line_end=value.get("line_end"),
        note=value.get("note"),
    )


def _recipe_input(value: RecipeInput | Mapping[str, Any]) -> RecipeInput:
    if isinstance(value, RecipeInput):
        return value
    return RecipeInput(
        name=str(value.get("name", "")),
        kind=str(value.get("kind", "")),
        crs=value.get("crs"),
        shape=value.get("shape"),
        format=value.get("format"),
        role=value.get("role"),
    )


def _recipe_output(value: RecipeOutput | Mapping[str, Any]) -> RecipeOutput:
    if isinstance(value, RecipeOutput):
        return value
    return RecipeOutput(
        kind=str(value.get("kind", "")),
        format=str(value.get("format", "")),
        path=value.get("path"),
        deterministic=value.get("deterministic"),
    )


def _recipe_layer(value: RecipeLayer | Mapping[str, Any]) -> RecipeLayer:
    if isinstance(value, RecipeLayer):
        return value
    return RecipeLayer(
        layer_id=str(value.get("layer_id", "")),
        layer_type=str(value.get("layer_type", "")),
        role=str(value.get("role", "")),
        required=bool(value.get("required", False)),
        style_ref=value.get("style_ref"),
    )


def _golden_fixture_intent(value: GoldenFixtureIntent | Mapping[str, Any]) -> GoldenFixtureIntent:
    if isinstance(value, GoldenFixtureIntent):
        return value
    return GoldenFixtureIntent(
        status=str(value.get("status", "missing")),
        source_test=value.get("source_test"),
        source_file=value.get("source_file"),
        note=value.get("note"),
    )


def _source_evidence_to_dict(item: SourceEvidence) -> dict[str, Any]:
    data: dict[str, Any] = {"path": str(item.path)}
    if item.line_start is not None:
        data["line_start"] = int(item.line_start)
    if item.line_end is not None:
        data["line_end"] = int(item.line_end)
    if item.note is not None:
        data["note"] = str(item.note)
    return data


def _recipe_input_to_dict(item: RecipeInput) -> dict[str, Any]:
    data: dict[str, Any] = {"name": str(item.name), "kind": str(item.kind)}
    for key in ("crs", "shape", "format", "role"):
        value = getattr(item, key)
        if value is not None:
            data[key] = _json_value(value)
    return data


def _recipe_output_to_dict(item: RecipeOutput) -> dict[str, Any]:
    data: dict[str, Any] = {"kind": str(item.kind), "format": str(item.format)}
    if item.path is not None:
        data["path"] = str(item.path)
    if item.deterministic is not None:
        data["deterministic"] = bool(item.deterministic)
    return data


def _recipe_layer_to_dict(item: RecipeLayer) -> dict[str, Any]:
    data: dict[str, Any] = {
        "layer_id": str(item.layer_id),
        "layer_type": str(item.layer_type),
        "role": str(item.role),
        "required": bool(item.required),
    }
    if item.style_ref is not None:
        data["style_ref"] = str(item.style_ref)
    return data


def _golden_fixture_intent_to_dict(item: GoldenFixtureIntent) -> dict[str, Any]:
    data: dict[str, Any] = {"status": str(item.status)}
    for key in ("source_test", "source_file", "note"):
        value = getattr(item, key)
        if value is not None:
            data[key] = str(value)
    return data


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def _validate_string(value: Any, emit: Any) -> None:
    if not isinstance(value, str) or not value:
        emit("recipe_manifest_invalid_field")


def _validate_string_list(value: Any, emit: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        emit("recipe_manifest_invalid_field")


def _validate_mapping(value: Any, emit: Any) -> None:
    if not isinstance(value, Mapping):
        emit("recipe_manifest_invalid_field")


def _validate_evidence(value: Any, emit: Any) -> None:
    if not isinstance(value, list):
        emit("recipe_manifest_invalid_field")
        return
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not item.get("path"):
            emit("recipe_manifest_invalid_field")
            continue
        for key in ("line_start", "line_end"):
            if item.get(key) is not None and not isinstance(item.get(key), int):
                emit("recipe_manifest_invalid_field")
        if item.get("note") is not None and not isinstance(item.get("note"), str):
            emit("recipe_manifest_invalid_field")


def _validate_inputs(value: Any, emit: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        emit("recipe_manifest_invalid_field")
        return []
    valid: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            emit("recipe_manifest_invalid_field")
            continue
        if not isinstance(item.get("name"), str) or not isinstance(item.get("kind"), str):
            emit("recipe_manifest_invalid_field")
            continue
        valid.append(item)
        for key in ("crs", "format", "role"):
            if item.get(key) is not None and not isinstance(item.get(key), str):
                emit("recipe_manifest_invalid_field")
    return valid


def _validate_outputs(value: Any, render_defaults: Any, emit: Any) -> None:
    if not isinstance(value, list):
        emit("recipe_manifest_invalid_field")
        return
    for item in value:
        if not isinstance(item, Mapping):
            emit("recipe_manifest_invalid_field")
            continue
        if not isinstance(item.get("kind"), str) or not isinstance(item.get("format"), str):
            emit("recipe_manifest_invalid_field")
            continue
        path = item.get("path")
        if path is not None and not isinstance(path, str):
            emit("recipe_manifest_invalid_field")
        if item.get("deterministic") is not None and not isinstance(item.get("deterministic"), bool):
            emit("recipe_manifest_invalid_field")
        if _requires_render_path(item) and path is None and not _has_example_defined_path_policy(render_defaults):
            emit("recipe_manifest_render_path_unspecified")


def _validate_layers(value: Any, emit: Any) -> None:
    if not isinstance(value, list):
        emit("recipe_manifest_invalid_field")
        return
    for item in value:
        if not isinstance(item, Mapping):
            emit("recipe_manifest_invalid_field")
            continue
        for key in ("layer_id", "layer_type", "role"):
            if not isinstance(item.get(key), str) or not item.get(key):
                emit("recipe_manifest_invalid_field")
        if not isinstance(item.get("required"), bool):
            emit("recipe_manifest_invalid_field")
        layer_type = item.get("layer_type")
        if isinstance(layer_type, str) and layer_type not in _ALLOWED_LAYER_TYPES:
            emit("recipe_manifest_unsupported_layer")


def _validate_support_status(value: Any, emit: Any) -> None:
    if not isinstance(value, Mapping):
        emit("recipe_manifest_invalid_field")
        return
    for status in value.values():
        if not isinstance(status, str):
            emit("recipe_manifest_invalid_field")
        elif status not in _ALLOWED_STATUSES:
            emit("recipe_manifest_invalid_status")


def _validate_diagnostics(value: Any, emit: Any) -> None:
    if not isinstance(value, list):
        emit("recipe_manifest_invalid_field")
        return
    for token in value:
        if not isinstance(token, str) or token not in _DIAGNOSTIC_SET:
            emit("recipe_manifest_invalid_field")


def _validate_golden(value: Any, emit: Any) -> None:
    if not isinstance(value, Mapping):
        emit("recipe_manifest_invalid_field")
        return
    status = value.get("status")
    if not isinstance(status, str) or status not in _GOLDEN_STATUSES:
        emit("recipe_manifest_invalid_field")
    elif status != "exists":
        emit("recipe_manifest_golden_not_selected")


def _validate_sources(data: Mapping[str, Any], repo_root: Path, emit: Any) -> None:
    for source in data.get("source_examples") if isinstance(data.get("source_examples"), list) else []:
        if isinstance(source, str) and source and not _is_repo_relative_path(source):
            emit("recipe_manifest_invalid_field")
            continue
        if isinstance(source, str) and source and not _local_path_exists(repo_root, source):
            emit("recipe_manifest_missing_source")
    evidence = data.get("source_evidence")
    if isinstance(evidence, list):
        for item in evidence:
            path = item.get("path") if isinstance(item, Mapping) else None
            if isinstance(path, str) and path and not _is_repo_relative_path(path):
                emit("recipe_manifest_invalid_field")
                continue
            if isinstance(path, str) and path and not _local_path_exists(repo_root, path):
                emit("recipe_manifest_missing_source")


def _is_repo_relative_path(path: str) -> bool:
    candidate = Path(path)
    return not candidate.is_absolute() and not candidate.drive and ".." not in candidate.parts


def _local_path_exists(repo_root: Path, path: str) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.exists()


def _spatial_input_count(inputs: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for item in inputs:
        text = " ".join(str(item.get(key, "")).lower() for key in ("name", "kind", "role"))
        if any(hint in text for hint in _SPATIAL_INPUT_HINTS):
            count += 1
    return count


def _has_alignment(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return any(value.get(key) not in (None, "", [], {}) for key in ("crs", "extent", "resolution", "transform", "notes"))


def _requires_render_path(output: Mapping[str, Any]) -> bool:
    kind = str(output.get("kind", "")).lower()
    fmt = str(output.get("format", "")).lower()
    return kind in {"render", "export", "snapshot", "poster", "bundle"} or fmt in {"png", "rgba", "jpg", "jpeg", "tif", "tiff"}


def _has_example_defined_path_policy(render_defaults: Any) -> bool:
    return isinstance(render_defaults, Mapping) and (
        render_defaults.get("path") == "example_defined"
        or render_defaults.get("path_policy") == "example_defined"
    )


class _CallableRecipeManifestModule(types.ModuleType):
    def __call__(
        self,
        value: Any,
        *,
        golden_fixture_intent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return recipe_manifest(value, golden_fixture_intent=golden_fixture_intent)


sys.modules[__name__].__class__ = _CallableRecipeManifestModule
