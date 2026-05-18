"""Typed MapScene recipe models for offline map-production workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diagnostics import (
    Diagnostic,
    LayerSummary,
    RenderFailurePolicy,
    ValidationReport,
    crs_mismatch_diagnostic,
    estimated_gpu_memory_diagnostic,
    experimental_feature_diagnostic,
    missing_texture_path_diagnostic,
    missing_external_asset_diagnostic,
    missing_label_field_diagnostic,
    missing_uvs_diagnostic,
    placeholder_fallback_diagnostic,
    pro_gated_path_diagnostic,
    python_public_3dtiles_incomplete_diagnostic,
    unicode_coverage_gap_diagnostic,
    unavailable_cache_lod_stats_diagnostic,
    unavailable_terrain_sampler_diagnostic,
    unsupported_instancing_path_diagnostic,
    unsupported_texture_format_diagnostic,
    unsupported_tile_feature_diagnostic,
    unsupported_tile_format_diagnostic,
    validate_label_support,
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value.keys(), key=str):
            result[str(key)] = _json_safe(value[key])
        return result
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"MapScene recipe values must be JSON-serializable, got {type(value).__name__}")


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return _json_safe(dict(value or {}))


def _stable_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sequence(value: Sequence[Any] | None) -> list[Any]:
    return [_json_safe(item) for item in (value or ())]


def _layer_id(layer: Any, fallback: str) -> str:
    return str(getattr(layer, "layer_id", None) or getattr(layer, "name", None) or fallback)


def _same_crs(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return True
    try:
        from .crs import _crs_equal

        return bool(_crs_equal(str(left), str(right)))
    except Exception:
        return str(left).strip().lower() == str(right).strip().lower()


def _metadata_dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _has_explicit_crs_policy(layer: Any) -> bool:
    metadata = _metadata_dict(getattr(layer, "metadata", None))
    policy = str(metadata.get("crs_policy", "")).lower()
    return bool(metadata.get("crs_transform")) or policy in {
        "compatible",
        "explicit_transform",
        "transform_provided",
    }


def _glyphs_from_atlas(glyph_atlas: Mapping[str, Any] | None) -> set[str] | None:
    if not glyph_atlas:
        return None
    glyphs = glyph_atlas.get("glyphs")
    if glyphs is None:
        return None
    return {str(glyph) for glyph in glyphs}


def _dimension_memory_bytes(metadata: Mapping[str, Any] | None, *, channels: int = 4) -> int | None:
    data = _metadata_dict(metadata)
    explicit = _explicit_memory_bytes(data)
    if explicit is not None:
        return explicit
    width = data.get("width")
    height = data.get("height")
    if width is None or height is None:
        dimensions = data.get("dimensions")
        if isinstance(dimensions, Sequence) and not isinstance(dimensions, (str, bytes)) and len(dimensions) >= 2:
            width, height = dimensions[0], dimensions[1]
    try:
        width_i = int(width)
        height_i = int(height)
    except (TypeError, ValueError):
        return None
    if width_i <= 0 or height_i <= 0:
        return None
    return width_i * height_i * channels


def _explicit_memory_bytes(metadata: Mapping[str, Any] | None) -> int | None:
    data = _metadata_dict(metadata)
    for key in ("memory_estimate_bytes", "estimated_memory_bytes", "gpu_memory_bytes"):
        if key not in data:
            continue
        try:
            value = int(data[key])
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


def _point_cloud_memory_bytes(point_count: int | None) -> int | None:
    if point_count is None:
        return None
    try:
        count = int(point_count)
    except (TypeError, ValueError):
        return None
    if count < 0:
        return None
    return count * 24


def _geometry_memory_bytes(geometry_count: int | None) -> int | None:
    if geometry_count is None:
        return None
    try:
        count = int(geometry_count)
    except (TypeError, ValueError):
        return None
    if count < 0:
        return None
    return count * 96


def _output_memory_bytes(output: "OutputSpec | None") -> int | None:
    if output is None:
        return None
    return int(output.width) * int(output.height) * 4


def _diagnostic_codes_for_layer(diagnostics: Sequence[Diagnostic], layer_id: str) -> list[str]:
    return sorted({diagnostic.code for diagnostic in diagnostics if diagnostic.layer_id == layer_id})


def _merge_report(
    report: ValidationReport,
    *,
    diagnostics: list[Diagnostic],
    layer_summaries: list[LayerSummary],
    supported_features: dict[str, str],
    unsupported_features: dict[str, str],
) -> None:
    diagnostics.extend(report.diagnostics)
    layer_summaries.extend(report.layer_summaries)
    supported_features.update(report.supported_features or {})
    unsupported_features.update(report.unsupported_features or {})


def _virtual_texture_report_from_metadata(metadata: Mapping[str, Any] | None) -> ValidationReport | None:
    data = _metadata_dict(metadata)
    config = data.get("virtual_texture") or data.get("vt")
    if not isinstance(config, Mapping):
        return None

    families = config.get("families") or config.get("layers") or ("albedo",)
    normalized_families: list[str] = []
    for item in families:
        if isinstance(item, Mapping):
            normalized_families.append(str(item.get("family", "albedo")))
        else:
            normalized_families.append(str(item))
    if not normalized_families:
        normalized_families = ["albedo"]

    from .terrain_params import TerrainVTSettings, VTLayerFamily, validate_terrain_vt_support

    settings = TerrainVTSettings(
        enabled=bool(config.get("enabled", True)),
        layers=[VTLayerFamily(family=family) for family in normalized_families],
        atlas_size=int(config.get("atlas_size", 4096)),
        residency_budget_mb=float(config.get("residency_budget_mb", 256.0)),
        max_mip_levels=int(config.get("max_mip_levels", 8)),
        use_feedback=bool(config.get("use_feedback", True)),
    )
    return validate_terrain_vt_support(settings, layer_id="terrain.vt")


def _unsupported_feature_diagnostic(
    feature: str,
    *,
    layer_id: str | None = None,
    object_id: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code="unsupported_feature",
        severity="error",
        message="Requested MapScene feature is not supported by the MVP workflow.",
        remediation="Remove the feature or use a documented supported MapScene path.",
        support_level="unsupported",
        layer_id=layer_id,
        object_id=object_id,
        details={"feature": feature},
    )


def _unsupported_layer_type_diagnostic(layer: Any, *, layer_id: str) -> Diagnostic:
    return Diagnostic(
        code="unsupported_layer_type",
        severity="error",
        message="Recipe layer type is not supported by MapScene validation.",
        remediation="Use a typed MapScene recipe layer or keep this path out of the MVP workflow.",
        support_level="unsupported",
        layer_id=layer_id,
        details={"python_type": type(layer).__name__},
    )


def _unsupported_output_format_diagnostic(output_format: str) -> Diagnostic:
    return Diagnostic(
        code="unsupported_output_format",
        severity="fatal",
        message="MapScene MVP rendering supports PNG output only.",
        remediation="Set OutputSpec.format to 'png' before rendering.",
        support_level="unsupported",
        details={"format": output_format, "supported_formats": ["png"]},
    )


def _missing_source_identity_diagnostic(
    layer_type: str,
    *,
    layer_id: str,
    source_fields: Sequence[str],
) -> Diagnostic:
    return Diagnostic(
        code="missing_source_identity",
        severity="error",
        message="MapScene layer is missing source identity required for review and render preparation.",
        remediation="Set a source path, stable source metadata, or an equivalent source identifier.",
        support_level="unsupported",
        layer_id=layer_id,
        details={"layer_type": layer_type, "source_fields": [str(field) for field in source_fields]},
    )


def _missing_renderable_data_diagnostic(
    layer_type: str,
    *,
    layer_id: str,
    required_any_of: Sequence[str],
) -> Diagnostic:
    return Diagnostic(
        code="missing_renderable_data",
        severity="error",
        message="MapScene layer has no renderable source data for the requested workflow.",
        remediation="Provide at least one supported source field before validation, render, or bundle save.",
        support_level="unsupported",
        layer_id=layer_id,
        details={"layer_type": layer_type, "required_any_of": [str(field) for field in required_any_of]},
    )


def _missing_crs_diagnostic(scene_crs: str | None, *, layer_id: str) -> Diagnostic:
    return Diagnostic(
        code="missing_crs",
        severity="error",
        message="Layer CRS metadata is missing and cannot be assumed compatible with the scene CRS.",
        remediation="Set the layer CRS or provide an explicit compatible CRS policy before rendering.",
        support_level="unsupported",
        layer_id=layer_id,
        details={"layer_crs": None, "scene_crs": scene_crs},
    )


_DECLARED_AVAILABLE_ASSET_STATUSES = frozenset({"available", "fixture", "external_reference", "generated"})
_TERRAIN_ASSET_EXTENSIONS = (".tif", ".tiff", ".npy")
_RASTER_ASSET_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
_POINT_CLOUD_ASSET_EXTENSIONS = (".las", ".laz", ".copc", ".ept.json")
_BUILDING_ASSET_EXTENSIONS = (".geojson", ".json", ".fgb", ".gpkg", ".shp")
_BUILDING_TEXTURE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
def _unsupported_asset_format_diagnostic(
    layer_type: str,
    *,
    layer_id: str,
    path: str,
    supported_extensions: Sequence[str],
) -> Diagnostic:
    return Diagnostic(
        code="unsupported_asset_format",
        severity="error",
        message="MapScene layer references an asset format that is not supported by the MVP workflow.",
        remediation="Use a supported asset format or keep this layer out of the MVP render path.",
        support_level="unsupported",
        layer_id=layer_id,
        details={
            "layer_type": layer_type,
            "path": path,
            "supported_extensions": sorted(str(extension) for extension in supported_extensions),
        },
    )


def _declares_available_asset(metadata: Mapping[str, Any] | None) -> bool:
    status = _metadata_dict(metadata).get("asset_status")
    return str(status).strip().lower() in _DECLARED_AVAILABLE_ASSET_STATUSES


def _is_external_uri(path: str) -> bool:
    return "://" in path or path.startswith("urn:")


def _has_supported_extension(path: str, supported_extensions: Sequence[str]) -> bool:
    normalized = path.lower()
    return any(normalized.endswith(extension.lower()) for extension in supported_extensions)


def _asset_path_diagnostics(
    layer_type: str,
    *,
    layer_id: str,
    path: str | None,
    metadata: Mapping[str, Any] | None,
    supported_extensions: Sequence[str],
) -> list[Diagnostic]:
    if not path:
        return []

    diagnostics: list[Diagnostic] = []
    path_value = str(path)
    if not _has_supported_extension(path_value, supported_extensions):
        diagnostics.append(
            _unsupported_asset_format_diagnostic(
                layer_type,
                layer_id=layer_id,
                path=path_value,
                supported_extensions=supported_extensions,
            )
        )

    if _is_external_uri(path_value) or _declares_available_asset(metadata):
        return diagnostics

    if not Path(path_value).exists():
        diagnostics.append(missing_external_asset_diagnostic(layer_type, layer_id=layer_id, path=path_value))

    return diagnostics


def _texture_format_from_path(path: str | None) -> str | None:
    if not path:
        return None
    suffix = Path(str(path)).suffix.lower().lstrip(".")
    return suffix or None


def _texture_path_available(path: str | None, metadata: Mapping[str, Any] | None) -> bool:
    if not path:
        return False
    path_value = str(path)
    return _is_external_uri(path_value) or _declares_available_asset(metadata) or Path(path_value).exists()


def _iter_textured_material_intents(metadata: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    data = _metadata_dict(metadata)
    intents = data.get("textured_materials")
    if intents is None:
        single = data.get("textured_material") or data.get("texture_material")
        intents = [single] if isinstance(single, Mapping) else []
    if not isinstance(intents, Sequence) or isinstance(intents, (str, bytes)):
        return []
    result: list[dict[str, Any]] = []
    for item in intents:
        if isinstance(item, Mapping):
            result.append(_metadata(item))
    return result


def _p2_building_texture_diagnostics(
    metadata: Mapping[str, Any] | None,
    *,
    layer_id: str,
) -> tuple[list[Diagnostic], dict[str, Any], str | None]:
    diagnostics: list[Diagnostic] = []
    details: dict[str, Any] = {}
    strongest_support: str | None = None
    intents = _iter_textured_material_intents(metadata)
    if not intents:
        return diagnostics, details, strongest_support

    details["textured_materials"] = []
    for index, intent in enumerate(intents):
        material_id = str(intent.get("material_id") or f"material_{index}")
        object_id = str(intent.get("object_id") or material_id)
        texture_path = intent.get("albedo_texture") or intent.get("texture_path")
        texture_path_value = str(texture_path) if texture_path else ""
        texture_format = str(
            intent.get("texture_format") or _texture_format_from_path(texture_path_value) or "unknown"
        ).lower().lstrip(".")
        uv_available = bool(intent.get("uv_available", intent.get("has_uvs", False)))
        scalar_fallback = bool(intent.get("scalar_fallback") or intent.get("fallback") == "scalar")

        details["textured_materials"].append(
            {
                "material_id": material_id,
                "object_id": object_id,
                "albedo_texture": texture_path_value,
                "texture_format": texture_format,
                "uv_available": uv_available,
                "scalar_fallback": scalar_fallback,
            }
        )

        if not _texture_path_available(texture_path_value, metadata):
            diagnostics.append(
                missing_texture_path_diagnostic(
                    texture_path_value,
                    layer_id=layer_id,
                    object_id=object_id,
                    material_id=material_id,
                )
            )
            strongest_support = "unsupported"
        if not uv_available:
            diagnostics.append(missing_uvs_diagnostic(layer_id=layer_id, object_id=object_id, material_id=material_id))
            strongest_support = "unsupported"
        if texture_format and f".{texture_format}" not in _BUILDING_TEXTURE_EXTENSIONS:
            diagnostics.append(
                unsupported_texture_format_diagnostic(
                    texture_format,
                    layer_id=layer_id,
                    object_id=object_id,
                    path=texture_path_value or None,
                )
            )
            strongest_support = "unsupported"
        if scalar_fallback:
            diagnostics.append(
                placeholder_fallback_diagnostic(
                    "building textured material scalar fallback",
                    layer_id=layer_id,
                    object_id=object_id,
                )
            )
            strongest_support = "placeholder/fallback"

    if not diagnostics:
        first_intent = details["textured_materials"][0]
        diagnostics.append(
            _unsupported_feature_diagnostic(
                "building textured PBR render path",
                layer_id=layer_id,
                object_id=str(first_intent["object_id"]),
            )
        )
        strongest_support = "unsupported"

    details["textured_material_status"] = strongest_support or "unsupported"
    return diagnostics, details, strongest_support


def _p2_resource_availability_diagnostics(
    metadata: Mapping[str, Any] | None,
    *,
    layer_id: str,
    layer_type: str,
) -> tuple[list[Diagnostic], dict[str, Any], str | None]:
    data = _metadata_dict(metadata)
    diagnostics: list[Diagnostic] = []
    details: dict[str, Any] = {}
    strongest_support: str | None = None

    unavailable_stats = data.get("unavailable_cache_lod_stats") or data.get("unavailable_stats")
    if unavailable_stats:
        if isinstance(unavailable_stats, str):
            stats = [unavailable_stats]
        else:
            stats = [str(item) for item in unavailable_stats]
        diagnostics.append(
            unavailable_cache_lod_stats_diagnostic(layer_type, stats, layer_id=layer_id)
        )
        details["unavailable_cache_lod_stats"] = sorted(stats)
        strongest_support = "underdeveloped"

    instancing = data.get("instancing") or data.get("instancing_status")
    if isinstance(instancing, Mapping):
        requested = bool(instancing.get("requested", True))
        support_level = str(instancing.get("support_level", instancing.get("status", ""))).strip()
        path = str(instancing.get("path") or f"{layer_type} instancing")
        details["instancing_status"] = _metadata(instancing)
        if requested and support_level in {"unsupported", "missing", "unavailable"}:
            diagnostics.append(
                unsupported_instancing_path_diagnostic(
                    path,
                    layer_id=layer_id,
                    object_id=str(instancing.get("object_id") or "instancing"),
                )
            )
            strongest_support = "unsupported"

    return diagnostics, details, strongest_support


def _large_scene_resource_summary(
    layer_summaries: Sequence[LayerSummary],
    *,
    estimated_memory: int | None,
    diagnostics: Sequence[Diagnostic],
) -> LayerSummary:
    memory_estimates = [
        {
            "layer_id": summary.layer_id,
            "layer_type": summary.layer_type,
            "memory_estimate_bytes": int(summary.memory_estimate_bytes),
        }
        for summary in layer_summaries
        if summary.memory_estimate_bytes is not None
    ]
    memory_estimates.sort(key=lambda item: (item["layer_id"], item["layer_type"]))

    cache_lod_status: list[dict[str, Any]] = []
    unavailable_stats: list[dict[str, Any]] = []
    instancing_status: dict[str, Any] = {}
    for summary in layer_summaries:
        details = _metadata_dict(summary.details)
        cache_lod: dict[str, Any] = {"layer_id": summary.layer_id, "layer_type": summary.layer_type}
        has_cache_lod = False
        for key in ("cache_stats", "cache_budget", "lod", "unavailable_cache_lod_stats"):
            if key in details:
                cache_lod[key] = _json_safe(details[key])
                has_cache_lod = True
        if has_cache_lod:
            cache_lod_status.append(cache_lod)
        if "unavailable_cache_lod_stats" in details:
            unavailable_stats.append(
                {
                    "layer_id": summary.layer_id,
                    "stats": sorted(str(item) for item in details["unavailable_cache_lod_stats"]),
                }
            )
        if "instancing_status" in details:
            instancing_status[summary.layer_id] = _json_safe(details["instancing_status"])
        if summary.memory_estimate_bytes is None:
            unavailable_stats.append({"layer_id": summary.layer_id, "stats": ["memory"]})

    bottleneck_layers = [
        {
            "layer_id": item["layer_id"],
            "layer_type": item["layer_type"],
            "memory_estimate_bytes": item["memory_estimate_bytes"],
        }
        for item in sorted(
            memory_estimates,
            key=lambda value: (-int(value["memory_estimate_bytes"]), value["layer_type"], value["layer_id"]),
        )
        if int(item["memory_estimate_bytes"]) > 0
    ]
    bottleneck_layer_types: list[str] = []
    for item in bottleneck_layers:
        layer_type = str(item["layer_type"])
        if layer_type not in bottleneck_layer_types:
            bottleneck_layer_types.append(layer_type)

    diagnostic_codes = sorted({diagnostic.code for diagnostic in diagnostics})
    if any(diagnostic.code == "unsupported_instancing_path" for diagnostic in diagnostics):
        support_level = "unsupported"
    elif unavailable_stats or any(diagnostic.code == "unavailable_cache_lod_stats" for diagnostic in diagnostics):
        support_level = "underdeveloped"
    else:
        support_level = "supported"

    return LayerSummary(
        layer_id="large_scene.resources",
        layer_type="large_scene_resource_summary",
        support_level=support_level,
        diagnostic_codes=diagnostic_codes,
        memory_estimate_bytes=estimated_memory,
        details={
            "memory_estimates": memory_estimates,
            "total_estimated_gpu_memory_bytes": estimated_memory,
            "cache_lod_status": sorted(cache_lod_status, key=lambda item: item["layer_id"]),
            "instancing_status": instancing_status,
            "unavailable_stats": sorted(unavailable_stats, key=lambda item: item["layer_id"]),
            "bottleneck_layers": bottleneck_layers,
            "bottleneck_layer_types": bottleneck_layer_types,
        },
    )


def _source_kind(source: Any, metadata: Mapping[str, Any] | None) -> str | None:
    if isinstance(source, Mapping):
        for key in ("kind", "type", "source_format", "format"):
            if source.get(key):
                return str(source[key]).lower()
        path = source.get("path")
        if path and str(path).lower().endswith("tileset.json"):
            return "3dtiles"
    data = _metadata_dict(metadata)
    for key in ("kind", "type", "source_format", "format"):
        if data.get(key):
            return str(data[key]).lower()
    if isinstance(source, str) and source.lower().endswith("tileset.json"):
        return "3dtiles"
    return None


def _source_path(source: Any) -> str | None:
    if isinstance(source, str):
        return source
    if isinstance(source, Mapping):
        path = source.get("path")
        return str(path) if path else None
    return None


def _has_identity_path_or_metadata(path: Any, metadata: Mapping[str, Any] | None) -> bool:
    return bool(path) or bool(_metadata_dict(metadata))


def _has_labels_or_plan(layer: "LabelLayer") -> bool:
    return bool(layer.labels) or layer.plan is not None


def _label_plan_seed(recipe: "SceneRecipe", layer: "LabelLayer") -> int:
    metadata = _metadata_dict(layer.metadata)
    if "seed" in metadata:
        try:
            return int(metadata["seed"])
        except (TypeError, ValueError):
            pass
    if recipe.reproducibility_profile is not None:
        return int(recipe.reproducibility_profile.seed)
    return 0


def _label_plan_from_layer(
    layer: "LabelLayer",
    *,
    recipe: "SceneRecipe",
    terrain: "TerrainSource",
) -> Any:
    from .label_plan import LabelPlan

    if layer.plan is not None:
        if isinstance(layer.plan, LabelPlan):
            return layer.plan
        if isinstance(layer.plan, Mapping):
            return LabelPlan.from_dict(layer.plan)
        if hasattr(layer.plan, "to_dict") and callable(layer.plan.to_dict):
            return LabelPlan.from_dict(layer.plan.to_dict())
        raise TypeError("LabelLayer.plan must be a LabelPlan or LabelPlan-compatible mapping")

    keepouts = tuple(recipe.map_furniture.keepouts or ()) if recipe.map_furniture is not None else ()
    return LabelPlan.compile(
        labels=layer.labels or (),
        camera=recipe.camera,
        viewport=recipe.output,
        terrain=terrain,
        keepouts=keepouts,
        priority_rules=layer.priority_rules or (),
        typography=layer.typography or {},
        glyph_atlas=layer.glyph_atlas,
        seed=_label_plan_seed(recipe, layer),
    )


def _diagnostic_for_layer(diagnostic: Diagnostic, layer_id: str) -> Diagnostic:
    if diagnostic.layer_id in {None, "labels", layer_id}:
        payload = diagnostic.to_dict()
        payload["layer_id"] = layer_id
        return Diagnostic.from_dict(payload)
    return diagnostic


def _feature_geometry(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    geometry = feature.get("geometry")
    if isinstance(geometry, Mapping):
        return geometry
    geometry_type = feature.get("geometry_type") or feature.get("type")
    coordinates = feature.get("coordinates") or feature.get("position") or feature.get("world_pos")
    if geometry_type and coordinates is not None:
        return {"type": str(geometry_type), "coordinates": coordinates}
    return {}


def _is_coordinate(value: Any) -> bool:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return False
    if len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _valid_geometry(geometry: Mapping[str, Any]) -> tuple[str | None, str | None]:
    geometry_type = str(geometry.get("type", "") or "")
    coordinates = geometry.get("coordinates")
    geometry_key = geometry_type.lower()
    if geometry_key == "point":
        return ("Point", "point") if _is_coordinate(coordinates) else (None, None)
    if geometry_key == "linestring":
        if (
            isinstance(coordinates, Sequence)
            and not isinstance(coordinates, (str, bytes))
            and len(coordinates) >= 2
            and all(_is_coordinate(point) for point in coordinates)
        ):
            return "LineString", "line"
        return None, None
    if geometry_key == "polygon":
        if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes)) or not coordinates:
            return None, None
        outer = coordinates[0]
        if (
            isinstance(outer, Sequence)
            and not isinstance(outer, (str, bytes))
            and len(outer) >= 4
            and all(_is_coordinate(point) for point in outer)
        ):
            return "Polygon", "polygon"
        return None, None
    if geometry_type:
        return geometry_type, None
    return None, None


def _feature_properties(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    properties = feature.get("properties")
    if isinstance(properties, Mapping):
        return properties
    return {str(key): value for key, value in feature.items() if key not in {"geometry", "coordinates", "type"}}


def _feature_id(feature: Mapping[str, Any], index: int) -> str:
    return str(feature.get("id") or feature.get("feature_id") or feature.get("source_id") or f"feature-{index}")


def _style_text_expression(style_layer: Any) -> Any:
    if isinstance(style_layer, Mapping):
        layout = style_layer.get("layout")
        if isinstance(layout, Mapping):
            return layout.get("text-field")
        return style_layer.get("text") or style_layer.get("text_field")
    layout = getattr(style_layer, "layout", None)
    if layout is not None:
        return getattr(layout, "text_field", None)
    return getattr(style_layer, "text", None) or getattr(style_layer, "text_field", None)


def _label_text_from_expression(expression: Any, properties: Mapping[str, Any]) -> tuple[str, str | None]:
    if expression is None:
        expression = "name"
    if isinstance(expression, str):
        if expression.startswith("{") and expression.endswith("}") and len(expression) > 2:
            field = expression[1:-1]
            if field not in properties:
                return "", field
            return str(properties[field]), None
        if expression in properties:
            return str(properties[expression]), None
        return str(expression), None
    try:
        from .style_expressions import EvalContext, evaluate

        value = evaluate(expression, EvalContext(properties=dict(properties)))
    except Exception:
        return "", str(expression)
    if value is None:
        if isinstance(expression, Sequence) and not isinstance(expression, (str, bytes)) and len(expression) >= 2:
            return "", str(expression[1])
        return "", str(expression)
    return str(value), None


def _transform_label_geometry(
    geometry: Mapping[str, Any],
    *,
    from_crs: str | None,
    to_crs: str | None,
) -> Mapping[str, Any]:
    if not from_crs or not to_crs or _same_crs(from_crs, to_crs):
        return geometry

    import numpy as np

    from .crs import transform_coords

    geometry_type = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates")
    if geometry_type == "Point" and _is_coordinate(coordinates):
        xy = np.asarray([[float(coordinates[0]), float(coordinates[1])]], dtype=np.float64)
        transformed = transform_coords(xy, from_crs, to_crs)[0]
        updated = [float(transformed[0]), float(transformed[1])]
        if len(coordinates) > 2:
            updated.append(float(coordinates[2]))
        return {"type": geometry_type, "coordinates": updated}
    if geometry_type == "LineString" and isinstance(coordinates, Sequence) and not isinstance(coordinates, (str, bytes)):
        points = [point for point in coordinates if _is_coordinate(point)]
        transformed = transform_coords(
            np.asarray([[float(point[0]), float(point[1])] for point in points], dtype=np.float64),
            from_crs,
            to_crs,
        )
        updated = []
        for point, xy in zip(points, transformed):
            new_point = [float(xy[0]), float(xy[1])]
            if len(point) > 2:
                new_point.append(float(point[2]))
            updated.append(new_point)
        return {"type": geometry_type, "coordinates": updated}
    if geometry_type == "Polygon" and isinstance(coordinates, Sequence) and not isinstance(coordinates, (str, bytes)):
        rings = []
        for ring in coordinates:
            if not isinstance(ring, Sequence) or isinstance(ring, (str, bytes)):
                rings.append(ring)
                continue
            points = [point for point in ring if _is_coordinate(point)]
            if not points:
                rings.append([])
                continue
            transformed = transform_coords(
                np.asarray([[float(point[0]), float(point[1])] for point in points], dtype=np.float64),
                from_crs,
                to_crs,
            )
            updated_ring = []
            for point, xy in zip(points, transformed):
                new_point = [float(xy[0]), float(xy[1])]
                if len(point) > 2:
                    new_point.append(float(point[2]))
                updated_ring.append(new_point)
            rings.append(updated_ring)
        return {"type": geometry_type, "coordinates": rings}
    return geometry


def _call_label_terrain_sampler(sampler: Any, coordinates: Sequence[Any]) -> Mapping[str, Any]:
    if sampler is None or not callable(sampler):
        return {}
    x = float(coordinates[0])
    y = float(coordinates[1])
    z = float(coordinates[2]) if len(coordinates) > 2 else 0.0
    for args in ((x, y, z), (x, y), (coordinates,)):
        try:
            result = sampler(*args)
        except TypeError:
            continue
        if isinstance(result, Mapping):
            return _metadata(result)
        if result is not None:
            return {"elevation": float(result), "source": type(sampler).__name__, "visible": True}
    return {}


def _sample_label_geometry(
    geometry: Mapping[str, Any],
    *,
    terrain_sampler: Any | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    coordinates = geometry.get("coordinates")
    if str(geometry.get("type")) != "Point" or not _is_coordinate(coordinates):
        return geometry, {}
    sample = _call_label_terrain_sampler(terrain_sampler, coordinates)
    if "elevation" not in sample:
        return geometry, sample
    updated = [float(coordinates[0]), float(coordinates[1]), float(sample["elevation"])]
    return {"type": "Point", "coordinates": updated}, sample


def _atlas_glyph_set(glyph_atlas: Any) -> set[str] | None:
    if glyph_atlas is None:
        return None
    if hasattr(glyph_atlas, "glyphs"):
        return {str(glyph) for glyph in getattr(glyph_atlas, "glyphs")}
    if isinstance(glyph_atlas, Mapping):
        glyphs = glyph_atlas.get("glyphs")
        if isinstance(glyphs, Mapping):
            result = set()
            for key in glyphs:
                try:
                    result.add(chr(int(key)))
                except (TypeError, ValueError):
                    result.add(str(key))
            return result
        if isinstance(glyphs, (set, frozenset, list, tuple)):
            return {chr(glyph) if isinstance(glyph, int) else str(glyph) for glyph in glyphs}
    if isinstance(glyph_atlas, (set, frozenset, list, tuple)):
        return {chr(glyph) if isinstance(glyph, int) else str(glyph) for glyph in glyph_atlas}
    return None


def _render_payload(recipe: "SceneRecipe") -> dict[str, Any]:
    payload = recipe.to_dict()
    output = dict(payload.get("output") or {})
    output["path"] = None
    payload["output"] = output
    return payload


def _hash_int(value: Any, *, salt: str = "") -> int:
    return int(_stable_hash({"salt": salt, "value": value})[:8], 16)


def _rgb(value: Any, *, salt: str = "") -> tuple[int, int, int]:
    digest = _stable_hash({"salt": salt, "value": value})
    return int(digest[0:2], 16), int(digest[2:4], 16), int(digest[4:6], 16)


def _point_to_pixel(point: Sequence[Any], width: int, height: int) -> tuple[int, int]:
    x = float(point[0]) if len(point) > 0 else 0.0
    y = float(point[1]) if len(point) > 1 else 0.0
    px = int(round(x * (width - 1))) if 0.0 <= x <= 1.0 else int(round(x)) % max(1, width)
    py = int(round(y * (height - 1))) if 0.0 <= y <= 1.0 else int(round(y)) % max(1, height)
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))


def _draw_pixel_block(image: Any, x: int, y: int, color: tuple[int, int, int], radius: int = 1) -> None:
    height, width = image.shape[:2]
    x0 = max(0, int(x) - radius)
    x1 = min(width, int(x) + radius + 1)
    y0 = max(0, int(y) - radius)
    y1 = min(height, int(y) + radius + 1)
    image[y0:y1, x0:x1, :3] = color


def _draw_line(image: Any, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    import numpy as np

    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    for x, y in zip(xs, ys):
        _draw_pixel_block(image, int(round(x)), int(round(y)), color, radius=1)


def _geometry_points(geometry: Mapping[str, Any]) -> list[Sequence[Any]]:
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return []
    geometry_type = str(geometry.get("type", "")).lower()
    if geometry_type == "point":
        return [coordinates]
    if geometry_type == "linestring":
        return list(coordinates)
    if geometry_type == "polygon":
        rings = list(coordinates)
        return list(rings[0]) if rings else []
    return []


def _composite_recipe_layers(
    base: Any,
    recipe: "SceneRecipe",
    plans: Mapping[str, Any],
    *,
    include_raster: bool = True,
    include_point_cloud: bool = True,
) -> Any:
    import numpy as np

    output = recipe.output
    width = int(output.width)
    height = int(output.height)
    yy, xx = np.mgrid[0:height, 0:width]

    for layer in recipe.layers:
        layer_id = _layer_id(layer, "layer")
        if isinstance(layer, RasterOverlay) and include_raster:
            overlay = _load_native_raster_overlay(layer)
            alpha = max(0.0, min(1.0, float(layer.opacity)))
            if overlay is not None:
                src_h, src_w = overlay.shape[:2]
                sample_y = np.clip((yy * src_h // max(height, 1)), 0, src_h - 1)
                sample_x = np.clip((xx * src_w // max(width, 1)), 0, src_w - 1)
                sampled = overlay[sample_y, sample_x]
                sampled_alpha = (sampled[..., 3:4].astype(np.float32) / 255.0) * alpha
                blended = (
                    base[..., :3].astype(np.float32) * (1.0 - sampled_alpha)
                    + sampled[..., :3].astype(np.float32) * sampled_alpha
                )
                base[..., :3] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            else:
                color = np.array(_rgb(layer.to_dict(), salt="raster"), dtype=np.uint8)
                fallback_alpha = alpha * 0.45
                mask = ((xx + yy + _hash_int(layer.to_dict(), salt="raster-mask")) % 5) < 3
                blended = (base[..., :3].astype(np.float32) * (1.0 - fallback_alpha) + color * fallback_alpha).astype(np.uint8)
                base[..., :3] = np.where(mask[..., None], blended, base[..., :3])
        elif isinstance(layer, VectorOverlay):
            color = _rgb(layer.to_dict(), salt="vector")
            for feature in layer.features or ():
                geometry = feature.get("geometry") if isinstance(feature, Mapping) else None
                if not isinstance(geometry, Mapping):
                    continue
                points = [_point_to_pixel(point, width, height) for point in _geometry_points(geometry)]
                if len(points) == 1:
                    _draw_pixel_block(base, points[0][0], points[0][1], color, radius=2)
                else:
                    for start, end in zip(points, points[1:]):
                        _draw_line(base, start, end, color)
        elif isinstance(layer, LabelLayer):
            plan = plans.get(layer_id)
            if plan is None:
                continue
            color = _rgb(layer.to_dict(), salt="label")
            for accepted in plan.accepted:
                bounds = accepted.screen_bounds or accepted.world_bounds
                if bounds and len(bounds) >= 4:
                    x0, y0 = _point_to_pixel((bounds[0], bounds[1]), width, height)
                    x1, y1 = _point_to_pixel((bounds[2], bounds[3]), width, height)
                    left, right = sorted((x0, x1))
                    top, bottom = sorted((y0, y1))
                    base[top : bottom + 1, left : right + 1, :3] = color
                else:
                    cx, cy = _point_to_pixel((len(str(accepted.label_id)) * 7, len(str(accepted.text)) * 5), width, height)
                    _draw_pixel_block(base, cx, cy, color, radius=2)
        elif isinstance(layer, PointCloudLayer) and include_point_cloud and layer.point_count:
            color = _rgb(layer.to_dict(), salt="point-cloud")
            count = min(int(layer.point_count), 64)
            layer_seed = _hash_int(layer.to_dict(), salt="point-cloud")
            for index in range(count):
                x = (layer_seed + index * 17) % width
                y = ((layer_seed >> 8) + index * 29) % height
                _draw_pixel_block(base, x, y, color, radius=0)

    return base


def _render_source_derived_rgba(recipe: "SceneRecipe", plans: Mapping[str, Any]) -> Any:
    import numpy as np

    output = recipe.output
    width = int(output.width)
    height = int(output.height)
    payload = _render_payload(recipe)
    seed = _hash_int(payload, salt="mapscene-source-render")

    yy, xx = np.mgrid[0:height, 0:width]
    base = np.empty((height, width, 4), dtype=np.uint8)
    base[..., 0] = ((xx * ((seed & 0x0F) + 3) + (seed >> 8)) % 256).astype(np.uint8)
    base[..., 1] = ((yy * (((seed >> 4) & 0x0F) + 5) + (seed >> 16)) % 256).astype(np.uint8)
    base[..., 2] = (((xx + yy) * (((seed >> 12) & 0x0F) + 7) + (seed >> 24)) % 256).astype(np.uint8)
    base[..., 3] = 255
    return _composite_recipe_layers(base, recipe, plans)


def _native_scene_class() -> Any | None:
    try:
        from ._native import get_native_module

        native_module = get_native_module()
    except Exception:
        return None
    return getattr(native_module, "Scene", None)


def _is_native_adapter_unavailable(exc: BaseException) -> bool:
    exc_type = type(exc)
    return (
        exc_type.__module__ == "pyo3_runtime"
        and exc_type.__name__ == "PanicException"
        and "No suitable GPU adapter" in str(exc)
    )


def _load_native_heightmap(terrain: "TerrainSource") -> Any | None:
    if not terrain.path:
        return None
    path = Path(str(terrain.path))
    if path.suffix.lower() != ".npy" or not path.exists():
        return None

    import numpy as np

    heightmap = np.load(path)
    if heightmap.ndim != 2:
        raise ValueError("MapScene native/offscreen terrain .npy input must be a 2D heightmap")
    return np.ascontiguousarray(heightmap.astype(np.float32, copy=False))


def _load_native_raster_overlay(layer: "RasterOverlay") -> Any | None:
    if not layer.path:
        return None
    path = Path(str(layer.path))
    if path.suffix.lower() != ".png" or not path.exists():
        return None

    import numpy as np

    from ._png import load_png_rgba

    return np.ascontiguousarray(load_png_rgba(path).astype(np.uint8, copy=False))


def _apply_native_camera(native_scene: Any, camera: "OrbitCamera") -> None:
    if not hasattr(native_scene, "set_camera_look_at"):
        return

    import math

    target_values = list(camera.target or (0.0, 0.0, 0.0))
    while len(target_values) < 3:
        target_values.append(0.0)
    target = tuple(float(value) for value in target_values[:3])
    distance = max(1.0e-3, float(camera.distance))
    azimuth = math.radians(float(camera.azimuth_deg))
    elevation = math.radians(float(camera.elevation_deg))
    horizontal = distance * math.cos(elevation)
    eye = (
        target[0] + horizontal * math.sin(azimuth),
        target[1] + distance * math.sin(elevation),
        target[2] + horizontal * math.cos(azimuth),
    )
    near = float(camera.near) if camera.near is not None else max(0.01, distance * 0.001)
    far = float(camera.far) if camera.far is not None else max(distance * 4.0, near * 2.0)
    native_scene.set_camera_look_at(eye, target, (0.0, 1.0, 0.0), float(camera.fov_deg), near, far)


def _render_native_offscreen_rgba(recipe: "SceneRecipe", plans: Mapping[str, Any]) -> Any | None:
    scene_cls = _native_scene_class()
    heightmap = _load_native_heightmap(recipe.terrain)
    if scene_cls is None or heightmap is None or recipe.output is None:
        return None

    import numpy as np

    output = recipe.output
    try:
        native_scene = scene_cls(int(output.width), int(output.height))
        native_scene.set_height_from_r32f(heightmap)
        _apply_native_camera(native_scene, recipe.camera)

        for layer in recipe.layers:
            if not isinstance(layer, RasterOverlay):
                continue
            overlay = _load_native_raster_overlay(layer)
            if overlay is None:
                return None
            alpha = max(0.0, min(1.0, float(layer.opacity)))
            native_scene.set_raster_overlay(overlay, alpha, None, None)

        rgba = np.asarray(native_scene.render_rgba())
    except BaseException as exc:
        if _is_native_adapter_unavailable(exc):
            return None
        raise
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise RuntimeError("MapScene native/offscreen render returned an invalid RGBA image")
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8)
    base = np.ascontiguousarray(rgba.copy())
    return _composite_recipe_layers(base, recipe, plans, include_raster=True, include_point_cloud=False)


@dataclass
class TerrainSource:
    path: str | None = None
    crs: str | None = None
    metadata: Mapping[str, Any] | None = None
    elevation_sampling_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "terrain_source",
            "path": self.path,
            "crs": self.crs,
            "metadata": _metadata(self.metadata),
            "elevation_sampling_available": bool(self.elevation_sampling_available),
        }


@dataclass
class RasterOverlay:
    layer_id: str
    path: str | None = None
    crs: str | None = None
    opacity: float = 1.0
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "raster_overlay",
            "layer_id": str(self.layer_id),
            "path": self.path,
            "crs": self.crs,
            "opacity": float(self.opacity),
            "metadata": _metadata(self.metadata),
        }


@dataclass
class VectorOverlay:
    layer_id: str
    path: str | None = None
    features: Sequence[Mapping[str, Any]] | None = None
    crs: str | None = None
    style: Mapping[str, Any] | None = None
    style_support: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "vector_overlay",
            "layer_id": str(self.layer_id),
            "path": self.path,
            "features": _sequence(self.features),
            "crs": self.crs,
            "style": _metadata(self.style),
            "style_support": _metadata(self.style_support),
            "metadata": _metadata(self.metadata),
        }


@dataclass(frozen=True)
class FontFallbackRange:
    name: str
    start: int
    end: int
    font_family: str

    def __post_init__(self) -> None:
        if int(self.end) < int(self.start):
            raise ValueError("FontFallbackRange end must be greater than or equal to start")

    def covers(self, char: str) -> bool:
        if not char:
            return False
        codepoint = ord(str(char)[0])
        return int(self.start) <= codepoint <= int(self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "start": int(self.start),
            "end": int(self.end),
            "font_family": str(self.font_family),
        }


@dataclass
class FontAtlas:
    glyphs: set[str] = field(default_factory=set)
    font_size: int = 24
    line_height: int = 32
    baseline: int = 24
    coverage: Mapping[str, Any] | None = None
    source_path: str | None = None
    fallbacks: Sequence[FontFallbackRange | Mapping[str, Any]] = field(default_factory=tuple)
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.glyphs = {str(glyph) for glyph in self.glyphs}
        self.coverage = _metadata(self.coverage)
        self.fallbacks = tuple(
            fallback
            if isinstance(fallback, FontFallbackRange)
            else FontFallbackRange(
                str(fallback["name"]),
                int(fallback["start"]),
                int(fallback["end"]),
                str(fallback["font_family"]),
            )
            for fallback in self.fallbacks
        )
        self.fallbacks = tuple(sorted(self.fallbacks, key=lambda item: (item.start, item.end, item.name)))
        self.diagnostics = [
            diagnostic if isinstance(diagnostic, Diagnostic) else Diagnostic.from_dict(diagnostic)
            for diagnostic in self.diagnostics
        ]

    @classmethod
    def default_latin(
        cls,
        *,
        fallbacks: Sequence[FontFallbackRange | Mapping[str, Any]] | None = None,
    ) -> "FontAtlas":
        atlas_path = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "default_atlas.json"
        if atlas_path.exists():
            with atlas_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            glyphs = {chr(int(key)) for key in payload.get("glyphs", {})}
            source_path = str(atlas_path)
        else:
            payload = {"font_size": 24, "line_height": 32, "baseline": 24}
            glyphs = {chr(codepoint) for codepoint in range(32, 128)}
            source_path = None
        return cls(
            glyphs=glyphs,
            font_size=int(payload.get("font_size", 24)),
            line_height=int(payload.get("line_height", 32)),
            baseline=int(payload.get("baseline", 24)),
            coverage={"start": 32, "end": 127, "name": "Basic Latin"},
            source_path=source_path,
            fallbacks=fallbacks or (),
        )

    @classmethod
    def from_font(
        cls,
        path: str | Path,
        *,
        ranges: Sequence[FontFallbackRange | Mapping[str, Any]] | None = None,
        font_size: int = 24,
        line_height: int | None = None,
    ) -> "FontAtlas":
        font_path = Path(path)
        if not font_path.exists():
            return cls(
                font_size=font_size,
                line_height=line_height or int(round(font_size * 4 / 3)),
                baseline=font_size,
                source_path=str(font_path),
                fallbacks=ranges or (),
                diagnostics=[
                    missing_external_asset_diagnostic(
                        "font_atlas",
                        object_id=str(path),
                        path=str(font_path),
                    )
                ],
            )
        glyphs: set[str] = set()
        for fallback in ranges or ():
            item = fallback if isinstance(fallback, FontFallbackRange) else FontFallbackRange(
                str(fallback["name"]),
                int(fallback["start"]),
                int(fallback["end"]),
                str(fallback["font_family"]),
            )
            glyphs.update(chr(codepoint) for codepoint in range(item.start, item.end + 1))
        if not glyphs:
            glyphs = set("".join(chr(codepoint) for codepoint in range(32, 128)))
        return cls(
            glyphs=glyphs,
            font_size=font_size,
            line_height=line_height or int(round(font_size * 4 / 3)),
            baseline=font_size,
            coverage={"source": "font_file", "path": str(font_path)},
            source_path=str(font_path),
            fallbacks=ranges or (),
        )

    def covers(self, char: str) -> bool:
        return str(char) in self.glyphs or self.fallback_for(str(char)) is not None

    def fallback_for(self, char: str) -> FontFallbackRange | None:
        for fallback in self.fallbacks:
            if fallback.covers(char):
                return fallback
        return None

    def validate_text(self, text: str, *, layer_id: str | None = None, object_id: str | None = None) -> list[Diagnostic]:
        missing = sorted({char for char in str(text) if not self.covers(char)})
        if not missing:
            return []
        return [unicode_coverage_gap_diagnostic(missing, layer_id=layer_id, object_id=object_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "glyphs": sorted(self.glyphs),
            "font_size": int(self.font_size),
            "line_height": int(self.line_height),
            "baseline": int(self.baseline),
            "coverage": _metadata(self.coverage),
            "source_path": self.source_path,
            "fallbacks": [fallback.to_dict() for fallback in self.fallbacks],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class TypographySettings:
    font_size: int = 24
    kerning: bool = True
    tracking: float = 0.0
    line_height: float | None = None
    multiline: bool = False
    callout: bool = False
    callout_offset: Sequence[float] = (0.0, 0.0)

    def measure_text(self, text: str) -> dict[str, Any]:
        lines = str(text).splitlines() or [""]
        char_width = float(self.font_size) * 0.6
        line_widths = []
        kerning_applied = False
        for line in lines:
            width = len(line) * char_width
            if line:
                width += len(line) * float(self.tracking)
            if self.kerning:
                pair_count = sum(1 for pair in zip(line, line[1:]) if "".join(pair) in {"AV", "VA", "To"})
                if pair_count:
                    width -= pair_count * float(self.font_size) * 0.1
                    kerning_applied = True
            line_widths.append(max(0.0, width))
        line_height = float(self.line_height if self.line_height is not None else self.font_size * 4 / 3)
        return {
            "width": max(line_widths) if line_widths else 0.0,
            "height": line_height * len(lines),
            "line_count": len(lines),
            "line_height": line_height,
            "kerning_applied": kerning_applied,
            "tracking": float(self.tracking),
        }

    def layout_label(self, text: str, *, anchor: Sequence[float]) -> dict[str, Any]:
        lines = str(text).splitlines() or [""]
        if not self.multiline:
            lines = [" ".join(lines)]
        anchor_values = [float(value) for value in anchor]
        while len(anchor_values) < 3:
            anchor_values.append(0.0)
        offset = [float(value) for value in self.callout_offset[:2]]
        while len(offset) < 2:
            offset.append(0.0)
        label_anchor = [anchor_values[0] + offset[0], anchor_values[1] + offset[1], anchor_values[2]]
        return {
            "lines": lines,
            "metrics": self.measure_text("\n".join(lines)),
            "callout": {
                "enabled": bool(self.callout),
                "anchor": anchor_values[:3],
                "label_anchor": label_anchor,
                "offset": offset[:2],
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "font_size": int(self.font_size),
            "kerning": bool(self.kerning),
            "tracking": float(self.tracking),
            "line_height": self.line_height,
            "multiline": bool(self.multiline),
            "callout": bool(self.callout),
            "callout_offset": _sequence(self.callout_offset),
        }


@dataclass
class LabelLayer:
    layer_id: str
    labels: Sequence[Mapping[str, Any]] | None = None
    glyph_atlas: Mapping[str, Any] | None = None
    typography: Mapping[str, Any] | None = None
    priority_rules: Sequence[Any] | None = None
    plan: Any | None = None
    metadata: Mapping[str, Any] | None = None
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] | None = None

    @classmethod
    def from_features(
        cls,
        features: Sequence[Mapping[str, Any]],
        *,
        text: Any = "name",
        crs: str | None = None,
        target_crs: str | None = None,
        terrain_sampling: str = "auto",
        terrain_sampler: Any | None = None,
        typography: Mapping[str, Any] | None = None,
        layer_id: str = "labels",
        glyph_atlas: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LabelLayer":
        labels: list[dict[str, Any]] = []
        diagnostics: list[Diagnostic] = []
        for index, feature in enumerate(features or ()):
            feature_id = _feature_id(feature, index)
            geometry = _feature_geometry(feature)
            geometry_type, placement_kind = _valid_geometry(geometry)
            if geometry_type is None:
                diagnostics.append(
                    placeholder_fallback_diagnostic(
                        "label invalid geometry",
                        layer_id=layer_id,
                        object_id=feature_id,
                    )
                )
                continue
            if placement_kind is None:
                diagnostics.append(
                    _unsupported_feature_diagnostic(
                        f"label geometry type {geometry_type}",
                        layer_id=layer_id,
                        object_id=feature_id,
                    )
                )
                continue

            transformed_geometry = _transform_label_geometry(geometry, from_crs=crs, to_crs=target_crs)
            sampled_geometry = transformed_geometry
            terrain_sample: Mapping[str, Any] = {}
            if terrain_sampling == "required" and terrain_sampler is not None:
                sampled_geometry, terrain_sample = _sample_label_geometry(
                    transformed_geometry,
                    terrain_sampler=terrain_sampler,
                )

            properties = _feature_properties(feature)
            label_text, missing_field = _label_text_from_expression(text, properties)
            if missing_field is not None:
                diagnostics.append(
                    missing_label_field_diagnostic(
                        missing_field,
                        layer_id=layer_id,
                        object_id=feature_id,
                    )
                )
                continue

            if terrain_sampling == "required" and terrain_sampler is None:
                diagnostics.append(unavailable_terrain_sampler_diagnostic(layer_id=layer_id, object_id=feature_id))

            label_record = {
                "id": feature_id,
                "source_id": feature_id,
                "text": label_text,
                "geometry": _json_safe(sampled_geometry),
                "geometry_type": geometry_type,
                "placement_kind": placement_kind,
                "properties": _metadata(properties),
                "terrain_mode": "required" if terrain_sampling == "required" else terrain_sampling,
            }
            if terrain_sample:
                label_record["terrain_sample"] = _metadata(terrain_sample)
            labels.append(label_record)

        labels.sort(
            key=lambda label: (
                str(label.get("id", "")),
                str(label.get("geometry_type", "")),
                str(label.get("text", "")),
            )
        )
        layer_metadata = _metadata(metadata)
        if crs is not None:
            layer_metadata["crs"] = target_crs or crs
            if target_crs is not None and not _same_crs(crs, target_crs):
                layer_metadata["source_crs"] = crs
        layer_metadata["terrain_sampling"] = terrain_sampling
        return cls(
            layer_id=layer_id,
            labels=labels,
            glyph_atlas=glyph_atlas,
            typography=typography,
            metadata=layer_metadata,
            diagnostics=diagnostics,
        )

    @classmethod
    def from_geodataframe(
        cls,
        gdf: Any,
        *,
        text: Any = "name",
        crs: str | None = None,
        target_crs: str | None = None,
        terrain_sampling: str = "auto",
        terrain_sampler: Any | None = None,
        typography: Mapping[str, Any] | None = None,
        layer_id: str = "labels",
        glyph_atlas: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LabelLayer":
        if not hasattr(gdf, "iterrows"):
            raise TypeError("LabelLayer.from_geodataframe requires a GeoDataFrame-like object")
        features: list[dict[str, Any]] = []
        for index, row in gdf.iterrows():
            properties = {
                str(key): row[key]
                for key in getattr(gdf, "columns", ())
                if str(key) != "geometry"
            }
            geometry = row.get("geometry") if hasattr(row, "get") else getattr(row, "geometry", None)
            if hasattr(geometry, "__geo_interface__"):
                geometry_payload = geometry.__geo_interface__
            else:
                geometry_payload = geometry
            features.append(
                {
                    "type": "Feature",
                    "id": str(index),
                    "properties": properties,
                    "geometry": geometry_payload,
                }
            )
        effective_crs = crs or str(getattr(gdf, "crs", "") or "") or None
        return cls.from_features(
            features,
            text=text,
            crs=effective_crs,
            target_crs=target_crs,
            terrain_sampling=terrain_sampling,
            terrain_sampler=terrain_sampler,
            typography=typography,
            layer_id=layer_id,
            glyph_atlas=glyph_atlas,
            metadata=metadata,
        )

    @classmethod
    def from_style_layer(
        cls,
        features: Sequence[Mapping[str, Any]],
        style_layer: Any,
        *,
        crs: str | None = None,
        target_crs: str | None = None,
        terrain_sampling: str = "auto",
        terrain_sampler: Any | None = None,
        layer_id: str = "labels",
        glyph_atlas: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "LabelLayer":
        return cls.from_features(
            features,
            text=_style_text_expression(style_layer) or "name",
            crs=crs,
            target_crs=target_crs,
            terrain_sampling=terrain_sampling,
            terrain_sampler=terrain_sampler,
            layer_id=layer_id,
            glyph_atlas=glyph_atlas,
            metadata=metadata,
        )

    def compile_labels(self, camera: Any, viewport: Any, terrain: Any | None = None) -> Any:
        from .label_plan import LabelPlan

        return LabelPlan.compile(
            labels=self.labels or (),
            camera=camera,
            viewport=viewport,
            terrain=terrain,
            priority_rules=self.priority_rules or (),
            typography=self.typography or {},
            glyph_atlas=self.glyph_atlas,
            seed=int(_metadata_dict(self.metadata).get("seed", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "label_layer",
            "layer_id": str(self.layer_id),
            "labels": _sequence(self.labels),
            "glyph_atlas": _metadata(self.glyph_atlas),
            "typography": _metadata(self.typography),
            "priority_rules": _sequence(self.priority_rules),
            "plan": _json_safe(self.plan) if self.plan is not None else None,
            "metadata": _metadata(self.metadata),
            "diagnostics": [
                diagnostic.to_dict() if isinstance(diagnostic, Diagnostic) else _json_safe(diagnostic)
                for diagnostic in (self.diagnostics or ())
            ],
        }


@dataclass
class PointCloudLayer:
    layer_id: str
    path: str | None = None
    crs: str | None = None
    point_count: int | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "point_cloud_layer",
            "layer_id": str(self.layer_id),
            "path": self.path,
            "crs": self.crs,
            "point_count": self.point_count,
            "metadata": _metadata(self.metadata),
        }


@dataclass
class BuildingLayer:
    layer_id: str
    source: str | Mapping[str, Any] | None = None
    support_level: str = "underdeveloped"
    geometry_count: int | None = None
    bounds: Sequence[float] | None = None
    material_status: str | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_geojson(
        cls,
        path: str | Path,
        **options: Any,
    ) -> "BuildingLayer":
        metadata = _metadata(options.pop("metadata", None))
        metadata.update(_metadata(options))
        metadata.setdefault("source_format", "geojson")
        return cls(
            layer_id=str(metadata.pop("layer_id", None) or Path(path).stem or "buildings"),
            source={"path": str(path), "source_format": "geojson"},
            support_level=str(metadata.pop("support_level", "underdeveloped")),
            geometry_count=metadata.pop("geometry_count", None),
            bounds=metadata.pop("bounds", None),
            material_status=str(metadata.pop("material_status", "scalar_pbr_underdeveloped")),
            metadata=metadata,
        )

    @classmethod
    def from_cityjson(
        cls,
        path: str | Path,
        **options: Any,
    ) -> "BuildingLayer":
        metadata = _metadata(options.pop("metadata", None))
        metadata.update(_metadata(options))
        metadata.setdefault("source_format", "cityjson")
        return cls(
            layer_id=str(metadata.pop("layer_id", None) or Path(path).stem or "buildings"),
            source={"path": str(path), "source_format": "cityjson"},
            support_level=str(metadata.pop("support_level", "underdeveloped")),
            geometry_count=metadata.pop("geometry_count", None),
            bounds=metadata.pop("bounds", None),
            material_status=str(metadata.pop("material_status", "scalar_pbr_underdeveloped")),
            metadata=metadata,
        )

    @classmethod
    def from_mesh(
        cls,
        path: str | Path | None = None,
        **options: Any,
    ) -> "BuildingLayer":
        metadata = _metadata(options.pop("metadata", None))
        metadata.update(_metadata(options))
        metadata.setdefault("source_format", "mesh")
        return cls(
            layer_id=str(metadata.pop("layer_id", None) or (Path(path).stem if path else "buildings.mesh")),
            source={"path": str(path), "source_format": "mesh"} if path else {"source_format": "mesh"},
            support_level=str(metadata.pop("support_level", "unsupported")),
            geometry_count=metadata.pop("geometry_count", None),
            bounds=metadata.pop("bounds", None),
            material_status=str(metadata.pop("material_status", "unsupported")),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "building_layer",
            "layer_id": str(self.layer_id),
            "source": _json_safe(self.source),
            "support_level": self.support_level,
            "geometry_count": self.geometry_count,
            "bounds": _sequence(self.bounds),
            "material_status": self.material_status,
            "metadata": _metadata(self.metadata),
        }


MapSceneBuildingLayer = BuildingLayer


@dataclass
class Tiles3DLayer:
    layer_id: str
    source: str | Mapping[str, Any] | None = None
    support_level: str = "underdeveloped"
    lod: Mapping[str, Any] | None = None
    cache_budget: int | None = None
    cache_stats: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] | None = None

    @classmethod
    def from_tileset_json(
        cls,
        path: str | Path,
        *,
        lod: Mapping[str, Any] | None = None,
        cache_budget: int | None = None,
        cache_stats: Mapping[str, Any] | None = None,
        layer_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Tiles3DLayer":
        data = _metadata(metadata)
        data.setdefault("source_format", "tileset.json")
        return cls(
            layer_id=layer_id or Path(path).stem or "tiles3d",
            source={"path": str(path), "source_format": "tileset.json"},
            lod=lod,
            cache_budget=cache_budget,
            cache_stats=cache_stats,
            metadata=data,
        )

    @classmethod
    def from_b3dm(
        cls,
        path: str | Path,
        *,
        lod: Mapping[str, Any] | None = None,
        cache_budget: int | None = None,
        cache_stats: Mapping[str, Any] | None = None,
        layer_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Tiles3DLayer":
        data = _metadata(metadata)
        data.setdefault("source_format", "b3dm")
        return cls(
            layer_id=layer_id or Path(path).stem or "tiles3d",
            source={"path": str(path), "source_format": "b3dm"},
            lod=lod,
            cache_budget=cache_budget,
            cache_stats=cache_stats,
            metadata=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "tiles3d_layer",
            "layer_id": str(self.layer_id),
            "source": _json_safe(self.source),
            "support_level": self.support_level,
            "lod": _metadata(self.lod),
            "cache_budget": self.cache_budget,
            "cache_stats": _metadata(self.cache_stats),
            "metadata": _metadata(self.metadata),
            "diagnostics": [
                diagnostic.to_dict() if isinstance(diagnostic, Diagnostic) else _json_safe(diagnostic)
                for diagnostic in (self.diagnostics or ())
            ],
        }


@dataclass
class MapFurnitureLayer:
    title: str | None = None
    legend: Mapping[str, Any] | None = None
    scale_bar: Mapping[str, Any] | None = None
    north_arrow: Mapping[str, Any] | None = None
    keepouts: Sequence[Any] | None = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "map_furniture_layer",
            "title": self.title,
            "legend": _metadata(self.legend),
            "scale_bar": _metadata(self.scale_bar),
            "north_arrow": _metadata(self.north_arrow),
            "keepouts": _sequence(self.keepouts),
            "metadata": _metadata(self.metadata),
        }


@dataclass
class OrbitCamera:
    target: Sequence[float] = (0.0, 0.0, 0.0)
    distance: float = 1.0
    azimuth_deg: float = 0.0
    elevation_deg: float = 45.0
    fov_deg: float = 45.0
    near: float | None = None
    far: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "orbit_camera",
            "target": _sequence(self.target),
            "distance": float(self.distance),
            "azimuth_deg": float(self.azimuth_deg),
            "elevation_deg": float(self.elevation_deg),
            "fov_deg": float(self.fov_deg),
            "near": self.near,
            "far": self.far,
        }


@dataclass
class LightingPreset:
    name: str = "default"
    sun_direction: Sequence[float] | None = None
    intensity: float = 1.0
    settings: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "lighting_preset",
            "name": str(self.name),
            "sun_direction": _sequence(self.sun_direction),
            "intensity": float(self.intensity),
            "settings": _metadata(self.settings),
        }


@dataclass
class OutputSpec:
    width: int
    height: int
    format: str = "png"
    path: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("OutputSpec width and height must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "output_spec",
            "width": int(self.width),
            "height": int(self.height),
            "format": str(self.format),
            "path": self.path,
            "metadata": _metadata(self.metadata),
        }


@dataclass
class ReproducibilityProfile:
    seed: int = 0
    camera: Mapping[str, Any] | None = None
    output_size: Sequence[int] | None = None
    terrain_transform: Mapping[str, Any] | None = None
    style_hashes: Mapping[str, str] | None = None
    asset_hashes_or_ids: Mapping[str, str] | None = None
    renderer_backend: str | None = None
    pixel_tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "reproducibility_profile",
            "seed": int(self.seed),
            "camera": _metadata(self.camera),
            "output_size": _sequence(self.output_size),
            "terrain_transform": _metadata(self.terrain_transform),
            "style_hashes": _metadata(self.style_hashes),
            "asset_hashes_or_ids": _metadata(self.asset_hashes_or_ids),
            "renderer_backend": self.renderer_backend,
            "pixel_tolerance": self.pixel_tolerance,
        }


@dataclass
class SceneRecipe:
    terrain: TerrainSource
    camera: OrbitCamera
    lighting: LightingPreset
    layers: Sequence[Any] = field(default_factory=tuple)
    output: OutputSpec | None = None
    map_furniture: MapFurnitureLayer | None = None
    render_policy: str = RenderFailurePolicy.CONTINUE_ON_WARNING
    diagnostics_policy: Mapping[str, Any] | None = None
    reproducibility_profile: ReproducibilityProfile | None = None

    def __post_init__(self) -> None:
        self.render_policy = RenderFailurePolicy.validate(self.render_policy)
        self.layers = tuple(self.layers or ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "scene_recipe",
            "terrain": _json_safe(self.terrain),
            "camera": _json_safe(self.camera),
            "lighting": _json_safe(self.lighting),
            "layers": _sequence(self.layers),
            "output": _json_safe(self.output) if self.output is not None else None,
            "map_furniture": _json_safe(self.map_furniture) if self.map_furniture is not None else None,
            "render_policy": self.render_policy,
            "diagnostics_policy": _metadata(self.diagnostics_policy),
            "reproducibility_profile": (
                _json_safe(self.reproducibility_profile)
                if self.reproducibility_profile is not None
                else None
            ),
        }


class MapScene:
    def __init__(
        self,
        recipe: SceneRecipe | None = None,
        *,
        terrain: TerrainSource | None = None,
        camera: OrbitCamera | None = None,
        lighting: LightingPreset | None = None,
        layers: Sequence[Any] | None = None,
        output: OutputSpec | None = None,
        map_furniture: MapFurnitureLayer | None = None,
        render_policy: str = RenderFailurePolicy.CONTINUE_ON_WARNING,
        diagnostics_policy: Mapping[str, Any] | None = None,
        reproducibility_profile: ReproducibilityProfile | None = None,
    ) -> None:
        if recipe is not None and any(
            value is not None
            for value in (
                terrain,
                camera,
                lighting,
                layers,
                output,
                map_furniture,
                diagnostics_policy,
                reproducibility_profile,
            )
        ):
            raise TypeError("Pass either recipe or recipe keyword components, not both")
        if recipe is None:
            if terrain is None or camera is None or lighting is None or output is None:
                raise TypeError("terrain, camera, lighting, and output are required when recipe is not provided")
            recipe = SceneRecipe(
                terrain=terrain,
                camera=camera,
                lighting=lighting,
                layers=layers or (),
                output=output,
                map_furniture=map_furniture,
                render_policy=render_policy,
                diagnostics_policy=diagnostics_policy,
                reproducibility_profile=reproducibility_profile,
            )
        self.recipe = recipe
        self.render_policy = recipe.render_policy
        self.reproducibility_profile = recipe.reproducibility_profile
        self.last_validation_report: ValidationReport | None = None
        self.compiled_label_plans: dict[str, Any] = {}
        self.last_render_path: str | None = None
        self.last_render_backend: str | None = None
        self.last_bundle_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "map_scene", "recipe": self.recipe.to_dict()}

    @staticmethod
    def _layer_from_dict(data: Mapping[str, Any]) -> Any:
        kind = str(data.get("kind", ""))
        if kind == "raster_overlay":
            return RasterOverlay(
                layer_id=str(data["layer_id"]),
                path=data.get("path"),
                crs=data.get("crs"),
                opacity=float(data.get("opacity", 1.0)),
                metadata=data.get("metadata") or {},
            )
        if kind == "vector_overlay":
            return VectorOverlay(
                layer_id=str(data["layer_id"]),
                path=data.get("path"),
                features=data.get("features") or (),
                crs=data.get("crs"),
                style=data.get("style") or {},
                style_support=data.get("style_support") or {},
                metadata=data.get("metadata") or {},
            )
        if kind == "label_layer":
            return LabelLayer(
                layer_id=str(data["layer_id"]),
                labels=data.get("labels") or (),
                glyph_atlas=data.get("glyph_atlas") or {},
                typography=data.get("typography") or {},
                priority_rules=data.get("priority_rules") or (),
                plan=data.get("plan"),
                metadata=data.get("metadata") or {},
                diagnostics=data.get("diagnostics") or (),
            )
        if kind == "point_cloud_layer":
            return PointCloudLayer(
                layer_id=str(data["layer_id"]),
                path=data.get("path"),
                crs=data.get("crs"),
                point_count=data.get("point_count"),
                metadata=data.get("metadata") or {},
            )
        if kind == "building_layer":
            return BuildingLayer(
                layer_id=str(data["layer_id"]),
                source=data.get("source"),
                support_level=str(data.get("support_level", "underdeveloped")),
                geometry_count=data.get("geometry_count"),
                bounds=data.get("bounds"),
                material_status=data.get("material_status"),
                metadata=data.get("metadata") or {},
            )
        if kind == "tiles3d_layer":
            return Tiles3DLayer(
                layer_id=str(data["layer_id"]),
                source=data.get("source"),
                support_level=str(data.get("support_level", "underdeveloped")),
                lod=data.get("lod") or {},
                cache_budget=data.get("cache_budget"),
                cache_stats=data.get("cache_stats") or {},
                metadata=data.get("metadata") or {},
                diagnostics=data.get("diagnostics") or (),
            )
        return dict(data)

    @classmethod
    def _recipe_from_dict(cls, data: Mapping[str, Any]) -> SceneRecipe:
        terrain_data = data.get("terrain") or {}
        camera_data = data.get("camera") or {}
        lighting_data = data.get("lighting") or {}
        output_data = data.get("output") or {}
        furniture_data = data.get("map_furniture")
        reproducibility_data = data.get("reproducibility_profile")
        return SceneRecipe(
            terrain=TerrainSource(
                path=terrain_data.get("path"),
                crs=terrain_data.get("crs"),
                metadata=terrain_data.get("metadata") or {},
                elevation_sampling_available=bool(terrain_data.get("elevation_sampling_available", False)),
            ),
            camera=OrbitCamera(
                target=camera_data.get("target") or (0.0, 0.0, 0.0),
                distance=float(camera_data.get("distance", 1.0)),
                azimuth_deg=float(camera_data.get("azimuth_deg", 0.0)),
                elevation_deg=float(camera_data.get("elevation_deg", 45.0)),
                fov_deg=float(camera_data.get("fov_deg", 45.0)),
                near=camera_data.get("near"),
                far=camera_data.get("far"),
            ),
            lighting=LightingPreset(
                name=str(lighting_data.get("name", "default")),
                sun_direction=lighting_data.get("sun_direction"),
                intensity=float(lighting_data.get("intensity", 1.0)),
                settings=lighting_data.get("settings") or {},
            ),
            layers=tuple(cls._layer_from_dict(layer) for layer in data.get("layers") or ()),
            output=OutputSpec(
                width=int(output_data.get("width", 1)),
                height=int(output_data.get("height", 1)),
                format=str(output_data.get("format", "png")),
                path=output_data.get("path"),
                metadata=output_data.get("metadata") or {},
            ),
            map_furniture=(
                MapFurnitureLayer(
                    title=furniture_data.get("title"),
                    legend=furniture_data.get("legend") or {},
                    scale_bar=furniture_data.get("scale_bar") or {},
                    north_arrow=furniture_data.get("north_arrow") or {},
                    keepouts=furniture_data.get("keepouts") or (),
                    metadata=furniture_data.get("metadata") or {},
                )
                if isinstance(furniture_data, Mapping)
                else None
            ),
            render_policy=str(data.get("render_policy", RenderFailurePolicy.CONTINUE_ON_WARNING)),
            diagnostics_policy=data.get("diagnostics_policy") or {},
            reproducibility_profile=(
                ReproducibilityProfile(
                    seed=int(reproducibility_data.get("seed", 0)),
                    camera=reproducibility_data.get("camera") or {},
                    output_size=reproducibility_data.get("output_size") or (),
                    terrain_transform=reproducibility_data.get("terrain_transform") or {},
                    style_hashes=reproducibility_data.get("style_hashes") or {},
                    asset_hashes_or_ids=reproducibility_data.get("asset_hashes_or_ids") or {},
                    renderer_backend=reproducibility_data.get("renderer_backend"),
                    pixel_tolerance=reproducibility_data.get("pixel_tolerance"),
                )
                if isinstance(reproducibility_data, Mapping)
                else None
            ),
        )

    @classmethod
    def load_bundle(cls, path: str | Path) -> "MapScene":
        bundle_path = Path(path)
        recipe_path = bundle_path / "scene" / "mapscene_recipe.json"
        if not recipe_path.exists():
            raise FileNotFoundError(f"MapScene bundle recipe not found: {recipe_path}")
        with recipe_path.open("r", encoding="utf-8") as handle:
            recipe_payload = json.load(handle)
        scene = cls(recipe=cls._recipe_from_dict(recipe_payload))
        scene.last_bundle_path = str(bundle_path)
        state_path = bundle_path / "scene" / "state.json"
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as handle:
                state_payload = json.load(handle)
            report_payload = state_payload.get("validation_report")
            if isinstance(report_payload, Mapping):
                scene.last_validation_report = ValidationReport.from_dict(report_payload)
        return scene

    def validate(self) -> ValidationReport:
        self.compiled_label_plans = {}
        diagnostics: list[Diagnostic] = []
        layer_summaries: list[LayerSummary] = []
        supported_features: dict[str, str] = {
            "mapscene.recipe": "underdeveloped",
            "mapscene.validation": "supported",
        }
        unsupported_features: dict[str, str] = {}
        total_memory = 0
        memory_known = False

        output = self.recipe.output
        if output is not None and str(output.format).lower() != "png":
            diagnostics.append(_unsupported_output_format_diagnostic(str(output.format).lower()))
            unsupported_features["output.format"] = "unsupported"

        output_bytes = _output_memory_bytes(output)
        if output_bytes is not None:
            total_memory += output_bytes
            memory_known = True

        terrain = self.recipe.terrain
        scene_crs = terrain.crs
        terrain_memory = _dimension_memory_bytes(terrain.metadata)
        if terrain_memory is not None:
            total_memory += terrain_memory
            memory_known = True

        vt_report = _virtual_texture_report_from_metadata(terrain.metadata)
        if vt_report is not None:
            _merge_report(
                vt_report,
                diagnostics=diagnostics,
                layer_summaries=layer_summaries,
                supported_features=supported_features,
                unsupported_features=unsupported_features,
            )

        terrain_support_level = "supported"
        terrain_details = {
            "crs": scene_crs,
            "elevation_sampling_available": bool(terrain.elevation_sampling_available),
            "path": terrain.path,
        }
        terrain_resource_diagnostics, terrain_resource_details, terrain_resource_support = (
            _p2_resource_availability_diagnostics(
                terrain.metadata,
                layer_id="terrain",
                layer_type="terrain_source",
            )
        )
        if terrain_resource_diagnostics:
            diagnostics.extend(terrain_resource_diagnostics)
            terrain_details.update(terrain_resource_details)
            if any(diagnostic.code == "unavailable_cache_lod_stats" for diagnostic in terrain_resource_diagnostics):
                unsupported_features["terrain.cache_lod_stats"] = "underdeveloped"
            if terrain_resource_support == "underdeveloped":
                terrain_support_level = "underdeveloped"
        if not _has_identity_path_or_metadata(terrain.path, terrain.metadata):
            diagnostics.append(
                _missing_source_identity_diagnostic(
                    "terrain_source",
                    layer_id="terrain",
                    source_fields=("path", "metadata"),
                )
            )
            unsupported_features["terrain.source_identity"] = "unsupported"
            terrain_support_level = "unsupported"
        else:
            terrain_asset_diagnostics = _asset_path_diagnostics(
                "terrain_source",
                layer_id="terrain",
                path=terrain.path,
                metadata=terrain.metadata,
                supported_extensions=_TERRAIN_ASSET_EXTENSIONS,
            )
            if terrain_asset_diagnostics:
                diagnostics.extend(terrain_asset_diagnostics)
                unsupported_features["terrain.asset"] = "unsupported"
                terrain_support_level = "unsupported"
        if not scene_crs:
            diagnostics.append(_missing_crs_diagnostic(None, layer_id="terrain"))
            unsupported_features["terrain.crs"] = "unsupported"
            terrain_support_level = "unsupported"

        layer_summaries.append(
            LayerSummary(
                layer_id="terrain",
                layer_type="terrain_source",
                support_level=terrain_support_level,
                diagnostic_codes=_diagnostic_codes_for_layer(diagnostics, "terrain"),
                memory_estimate_bytes=terrain_memory,
                details=terrain_details,
            )
        )
        supported_features["layer.terrain"] = "supported"

        for index, layer in enumerate(self.recipe.layers):
            layer_id = _layer_id(layer, f"layer_{index}")
            layer_diagnostics: list[Diagnostic] = []
            layer_memory: int | None = None
            support_level = "supported"
            layer_type = type(layer).__name__
            object_count: int | None = None
            details: dict[str, Any] = {}

            layer_crs = getattr(layer, "crs", None)
            has_crs_field = hasattr(layer, "crs")
            if scene_crs and has_crs_field and layer_crs is None and not _has_explicit_crs_policy(layer):
                layer_diagnostics.append(_missing_crs_diagnostic(str(scene_crs), layer_id=layer_id))
                support_level = "unsupported"
            elif scene_crs and layer_crs and not _same_crs(scene_crs, str(layer_crs)) and not _has_explicit_crs_policy(layer):
                layer_diagnostics.append(
                    crs_mismatch_diagnostic(
                        str(scene_crs),
                        str(layer_crs),
                        layer_id=layer_id,
                    )
                )
                support_level = "unsupported"

            if isinstance(layer, RasterOverlay):
                layer_type = "raster_overlay"
                layer_memory = _dimension_memory_bytes(layer.metadata)
                details = {"path": layer.path, "crs": layer.crs, "opacity": float(layer.opacity)}
                supported_features["layer.raster_overlay"] = "supported"
                if not _has_identity_path_or_metadata(layer.path, layer.metadata):
                    layer_diagnostics.append(
                        _missing_source_identity_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            source_fields=("path", "metadata"),
                        )
                    )
                    unsupported_features["raster.source_identity"] = "unsupported"
                    support_level = "unsupported"
                else:
                    layer_diagnostics.extend(
                        _asset_path_diagnostics(
                            layer_type,
                            layer_id=layer_id,
                            path=layer.path,
                            metadata=layer.metadata,
                            supported_extensions=_RASTER_ASSET_EXTENSIONS,
                        )
                    )
                    if any(
                        diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                        for diagnostic in layer_diagnostics
                    ):
                        unsupported_features["raster.asset"] = "unsupported"
                        support_level = "unsupported"
            elif isinstance(layer, VectorOverlay):
                layer_type = "vector_overlay"
                object_count = len(layer.features or ())
                details = {"path": layer.path, "crs": layer.crs, "feature_count": object_count}
                supported_features["layer.vector_overlay"] = "supported"
                if not layer.path and not layer.features:
                    layer_diagnostics.append(
                        _missing_renderable_data_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            required_any_of=("path", "features"),
                        )
                    )
                    unsupported_features["vector.renderable_data"] = "unsupported"
                    support_level = "unsupported"
                elif layer.path and not layer.features:
                    layer_diagnostics.append(placeholder_fallback_diagnostic("vector path loader", layer_id=layer_id))
                    unsupported_features["vector.path_loader"] = "placeholder/fallback"
                    support_level = "placeholder/fallback"
                if layer.style:
                    from .style import validate_style_support

                    style_report = validate_style_support(dict(layer.style))
                    _merge_report(
                        style_report,
                        diagnostics=diagnostics,
                        layer_summaries=layer_summaries,
                        supported_features=supported_features,
                        unsupported_features=unsupported_features,
                    )
                    if style_report.status in {"error", "fatal"}:
                        support_level = "unsupported"
                    elif style_report.status == "warning" and support_level == "supported":
                        support_level = "underdeveloped"
            elif isinstance(layer, LabelLayer):
                layer_type = "label_layer"
                labels = list(layer.labels or ())
                object_count = len(labels)
                details = {"label_count": object_count}
                supported_features["layer.label_layer"] = "supported"
                layer_diagnostics.extend(
                    diagnostic if isinstance(diagnostic, Diagnostic) else Diagnostic.from_dict(diagnostic)
                    for diagnostic in (layer.diagnostics or ())
                )
                if not _has_labels_or_plan(layer):
                    layer_diagnostics.append(
                        _missing_renderable_data_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            required_any_of=("labels", "plan"),
                        )
                    )
                    unsupported_features["labels.renderable_data"] = "unsupported"
                    support_level = "unsupported"
                label_report = validate_label_support(
                    labels,
                    atlas_glyphs=_atlas_glyph_set(layer.glyph_atlas),
                    layer_id=layer_id,
                )
                _merge_report(
                    label_report,
                    diagnostics=diagnostics,
                    layer_summaries=layer_summaries,
                    supported_features=supported_features,
                    unsupported_features=unsupported_features,
                )
                if _has_labels_or_plan(layer):
                    plan = _label_plan_from_layer(layer, recipe=self.recipe, terrain=terrain)
                    self.compiled_label_plans[layer_id] = plan
                    plan_diagnostics = [_diagnostic_for_layer(diagnostic, layer_id) for diagnostic in plan.diagnostics]
                    existing_diagnostic_keys = {
                        (diagnostic.code, diagnostic.layer_id, diagnostic.object_id)
                        for diagnostic in diagnostics
                    }
                    plan_diagnostics = [
                        diagnostic
                        for diagnostic in plan_diagnostics
                        if (diagnostic.code, diagnostic.layer_id, diagnostic.object_id) not in existing_diagnostic_keys
                    ]
                    diagnostics.extend(plan_diagnostics)
                    details["compiled_label_plan"] = {
                        "accepted_count": len(plan.accepted),
                        "rejected_count": len(plan.rejected),
                        "diagnostic_codes": sorted({diagnostic.code for diagnostic in plan_diagnostics}),
                        "seed": plan.seed,
                    }
                    if any(diagnostic.severity in {"error", "fatal"} for diagnostic in plan_diagnostics):
                        support_level = "underdeveloped"
                    elif support_level == "supported":
                        support_level = "supported"
            elif isinstance(layer, PointCloudLayer):
                layer_type = "point_cloud_layer"
                object_count = layer.point_count
                layer_memory = _point_cloud_memory_bytes(layer.point_count)
                details = {"path": layer.path, "crs": layer.crs, "point_count": layer.point_count}
                supported_features["layer.point_cloud"] = "underdeveloped"
                support_level = "underdeveloped"
                if not layer.path and layer.point_count is None and not _metadata_dict(layer.metadata):
                    layer_diagnostics.append(
                        _missing_source_identity_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            source_fields=("path", "point_count", "metadata"),
                        )
                    )
                    unsupported_features["point_cloud.source_identity"] = "unsupported"
                    support_level = "unsupported"
                else:
                    layer_diagnostics.extend(
                        _asset_path_diagnostics(
                            layer_type,
                            layer_id=layer_id,
                            path=layer.path,
                            metadata=layer.metadata,
                            supported_extensions=_POINT_CLOUD_ASSET_EXTENSIONS,
                        )
                    )
                    if any(
                        diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                        for diagnostic in layer_diagnostics
                    ):
                        unsupported_features["point_cloud.asset"] = "unsupported"
                        support_level = "unsupported"
                if support_level == "underdeveloped":
                    layer_diagnostics.append(
                        placeholder_fallback_diagnostic("point cloud MapScene render path", layer_id=layer_id)
                    )
                    unsupported_features["point_cloud.mapscene_render"] = "placeholder/fallback"
                    support_level = "placeholder/fallback"
            elif isinstance(layer, Tiles3DLayer):
                layer_type = "tiles3d_layer"
                object_count = None
                layer_memory = _explicit_memory_bytes(layer.metadata)
                source_path = _source_path(layer.source)
                source_kind = _source_kind(layer.source, layer.metadata)
                details = {
                    "source_kind": source_kind,
                    "cache_budget": layer.cache_budget,
                    "cache_stats": _metadata(layer.cache_stats),
                    "lod": _metadata(layer.lod),
                }
                supported_features["layer.tiles3d_intent"] = "underdeveloped"
                support_level = layer.support_level
                layer_diagnostics.extend(
                    diagnostic if isinstance(diagnostic, Diagnostic) else Diagnostic.from_dict(diagnostic)
                    for diagnostic in (layer.diagnostics or ())
                )
                if layer.source is None and not _metadata_dict(layer.metadata):
                    layer_diagnostics.append(
                        _missing_source_identity_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            source_fields=("source", "metadata"),
                        )
                    )
                    unsupported_features["tiles3d.source_identity"] = "unsupported"
                    support_level = "unsupported"
                else:
                    if source_path:
                        if not source_path.lower().endswith(("tileset.json", ".b3dm")):
                            layer_diagnostics.append(
                                unsupported_tile_format_diagnostic(
                                    Path(source_path).suffix.lstrip(".") or source_kind or "unknown",
                                    layer_id=layer_id,
                                    object_id=source_path,
                                )
                            )
                            unsupported_features["tiles3d.format"] = "unsupported"
                            support_level = "unsupported"
                        layer_diagnostics.extend(
                            _asset_path_diagnostics(
                                layer_type,
                                layer_id=layer_id,
                                path=source_path,
                                metadata=layer.metadata,
                                supported_extensions=("tileset.json", ".b3dm"),
                            )
                        )
                    for feature in _metadata_dict(layer.metadata).get("unsupported_features", ()) or ():
                        layer_diagnostics.append(
                            unsupported_tile_feature_diagnostic(str(feature), layer_id=layer_id)
                        )
                        unsupported_features["tiles3d.feature"] = "unsupported"
                        support_level = "unsupported"
                    if support_level != "unsupported":
                        layer_diagnostics.append(python_public_3dtiles_incomplete_diagnostic(layer_id=layer_id))
                        unsupported_features["tiles3d.public_python_render"] = "underdeveloped"
                        support_level = "underdeveloped"
            elif isinstance(layer, BuildingLayer):
                layer_type = "building_layer"
                object_count = layer.geometry_count
                layer_memory = _geometry_memory_bytes(layer.geometry_count)
                source_kind = _source_kind(layer.source, layer.metadata)
                support_level = layer.support_level
                details = {
                    "geometry_count": layer.geometry_count,
                    "material_status": layer.material_status,
                    "source_kind": source_kind,
                }
                supported_features["layer.building_intent"] = "underdeveloped"
                if layer.source is None and not _metadata_dict(layer.metadata):
                    layer_diagnostics.append(
                        _missing_source_identity_diagnostic(
                            layer_type,
                            layer_id=layer_id,
                            source_fields=("source", "metadata"),
                        )
                    )
                    unsupported_features["buildings.source_identity"] = "unsupported"
                    support_level = "unsupported"
                elif layer.support_level == "supported":
                    layer_diagnostics.extend(
                        _asset_path_diagnostics(
                            layer_type,
                            layer_id=layer_id,
                            path=_source_path(layer.source),
                            metadata=layer.metadata,
                            supported_extensions=_BUILDING_ASSET_EXTENSIONS,
                        )
                    )
                    if any(
                        diagnostic.code in {"missing_external_asset", "unsupported_asset_format"}
                        for diagnostic in layer_diagnostics
                    ):
                        unsupported_features["buildings.asset"] = "unsupported"
                        support_level = "unsupported"
                    else:
                        layer_diagnostics.append(
                            placeholder_fallback_diagnostic(
                                "building layer MapScene render adapter",
                                layer_id=layer_id,
                            )
                        )
                        unsupported_features["buildings.mapscene_render"] = "placeholder/fallback"
                        support_level = "placeholder/fallback"
                if source_kind == "3dtiles":
                    layer_diagnostics.append(python_public_3dtiles_incomplete_diagnostic(layer_id=layer_id))
                    unsupported_features["tiles3d.public_python_render"] = "underdeveloped"
                    support_level = "underdeveloped"
                if layer.support_level == "Pro-gated":
                    layer_diagnostics.append(pro_gated_path_diagnostic("building layer", layer_id=layer_id))
                    unsupported_features["buildings.pro_gated_path"] = "Pro-gated"
                if str(layer.material_status or "").lower() in {"textured_pbr_unsupported", "textured pbr unsupported"}:
                    layer_diagnostics.append(
                        _unsupported_feature_diagnostic("building textured PBR", layer_id=layer_id)
                    )
                    unsupported_features["buildings.textured_pbr"] = "unsupported"
                elif layer.support_level == "placeholder/fallback" or (
                    layer.geometry_count == 0 and layer.support_level not in {"experimental", "underdeveloped"}
                ):
                    layer_diagnostics.append(placeholder_fallback_diagnostic("building layer", layer_id=layer_id))
                    unsupported_features["buildings.placeholder_fallback"] = "placeholder/fallback"
                    support_level = "placeholder/fallback"
                elif layer.support_level == "experimental":
                    layer_diagnostics.append(experimental_feature_diagnostic("building layer", layer_id=layer_id))
                    unsupported_features["buildings.experimental"] = "experimental"
                elif layer.support_level == "unsupported":
                    layer_diagnostics.append(_unsupported_feature_diagnostic("building layer", layer_id=layer_id))
                    unsupported_features["buildings.unsupported"] = "unsupported"
                texture_diagnostics, texture_details, texture_support = _p2_building_texture_diagnostics(
                    layer.metadata,
                    layer_id=layer_id,
                )
                if texture_details:
                    details.update(texture_details)
                if texture_diagnostics:
                    layer_diagnostics.extend(texture_diagnostics)
                    if any(diagnostic.code == "missing_texture_path" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.missing_texture_path"] = "unsupported"
                    if any(diagnostic.code == "missing_uvs" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.missing_uvs"] = "unsupported"
                    if any(diagnostic.code == "unsupported_texture_format" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.unsupported_texture_format"] = "unsupported"
                    if any(diagnostic.code == "placeholder_fallback" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.textured_material_fallback"] = "placeholder/fallback"
                    if any(diagnostic.code == "unsupported_feature" for diagnostic in texture_diagnostics):
                        unsupported_features["buildings.textured_pbr"] = "unsupported"
                    if texture_support == "unsupported":
                        support_level = "unsupported"
                    elif texture_support == "placeholder/fallback" and support_level not in {"unsupported", "Pro-gated"}:
                        support_level = "placeholder/fallback"
            else:
                layer_diagnostics.append(_unsupported_layer_type_diagnostic(layer, layer_id=layer_id))
                unsupported_features["layer.unknown"] = "unsupported"
                support_level = "unsupported"
                details = {"python_type": type(layer).__name__}

            resource_diagnostics, resource_details, resource_support = _p2_resource_availability_diagnostics(
                getattr(layer, "metadata", None),
                layer_id=layer_id,
                layer_type=layer_type,
            )
            if resource_diagnostics:
                layer_diagnostics.extend(resource_diagnostics)
                details.update(resource_details)
                if any(diagnostic.code == "unavailable_cache_lod_stats" for diagnostic in resource_diagnostics):
                    unsupported_features[f"{layer_type}.cache_lod_stats"] = "underdeveloped"
                if any(diagnostic.code == "unsupported_instancing_path" for diagnostic in resource_diagnostics):
                    unsupported_features[f"{layer_type}.instancing"] = "unsupported"
                if resource_support == "unsupported":
                    support_level = "unsupported"
                elif resource_support == "underdeveloped" and support_level == "supported":
                    support_level = "underdeveloped"

            if layer_memory is not None:
                total_memory += layer_memory
                memory_known = True
            if any(diagnostic.code in {"crs_mismatch", "missing_crs"} for diagnostic in layer_diagnostics):
                support_level = "unsupported"

            diagnostics.extend(layer_diagnostics)
            layer_summaries.append(
                LayerSummary(
                    layer_id=layer_id,
                    layer_type=layer_type,
                    support_level=support_level,
                    diagnostic_codes=_diagnostic_codes_for_layer(diagnostics, layer_id),
                    object_count=object_count,
                    bounds=getattr(layer, "bounds", None),
                    memory_estimate_bytes=layer_memory,
                    details=details,
                )
            )

        estimated_memory = total_memory if memory_known else None
        budget = None
        diagnostics_policy = _metadata_dict(self.recipe.diagnostics_policy)
        if "gpu_memory_budget_bytes" in diagnostics_policy:
            try:
                budget = int(diagnostics_policy["gpu_memory_budget_bytes"])
            except (TypeError, ValueError):
                budget = None
        if estimated_memory is not None and budget is not None:
            diagnostics.append(
                estimated_gpu_memory_diagnostic(
                    estimated_memory,
                    budget,
                    layer_id="scene",
                )
            )

        if bool(diagnostics_policy.get("large_scene_summary")):
            layer_summaries.append(
                _large_scene_resource_summary(
                    layer_summaries,
                    estimated_memory=estimated_memory,
                    diagnostics=diagnostics,
                )
            )

        report = ValidationReport(
            diagnostics=diagnostics,
            layer_summaries=layer_summaries,
            estimated_gpu_memory_bytes=estimated_memory,
            supported_features=supported_features,
            unsupported_features=unsupported_features,
        )
        self.last_validation_report = report
        return report

    def _report_with_feature(self, report: ValidationReport, feature: str, support_level: str) -> ValidationReport:
        payload = report.to_dict()
        supported = dict(payload.get("supported_features") or {})
        supported[feature] = support_level
        payload["supported_features"] = supported
        updated = ValidationReport.from_dict(payload)
        self.last_validation_report = updated
        return updated

    def render(self, path: str | None = None) -> ValidationReport:
        output = self.recipe.output
        target = path or (output.path if output is not None else None)
        if not target:
            raise ValueError("MapScene.render requires a render path or OutputSpec.path")

        report = self.validate()
        if report.render_blocked(self.render_policy):
            if report.status == "warning" and self.render_policy == RenderFailurePolicy.FAIL_ON_WARNING:
                raise RuntimeError("MapScene.render blocked by warning diagnostics")
            raise RuntimeError("MapScene.render blocked by blocking diagnostics")

        from .helpers.offscreen import save_png_deterministic

        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        rgba = _render_native_offscreen_rgba(self.recipe, self.compiled_label_plans)
        if rgba is None:
            rgba = _render_source_derived_rgba(self.recipe, self.compiled_label_plans)
            backend = "source-derived"
        else:
            backend = "native/offscreen"
        save_png_deterministic(target_path, rgba)
        self.last_render_path = str(target_path)
        self.last_render_backend = backend
        report = self._report_with_feature(report, "mapscene.render_backend", "supported")
        return self._report_with_feature(report, "mapscene.render_png", "supported")

    def _bundle_path(self, path: str | Path) -> Path:
        bundle_path = Path(path)
        if not bundle_path.suffix:
            bundle_path = bundle_path.with_suffix(".forge3d")
        return bundle_path

    def _bundle_manifest(self, checksums: Mapping[str, str]) -> Any:
        from .bundle import BUNDLE_VERSION, BundleManifest

        manifest = BundleManifest(
            version=BUNDLE_VERSION,
            name="mapscene_review",
            created_at="1970-01-01T00:00:00+00:00",
            description="Deterministic MapScene review bundle",
            checksums={key: checksums[key] for key in sorted(checksums)},
        )
        return manifest

    def _write_bundle_json(self, bundle_path: Path, rel_path: str, payload: Any, checksums: dict[str, str]) -> None:
        file_path = bundle_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        checksums[rel_path.replace("\\", "/")] = hashlib.sha256(file_path.read_bytes()).hexdigest()

    def _label_source_payload(self, layer: LabelLayer, layer_id: str) -> dict[str, Any]:
        metadata = _metadata_dict(layer.metadata)
        source_reference = {
            "layer_id": layer_id,
            "source_id": str(metadata.get("source_id") or layer_id),
            "source_hash": _stable_hash(layer.labels or layer.plan or {}),
        }
        return {
            "kind": "mapscene_label_source",
            "layer_id": layer_id,
            "source_reference": source_reference,
            "labels": _sequence(layer.labels),
            "metadata": _metadata(layer.metadata),
        }

    def _source_payload(self, layer: Any, layer_id: str, layer_type: str) -> dict[str, Any]:
        if hasattr(layer, "to_dict") and callable(layer.to_dict):
            payload = layer.to_dict()
        else:
            payload = _json_safe(layer)
        metadata = _metadata_dict(getattr(layer, "metadata", None))
        source_value = getattr(layer, "path", None)
        if source_value is None:
            source_value = getattr(layer, "source", None)
        source_id = metadata.get("source_id") or source_value or layer_id
        result = {
            "kind": "mapscene_layer_source",
            "layer_id": layer_id,
            "layer_type": layer_type,
            "source_reference": {
                "layer_id": layer_id,
                "source_id": str(source_id),
                "source_hash": _stable_hash(payload),
            },
            "payload": payload,
            "metadata": _metadata(metadata),
        }
        if isinstance(layer, VectorOverlay):
            result["features"] = _sequence(layer.features)
            result["style"] = _metadata(layer.style)
        return result

    def save_bundle(self, path: str | Path) -> ValidationReport:
        report = self.validate()
        report = self._report_with_feature(report, "mapscene.save_bundle", "supported")
        bundle_path = self._bundle_path(path)
        bundle_path.mkdir(parents=True, exist_ok=True)
        checksums: dict[str, str] = {}

        recipe_payload = self.recipe.to_dict()
        self._write_bundle_json(bundle_path, "scene/mapscene_recipe.json", recipe_payload, checksums)

        renderable = not report.render_blocked(self.render_policy)
        review_payload = {
            "kind": "mapscene_review_bundle",
            "schema": "forge3d.mapscene.review.v1",
            "renderable": renderable,
            "render_status": "ready_for_render" if renderable else "blocked_by_diagnostics",
            "recipe_hash": _stable_hash(recipe_payload),
            "validation_report_hash": _stable_hash(report.to_dict()),
            "camera": self.recipe.camera.to_dict(),
            "lighting": self.recipe.lighting.to_dict(),
            "output": self.recipe.output.to_dict() if self.recipe.output is not None else None,
            "map_furniture": self.recipe.map_furniture.to_dict() if self.recipe.map_furniture is not None else None,
            "supported_features": dict(report.supported_features or {}),
            "unsupported_features": dict(report.unsupported_features or {}),
            "supported_export_settings": {
                "bundle_schema": "forge3d.mapscene.review.v1",
                "label_plan_persistence": True,
                "output_formats": ["png"],
            },
            "compiled_label_plan_ids": sorted(self.compiled_label_plans),
            "source_layer_ids": [],
            "last_render_path": self.last_render_path,
            "last_render_backend": self.last_render_backend,
        }

        layer_source_ids = {"terrain"}
        self._write_bundle_json(
            bundle_path,
            "scene/layer_sources/terrain.json",
            self._source_payload(self.recipe.terrain, "terrain", "terrain_source"),
            checksums,
        )

        for layer in self.recipe.layers:
            layer_source_ids.add(_layer_id(layer, "layer"))
            layer_type = str(layer.to_dict().get("kind")) if hasattr(layer, "to_dict") else type(layer).__name__
            self._write_bundle_json(
                bundle_path,
                f"scene/layer_sources/{_layer_id(layer, 'layer')}.json",
                self._source_payload(layer, _layer_id(layer, "layer"), layer_type),
                checksums,
            )

        review_payload["source_layer_ids"] = sorted(layer_source_ids)
        self._write_bundle_json(bundle_path, "scene/mapscene_review.json", review_payload, checksums)

        for layer in self.recipe.layers:
            if isinstance(layer, LabelLayer) and _has_labels_or_plan(layer):
                layer_id = _layer_id(layer, "labels")
                plan = self.compiled_label_plans.get(layer_id)
                if plan is not None:
                    self._write_bundle_json(
                        bundle_path,
                        f"scene/label_plans/{layer_id}.json",
                        plan.to_dict(),
                        checksums,
                    )
                self._write_bundle_json(
                    bundle_path,
                    f"scene/label_sources/{layer_id}.json",
                    self._label_source_payload(layer, layer_id),
                    checksums,
                )

        from .bundle import SceneState

        state = SceneState(validation_report=report)
        self._write_bundle_json(bundle_path, "scene/state.json", state.to_dict(), checksums)

        manifest = self._bundle_manifest(checksums)
        with (bundle_path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

        self.last_bundle_path = str(bundle_path)
        return report


__all__ = [
    "MapScene",
    "SceneRecipe",
    "TerrainSource",
    "RasterOverlay",
    "VectorOverlay",
    "FontAtlas",
    "FontFallbackRange",
    "TypographySettings",
    "LabelLayer",
    "PointCloudLayer",
    "Tiles3DLayer",
    "BuildingLayer",
    "MapSceneBuildingLayer",
    "MapFurnitureLayer",
    "OrbitCamera",
    "LightingPreset",
    "OutputSpec",
    "ReproducibilityProfile",
]
