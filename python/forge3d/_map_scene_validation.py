"""Validation support helpers for private MapScene orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ._map_scene_common import _json_safe, _metadata, _metadata_dict
from .diagnostics import (
    Diagnostic,
    LayerSummary,
    ValidationReport,
    missing_external_asset_diagnostic,
    missing_texture_path_diagnostic,
    missing_uvs_diagnostic,
    placeholder_fallback_diagnostic,
    unavailable_cache_lod_stats_diagnostic,
    unsupported_instancing_path_diagnostic,
    unsupported_texture_format_diagnostic,
)


# ---------------------------------------------------------------------------
# SUTURA: native-required layer classification (zero silent placeholders)
# ---------------------------------------------------------------------------
#
# Every MapScene layer either renders through a concrete native symbol or the
# render is blocked with a structured fatal diagnostic — with two explicitly
# named deterministic CPU compositor exceptions (NOT placeholders and NOT
# native-only): loaded raster overlays composite through the Python
# resample compositor (render metadata ``raster_overlay_backend =
# "python_resample_composite"``), and dashed/mitered precise vectors route
# through the Python precise raster path (``vector_backend =
# "python_precise_raster"``). Both are surfaced in render metadata and
# support features so nothing can claim "all layers native-only".
# This mirrors the diagnose-before-render precedent documented in
# docs/guides/competitive_positioning.md (textured PBR buildings, VT runtime).

_NATIVE_CAPABILITY_SYMBOLS: dict[str, tuple[str, ...]] = {
    "terrain": ("TerrainRenderer", "Session"),
    "raster": ("Scene",),
    "labels": ("Scene",),
    "vector": ("vector_render_oit_py", "vector_render_polygons_fill_py"),
    "buildings": ("Scene",),
    "point_cloud": ("vector_render_oit_py",),
    "tiles3d": ("vector_render_oit_py",),
}

_LAYER_CAPABILITY_KINDS: dict[str, str] = {
    "RasterOverlay": "raster",
    "VectorOverlay": "vector",
    "LabelLayer": "labels",
    "BuildingLayer": "buildings",
    "PointCloudLayer": "point_cloud",
    "Tiles3DLayer": "tiles3d",
}


def required_native_symbols(kind: str) -> tuple[str, ...]:
    """Concrete native symbols a capability kind needs to render."""
    return _NATIVE_CAPABILITY_SYMBOLS.get(str(kind), ())


def probe_native_capability(kind: str) -> bool:
    """Return True when the concrete native symbols for ``kind`` are importable."""
    symbols = required_native_symbols(kind)
    if not symbols:
        return False
    try:
        from ._native import get_native_module

        native_module = get_native_module()
    except Exception:
        return False
    if native_module is None:
        return False
    return all(hasattr(native_module, symbol) for symbol in symbols)


def classify_layer(layer: Any) -> str:
    """Classify a recipe layer as ``"native"`` or ``"diagnostic_block"``.

    A layer classifies ``diagnostic_block`` when the concrete native symbol it
    needs is absent (or the layer type is unknown); there is no CPU-placeholder
    third outcome.
    """
    kind = _LAYER_CAPABILITY_KINDS.get(type(layer).__name__)
    if kind is None:
        return "diagnostic_block"
    return "native" if probe_native_capability(kind) else "diagnostic_block"


def diagnostic_block(
    *,
    layer: str,
    reason: str,
    required_native: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured fatal diagnostic for a layer that cannot render natively."""
    block: dict[str, Any] = {
        "status": "diagnostic_block",
        "layer": str(layer),
        "reason": str(reason),
        "required_native": str(required_native),
    }
    if details:
        block["details"] = _json_safe(dict(details))
    return block


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
        strongest_support = "supported"

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
