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
    placeholder_fallback_diagnostic,
    pro_gated_path_diagnostic,
    python_public_3dtiles_incomplete_diagnostic,
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


def _unsupported_feature_diagnostic(feature: str, *, layer_id: str | None = None) -> Diagnostic:
    return Diagnostic(
        code="unsupported_feature",
        severity="error",
        message="Requested MapScene feature is not supported by the MVP workflow.",
        remediation="Remove the feature or use a documented supported MapScene path.",
        support_level="unsupported",
        layer_id=layer_id,
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
def _missing_external_asset_diagnostic(layer_type: str, *, layer_id: str, path: str) -> Diagnostic:
    return Diagnostic(
        code="missing_external_asset",
        severity="error",
        message="MapScene layer references an external asset that cannot be found before rendering.",
        remediation="Provide an existing local asset path or mark intentional fixtures with asset_status='fixture'.",
        support_level="unsupported",
        layer_id=layer_id,
        details={"layer_type": layer_type, "path": path},
    )


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
        diagnostics.append(_missing_external_asset_diagnostic(layer_type, layer_id=layer_id, path=path_value))

    return diagnostics


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
            color = np.array(_rgb(layer.to_dict(), salt="raster"), dtype=np.uint8)
            alpha = max(0.0, min(1.0, float(layer.opacity))) * 0.45
            mask = ((xx + yy + _hash_int(layer.to_dict(), salt="raster-mask")) % 5) < 3
            blended = (base[..., :3].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)
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
    return _composite_recipe_layers(base, recipe, plans, include_raster=False, include_point_cloud=False)


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


@dataclass
class LabelLayer:
    layer_id: str
    labels: Sequence[Mapping[str, Any]] | None = None
    glyph_atlas: Mapping[str, Any] | None = None
    typography: Mapping[str, Any] | None = None
    priority_rules: Sequence[Any] | None = None
    plan: Any | None = None
    metadata: Mapping[str, Any] | None = None

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
                details={
                    "crs": scene_crs,
                    "elevation_sampling_available": bool(terrain.elevation_sampling_available),
                    "path": terrain.path,
                },
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
                    atlas_glyphs=None,
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
            else:
                layer_diagnostics.append(_unsupported_layer_type_diagnostic(layer, layer_id=layer_id))
                unsupported_features["layer.unknown"] = "unsupported"
                support_level = "unsupported"
                details = {"python_type": type(layer).__name__}

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
    "LabelLayer",
    "PointCloudLayer",
    "BuildingLayer",
    "MapSceneBuildingLayer",
    "MapFurnitureLayer",
    "OrbitCamera",
    "LightingPreset",
    "OutputSpec",
    "ReproducibilityProfile",
]
