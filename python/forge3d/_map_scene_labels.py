"""Label planning helpers for private MapScene orchestration."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ._map_scene_common import _metadata, _metadata_dict, _same_crs
from .diagnostics import (
    Diagnostic,
    missing_external_asset_diagnostic,
    missing_label_field_diagnostic,
    unavailable_terrain_sampler_diagnostic,
    unicode_coverage_gap_diagnostic,
)


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
