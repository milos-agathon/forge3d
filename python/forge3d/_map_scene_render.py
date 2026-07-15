"""Source-derived MapScene rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._map_scene_common import _layer_id, _stable_hash
from .style import evaluate_color_expr, evaluate_number_expr


@dataclass(frozen=True)
class MapSceneRenderLayerTypes:
    raster_overlay: type
    vector_overlay: type
    label_layer: type
    point_cloud_layer: type
    building_layer: type


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


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _color(value: Any, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        item = value.strip()
        if item.startswith("#"):
            item = item[1:]
        if len(item) == 3:
            item = "".join(ch * 2 for ch in item)
        if len(item) in {6, 8}:
            try:
                r = int(item[0:2], 16)
                g = int(item[2:4], 16)
                b = int(item[4:6], 16)
                a = int(item[6:8], 16) if len(item) == 8 else fallback[3]
                return r, g, b, a
            except ValueError:
                return fallback
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        values = [float(component) for component in value[:4]]
        scale = 255.0 if max(values[:3]) <= 1.0 else 1.0
        r = max(0, min(255, int(round(values[0] * scale))))
        g = max(0, min(255, int(round(values[1] * scale))))
        b = max(0, min(255, int(round(values[2] * scale))))
        a = max(0, min(255, int(round((values[3] if len(values) > 3 else fallback[3]) * (255.0 if len(values) > 3 and values[3] <= 1.0 else 1.0)))))
        return r, g, b, a
    return fallback


def _is_style_expression(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], str)


def _properties(feature: Any) -> dict[str, Any]:
    if not isinstance(feature, Mapping):
        return {}
    properties = feature.get("properties")
    return dict(properties) if isinstance(properties, Mapping) else {}


def _feature_color(value: Any, properties: Mapping[str, Any], fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if _is_style_expression(value):
        evaluated = evaluate_color_expr(value, dict(properties))
        return _color(evaluated, fallback) if evaluated is not None else fallback
    return _color(value, fallback)


def _feature_number(value: Any, properties: Mapping[str, Any], default: float) -> float:
    if _is_style_expression(value):
        evaluated = evaluate_number_expr(value, dict(properties))
        return float(evaluated) if evaluated is not None else float(default)
    return _number(value, default)


def _style_layers(layer: Any, layer_type: str) -> list[Mapping[str, Any]]:
    style = getattr(layer, "style", None)
    if not isinstance(style, Mapping):
        return []
    return [
        item
        for item in style.get("layers", ()) or ()
        if isinstance(item, Mapping) and str(item.get("type", "")).lower() == layer_type
    ]


def _paint(layer: Any, layer_type: str) -> Mapping[str, Any]:
    layers = _style_layers(layer, layer_type)
    if layers:
        return dict(layers[0].get("paint") or {})
    return {}


def _layout(layer: Any, layer_type: str) -> Mapping[str, Any]:
    layers = _style_layers(layer, layer_type)
    if layers:
        return dict(layers[0].get("layout") or {})
    return {}


def _point_to_pixel(point: Sequence[Any], width: int, height: int) -> tuple[int, int]:
    x = float(point[0]) if len(point) > 0 else 0.0
    y = float(point[1]) if len(point) > 1 else 0.0
    px = int(round(x * (width - 1))) if 0.0 <= x <= 1.0 else int(round(x)) % max(1, width)
    py = int(round(y * (height - 1))) if 0.0 <= y <= 1.0 else int(round(y)) % max(1, height)
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))


def _blend_region(image: Any, mask: Any, color: tuple[int, int, int, int]) -> None:
    import numpy as np

    coverage = np.asarray(mask, dtype=np.float32)
    if coverage.dtype == np.bool_:
        coverage = coverage.astype(np.float32)
    coverage = np.clip(coverage, 0.0, 1.0)
    if not np.any(coverage > 0.0):
        return
    src_alpha = coverage * (float(color[3]) / 255.0)
    dst_alpha = image[..., 3].astype(np.float32) / 255.0
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)

    src_rgb = np.array(color[:3], dtype=np.float32) / 255.0
    dst_rgb = image[..., :3].astype(np.float32) / 255.0
    out_rgb_premul = (
        src_rgb[None, None, :] * src_alpha[..., None]
        + dst_rgb * dst_alpha[..., None] * (1.0 - src_alpha[..., None])
    )
    out_rgb = np.divide(
        out_rgb_premul,
        np.maximum(out_alpha[..., None], 1.0e-6),
        out=np.zeros_like(out_rgb_premul),
        where=out_alpha[..., None] > 1.0e-6,
    )
    image[..., :3] = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    image[..., 3] = np.clip(out_alpha * 255.0, 0.0, 255.0).astype(np.uint8)


def _draw_pixel_block(image: Any, x: int, y: int, color: tuple[int, int, int, int], radius: int = 1) -> None:
    import numpy as np

    height, width = image.shape[:2]
    x0 = max(0, int(x) - radius)
    x1 = min(width, int(x) + radius + 1)
    y0 = max(0, int(y) - radius)
    y1 = min(height, int(y) + radius + 1)
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    _blend_region(image, mask, color)


def _draw_disc(image: Any, x: float, y: float, color: tuple[int, int, int, int], radius: float) -> None:
    import numpy as np

    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.sqrt((xx.astype(np.float32) - float(x)) ** 2 + (yy.astype(np.float32) - float(y)) ** 2)
    coverage = np.clip(float(radius) + 0.5 - distance, 0.0, 1.0)
    _blend_region(image, coverage, color)


def _draw_line(
    image: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int, int],
    *,
    width_px: float = 1.0,
    cap: str = "round",
) -> None:
    import numpy as np

    x0, y0 = start
    x1, y1 = end
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    px = xx.astype(np.float32)
    py = yy.astype(np.float32)
    vx = float(x1 - x0)
    vy = float(y1 - y0)
    length_sq = max(vx * vx + vy * vy, 1.0)
    segment_len = float(np.sqrt(length_sq))
    half_width = max(0.5, float(width_px) * 0.5)
    t_raw = ((px - x0) * vx + (py - y0) * vy) / length_sq
    t_min = 0.0
    t_max = 1.0
    cap_key = str(cap or "round").lower()
    if cap_key == "square":
        extension = half_width / max(segment_len, 1.0)
        t_min -= extension
        t_max += extension
    t = np.clip(t_raw, t_min, t_max)
    nearest_x = x0 + t * vx
    nearest_y = y0 + t * vy
    distance = np.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)
    coverage = np.clip(half_width + 0.5 - distance, 0.0, 1.0)
    if cap_key == "butt":
        coverage *= ((t_raw >= 0.0) & (t_raw <= 1.0)).astype(np.float32)
    _blend_region(image, coverage, color)


def _segment_length(start: tuple[int, int] | tuple[float, float], end: tuple[int, int] | tuple[float, float]) -> float:
    import math

    return math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))


def _lerp_point(
    start: tuple[int, int] | tuple[float, float],
    end: tuple[int, int] | tuple[float, float],
    t: float,
) -> tuple[float, float]:
    return (
        float(start[0]) + (float(end[0]) - float(start[0])) * float(t),
        float(start[1]) + (float(end[1]) - float(start[1])) * float(t),
    )


def _dash_pattern(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pattern = tuple(float(item) for item in value if float(item) > 0.0)
    else:
        pattern = ()
    if len(pattern) == 1:
        pattern = (pattern[0], pattern[0])
    if len(pattern) % 2 == 1:
        pattern = pattern + pattern
    return pattern


def _dash_segments(
    points: Sequence[tuple[int, int] | tuple[float, float]],
    dash_array: Sequence[float] | None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pattern = _dash_pattern(dash_array)
    if len(points) < 2:
        return []
    if not pattern:
        return [(_lerp_point(start, start, 0.0), _lerp_point(end, end, 0.0)) for start, end in zip(points, points[1:])]

    result: list[tuple[tuple[float, float], tuple[float, float]]] = []
    pattern_index = 0
    remaining = pattern[0]
    draw = True
    for start, end in zip(points, points[1:]):
        length = _segment_length(start, end)
        if length <= 1.0e-6:
            continue
        offset = 0.0
        while offset < length:
            run = min(remaining, length - offset)
            next_offset = offset + run
            if draw and run > 1.0e-6:
                result.append((
                    _lerp_point(start, end, offset / length),
                    _lerp_point(start, end, next_offset / length),
                ))
            offset = next_offset
            remaining -= run
            if remaining <= 1.0e-6:
                pattern_index = (pattern_index + 1) % len(pattern)
                remaining = pattern[pattern_index]
                draw = pattern_index % 2 == 0
    return result


def _normalize_2d(dx: float, dy: float) -> tuple[float, float] | None:
    import math

    length = math.hypot(float(dx), float(dy))
    if length <= 1.0e-6:
        return None
    return (float(dx) / length, float(dy) / length)


def _line_intersection(
    point_a: tuple[float, float],
    dir_a: tuple[float, float],
    point_b: tuple[float, float],
    dir_b: tuple[float, float],
) -> tuple[float, float] | None:
    denominator = dir_a[0] * dir_b[1] - dir_a[1] * dir_b[0]
    if abs(denominator) <= 1.0e-6:
        return None
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    t = (dx * dir_b[1] - dy * dir_b[0]) / denominator
    return (point_a[0] + dir_a[0] * t, point_a[1] + dir_a[1] * t)


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for left, right in zip(points, [*points[1:], points[0]]):
        total += left[0] * right[1] - right[0] * left[1]
    return total * 0.5


def _draw_polyline_join(
    image: Any,
    previous: tuple[int, int] | tuple[float, float],
    point: tuple[int, int] | tuple[float, float],
    next_point: tuple[int, int] | tuple[float, float],
    color: tuple[int, int, int, int],
    *,
    radius: float,
    join: str,
    miter_limit: float,
) -> None:
    import math

    incoming = _normalize_2d(float(point[0]) - float(previous[0]), float(point[1]) - float(previous[1]))
    outgoing = _normalize_2d(float(next_point[0]) - float(point[0]), float(next_point[1]) - float(point[1]))
    if incoming is None or outgoing is None:
        return
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    if dot > 0.999:
        return
    join_key = str(join or "miter").lower()
    if join_key == "round" or dot < -0.999:
        _draw_disc(image, float(point[0]), float(point[1]), color, radius)
        return

    normal_in = (-incoming[1], incoming[0])
    normal_out = (-outgoing[1], outgoing[0])
    px = float(point[0])
    py = float(point[1])
    limit = max(1.0, float(miter_limit)) * float(radius)
    for side in (-1.0, 1.0):
        start_offset = (px + normal_in[0] * radius * side, py + normal_in[1] * radius * side)
        end_offset = (px + normal_out[0] * radius * side, py + normal_out[1] * radius * side)
        polygon: list[tuple[float, float]]
        miter = None
        if join_key == "miter":
            candidate = _line_intersection(start_offset, incoming, end_offset, outgoing)
            if candidate is not None and math.hypot(candidate[0] - px, candidate[1] - py) <= limit:
                miter = candidate
        if miter is None:
            polygon = [(px, py), start_offset, end_offset]
        else:
            polygon = [(px, py), start_offset, miter, end_offset]
        if abs(_polygon_area(polygon)) > 1.0e-3:
            _draw_polygon_fill(image, [polygon], color)


def _resolve_line_width_px(layer: Any, line_paint: Mapping[str, Any], recipe: Any, width: int, height: int) -> float:
    width_px = getattr(layer, "width_px", None)
    if width_px is not None:
        return max(1.0, float(width_px))

    paint_width = line_paint.get("line-width") if isinstance(line_paint, Mapping) else None
    if paint_width is not None:
        return max(1.0, _number(paint_width, 2.0))

    width_world = getattr(layer, "width_world", None)
    if width_world is not None:
        terrain = getattr(recipe, "terrain", None)
        metadata = getattr(terrain, "metadata", None)
        bounds = _coerce_bounds(metadata.get("bounds") if isinstance(metadata, Mapping) else None)
        if bounds is not None:
            xmin, ymin, xmax, ymax = bounds
            span_x = abs(float(xmax) - float(xmin))
            span_y = abs(float(ymax) - float(ymin))
            if span_x > 0.0 and span_y > 0.0:
                px_per_world_x = max(1, int(width)) / span_x
                px_per_world_y = max(1, int(height)) / span_y
                return max(1.0, float(width_world) * 0.5 * (px_per_world_x + px_per_world_y))
        return max(1.0, float(width_world))

    return max(1.0, _number(None, 2.0))


def _draw_polyline(
    image: Any,
    points: Sequence[tuple[int, int]],
    color: tuple[int, int, int, int],
    *,
    width_px: float = 1.0,
    cap: str = "butt",
    join: str = "miter",
    dash_array: Sequence[float] | None = None,
    miter_limit: float = 4.0,
) -> None:
    if len(points) < 2:
        return
    segments = _dash_segments(points, dash_array)
    cap_key = str(cap or "butt").lower()
    join_key = str(join or "miter").lower()
    radius = max(0.5, float(width_px) * 0.5)
    for start, end in segments:
        _draw_line(
            image,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            color,
            width_px=width_px,
            cap=cap_key,
        )
        if cap_key == "round":
            _draw_disc(image, start[0], start[1], color, radius)
            _draw_disc(image, end[0], end[1], color, radius)
    if dash_array:
        return
    for previous, point, next_point in zip(points, points[1:], points[2:]):
        _draw_polyline_join(
            image,
            previous,
            point,
            next_point,
            color,
            radius=radius,
            join=join_key,
            miter_limit=miter_limit,
        )


def _is_point_like(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _polygon_rings(points_or_rings: Sequence[Any]) -> list[list[tuple[float, float]]]:
    if not points_or_rings:
        return []
    if _is_point_like(points_or_rings[0]):
        rings = [points_or_rings]
    else:
        rings = list(points_or_rings)
    normalized: list[list[tuple[float, float]]] = []
    for ring in rings:
        if not isinstance(ring, Sequence) or isinstance(ring, (str, bytes)) or len(ring) < 3:
            continue
        normalized.append([(float(point[0]), float(point[1])) for point in ring if _is_point_like(point)])
    return [ring for ring in normalized if len(ring) >= 3]


def _ring_contains(ring: Sequence[tuple[float, float]], sample_x: Any, sample_y: Any) -> Any:
    import numpy as np

    inside = np.zeros_like(sample_x, dtype=bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = ((yi > sample_y) != (yj > sample_y)) & (
            sample_x < (xj - xi) * (sample_y - yi) / (yj - yi + 1.0e-9) + xi
        )
        inside ^= intersects
        j = i
    return inside


def _draw_polygon_fill(image: Any, points: Sequence[Any], color: tuple[int, int, int, int]) -> None:
    import numpy as np

    rings = _polygon_rings(points)
    if not rings:
        return
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    samples = 4
    offsets = (np.arange(samples, dtype=np.float32) + 0.5) / float(samples) - 0.5
    coverage = np.zeros((height, width), dtype=np.float32)
    base_x = xx.astype(np.float32)
    base_y = yy.astype(np.float32)
    for dy in offsets:
        for dx in offsets:
            sample_x = base_x + float(dx)
            sample_y = base_y + float(dy)
            inside = np.zeros((height, width), dtype=bool)
            for ring in rings:
                inside ^= _ring_contains(ring, sample_x, sample_y)
            coverage += inside.astype(np.float32)
    coverage /= float(samples * samples)
    _blend_region(image, coverage, color)


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


def _building_features(layer: Any) -> list[Mapping[str, Any]]:
    features = getattr(layer, "features", None)
    if features:
        return [feature for feature in features if isinstance(feature, Mapping)]
    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("features", "buildings"):
            items = metadata.get(key)
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                return [feature for feature in items if isinstance(feature, Mapping)]
    source = getattr(layer, "source", None)
    path = source.get("path") if isinstance(source, Mapping) else source
    if not path:
        return []
    try:
        import json
        from pathlib import Path

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, Mapping) and isinstance(payload.get("features"), Sequence):
        return [feature for feature in payload.get("features", ()) if isinstance(feature, Mapping)]
    return []


def _building_rings(geometry: Mapping[str, Any]) -> list[list[Sequence[Any]]]:
    coordinates = geometry.get("coordinates")
    geometry_type = str(geometry.get("type", "")).lower()
    if geometry_type == "polygon" and isinstance(coordinates, Sequence) and coordinates:
        return [list(coordinates[0])]
    if geometry_type == "multipolygon" and isinstance(coordinates, Sequence):
        rings = []
        for polygon in coordinates:
            if isinstance(polygon, Sequence) and polygon:
                rings.append(list(polygon[0]))
        return rings
    return []


def _geometry_polygon_rings(geometry: Mapping[str, Any]) -> list[list[Sequence[Any]]]:
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return []
    geometry_type = str(geometry.get("type", "")).lower()
    if geometry_type == "polygon" and isinstance(coordinates, Sequence):
        return [[list(ring) for ring in coordinates if isinstance(ring, Sequence) and ring]]
    if geometry_type == "multipolygon" and isinstance(coordinates, Sequence):
        return [
            [list(ring) for ring in polygon if isinstance(ring, Sequence) and ring]
            for polygon in coordinates
            if isinstance(polygon, Sequence) and polygon
        ]
    return []


def _building_properties(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    properties = feature.get("properties")
    return properties if isinstance(properties, Mapping) else {}


def _building_height(properties: Mapping[str, Any]) -> float:
    for key in ("height", "building:height", "render_height"):
        if key in properties:
            return max(1.0, _number(properties.get(key), 12.0))
    levels = _number(properties.get("building:levels", properties.get("levels")), 0.0)
    return max(1.0, levels * 3.0) if levels > 0.0 else 12.0


def _building_roof_type(properties: Mapping[str, Any]) -> str:
    for key in ("roof_type", "roof:shape", "building:roof:shape", "roof_shape"):
        value = properties.get(key)
        if value:
            roof = str(value).lower()
            if roof in {"flat", "gabled", "hipped", "pyramidal"}:
                return roof
    return "flat"


def _building_fill_color(properties: Mapping[str, Any]) -> tuple[int, int, int, int]:
    material = str(properties.get("building:material", properties.get("material", "concrete"))).lower()
    palette = {
        "brick": (166, 82, 58, 235),
        "concrete": (158, 154, 145, 235),
        "glass": (112, 159, 184, 220),
        "stone": (132, 128, 118, 235),
        "wood": (143, 101, 65, 235),
    }
    return palette.get(material, (150, 143, 132, 235))


def _draw_building_roof(
    image: Any,
    points: Sequence[tuple[int, int]],
    roof_type: str,
    color: tuple[int, int, int, int],
) -> None:
    if len(points) < 3 or roof_type == "flat":
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    cx = int(round(sum(xs) / len(xs)))
    cy = int(round(sum(ys) / len(ys)))
    roof_line = (max(0, color[0] - 35), max(0, color[1] - 35), max(0, color[2] - 35), min(255, color[3] + 20))
    if roof_type == "gabled":
        horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))
        if horizontal:
            _draw_polyline(image, [(min(xs), cy), (max(xs), cy)], roof_line, width_px=1.5, cap="butt")
        else:
            _draw_polyline(image, [(cx, min(ys)), (cx, max(ys))], roof_line, width_px=1.5, cap="butt")
    elif roof_type in {"hipped", "pyramidal"}:
        for point in points[:4]:
            _draw_polyline(image, [point, (cx, cy)], roof_line, width_px=1.0, cap="butt")


def _draw_buildings(image: Any, layer: Any, width: int, height: int) -> None:
    features = _building_features(layer)
    if not features:
        return
    for feature in features:
        geometry = feature.get("geometry") if isinstance(feature.get("geometry"), Mapping) else {}
        properties = _building_properties(feature)
        height_m = _building_height(properties)
        roof_type = _building_roof_type(properties)
        fill = _building_fill_color(properties)
        roof = (min(255, fill[0] + 28), min(255, fill[1] + 28), min(255, fill[2] + 28), fill[3])
        shadow = (28, 26, 24, min(150, int(55 + min(height_m, 60.0))))
        shadow_offset = max(1, min(10, int(round(height_m / 8.0))))
        for ring in _building_rings(geometry):
            points = [_point_to_pixel(point, width, height) for point in ring]
            if len(points) < 3:
                continue
            shadow_points = [
                (max(0, min(width - 1, x + shadow_offset)), max(0, min(height - 1, y + shadow_offset)))
                for x, y in points
            ]
            _draw_polygon_fill(image, shadow_points, shadow)
            _draw_polygon_fill(image, points, fill)
            inset_points = points[:: max(1, len(points) // 8)]
            _draw_polygon_fill(image, inset_points if len(inset_points) >= 3 else points, roof)
            outline = (70, 64, 58, 210)
            if points and points[0] != points[-1]:
                points = [*points, points[0]]
            _draw_polyline(image, points, outline, width_px=1.0, cap="butt", join="miter")
            _draw_building_roof(image, points[:-1], roof_type, roof)


def _label_anchor(label: Any, width: int, height: int) -> tuple[int, int]:
    candidate = getattr(label, "candidate", None)
    anchor = getattr(candidate, "anchor", None)
    if anchor is not None:
        return _point_to_pixel(anchor, width, height)
    bounds = getattr(label, "screen_bounds", None) or getattr(label, "world_bounds", None)
    if bounds and len(bounds) >= 4:
        return _point_to_pixel(((float(bounds[0]) + float(bounds[2])) * 0.5, (float(bounds[1]) + float(bounds[3])) * 0.5), width, height)
    return _point_to_pixel((len(str(getattr(label, "label_id", ""))) * 7, len(str(getattr(label, "text", ""))) * 5), width, height)


def _text_font_chain(font_chain: Sequence[str] | None = None) -> list[str]:
    root = Path(__file__).resolve().parent / "data" / "fonts"
    bundled = [
        root / name
        for name in (
            "NotoSansLatin-subset.ttf",
            "NotoSansArabic-subset.ttf",
            "NotoSansHebrew-subset.ttf",
            "NotoSansDevanagari-subset.ttf",
            "NotoSansSC-subset.ttf",
        )
    ]
    return [*(str(path) for path in font_chain or ()), *(str(path) for path in bundled)]


def _composite_text_mask(image: Any, mask: Any, color: tuple[int, int, int, int]) -> None:
    import numpy as np

    coverage = np.asarray(mask, dtype=np.float32)[..., None]
    source_alpha = coverage * (float(color[3]) / 255.0)
    destination = np.asarray(image, dtype=np.float32) / 255.0
    destination_alpha = destination[..., 3:4]
    output_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    source_rgb = np.asarray(color[:3], dtype=np.float32).reshape(1, 1, 3) / 255.0
    numerator = (
        source_rgb * source_alpha
        + destination[..., :3] * destination_alpha * (1.0 - source_alpha)
    )
    output_rgb = np.divide(
        numerator,
        output_alpha,
        out=np.zeros_like(numerator),
        where=output_alpha > 0.0,
    )
    image[..., :3] = np.clip(output_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    image[..., 3] = np.clip(output_alpha[..., 0] * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _expanded_mask(mask: Any, radius: int) -> Any:
    import numpy as np

    source = np.asarray(mask, dtype=np.float32)
    if radius <= 0:
        return source
    padded = np.pad(source, radius)
    return np.maximum.reduce(
        [
            padded[
                radius + dy : radius + dy + source.shape[0],
                radius + dx : radius + dx + source.shape[1],
            ]
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
    )


def _draw_text(
    image: Any,
    text: str,
    anchor: tuple[int, int],
    *,
    color: tuple[int, int, int, int],
    halo: tuple[int, int, int, int],
    halo_width_px: float = 1.0,
    font_size: float = 12.0,
    font_chain: Sequence[str] | None = None,
) -> None:
    from .text import rasterize_shaped_run, shape

    x, y = anchor
    shaped = shape(str(text), _text_font_chain(font_chain), float(font_size))
    mask = rasterize_shaped_run(
        shaped,
        int(image.shape[1]),
        int(image.shape[0]),
        origin=(float(x), float(y) + float(font_size)),
    )
    radius = max(0, int(round(float(halo_width_px))))
    if halo[3] > 0 and radius > 0:
        _composite_text_mask(image, _expanded_mask(mask, radius), halo)
    _composite_text_mask(image, mask, color)


def _overlay_rgba(image: Any, overlay: Any, x: int, y: int) -> None:
    import numpy as np

    src = np.asarray(overlay, dtype=np.uint8)
    if src.ndim != 3 or src.shape[2] != 4:
        return
    height, width = image.shape[:2]
    src_h, src_w = src.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(width, int(x) + src_w)
    y1 = min(height, int(y) + src_h)
    if x0 >= x1 or y0 >= y1:
        return
    sx0 = x0 - int(x)
    sy0 = y0 - int(y)
    clipped = src[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
    alpha = clipped[..., 3:4].astype(np.float32) / 255.0
    dst = image[y0:y1, x0:x1, :3].astype(np.float32)
    blended = dst * (1.0 - alpha) + clipped[..., :3].astype(np.float32) * alpha
    image[y0:y1, x0:x1, :3] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    image[y0:y1, x0:x1, 3] = 255


def _config_kwargs(config_type: type, options: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(config_type)}
    return {str(key): value for key, value in options.items() if str(key) in allowed}


def _config_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _coerce_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        keys = ("west", "south", "east", "north")
        if all(key in value for key in keys):
            return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
        keys = ("left", "bottom", "right", "top")
        if all(key in value for key in keys):
            return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    return None


def _furniture_bounds(recipe: Any, options: Mapping[str, Any] | None = None) -> tuple[float, float, float, float]:
    candidates: list[Any] = []
    if options is not None:
        candidates.extend([options.get("bbox"), options.get("bounds")])
    furniture = getattr(recipe, "map_furniture", None)
    if furniture is not None:
        for field_name in ("scale_bar", "graticule"):
            item = getattr(furniture, field_name, None)
            item_options = _config_mapping(item)
            if item_options:
                candidates.extend([item_options.get("bbox"), item_options.get("bounds")])
    terrain_metadata = getattr(getattr(recipe, "terrain", None), "metadata", None)
    if isinstance(terrain_metadata, Mapping):
        candidates.extend([
            terrain_metadata.get("bbox"),
            terrain_metadata.get("bounds"),
            terrain_metadata.get("extent"),
        ])
    for candidate in candidates:
        bounds = _coerce_bounds(candidate)
        if bounds is not None and bounds[0] < bounds[2] and bounds[1] < bounds[3]:
            return bounds
    return (0.0, 0.0, 1.0, 1.0)


def _same_crs_name(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return True
    lhs = str(left).strip().upper()
    rhs = str(right).strip().upper()
    geo_aliases = {"EPSG:4326", "WGS84", "WGS 84"}
    return lhs == rhs or (lhs in geo_aliases and rhs in geo_aliases)


def _is_geographic_crs(crs: str | None) -> bool:
    return _same_crs_name(crs, "EPSG:4326")


def _furniture_target_crs(recipe: Any, options: Mapping[str, Any] | None = None) -> str:
    if options is not None and options.get("target_crs"):
        return str(options["target_crs"])
    target = getattr(recipe, "target_crs", None)
    if target:
        return str(target)
    terrain = getattr(recipe, "terrain", None)
    terrain_crs = getattr(terrain, "crs", None)
    return str(terrain_crs or "EPSG:4326")


def _terrain_bounds(recipe: Any) -> tuple[float, float, float, float] | None:
    terrain_metadata = getattr(getattr(recipe, "terrain", None), "metadata", None)
    if isinstance(terrain_metadata, Mapping):
        for key in ("bbox", "bounds", "extent"):
            bounds = _coerce_bounds(terrain_metadata.get(key))
            if bounds is not None and bounds[0] < bounds[2] and bounds[1] < bounds[3]:
                return bounds
    return None


def _bounds_edge_samples(bounds: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    west, south, east, north = bounds
    samples: list[tuple[float, float]] = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = west + (east - west) * t
        y = south + (north - south) * t
        samples.extend([(x, south), (x, north), (west, y), (east, y)])
    return samples


def _transform_bounds(
    bounds: tuple[float, float, float, float],
    source_crs: str,
    target_crs: str,
) -> tuple[float, float, float, float] | None:
    if _same_crs_name(source_crs, target_crs):
        return bounds
    try:
        import numpy as np

        from .crs import transform_coords

        transformed = transform_coords(
            np.asarray(_bounds_edge_samples(bounds), dtype=np.float64),
            str(source_crs),
            str(target_crs),
        )
        xs = transformed[:, 0].astype(np.float64)
        ys = transformed[:, 1].astype(np.float64)
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            return None
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    except Exception:
        return None


def _graticule_source_bounds(recipe: Any, options: Mapping[str, Any]) -> tuple[float, float, float, float]:
    explicit = _coerce_bounds(options.get("bbox")) or _coerce_bounds(options.get("bounds"))
    if explicit is not None:
        return explicit
    terrain_bounds = _terrain_bounds(recipe)
    if terrain_bounds is None:
        return _furniture_bounds(recipe, options)
    target_crs = _furniture_target_crs(recipe, options)
    if _is_geographic_crs(target_crs):
        return terrain_bounds
    transformed = _transform_bounds(terrain_bounds, target_crs, "EPSG:4326")
    return transformed or _furniture_bounds(recipe, options)


def _graticule_pixel_bounds(
    source_bounds: tuple[float, float, float, float],
    target_crs: str,
    options: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    for key in ("target_bounds", "projected_bounds", "render_bounds"):
        bounds = _coerce_bounds(options.get(key))
        if bounds is not None and bounds[0] < bounds[2] and bounds[1] < bounds[3]:
            return bounds
    transformed = _transform_bounds(source_bounds, "EPSG:4326", target_crs)
    return transformed or source_bounds


def _coord_to_pixel(
    point: Sequence[Any],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    west, south, east, north = bounds
    x_coord = float(point[0])
    y_coord = float(point[1])
    x = (x_coord - west) / max(east - west, 1.0e-9)
    y = (north - y_coord) / max(north - south, 1.0e-9)
    px = int(round(x * (width - 1)))
    py = int(round(y * (height - 1)))
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))


def _lonlat_to_pixel(
    point: Sequence[Any],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    return _coord_to_pixel(point, bounds, width, height)


def _overlay_position(
    image_shape: Sequence[int],
    overlay_shape: Sequence[int],
    position: str,
    margin: int,
) -> tuple[int, int]:
    src_h, src_w = int(overlay_shape[0]), int(overlay_shape[1])
    height, width = int(image_shape[0]), int(image_shape[1])
    key = str(position or "bottom-left").lower()
    if "right" in key:
        x = width - src_w - margin
    elif "center" in key:
        x = (width - src_w) // 2
    else:
        x = margin
    if "bottom" in key:
        y = height - src_h - margin
    elif "center" in key and "top" not in key:
        y = (height - src_h) // 2
    else:
        y = margin
    return int(x), int(y)


def _position_candidates(position: str) -> list[str]:
    requested = str(position or "bottom-left").lower()
    candidates = [
        requested,
        "bottom-right",
        "bottom-left",
        "top-right",
        "top-left",
        "bottom-center",
        "top-center",
        "center",
    ]
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _coerce_keepout_rect(value: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    padding = 0.0
    normalized = False
    candidate = value
    if isinstance(value, Mapping):
        candidate = value.get("bounds", value.get("screen_bounds", value.get("rect", value.get("bbox"))))
        padding = _number(value.get("padding_px", value.get("padding", 0.0)), 0.0)
        space = str(value.get("units", value.get("space", value.get("coordinate_space", "")))).lower()
        normalized = space in {"normalized", "relative", "fraction", "fractions"}
    bounds = _coerce_bounds(candidate)
    if bounds is None:
        return None
    x0, y0, x1, y1 = bounds
    if normalized:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    return left - padding, top - padding, right + padding, bottom + padding


def _rects_intersect(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] <= right[0]
        or right[2] <= left[0]
        or left[3] <= right[1]
        or right[3] <= left[1]
    )


def _furniture_keepouts(recipe: Any, image: Any) -> list[tuple[float, float, float, float]]:
    furniture = getattr(recipe, "map_furniture", None)
    raw_keepouts = getattr(furniture, "keepouts", None) if furniture is not None else None
    if not raw_keepouts:
        return []
    height, width = image.shape[:2]
    keepouts: list[tuple[float, float, float, float]] = []
    for item in raw_keepouts:
        rect = _coerce_keepout_rect(item, int(width), int(height))
        if rect is not None:
            keepouts.append(rect)
    return keepouts


def _place_overlay(
    image: Any,
    overlay: Any,
    position: str,
    margin: int,
    *,
    keepouts: Sequence[Any] | None = None,
) -> None:
    src_h, src_w = overlay.shape[:2]
    height, width = image.shape[:2]
    keepout_rects = [
        rect if isinstance(rect, tuple) else _coerce_keepout_rect(rect, int(width), int(height))
        for rect in (keepouts or ())
    ]
    keepout_rects = [rect for rect in keepout_rects if rect is not None]
    fallback = _overlay_position((height, width), (src_h, src_w), position, margin)
    for candidate in _position_candidates(position):
        x, y = _overlay_position((height, width), (src_h, src_w), candidate, margin)
        rect = (float(x), float(y), float(x + src_w), float(y + src_h))
        if not any(_rects_intersect(rect, keepout) for keepout in keepout_rects):
            _overlay_rgba(image, overlay, x, y)
            return
    x, y = fallback
    _overlay_rgba(image, overlay, x, y)


def _output_dpi_scale(recipe: Any, options: Mapping[str, Any] | None = None) -> float:
    if options is not None and options.get("scale_with_dpi") is False:
        return 1.0
    output = getattr(recipe, "output", None)
    metadata = getattr(output, "metadata", None)
    dpi_value = options.get("dpi") if options is not None and options.get("dpi") is not None else None
    base_value = options.get("base_dpi") if options is not None and options.get("base_dpi") is not None else None
    if dpi_value is None and isinstance(metadata, Mapping):
        dpi_value = metadata.get("dpi", metadata.get("resolution_dpi"))
        base_value = base_value if base_value is not None else metadata.get("base_dpi")
    dpi = _number(dpi_value, 150.0)
    base = max(1.0, _number(base_value, 150.0))
    return max(0.1, min(8.0, dpi / base))


def _scaled_furniture_options(
    options: Mapping[str, Any],
    scale: float,
    *,
    integer_fields: Sequence[str] = (),
    float_fields: Sequence[str] = (),
) -> dict[str, Any]:
    scaled = dict(options)
    if abs(float(scale) - 1.0) < 1.0e-9:
        return scaled
    for key in integer_fields:
        if key in scaled and scaled[key] is not None:
            scaled[key] = max(1, int(round(_number(scaled[key], 1.0) * float(scale))))
    for key in float_fields:
        if key in scaled and scaled[key] is not None:
            scaled[key] = _number(scaled[key], 1.0) * float(scale)
    return scaled


def _scaled_margin(options: Mapping[str, Any], scale: float, default: int = 12) -> int:
    return max(0, int(round(_number(options.get("margin_px"), float(default)) * float(scale))))


def _draw_graticule(image: Any, recipe: Any, options: Mapping[str, Any]) -> None:
    from .graticule import generate_graticule

    height, width = image.shape[:2]
    bounds = _graticule_source_bounds(recipe, options)
    target_crs = _furniture_target_crs(recipe, options)
    pixel_bounds = _graticule_pixel_bounds(bounds, target_crs, options)
    interval = _number(options.get("interval_deg", options.get("interval")), 1.0)
    try:
        graticule = generate_graticule(
            bounds,
            interval_deg=interval,
            target_crs=target_crs,
            include_labels=bool(options.get("include_labels", True)),
            precision=int(options.get("precision", 3)),
            line_steps=int(options.get("line_steps", 32)),
        )
    except Exception:
        return
    color = _color(options.get("color"), (24, 24, 24, 90))
    label_color = _color(options.get("label_color"), (24, 24, 24, 220))
    scale = _output_dpi_scale(recipe, options)
    width_px = max(0.5, _number(options.get("width_px"), 1.0) * scale)
    dash_array = options.get("dash_array")
    for feature in graticule.get("features", ()):
        geometry = feature.get("geometry") if isinstance(feature, Mapping) else None
        if not isinstance(geometry, Mapping):
            continue
        coords = geometry.get("coordinates") or ()
        points = [_coord_to_pixel(point, pixel_bounds, width, height) for point in coords]
        _draw_polyline(image, points, color, width_px=width_px, cap="butt", join="miter", dash_array=dash_array)
    if bool(options.get("include_labels", True)):
        for label in graticule.get("labels", ()):
            if not isinstance(label, Mapping):
                continue
            coordinate = label.get("coordinate")
            if not isinstance(coordinate, Sequence):
                continue
            anchor = _coord_to_pixel(coordinate, pixel_bounds, width, height)
            _draw_text(
                image,
                str(label.get("text", "")),
                (anchor[0] + 3, anchor[1] - 10),
                color=label_color,
                halo=(255, 255, 255, 190),
                halo_width_px=1.0,
            )


def _draw_title(image: Any, title: str) -> None:
    if not title:
        return
    _draw_text(
        image,
        str(title),
        (12, 12),
        color=(20, 20, 20, 255),
        halo=(255, 255, 255, 220),
        halo_width_px=2.0,
    )


def _draw_simple_legend(image: Any, options: Mapping[str, Any]) -> None:
    items = options.get("items") or options.get("labels")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
        return
    height, width = image.shape[:2]
    rows = [str(item) for item in items[:8]]
    panel_w = min(180, max(80, max(len(row) for row in rows) * 7 + 28))
    panel_h = len(rows) * 13 + 12
    x0 = width - panel_w - 12
    y0 = height - panel_h - 12
    import numpy as np

    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y0) : min(height, y0 + panel_h), max(0, x0) : min(width, x0 + panel_w)] = True
    _blend_region(image, mask, _color(options.get("background"), (255, 255, 255, 205)))
    for index, row in enumerate(rows):
        y = y0 + 8 + index * 13
        swatch = _rgb(row, salt="legend")
        _draw_pixel_block(image, x0 + 8, y + 3, (*swatch, 255), radius=3)
        _draw_text(image, row, (x0 + 18, y), color=(20, 20, 20, 255), halo=(255, 255, 255, 180))


def _compose_scale_bar(image: Any, recipe: Any, options: Mapping[str, Any]) -> None:
    from .map_plate import BBox
    from .scale_bar import ScaleBar, ScaleBarConfig

    scale = _output_dpi_scale(recipe, options)
    scaled_options = _scaled_furniture_options(
        options,
        scale,
        integer_fields=("width_px", "height_px", "font_size", "padding", "bar_height"),
    )
    cfg = ScaleBarConfig(**_config_kwargs(ScaleBarConfig, scaled_options))
    meters_per_pixel = options.get("meters_per_pixel")
    if meters_per_pixel is None:
        west, south, east, north = _furniture_bounds(recipe, options)
        bbox = BBox(west=west, south=south, east=east, north=north, crs=str(options.get("crs", "EPSG:4326")))
        meters_per_pixel = ScaleBar.compute_meters_per_pixel(
            bbox,
            int(image.shape[1]),
            geodesic=bool(options.get("geodesic", cfg.geodesic)),
        )
    overlay, label, anchor = ScaleBar(float(meters_per_pixel), config=cfg).render_geometry()
    _draw_text(
        overlay,
        label,
        anchor,
        color=cfg.label_color,
        halo=(0, 0, 0, 0),
        halo_width_px=0.0,
        font_size=float(cfg.font_size),
    )
    _place_overlay(
        image,
        overlay,
        str(options.get("position", "bottom-left")),
        _scaled_margin(options, scale),
        keepouts=_furniture_keepouts(recipe, image),
    )


def _compose_north_arrow(image: Any, recipe: Any, options: Mapping[str, Any]) -> None:
    from .north_arrow import NorthArrow, NorthArrowConfig

    scale = _output_dpi_scale(recipe, options)
    scaled_options = _scaled_furniture_options(
        options,
        scale,
        integer_fields=("size", "font_size", "border_width"),
    )
    cfg = NorthArrowConfig(**_config_kwargs(NorthArrowConfig, scaled_options))
    overlay, label, anchor = NorthArrow(cfg).render_geometry()
    if label is not None and anchor is not None:
        _draw_text(
            overlay,
            label,
            anchor,
            color=cfg.color,
            halo=(0, 0, 0, 0),
            halo_width_px=0.0,
            font_size=float(cfg.font_size),
        )
    _place_overlay(
        image,
        overlay,
        str(options.get("position", "top-right")),
        _scaled_margin(options, scale),
        keepouts=_furniture_keepouts(recipe, image),
    )


def _compose_furniture(image: Any, recipe: Any) -> None:
    furniture = getattr(recipe, "map_furniture", None)
    if furniture is None:
        return
    graticule = _config_mapping(getattr(furniture, "graticule", None))
    if graticule:
        _draw_graticule(image, recipe, graticule)
    legend = _config_mapping(getattr(furniture, "legend", None))
    if legend:
        _draw_simple_legend(image, legend)
    scale_bar = _config_mapping(getattr(furniture, "scale_bar", None))
    if scale_bar:
        _compose_scale_bar(image, recipe, scale_bar)
    north_arrow = _config_mapping(getattr(furniture, "north_arrow", None))
    if north_arrow:
        _compose_north_arrow(image, recipe, north_arrow)
    _draw_title(image, str(getattr(furniture, "title", "") or ""))


def _composite_recipe_layers(
    base: Any,
    recipe: Any,
    plans: Mapping[str, Any],
    *,
    layer_types: MapSceneRenderLayerTypes,
    load_raster_overlay: Any,
    include_raster: bool = True,
    include_vectors: bool = True,
    include_labels: bool = True,
    include_buildings: bool = True,
    include_point_cloud: bool = True,
) -> Any:
    import numpy as np

    output = recipe.output
    width = int(output.width)
    height = int(output.height)
    yy, xx = np.mgrid[0:height, 0:width]

    for layer in recipe.layers:
        layer_id = _layer_id(layer, "layer")
        if isinstance(layer, layer_types.raster_overlay) and include_raster:
            overlay = load_raster_overlay(layer)
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
                blended = (
                    base[..., :3].astype(np.float32) * (1.0 - fallback_alpha)
                    + color * fallback_alpha
                ).astype(np.uint8)
                base[..., :3] = np.where(mask[..., None], blended, base[..., :3])
        elif isinstance(layer, layer_types.vector_overlay) and include_vectors:
            line_paint = _paint(layer, "line")
            line_layout = _layout(layer, "line")
            fill_paint = _paint(layer, "fill")
            fallback_rgb = _rgb(layer.to_dict(), salt="vector")
            line_color_value = line_paint.get("line-color")
            line_color = (
                (*fallback_rgb, 255)
                if _is_style_expression(line_color_value)
                else _color(line_color_value, (*fallback_rgb, 255))
            )
            line_opacity_value = line_paint.get("line-opacity")
            line_opacity = (
                line_color[3] / 255.0
                if _is_style_expression(line_opacity_value)
                else _number(line_opacity_value, line_color[3] / 255.0)
            )
            line_color = (line_color[0], line_color[1], line_color[2], max(0, min(255, int(round(line_opacity * 255.0)))))
            line_width = _resolve_line_width_px(layer, line_paint, recipe, width, height)
            line_cap = str(line_layout.get("line-cap") or getattr(layer, "line_cap", "butt") or "butt").lower()
            line_join = str(line_layout.get("line-join") or getattr(layer, "line_join", "miter") or "miter").lower()
            miter_limit = _number(line_layout.get("line-miter-limit"), 4.0)
            dash_array = getattr(layer, "dash_array", None) or line_paint.get("line-dasharray")
            fill_color_value = fill_paint.get("fill-color")
            fill_color = (
                (*fallback_rgb, 160)
                if _is_style_expression(fill_color_value)
                else _color(fill_color_value, (*fallback_rgb, 160))
            )
            fill_opacity_value = fill_paint.get("fill-opacity")
            fill_opacity = (
                fill_color[3] / 255.0
                if _is_style_expression(fill_opacity_value)
                else _number(fill_opacity_value, fill_color[3] / 255.0)
            )
            fill_color = (fill_color[0], fill_color[1], fill_color[2], max(0, min(255, int(round(fill_opacity * 255.0)))))
            for feature in layer.features or ():
                geometry = feature.get("geometry") if isinstance(feature, Mapping) else None
                if not isinstance(geometry, Mapping):
                    continue
                properties = _properties(feature)
                feature_line_color = _feature_color(line_color_value, properties, line_color)
                feature_line_opacity = _feature_number(
                    line_opacity_value,
                    properties,
                    feature_line_color[3] / 255.0,
                )
                feature_line_color = (
                    feature_line_color[0],
                    feature_line_color[1],
                    feature_line_color[2],
                    max(0, min(255, int(round(feature_line_opacity * 255.0)))),
                )
                feature_line_width = line_width
                if getattr(layer, "width_px", None) is None and _is_style_expression(line_paint.get("line-width")):
                    feature_line_width = max(1.0, _feature_number(line_paint.get("line-width"), properties, line_width))
                feature_fill_color = _feature_color(fill_color_value, properties, fill_color)
                feature_fill_opacity = _feature_number(
                    fill_opacity_value,
                    properties,
                    feature_fill_color[3] / 255.0,
                )
                feature_fill_color = (
                    feature_fill_color[0],
                    feature_fill_color[1],
                    feature_fill_color[2],
                    max(0, min(255, int(round(feature_fill_opacity * 255.0)))),
                )
                geometry_type = str(geometry.get("type", "")).lower()
                if geometry_type in {"polygon", "multipolygon"}:
                    for polygon_rings in _geometry_polygon_rings(geometry):
                        pixel_rings = [
                            [_point_to_pixel(point, width, height) for point in ring]
                            for ring in polygon_rings
                            if len(ring) >= 3
                        ]
                        _draw_polygon_fill(base, pixel_rings, feature_fill_color)
                        for ring_points in pixel_rings:
                            if ring_points and ring_points[0] != ring_points[-1]:
                                ring_points = [*ring_points, ring_points[0]]
                            if len(ring_points) >= 2:
                                _draw_polyline(
                                    base,
                                    ring_points,
                                    feature_line_color,
                                    width_px=feature_line_width,
                                    cap=line_cap,
                                    join=line_join,
                                    dash_array=dash_array,
                                    miter_limit=miter_limit,
                                )
                    continue

                points = [_point_to_pixel(point, width, height) for point in _geometry_points(geometry)]
                if len(points) == 1:
                    _draw_pixel_block(
                        base,
                        points[0][0],
                        points[0][1],
                        feature_line_color,
                        radius=max(1, int(round(feature_line_width))),
                    )
                else:
                    _draw_polyline(
                        base,
                        points,
                        feature_line_color,
                        width_px=feature_line_width,
                        cap=line_cap,
                        join=line_join,
                        dash_array=dash_array,
                        miter_limit=miter_limit,
                    )
        elif isinstance(layer, layer_types.label_layer) and include_labels:
            plan = plans.get(layer_id)
            if plan is None:
                continue
            label_rgb = _rgb(layer.to_dict(), salt="label")
            for accepted in plan.accepted:
                typography = dict(getattr(accepted, "typography", None) or {})
                text_color = _color(typography.get("color") or typography.get("text_color"), (*label_rgb, 255))
                halo_color = _color(typography.get("halo_color") or typography.get("text_halo_color"), (0, 0, 0, 190))
                halo_width = _number(
                    typography.get("halo_width_px")
                    if "halo_width_px" in typography
                    else typography.get("halo_width", typography.get("text_halo_width")),
                    1.0,
                )
                _draw_text(
                    base,
                    str(accepted.text),
                    _label_anchor(accepted, width, height),
                    color=text_color,
                    halo=halo_color,
                    halo_width_px=halo_width,
                    font_size=_number(
                        typography.get("size", typography.get("font_size")), 12.0
                    ),
                    font_chain=typography.get("font_chain"),
                )
        elif isinstance(layer, layer_types.building_layer) and include_buildings:
            _draw_buildings(base, layer, width, height)
        elif isinstance(layer, layer_types.point_cloud_layer) and include_point_cloud and layer.point_count:
            color = (*_rgb(layer.to_dict(), salt="point-cloud"), 255)
            count = min(int(layer.point_count), 64)
            layer_seed = _hash_int(layer.to_dict(), salt="point-cloud")
            for index in range(count):
                x = (layer_seed + index * 17) % width
                y = ((layer_seed >> 8) + index * 29) % height
                _draw_pixel_block(base, x, y, color, radius=0)

    _compose_furniture(base, recipe)
    return base


def render_source_derived_rgba(
    recipe: Any,
    plans: Mapping[str, Any],
    *,
    layer_types: MapSceneRenderLayerTypes,
    load_raster_overlay: Any,
) -> Any:
    import numpy as np

    output = recipe.output
    width = int(output.width)
    height = int(output.height)
    payload = _render_payload(recipe)
    seed = _hash_int(payload, salt="mapscene-source-render")

    yy, xx = np.mgrid[0:height, 0:width]
    base = np.empty((height, width, 4), dtype=np.uint8)
    base[..., 0] = (
        (xx * ((seed & 0x0F) + 3) + (seed >> 8)) % 256
    ).astype(np.uint8)
    base[..., 1] = (
        (yy * (((seed >> 4) & 0x0F) + 5) + (seed >> 16)) % 256
    ).astype(np.uint8)
    base[..., 2] = (
        ((xx + yy) * (((seed >> 12) & 0x0F) + 7) + (seed >> 24)) % 256
    ).astype(np.uint8)
    base[..., 3] = 255
    return _composite_recipe_layers(
        base,
        recipe,
        plans,
        layer_types=layer_types,
        load_raster_overlay=load_raster_overlay,
        include_labels=True,
    )
