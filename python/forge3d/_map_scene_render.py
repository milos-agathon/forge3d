"""Source-derived MapScene rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._map_scene_common import _layer_id, _stable_hash


@dataclass(frozen=True)
class MapSceneRenderLayerTypes:
    raster_overlay: type
    vector_overlay: type
    label_layer: type
    point_cloud_layer: type


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
    recipe: Any,
    plans: Mapping[str, Any],
    *,
    layer_types: MapSceneRenderLayerTypes,
    load_raster_overlay: Any,
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
        elif isinstance(layer, layer_types.vector_overlay):
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
        elif isinstance(layer, layer_types.label_layer):
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
                    cx, cy = _point_to_pixel(
                        (len(str(accepted.label_id)) * 7, len(str(accepted.text)) * 5),
                        width,
                        height,
                    )
                    _draw_pixel_block(base, cx, cy, color, radius=2)
        elif isinstance(layer, layer_types.point_cloud_layer) and include_point_cloud and layer.point_count:
            color = _rgb(layer.to_dict(), salt="point-cloud")
            count = min(int(layer.point_count), 64)
            layer_seed = _hash_int(layer.to_dict(), salt="point-cloud")
            for index in range(count):
                x = (layer_seed + index * 17) % width
                y = ((layer_seed >> 8) + index * 29) % height
                _draw_pixel_block(base, x, y, color, radius=0)

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
    )
