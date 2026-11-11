"""Compile route, landmark and locator decisions for an existing terrain scene."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from ._map_scene_projection import TerrainProjector
from .gis import simplify_geometry
from .map_scene import (
    LabelLayer, MapFurnitureLayer, MapScene, OrbitCamera, SceneRecipe, VectorOverlay,
    _is_3d_camera_mode, _mapscene_effective_camera_mode,
)

XY = tuple[float, float]
Bounds = tuple[float, float, float, float]

__all__ = ["RouteEdge", "Connection", "Landmark", "ScaleFrame", "ScalePolicy", "CompiledScale", "compile_scale"]


@dataclass(frozen=True)
class RouteEdge:
    source_id: str
    path: tuple[XY, ...]


@dataclass(frozen=True)
class Connection:
    """An authored, ordered walk through source edges, in either direction."""

    name: str
    edge_ids: tuple[str, ...]


@dataclass(frozen=True)
class Landmark:
    source_id: str
    name: str
    xy: XY
    required: bool
    reason: str
    # Optional local labels become eligible when this ground span projects
    # at least as wide as the authored label font size.
    detail_span_m: float | None = None


@dataclass(frozen=True)
class ScaleFrame:
    camera: OrbitCamera
    focus: XY
    # Snapshot from the existing residency owner, or an explicit scenario
    # input. Each entry names unavailable detail and why it is unavailable.
    unavailable_detail: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScalePolicy:
    crs: str
    vertical_reference: str
    routes: tuple[RouteEdge, ...]
    connections: tuple[Connection, ...]
    landmarks: tuple[Landmark, ...]
    locator_bounds: Bounds
    locator_rect: Bounds  # output fractions, left/top/right/bottom
    overview_tolerance_m: float
    pixel_tolerance: float
    stroke_width_px: float
    label_size_px: float
    reasons: tuple[str, ...]
    omissions: tuple[str, ...]
    simplification_crs: str | None = None
    route_color: str = "#f6be66"
    glyph_atlas: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CompiledScale:
    """Frozen frame decisions and recipes ready for MapScene and CHRONOS."""

    _recipes: tuple[SceneRecipe, ...]
    _policy_record: str
    _frame_records: tuple[str, ...]

    def to_dict(self) -> dict:
        return {**json.loads(self._policy_record), "frames": [json.loads(frame) for frame in self._frame_records]}

    def frame_record(self, index: int) -> dict:
        return json.loads(self._frame_records[self._index(index)])

    def _index(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < len(self._recipes):
            raise IndexError(f"no semantic-scale frame {index!r}")
        return index

    def scene(self, index: int) -> MapScene:
        recipe = self._recipes[self._index(index)]
        # The existing terrain and raster residency inputs are shared. Only
        # the frame-owned layers and camera need copies for the caller.
        return MapScene(replace(
            recipe, camera=copy.deepcopy(recipe.camera),
            layers=tuple(copy.deepcopy(layer) for layer in recipe.layers),
            map_furniture=copy.deepcopy(recipe.map_furniture), output=copy.deepcopy(recipe.output),
        ))

    def scenes(self) -> tuple[MapScene, ...]:
        return tuple(self.scene(i) for i in range(len(self._recipes)))


def _xy(point: XY) -> XY:
    if len(point) != 2 or not all(math.isfinite(v) for v in point):
        raise ValueError("source coordinates must be finite projected XY in metres")
    return tuple(map(float, point))


def _bounds(bounds: Bounds) -> Bounds:
    if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
        raise ValueError("bounds must contain four finite coordinates")
    x0, y0, x1, y1 = bounds
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bounds must have positive width and height")
    return bounds


def _check_connections(routes: dict[str, tuple[XY, ...]], connections: Sequence[Connection]) -> None:
    for connection in connections:
        ids = connection.edge_ids
        if not connection.name.strip() or not ids or any(key not in routes for key in ids):
            raise ValueError(f"required connection {connection.name!r} must reference known edges")
        possible = {routes[ids[0]][0], routes[ids[0]][-1]}
        for key in ids[1:]:
            start, end = routes[key][0], routes[key][-1]
            possible = {b for a, b in ((start, end), (end, start)) if a in possible}
        if not possible:
            raise ValueError(f"required connection {connection.name!r} is disconnected at source endpoints")


def _line(source_id, path):
    return {"type": "Feature", "id": source_id, "geometry": {
        "type": "LineString", "coordinates": path,
    }}


def _simplify(path: tuple[XY, ...], policy: ScalePolicy) -> tuple[XY, ...]:
    from rasterio.crs import CRS
    from rasterio.warp import transform

    crs = policy.simplification_crs or policy.crs
    if CRS.from_user_input(crs) != CRS.from_user_input(policy.crs):
        x, y = zip(*path, strict=True)
        coordinates = tuple(zip(*transform(policy.crs, crs, x, y), strict=True))
    else:
        coordinates = path
    candidate = simplify_geometry(
        _line("", coordinates)["geometry"], policy.overview_tolerance_m,
        preserve_topology=True, crs=crs,
    )
    # GENERALIS retains original vertices. Use that correspondence to keep
    # the exact prepared XY and network joins, without an inverse round trip.
    originals = dict(zip(coordinates, path, strict=True))
    if len(originals) != len(set(path)):
        raise ValueError("simplification CRS collapsed distinct route coordinates")
    result = tuple(originals[_xy(point)] for point in candidate["geometry"]["coordinates"])
    if len(result) < 2 or (result[0], result[-1]) != (path[0], path[-1]):
        raise ValueError("GENERALIS changed a route endpoint")
    return result


def _viewport_samples(line, width, height):
    """Sample the projected centerline inside the viewport, including crossings."""
    start = line[:-1]
    delta = line[1:] - start
    enter, leave = np.zeros(len(start)), np.ones(len(start))
    valid = np.ones(len(start), dtype=bool)
    for axis, limit in enumerate((width, height)):
        moving = delta[:, axis] != 0
        a = np.divide(-start[:, axis], delta[:, axis], out=np.zeros(len(start)), where=moving)
        b = np.divide(limit - start[:, axis], delta[:, axis], out=np.ones(len(start)), where=moving)
        enter = np.maximum(enter, np.minimum(a, b))
        leave = np.minimum(leave, np.maximum(a, b))
        valid &= moving | ((0 <= start[:, axis]) & (start[:, axis] <= limit))
    valid &= enter <= leave
    return np.concatenate((start[valid] + enter[valid, None] * delta[valid],
                           start[valid] + leave[valid, None] * delta[valid]))


def _screen_error(source, candidate, width, height) -> float | None:
    a = source[:, :2]
    b = a if candidate is source else candidate[:, :2]
    in_view_a = _viewport_samples(a, width, height)
    in_view_b = in_view_a if b is a else _viewport_samples(b, width, height)
    if not len(in_view_a) and not len(in_view_b):
        return None
    if a is b:
        return 0.0

    def directed(points, line):
        if not len(points):
            return 0.0
        starts = line[:-1]
        delta = line[1:] - starts
        lengths = np.sum(delta * delta, axis=1)
        nearest = np.full(len(points), np.inf)
        for start, step, squared_length in zip(starts, delta, lengths, strict=True):
            offset = points - start
            if squared_length:
                fraction = np.clip((offset @ step) / squared_length, 0.0, 1.0)
                offset -= fraction[:, None] * step
            nearest = np.minimum(nearest, np.sum(offset * offset, axis=1))
        return float(np.max(nearest))

    return math.sqrt(max(directed(in_view_a, b), directed(in_view_b, a)))


def _route_depth_diagnostics(projector, projected, width, height):
    """Record centerline samples rejected by MapScene's terrain depth test."""
    if not projected:
        return []
    depth = projector.depth_image()
    diagnostics = []
    for source_id, points in projected.items():
        x, y, z = points.T
        inside = (0 <= x) & (x < width) & (0 <= y) & (y < height)
        surface = depth[y[inside].astype(int), x[inside].astype(int)]
        hidden = int(np.count_nonzero(z[inside] > np.nextafter(surface, np.float32(np.inf))))
        if hidden:
            diagnostics.append({"source_id": source_id, "reason": "route_depth_rejected_samples",
                                "samples": hidden, "in_view_samples": int(np.count_nonzero(inside))})
    return diagnostics


def _validate_base(base, policy):
    from rasterio.crs import CRS

    recipe = base.recipe
    terrain = recipe.terrain
    crs = CRS.from_user_input(policy.crs)
    metadata = terrain.metadata or {}
    simplify_crs = CRS.from_user_input(policy.simplification_crs or policy.crs)
    if not simplify_crs.is_projected or simplify_crs.linear_units_factor[1] != 1.0:
        raise ValueError("GENERALIS requires a projected metre simplification CRS")
    if (not crs.is_projected or crs.linear_units_factor[1] != 1.0
            or CRS.from_user_input(terrain.crs) != crs
            or CRS.from_user_input(recipe.target_crs or terrain.crs) != crs):
        raise ValueError("semantic scale requires prepared terrain and vectors in one projected metre CRS")
    if (not policy.vertical_reference.strip() or metadata.get("units") != "m"
            or metadata.get("vertical_reference") != policy.vertical_reference):
        raise ValueError("prepared terrain height datum and metre units must match the policy")
    if not recipe.output or not _is_3d_camera_mode(_mapscene_effective_camera_mode(recipe)):
        raise ValueError("semantic scale requires a native 3D terrain camera and output viewport")
    if any(isinstance(layer, (VectorOverlay, LabelLayer)) for layer in recipe.layers):
        raise ValueError("supply the route network and named labels through ScalePolicy")
    if not math.isfinite(policy.overview_tolerance_m) or policy.overview_tolerance_m < 0:
        raise ValueError("overview tolerance must be finite and nonnegative")
    for value in (policy.pixel_tolerance,
                  policy.stroke_width_px, policy.label_size_px):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("scale tolerances and style sizes must be finite and positive")
    if not isinstance(policy.route_color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", policy.route_color):
        raise ValueError("route_color must be an opaque #RRGGBB value")
    if not policy.reasons or any(not reason.strip() for reason in (*policy.reasons, *policy.omissions)):
        raise ValueError("record reasons for scale choices and intentional omissions")
    return recipe


def _suppress_single_frame_labels(frame_recipes, records, landmarks):
    from .chronos import _label_timeline_events
    from .label_plan import LabelPlan, _plan_bounds

    layer_id = "semantic-scale-labels"
    optional = {place.source_id for place in landmarks if not place.required}
    keys = {i: {(layer_id, key) for key in row["labels_accepted"] if key in optional}
            for i, row in enumerate(records)}
    events = _label_timeline_events(keys, range(len(records)))
    suppressed: dict[int, set[str]] = {}
    for (_, key), timeline in events.items():
        for index, (start, end) in timeline.items():
            if start == index and end == index + 1:
                suppressed.setdefault(index, set()).add(key)

    for index, names in suppressed.items():
        recipe, record = frame_recipes[index], records[index]
        layer = recipe.layers[-1]
        plan = LabelPlan.from_dict(layer.plan)
        plan.accepted = tuple(label for label in plan.accepted if label.source_id not in names)
        plan.bounds = {**plan.bounds, **_plan_bounds(plan.accepted)}
        render_plan = plan.to_dict()
        render_plan.pop("rationale")
        frame_recipes[index] = replace(recipe, layers=(*recipe.layers[:-1], replace(layer, plan=render_plan)))
        record["labels_accepted"] = [key for key in record["labels_accepted"] if key not in names]
        record["labels_omitted"].update(dict.fromkeys(names, "single_frame_admission"))


def compile_scale(base: MapScene, policy: ScalePolicy, frames: Sequence[ScaleFrame]) -> CompiledScale:
    """Select GENERALIS candidates by measured output error, then freeze MapScene plans.

    Route and label XY are prepared in the same projected metre CRS. They
    drape on the prepared DEM, so source Z is never silently reinterpreted.
    Residency inputs describe detail availability; this compiler does not
    request or evict terrain or texture pages.
    """
    recipe = _validate_base(base, policy)
    if not frames or not policy.routes or not policy.connections or not any(p.required for p in policy.landmarks):
        raise ValueError("declare frames, a route network, required connections and required landmarks")
    west, south, east, north = _bounds(policy.locator_bounds)
    rect = _bounds(policy.locator_rect)
    width, height = recipe.output.width, recipe.output.height
    if not all(0 <= p <= 1 for p in rect):
        raise ValueError("locator rectangle must be inside the output")
    if round(rect[0] * width) == round(rect[2] * width) or round(rect[1] * height) == round(rect[3] * height):
        raise ValueError("locator rectangle is smaller than the output pixel grid")
    ids = [item.source_id for item in (*policy.routes, *policy.landmarks)]
    if any(not key.strip() for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("route and landmark source IDs must be nonempty and unique")
    routes = {edge.source_id: tuple(map(_xy, edge.path)) for edge in policy.routes}
    if any(len(path) < 2 or any(a == b for a, b in zip(path, path[1:])) for path in routes.values()):
        raise ValueError("route edges need at least two distinct consecutive points")
    _check_connections(routes, policy.connections)
    required_edges = {key for connection in policy.connections for key in connection.edge_ids}
    for place in policy.landmarks:
        _xy(place.xy)
        if (not place.name.strip() or not place.reason.strip()
                or (place.detail_span_m is not None and
                    (place.required or not math.isfinite(place.detail_span_m) or place.detail_span_m <= 0))):
            raise ValueError(f"invalid landmark definition for {place.source_id!r}")
    overview = {key: _simplify(path, policy) for key, path in routes.items()}

    def locator_xy(point):
        x, y = _xy(point)
        if not (west <= x <= east and south <= y <= north):
            raise ValueError(f"locator does not contain source coordinate {point!r}")
        return ((x - west) / (east - west), (north - y) / (north - south))

    locator_routes = {key: tuple(map(locator_xy, path)) for key, path in routes.items()}
    locator_landmarks = {place.source_id: locator_xy(place.xy) for place in policy.landmarks if place.required}
    frame_recipes, records = [], []
    for frame in frames:
        if any(not reason.strip() for reason in frame.unavailable_detail):
            raise ValueError("unavailable detail requires a reason")
        camera = frame.camera
        if (len(camera.target) != 3 or not all(math.isfinite(value) for value in
                (*camera.target, camera.distance, camera.azimuth_deg, camera.elevation_deg, camera.fov_deg))
                or camera.distance <= 0 or not 0 < camera.fov_deg < 180 or not 0 < camera.elevation_deg < 180):
            raise ValueError("semantic scale requires a finite perspective camera with positive distance")
        frame_recipe = replace(recipe, camera=copy.deepcopy(camera))
        projector = TerrainProjector(frame_recipe, crs=policy.crs)
        locator_focus = locator_xy(frame.focus)
        projector.uv(frame.focus)
        for point in (*[p for path in routes.values() for p in path], *(p.xy for p in policy.landmarks)):
            projector.uv(point)
        chosen, decisions, diagnostics = [], {}, []
        projected_required = {}
        locator_only = False
        projection_unavailable = False
        for key, path in routes.items():
            try:
                full_pixels = np.asarray(projector.path(path))
                overview_pixels = full_pixels if path == overview[key] else np.asarray(projector.path(overview[key]))
                error = _screen_error(full_pixels, overview_pixels, width, height)
            except ValueError as exc:
                decisions[key] = {"representation": "locator_only", "reason": str(exc)}
                diagnostics.append({"source_id": key, "reason": "route_projection_unavailable", "detail": str(exc)})
                projection_unavailable = True
                continue
            if error is None:
                decisions[key] = {"representation": "locator_only", "reason": "outside_viewport"}
                continue
            use_overview = error <= policy.pixel_tolerance
            reason = ("detail_unavailable" if frame.unavailable_detail else
                      "within_output_tolerance" if use_overview else "overview_exceeds_output_tolerance")
            decisions[key] = {"representation": "overview" if use_overview else "full",
                              "reason": reason, "overview_error_px": error}
            if frame.unavailable_detail and not use_overview:
                locator_only = True
                diagnostics.append({"source_id": key, "reason": "overview_exceeds_output_tolerance",
                                    "error_px": error})
            chosen.append(_line(key, overview[key] if use_overview else path))
            if key in required_edges:
                projected_required[key] = overview_pixels if use_overview else full_pixels
        if locator_only:
            # The locator retains the whole connected route. A partially drawn
            # main-view itinerary would imply connections had been lost.
            chosen.clear()
            for decision in decisions.values():
                if decision["representation"] != "locator_only":
                    decision.update(representation="locator_only", reason="network_fallback_outside_tolerance")
        else:
            diagnostics.extend(_route_depth_diagnostics(projector, projected_required, width, height))
        labels, omitted = [], {}
        # Clearance at the prepared grid spacing keeps the fixed anchor clear
        # of the DEM cell it labels. The label solver still tests terrain depth.
        label_clearance = projector.span / max(1, max(projector.heightmap.shape) - 1)
        for place in policy.landmarks:
            if place.detail_span_m is not None and frame.unavailable_detail:
                omitted[place.source_id] = "detail_unavailable"
                continue
            try:
                elevation = projector.height(projector.uv(place.xy)) + label_clearance
                anchor = (*place.xy, elevation)
                projector.project(anchor)
                if place.detail_span_m is not None:
                    span = projector.stroke_width(place.xy, None, place.detail_span_m)
                    if span < policy.label_size_px:
                        omitted[place.source_id] = "ground_span_smaller_than_label"
                        continue
            except ValueError as exc:
                omitted[place.source_id] = f"projection_unavailable: {exc}"
                continue
            labels.append({
                "id": place.source_id, "source_id": place.source_id, "text": place.name,
                "geometry": {"type": "Point", "coordinates": anchor},
                "priority_class": "required" if place.required else "default",
                "candidate_policy": {"offset_px": 0.0, "radial_count": 0},
            })

        locator = {"rect": rect, "aspect": (east - west) / (north - south),
                   "routes": locator_routes, "landmarks": locator_landmarks, "focus": locator_focus,
                   "route_color": policy.route_color}
        if locator_only or (projection_unavailable and not chosen):
            locator["route_in_locator_only"] = True
        elif projection_unavailable:
            locator["route_continues_in_locator"] = True
        keepout = {"region_id": "semantic-scale-locator", "kind": "locator",
                   "bounds": [rect[0] * width, rect[1] * height, rect[2] * width, rect[3] * height]}
        furniture = recipe.map_furniture or MapFurnitureLayer()
        furniture = replace(furniture, keepouts=(*(furniture.keepouts or ()), keepout),
                            metadata={**(furniture.metadata or {}), "semantic_scale_locator": locator})
        label_layer = LabelLayer(
            "semantic-scale-labels", labels=labels, glyph_atlas=copy.deepcopy(policy.glyph_atlas),
            typography={"font_size": policy.label_size_px},
            priority_rules=({"name": "required", "rank": 1},),
            metadata={"coordinate_space": "world"},
        )
        layers = (*recipe.layers, *((VectorOverlay(
            "semantic-scale-routes", features=chosen, crs=policy.crs,
            width_px=policy.stroke_width_px, line_join="round", line_cap="round",
            style={"version": 8, "layers": [{"id": "itinerary", "type": "line",
                                            "paint": {"line-color": policy.route_color}}]},
        ),) if chosen else ()), *((label_layer,) if labels else ()))
        frame_recipe = replace(frame_recipe, layers=layers, map_furniture=furniture)
        compiled = MapScene(frame_recipe).compile_plan()
        plan = compiled.label_plans.get(label_layer.layer_id)
        accepted = [label.source_id for label in plan.accepted] if plan is not None else []
        rejected = {label.source_id: label.reason for label in plan.rejected} if plan is not None else {}
        for place in policy.landmarks:
            if place.required and place.source_id not in accepted:
                diagnostics.append({"source_id": place.source_id, "reason": "required_label_unplaced",
                                    "detail": rejected.get(place.source_id) or omitted.get(place.source_id)})
        rationale = []
        if plan is not None:
            # CHRONOS rebinds the frame seed and output path. Carry placement
            # as a LabelPlan so that rebinding cannot run the solver again.
            # Solver rationale belongs in the semantic receipt. Rendering
            # consumes the placements, glyphs and rejection diagnostics.
            render_plan = plan.to_dict()
            rationale = render_plan.pop("rationale")
            frame_recipe = replace(frame_recipe, layers=(*layers[:-1], replace(label_layer, plan=render_plan)))
        frame_recipes.append(frame_recipe)
        records.append({"frame": len(records), "camera": frame.camera.to_dict(),
                        "focus": frame.focus, "unavailable_detail": frame.unavailable_detail,
                        "routes": decisions, "labels_placed": accepted.copy(), "labels_accepted": accepted,
                        "labels_rejected": rejected, "labels_omitted": omitted, "label_rationale": rationale,
                        "diagnostics": diagnostics, "locator": locator})

    _suppress_single_frame_labels(frame_recipes, records, policy.landmarks)

    def freeze(value):
        return json.dumps(value, allow_nan=False, sort_keys=True)

    return CompiledScale(tuple(frame_recipes), freeze({
        "policy": asdict(policy), "output": [width, height], "overview_routes": overview,
    }), tuple(map(freeze, records)))
