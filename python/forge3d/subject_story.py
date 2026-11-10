"""Compile a feature-bound reveal and hold for the MapScene terrain mesh."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import math
from operator import index
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from ._subject_clearance import BakedClearance
from ._subject_geometry import (
    _GRID, StoryMesh, box_triangles, intersects_rectangle, mesh_height as _mesh_height,
    obstacle_at_eye, point_blocker, segment_cuts, whole_blocker,
)
from .camera_rigs import TerrainOrbitRig
from .chronos import FlythroughManifest, render_flythrough
from ._map_scene_render import _point_to_pixel
from .label_plan import KeepoutRegion, LabelPlan, _shape_label_glyphs
from .map_scene import (
    LabelLayer, LightingPreset, MapFurnitureLayer, MapScene, OrbitCamera,
    OutputSpec, SceneRecipe, TerrainSource,
    _build_mapscene_terrain_params, _heightmap_domain, _load_native_heightmap,
    _native_glyph_box,
)
from .terrain_scatter import TerrainScatterSource
from .text_atlas import default_latin_atlas_paths, load_atlas_metrics


@dataclass(frozen=True)
class SubjectFeature:
    """Stable ID and orbit anchor, optionally with a terrain-draped extent."""

    feature_id: str
    name: str
    easting: float
    northing: float
    line_xy: tuple[tuple[float, float], ...] = ()
    area_triangles_xy: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass(frozen=True)
class StoryObstacle:
    """Opaque solid envelope in the same projected CRS and height datum as terrain."""

    obstacle_id: str
    kind: Literal["building", "vegetation"]
    bounds_xyz_m: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class StoryAnnotation:
    """One screen annotation, with a normalized baseline and a pixel font size."""

    text: str
    anchor: tuple[float, float]
    font_size: float


class SubjectShotInfeasible(ValueError):
    """A required constraint failed at authoring or at a specific output frame."""


@dataclass(frozen=True)
class CompiledSubjectShot:
    fps: int
    reveal_frames: int
    _recipes: tuple[SceneRecipe, ...]
    _checks: tuple[Mapping[str, Any], ...]
    _terrain_sha256: str

    @property
    def frame_count(self) -> int:
        return len(self._recipes)

    @property
    def key_frames(self) -> dict[str, int]:
        return {"establish": 0, "reveal": self.reveal_frames, "hold": self.frame_count - 1}

    @property
    def frame_checks(self) -> tuple[Mapping[str, Any], ...]:
        """Native camera inputs, terrain clearance and normalized feature projections."""
        return deepcopy(self._checks)

    def frame_scene(self, frame_index: int) -> MapScene:
        """Get a copy of the scene to use with CHRONOS manifest replay."""
        frame_index = index(frame_index)
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(f"no subject shot frame {frame_index}")
        self._verify_terrain()
        return MapScene(deepcopy(self._recipes[frame_index]))

    def _verify_terrain(self) -> None:
        path = Path(self._recipes[0].terrain.path)
        if sha256(path.read_bytes()).hexdigest() != self._terrain_sha256:
            raise ValueError("subject shot terrain changed since compilation")

    def render(
        self, out_dir: str | Path, *, base_seed: int, certificate: bool = False,
        cache: str | PathLike[str] | None = None,
    ) -> FlythroughManifest:
        """Render the checked sequence through the native CHRONOS frame route."""
        self._verify_terrain()
        manifest = render_flythrough(
            [MapScene(deepcopy(recipe)) for recipe in self._recipes],
            base_seed=base_seed, samples=1, out_dir=out_dir, certificate=certificate, cache=cache,
        )
        self._verify_terrain()
        return manifest


# A 512 × 512 texture maps to the native mesh one-to-one under nearest sampling.
# Its native row diagonal runs northwest–southeast, even after the north-up view.
_MESH_CAMERA_MODE = "mesh:zup:north"


def _terrain_segment_margins(
    heights: np.ndarray, start_point: np.ndarray, end_point: np.ndarray,
) -> np.ndarray:
    t = segment_cuts(start_point, end_point)[1:-1]
    points = start_point + t[:, None] * (end_point - start_point)
    return points[:, 2] - _mesh_height(heights, points[:, :2])


def _rectangle(values: Sequence[float], name: str) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = map(float, values)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"{name} must be a nonempty normalized viewport rectangle")
    return x0, y0, x1, y1


def _contains(rect: Sequence[float], x: float, y: float) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _annotation_layer(
    annotation: StoryAnnotation, subject: SubjectFeature, output: OutputSpec,
    reserve: tuple[float, float, float, float], label_id: str,
) -> LabelLayer:
    text = annotation.text.replace("{subject}", subject.name)
    size = float(annotation.font_size)
    width, height = output.width, output.height
    x, y = annotation.anchor
    if not text.strip() or not math.isfinite(size) or size <= 0 or not _contains(reserve, x, y):
        raise SubjectShotInfeasible(f"{label_id} annotation needs text, positive size and a baseline in reserved text space")
    record = {
        # This is one persistent editorial slot. Copy can change on the
        # reveal/hold cut without triggering a geographic label fade.
        "id": "story-caption", "source_id": subject.feature_id, "text": text,
        "geometry": {"type": "Point", "coordinates": (x * width, y * height)},
        "typography": {"font_size": size, "halo_width_px": 1.0},
    }
    anchor = _point_to_pixel(record["geometry"]["coordinates"], width, height)
    _, shaped = _shape_label_glyphs(text)
    metrics = load_atlas_metrics(default_latin_atlas_paths()[1])
    glyphs = metrics["glyphs_by_id"]
    font_hashes = tuple(shaped.get("font_sha256", ()))
    if font_hashes != tuple(metrics["font_sha256"]):
        raise SubjectShotInfeasible(f"{label_id} annotation requires glyphs outside the bundled Latin atlas")
    boxes = []
    for positioned in shaped.get("positioned_glyphs", ()):
        if not positioned.get("has_outline", True):
            continue
        glyph = glyphs.get(f"{positioned['font_index']}:{positioned['glyph_id']}")
        if glyph is None:
            raise SubjectShotInfeasible(f"{label_id} annotation has no glyph in the bundled Latin atlas")
        gx, gy, w, h = _native_glyph_box(anchor, positioned["origin"], glyph, size, metrics["font_size"])
        boxes.append((gx, gy, gx + w, gy + h))
    if not boxes or any(
        not (_contains(reserve, box[0] / width, box[1] / height)
             and _contains(reserve, box[2] / width, box[3] / height))
        for box in boxes
    ):
        raise SubjectShotInfeasible(f"{label_id} annotation glyphs do not fit the reserved text space at {width}×{height}")
    record["screen_bounds"] = (
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    )
    record["candidate_policy"] = {"offset_px": 0, "radial_count": 0}
    plan = LabelPlan.compile(labels=[record], camera=None, viewport=output)
    if not plan.accepted:
        raise SubjectShotInfeasible(f"{label_id} annotation rejected: {[item.reason for item in plan.rejected]}")
    return LabelLayer(layer_id="subject-story", labels=[record], plan=plan)


def _consumer_eye(params) -> np.ndarray:
    # The renderer's upload.rs evaluates the Z-up orbit in binary32.
    phi, theta = np.asarray((params.cam_phi_deg, params.cam_theta_deg), dtype=np.float32) * np.float32(math.pi / 180)
    radius = np.float32(params.cam_radius)
    sin_theta = np.float32(math.sin(theta))
    offset = np.array([
        radius * sin_theta * np.float32(math.cos(phi)),
        radius * sin_theta * np.float32(math.sin(phi)),
        radius * np.float32(math.cos(theta)),
    ], dtype=np.float32)
    return (np.asarray(params.cam_target, dtype=np.float32) + offset).astype(float)


def _consumer_params(recipe, heights):
    output = recipe.output
    params = _build_mapscene_terrain_params(recipe, heights, (output.width, output.height))
    if params is None or params.camera_mode != _MESH_CAMERA_MODE or params.height_curve_mode != "linear":
        raise RuntimeError("subject shot requires the native linear Z-up terrain mesh")
    return params


def _consumer_check(
    recipe, heights, grounds, extents, mesh, obstacles,
    minimum, framing, reserve, frame, previous,
) -> Mapping[str, Any]:
    params = _consumer_params(recipe, heights)
    eye = _consumer_eye(params)
    forward = np.asarray(params.cam_target) - eye
    forward /= np.linalg.norm(forward)
    up_axis = (0, 1, 0) if abs(math.sin(math.radians(params.cam_theta_deg))) < 1e-4 else (0, 0, 1)
    right = np.cross(forward, up_axis)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    width, height = params.size_px
    half_y = math.tan(math.radians(params.fov_y_deg) / 2)
    half_x = half_y * width / height
    near, far = params.clip
    obstacle_faces = [(name, box_triangles(lo, hi, eye, forward, near)) for name, lo, hi in obstacles] if extents else []

    def to_grid(point):
        return np.array([*mesh.grid(point[:2]), point[2]])

    def project(points):
        relative = points - eye
        depth = relative @ forward
        if not np.all((near < depth) & (depth < far)):
            raise SubjectShotInfeasible(f"frame {frame}: {feature_id} outside the consumer near/far planes")
        viewport = 0.5 + np.column_stack((relative @ right / (2 * half_x),
                                         relative @ up / (-2 * half_y))) / depth[:, None]
        return viewport, depth

    def check_projection(viewport):
        if not all(_contains(framing, *point) for point in viewport):
            raise SubjectShotInfeasible(f"frame {frame}: {feature_id} outside framing at {tuple(viewport[0])}")
        if intersects_rectangle(viewport, reserve):
            raise SubjectShotInfeasible(f"frame {frame}: {feature_id} overlaps reserved text space")

    eye_grid = to_grid(eye)
    if not np.all((0 <= eye_grid[:2]) & (eye_grid[:2] <= _GRID - 1)):
        raise SubjectShotInfeasible(f"frame {frame}: consumer camera eye left the terrain footprint")
    clearance = eye[2] - float(_mesh_height(mesh.heights, eye_grid[:2][None])[0])
    if clearance < minimum or clearance <= 0:
        raise SubjectShotInfeasible(f"frame {frame}: consumer terrain clearance {clearance:.3f} m below the required {minimum:g} m or intersects terrain")
    obstacle = obstacle_at_eye(eye, obstacles, minimum)
    if obstacle is not None:
        raise SubjectShotInfeasible(f"frame {frame}: consumer camera violates clearance of {obstacle}")

    linear_clearance = None
    if previous is not None:
        previous_grid = to_grid(np.asarray(previous["eye_xyz_m"]))
        margins = _terrain_segment_margins(mesh.heights, previous_grid, eye_grid)
        # Diagnostic only. A chord can cut across terrain the actual orbit clears.
        linear_clearance = min(clearance, previous["clearance_m"], float(margins.min(initial=np.inf)))

    projections = {}
    for feature_id, point in grounds.items():
        if feature_id in extents:
            lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
            for piece in extents[feature_id]:
                viewport, _ = project(piece)
                check_projection(viewport)
                blocker = whole_blocker(mesh, eye, piece, forward, near, obstacle_faces, obstacles)
                if blocker is not None:
                    raise SubjectShotInfeasible(f"frame {frame}: {blocker} occludes part of {feature_id} from the consumer camera")
                lo, hi = np.minimum(lo, viewport.min(axis=0)), np.maximum(hi, viewport.max(axis=0))
            projections[feature_id] = {"viewport_bounds": (*lo, *hi), "visible_pieces": len(extents[feature_id])}
            continue
        viewport, depth = project(point[None])
        check_projection(viewport)
        clipped_eye = eye + (point - eye) * (near / depth[0])
        clipped_grid = to_grid(clipped_eye)
        if (clipped_eye[2] <= _mesh_height(mesh.heights, clipped_grid[:2][None])[0]
                or np.any(_terrain_segment_margins(mesh.heights, clipped_grid, to_grid(point)) <= 0)):
            raise SubjectShotInfeasible(f"frame {frame}: terrain occludes {feature_id} from the consumer camera")
        blocker = point_blocker(eye, point, obstacles, near / depth[0])
        if blocker is not None:
            raise SubjectShotInfeasible(f"frame {frame}: {blocker} occludes {feature_id} from the consumer camera")
        projections[feature_id] = {"viewport_xy": tuple(viewport[0]), "mesh_xyz_m": tuple(point), "depth_m": float(depth[0])}
    return {
        "frame": frame, "eye_xyz_m": tuple(eye), "target_xyz_m": tuple(params.cam_target),
        "clip_m": (near, far), "fov_y_deg": params.fov_y_deg,
        "clearance_m": clearance, "linear_eye_clearance_from_previous_m": linear_clearance,
        "features": projections,
    }


def _registered_extent(feature, mesh, west, south, span):
    line = np.asarray(feature.line_xy, dtype=float)
    faces = np.asarray(feature.area_triangles_xy, dtype=float)
    has_line, has_faces = len(feature.line_xy) > 0, len(feature.area_triangles_xy) > 0
    if has_line and has_faces:
        raise ValueError(f"feature {feature.feature_id} must choose a line or an area")
    if has_line and (line.ndim != 2 or line.shape[1] != 2 or len(line) < 2
                      or np.any(np.all(np.diff(line, axis=0) == 0, axis=1))):
        raise ValueError(f"feature {feature.feature_id} requires a line with distinct consecutive XY vertices")
    if has_faces:
        if faces.ndim != 3 or faces.shape[1:] != (3, 2):
            raise ValueError(f"feature {feature.feature_id} requires area triangles with three XY vertices")
        ab, ac = faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]
        if np.any(ab[:, 0] * ac[:, 1] == ab[:, 1] * ac[:, 0]):
            raise ValueError(f"feature {feature.feature_id} has a degenerate area triangle")
    offset = np.array([west, south])
    for points in (line, faces):
        if points.size and not np.all((offset <= points) & (points <= offset + span)):
            raise SubjectShotInfeasible(f"feature {feature.feature_id} extent is outside the registered terrain footprint")
    if not (has_line or has_faces):
        return ()
    pieces = tuple(mesh.drape((line - offset) * (_GRID - 1) / span if has_line else (),
                              (faces - offset) * (_GRID - 1) / span if has_faces else ()))
    if not pieces:
        raise SubjectShotInfeasible(f"feature {feature.feature_id} has no extent at registered mesh precision")
    return pieces


def _registered_obstacles(obstacles, mesh, west, south, span, midpoint, scale):
    registered = []
    ids = set()
    for obstacle in obstacles:
        if not obstacle.obstacle_id or obstacle.obstacle_id in ids or obstacle.kind not in ("building", "vegetation"):
            raise ValueError("story obstacles require unique, nonempty IDs and a building or vegetation kind")
        ids.add(obstacle.obstacle_id)
        bounds = np.asarray(obstacle.bounds_xyz_m, dtype=float)
        if bounds.shape != (6,) or not np.isfinite(bounds).all() or not np.all(bounds[:3] < bounds[3:]):
            raise ValueError(f"obstacle {obstacle.obstacle_id} requires finite ordered XYZ bounds")
        xy = bounds.reshape(2, 3)[:, :2] - (west, south)
        if not np.all((0 <= xy) & (xy <= span)):
            raise ValueError(f"obstacle {obstacle.obstacle_id} lies outside the registered terrain footprint")
        z = (bounds[2::3] - midpoint) * scale
        if not np.isfinite(z).all():
            raise ValueError(f"obstacle {obstacle.obstacle_id} heights overflow the registered scene")
        lo, hi = np.column_stack(((xy / span - 0.5) * mesh.span, z))
        registered.append((f"{obstacle.kind} {obstacle.obstacle_id}", lo, hi))
    return registered


def _duration_frames(seconds: float, fps: int, name: str) -> int:
    count = float(seconds) * fps
    if not math.isfinite(count) or count <= 0 or not count.is_integer():
        raise ValueError(f"{name} must be a positive whole number of frames at {fps} fps")
    return int(count)


def _validate_mesh_geotransform(
    metadata: Mapping[str, Any], bounds: tuple[float, float, float, float],
) -> None:
    transform = metadata.get("geotransform")
    if not isinstance(transform, Sequence) or isinstance(transform, (str, bytes)) or len(transform) != 6:
        raise ValueError("subject terrain requires a six-value GDAL corner geotransform in metadata")
    x0, dx, x_skew, y0, y_skew, dy = map(float, transform)
    if not all(map(math.isfinite, (x0, dx, x_skew, y0, y_skew, dy))):
        raise ValueError("subject terrain geotransform must be finite")
    if dx <= 0 or dy >= 0 or x_skew != 0 or y_skew != 0:
        raise ValueError("subject terrain requires a north-up geotransform with eastward columns and southward rows")
    first, last = 0.5, _GRID - 0.5
    centres = (x0 + first * dx, y0 + last * dy, x0 + last * dx, y0 + first * dy)
    # Allow one representable coordinate step for corner-to-centre rounding.
    if dx != -dy or any(
        not math.isclose(a, b, rel_tol=0, abs_tol=math.ulp(b)) for a, b in zip(centres, bounds)
    ):
        raise ValueError("subject terrain geotransform must have square pixels and match bounds_wsen at the outer pixel centres")


def compile_subject_shot(
    terrain: TerrainSource, features: Sequence[SubjectFeature], *, subject_id: str,
    bounds_wsen: Sequence[float], feature_crs: str,
    orbit: Callable[[tuple[float, float]], TerrainOrbitRig],
    fps: int, hold_seconds: float, output: OutputSpec, lighting: LightingPreset,
    framing: Sequence[float], reserved_text: Sequence[float],
    reveal_annotation: StoryAnnotation, hold_annotation: StoryAnnotation | None = None,
    must_see: Sequence[str] = (), establish_must_see: Sequence[str] = (),
    obstacles: Sequence[StoryObstacle] = (),
) -> CompiledSubjectShot:
    """Bind a terrain orbit to a stable ID and verify each output camera.

    Bounds are the outer vertex coordinates (centres of the outer DEM samples).
    Texture columns increase east and rows increase south, as declared by
    ``TerrainSource.metadata["geotransform"]`` in GDAL corner order.
    Framing and reserve are normalized viewport rectangles in left/top order.
    Required points, lines and triangulated areas are draped at unchanged XY.
    Obstacles are solid building or vegetation envelopes in that CRS and the
    processed height field's metre datum. Clearance follows the baked orbit
    between output frames.
    """
    fps = index(fps)
    if fps <= 0:
        raise ValueError("fps must be a positive integer")
    hold_frames = _duration_frames(hold_seconds, fps, "hold duration")
    framing = _rectangle(framing, "framing")
    reserve = _rectangle(reserved_text, "reserved text space")
    width, height = index(output.width), index(output.height)
    if (min(width, height) < 64 or output.samples != 1 or output.format.lower() != "png"
            or output.bit_depth != 8 or output.hdr or output.aovs
            or str(output.denoiser).lower() not in {"", "none", "off"}):
        raise ValueError("subject shots require single-sample 8-bit PNG output, at least 64×64, without AOVs or denoising")
    try:
        from pyproj import CRS  # Existing optional raster extra.
    except ImportError as exc:
        raise RuntimeError("subject shots require pyproj from the Forge3D raster extra") from exc

    crs = CRS.from_user_input(feature_crs)
    if (not terrain.crs or CRS.from_user_input(terrain.crs) != crs or not crs.is_projected
            or not all(axis.unit_conversion_factor == 1 for axis in crs.axis_info)):
        raise ValueError("terrain and features must already share the same projected metre CRS")
    west, south, east, north = map(float, bounds_wsen)
    span = east - west
    if not all(map(math.isfinite, (west, south, east, north, span))) or span < 1 or north - south != span:
        raise ValueError("subject shot requires finite square terrain bounds at least 1 m wide (MapScene mesh span)")
    if terrain.data is not None or not terrain.path or Path(terrain.path).suffix.lower() != ".npy":
        raise ValueError("subject shot requires a replayable .npy terrain source")
    if terrain.style is not None and terrain.style.height_units != "m":
        raise ValueError("subject shot terrain style must produce metre heights (height_units='m')")
    metadata = dict(terrain.metadata or {})
    _validate_mesh_geotransform(metadata, (west, south, east, north))
    path = Path(terrain.path).resolve(strict=True)
    terrain_bytes = path.read_bytes()
    terrain_hash = sha256(terrain_bytes).hexdigest()
    # Validate the exact bytes whose hash is retained, using the same dtype
    # and style processing as MapScene's file-backed render path.
    samples = np.load(BytesIO(terrain_bytes), allow_pickle=False)
    if not isinstance(samples, np.ndarray) or samples.shape != (_GRID, _GRID):
        raise ValueError("subject shot requires a 512×512 terrain mesh")
    heightmap = _load_native_heightmap(replace(terrain, data=samples))
    if not np.isfinite(heightmap).all():
        raise ValueError("subject shot requires a complete finite 512×512 terrain mesh after styling")
    metadata.update(width=_GRID, height=_GRID, resolution=(span / _GRID,) * 2,
                    subject_mesh_bounds_wsen=(west, south, east, north),
                    subject_mesh_axes="east,north,up", subject_mesh_sha256=terrain_hash)
    terrain = replace(terrain, path=path, metadata=metadata)

    # MapScene reflects the north-first mesh in the native north-up view.
    # The rig and CPU geometry use geographic axes instead.
    heights = np.ascontiguousarray(heightmap[::-1])

    catalog = {feature.feature_id: feature for feature in features}
    if len(catalog) != len(features) or any(not key for key in catalog):
        raise ValueError("subject features require unique, nonempty IDs")
    grounds = {}
    for key in dict.fromkeys((subject_id, *must_see, *establish_must_see)):
        if key not in catalog:
            raise SubjectShotInfeasible(f"unknown feature ID {key!r}")
        feature = catalog[key]
        if not (west <= feature.easting <= east and south <= feature.northing <= north):
            raise SubjectShotInfeasible(f"feature {key} is outside the registered terrain footprint")
        grounds[key] = np.array([feature.easting - west - span / 2, feature.northing - south - span / 2, 0.0])

    lighting = deepcopy(lighting)
    settings = dict(lighting.settings or {})
    cli = dict(settings.get("cli_params") or {})
    camera_settings = dict(settings.get("camera") or {})
    requested_mode = cli.get("camera_mode") or camera_settings.get("camera_mode")
    if requested_mode not in (None, "mesh:zup", _MESH_CAMERA_MODE):
        raise ValueError("subject shots require the Z-up terrain mesh camera")
    cli["camera_mode"] = _MESH_CAMERA_MODE
    scale = float(np.float32(settings.get("exaggeration", 1.0)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("terrain exaggeration must be positive and finite")
    lighting = replace(lighting, settings={**settings, "exaggeration": scale, "cli_params": cli})
    low, high = _heightmap_domain(heights)
    midpoint = (low + high) / 2
    height_range = high - low
    mesh_heights = heights if height_range >= 1e-6 else low + (heights - low) * (height_range / 1e-6)
    mesh = StoryMesh((mesh_heights - midpoint) * scale, float(np.float32(span)))
    extents = {key: pieces for key in grounds
               if (pieces := _registered_extent(catalog[key], mesh, west, south, span))}
    registered = _registered_obstacles(obstacles, mesh, west, south, span, midpoint, scale)
    source = TerrainScatterSource(heights, terrain_width=span, z_scale=scale)
    target = tuple(float(v + span / 2) for v in grounds[subject_id][:2])
    rig = orbit(target)
    if not isinstance(rig, TerrainOrbitRig) or rig.target_xz != target:
        raise ValueError("orbit must return a TerrainOrbitRig using the bound subject target")
    reveal_frames = _duration_frames(rig.duration, fps, "reveal duration")
    try:
        animation = rig.bake(source, samples_per_second=fps)
    except ValueError as exc:
        raise SubjectShotInfeasible(f"terrain rig rejected the reveal: {exc}") from exc

    subject = catalog[subject_id]
    reveal_layer = _annotation_layer(reveal_annotation, subject, output, reserve, "reveal")
    hold_layer = (
        reveal_layer if hold_annotation is None or hold_annotation == reveal_annotation
        else _annotation_layer(hold_annotation, subject, output, reserve, "hold")
    )
    reserve_px = tuple(v * d for v, d in zip(reserve, (width, height, width, height)))
    furniture = MapFurnitureLayer(
        keepouts=[KeepoutRegion("story-text", "annotation", reserve_px)],
        metadata={"subject_id": subject_id, "required_feature_ids": list(dict.fromkeys((subject_id, *must_see))),
                  "establish_required_feature_ids": list(dict.fromkeys(establish_must_see)),
                  "fps": fps, "reveal_frames": reveal_frames, "hold_frames": hold_frames,
                  "framing": framing, "required_clearance_m": rig.clearance.minimum_height,
                  "feature_xy": {key: (catalog[key].easting, catalog[key].northing) for key in grounds}},
    )
    if extents:
        furniture.metadata["feature_extents_xy"] = {
            key: {"line": np.asarray(catalog[key].line_xy).tolist(),
                  "area_triangles": np.asarray(catalog[key].area_triangles_xy).tolist()}
            for key in extents
        }
    if obstacles:
        furniture.metadata["story_obstacles"] = [
            {"id": item.obstacle_id, "kind": item.kind, "bounds_xyz_m": list(map(float, item.bounds_xyz_m))}
            for item in obstacles
        ]
    vertical_shift = (source.min_height - midpoint) * scale
    try:
        continuous = BakedClearance(animation, mesh, registered, rig.clearance.minimum_height,
                                    (-span / 2, -span / 2, vertical_shift))
    except ValueError as exc:
        raise SubjectShotInfeasible(str(exc)) from exc
    grounds = {key: mesh.lift(mesh.grid(point[:2] * mesh.span / span)[None])[0]
               for key, point in grounds.items()}

    def camera_at(t):
        state = animation.evaluate(t)
        if state is None or state.target is None:
            raise SubjectShotInfeasible(f"terrain rig has no evaluated camera target at {t:g} s")
        x, up, northward = state.target
        return OrbitCamera(
            target=(x - span / 2, northward - span / 2, up + vertical_shift),
            distance=state.radius, azimuth_deg=state.phi_deg,
            elevation_deg=state.theta_deg, fov_deg=state.fov_deg,
        )
    recipes = []
    checks = []
    previous_time = 0.
    for frame in range(reveal_frames + hold_frames):
        # Monotone easing changes when the rig is sampled, not its swept path.
        u = min(frame, reveal_frames) / reveal_frames
        t = rig.duration * (u * u * (3 - 2 * u))
        recipe = SceneRecipe(
            terrain=terrain, camera=camera_at(t), lighting=lighting, output=deepcopy(output),
            layers=[reveal_layer if frame < reveal_frames else hold_layer], map_furniture=furniture,
        )
        required = grounds if frame == 0 else {key: grounds[key] for key in (subject_id, *must_see)}
        check = _consumer_check(
            recipe, heightmap, required, extents, mesh, registered,
            rig.clearance.minimum_height, framing, reserve, frame, checks[-1] if checks else None,
        )

        def eye_at(rig_time):
            return _consumer_eye(_consumer_params(replace(recipe, camera=camera_at(rig_time)), heightmap))

        def fail(message):
            raise SubjectShotInfeasible(f"frames {frame - 1}–{frame}: nonlinear path {message}")

        lower_bound = None
        if frame:
            lower_bound = continuous.check(previous_time, t, eye_at, fail)
            if lower_bound is None:
                lower_bound = check["clearance_m"]  # Stationary hold or equal native times.
        check["nonlinear_terrain_clearance_lower_bound_from_previous_m"] = lower_bound
        checks.append(check)
        recipes.append(recipe)
        previous_time = t
    return CompiledSubjectShot(fps, reveal_frames, tuple(recipes), tuple(checks), terrain_hash)


__all__ = ["CompiledSubjectShot", "StoryAnnotation", "StoryObstacle", "SubjectFeature",
           "SubjectShotInfeasible", "compile_subject_shot"]
