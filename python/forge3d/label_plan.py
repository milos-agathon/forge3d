"""Deterministic label-plan compiler contract for offline map rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .diagnostics import (
    Diagnostic,
    label_rejection_summary_diagnostic,
    missing_glyphs_diagnostic,
    placeholder_fallback_diagnostic,
)


PAYLOAD_VERSION = 1
REJECTION_REASONS = (
    "collision",
    "outside_view",
    "missing_glyph",
    "priority_lost",
    "keepout_region",
    "terrain_occluded",
    "invalid_geometry",
    "unsupported_geometry_type",
    "empty_text",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"LabelPlan payload value is not JSON-serializable: {type(value).__name__}")


def _stable_json(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_unit_interval(*parts: Any) -> float:
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _number(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coordinates(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    coords = [_number(item, default=float("nan")) for item in value]
    if len(coords) < 2 or any(coord != coord for coord in coords):
        return None
    while len(coords) < 3:
        coords.append(0.0)
    return coords[:3]


def _viewport_size(viewport: Any) -> tuple[float, float] | None:
    if isinstance(viewport, Mapping):
        if "width" in viewport and "height" in viewport:
            return (_number(viewport["width"]), _number(viewport["height"]))
    if isinstance(viewport, Sequence) and not isinstance(viewport, (str, bytes)) and len(viewport) >= 2:
        return (_number(viewport[0]), _number(viewport[1]))
    width = getattr(viewport, "width", None)
    height = getattr(viewport, "height", None)
    if width is not None and height is not None:
        return (_number(width), _number(height))
    return None


def _iter_label_records(labels: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(labels, Mapping):
        for key in sorted(labels.keys(), key=str):
            value = labels[key]
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("id", str(key))
                yield str(key), record
        return
    for index, value in enumerate(labels or ()):
        if isinstance(value, Mapping):
            yield str(index), dict(value)


def _glyph_set(glyph_atlas: Any) -> set[str] | None:
    if glyph_atlas is None:
        return None
    if isinstance(glyph_atlas, Mapping):
        glyphs = glyph_atlas.get("glyphs")
        if glyphs is not None:
            return {str(glyph) for glyph in glyphs}
    if isinstance(glyph_atlas, (set, frozenset, list, tuple)):
        return {str(glyph) for glyph in glyph_atlas}
    return None


def _rect_bounds(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    bounds = [_number(item, default=float("nan")) for item in value[:4]]
    if any(coord != coord for coord in bounds):
        return None
    x0, y0, x1, y1 = bounds
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _rects_intersect(left: Sequence[float] | None, right: Sequence[float] | None) -> bool:
    left_bounds = _rect_bounds(left)
    right_bounds = _rect_bounds(right)
    if left_bounds is None or right_bounds is None:
        return False
    return (
        left_bounds[0] <= right_bounds[2]
        and left_bounds[2] >= right_bounds[0]
        and left_bounds[1] <= right_bounds[3]
        and left_bounds[3] >= right_bounds[1]
    )


def _requires_terrain(record: Mapping[str, Any]) -> bool:
    mode = str(record.get("terrain_mode", "")).lower()
    return bool(record.get("requires_terrain")) or mode in {"required", "sample", "terrain"}


def _call_terrain_sampler(terrain: Any, coords: Sequence[float]) -> Mapping[str, Any]:
    sampler = getattr(terrain, "sample", None) or (terrain if callable(terrain) else None)
    if sampler is None:
        return {}
    x, y, z = coords
    for args in ((x, y, z), (x, y), (coords,)):
        try:
            result = sampler(*args)
        except TypeError:
            continue
        if isinstance(result, Mapping):
            return dict(result)
        if result is not None:
            return {"elevation": _number(result), "source": type(terrain).__name__, "visible": True}
    return {"source": type(terrain).__name__, "unavailable": True, "visible": False}


def _terrain_sample(
    record: Mapping[str, Any],
    terrain: Any,
    label_id: str,
    coords: Sequence[float] | None = None,
) -> Mapping[str, Any]:
    sample = record.get("terrain_sample")
    if isinstance(sample, Mapping):
        return dict(sample)
    terrain_record = record.get("terrain")
    if isinstance(terrain_record, Mapping):
        return dict(terrain_record)
    if isinstance(terrain, Mapping):
        samples = terrain.get("samples")
        if isinstance(samples, Mapping) and isinstance(samples.get(label_id), Mapping):
            return dict(samples[label_id])
        if isinstance(terrain.get(label_id), Mapping):
            return dict(terrain[label_id])
    if coords is not None and _requires_terrain(record):
        if terrain is None:
            return {"source": "terrain_sampler", "unavailable": True, "visible": False}
        return _call_terrain_sampler(terrain, coords)
    return {}


def _candidate_policy(record: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = record.get("candidate_policy")
    return dict(policy) if isinstance(policy, Mapping) else {}


def _priority_score(record: Mapping[str, Any], priority_ranks: Mapping[str, int]) -> float:
    priority_class = str(record.get("priority_class", "default"))
    rank = int(priority_ranks.get(priority_class, 0))
    local_priority = _number(record.get("priority", 0))
    return (rank * 1_000_000.0) + local_priority


def _point_label_candidates(
    *,
    label_id: str,
    coords: Sequence[float],
    score: float,
    ordering_key: str,
    record: Mapping[str, Any],
    seed: int,
    terrain_sample: Mapping[str, Any],
) -> list[LabelCandidate]:
    x, y, z = coords
    policy = _candidate_policy(record)
    offset = _number(policy.get("offset_px", record.get("candidate_offset_px", 12.0)), default=12.0)
    radial_radius = _number(
        policy.get("radial_radius_px", record.get("radial_radius_px", offset * 1.5)),
        default=offset * 1.5,
    )
    radial_count = max(0, int(_number(policy.get("radial_count", record.get("radial_count", 4)), default=4.0)))
    radial_jitter_deg = max(
        0.0,
        _number(policy.get("radial_jitter_deg", record.get("radial_jitter_deg", 0.0)), default=0.0),
    )

    candidates: list[LabelCandidate] = []

    def add_candidate(
        suffix: str,
        candidate_type: str,
        anchor: Sequence[float],
        order: int,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        candidates.append(
            LabelCandidate(
                candidate_id=f"{label_id}:{suffix}",
                candidate_type=candidate_type,
                anchor=anchor,
                score=score - (order * 0.001),
                bounds=[anchor[0], anchor[1], anchor[0], anchor[1]],
                terrain_sample=terrain_sample,
                details=details or {},
                ordering_key=f"{ordering_key}:{order:02d}:{suffix}",
            )
        )

    add_candidate("center", "center", [x, y, z], 0)
    add_candidate("above", "above", [x, y - offset, z], 1, details={"offset_px": offset})
    add_candidate("below", "below", [x, y + offset, z], 2, details={"offset_px": offset})
    add_candidate("left", "left", [x - offset, y, z], 3, details={"offset_px": offset})
    add_candidate("right", "right", [x + offset, y, z], 4, details={"offset_px": offset})

    for index in range(radial_count):
        base_angle = (360.0 / radial_count) * index if radial_count else 0.0
        jitter = (_stable_unit_interval(seed, label_id, index, "radial") - 0.5) * 2.0 * radial_jitter_deg
        angle = (base_angle + jitter) % 360.0
        radians = math.radians(angle)
        anchor = [
            x + math.cos(radians) * radial_radius,
            y + math.sin(radians) * radial_radius,
            z,
        ]
        add_candidate(
            f"radial-{index}",
            "radial",
            anchor,
            5 + index,
            details={
                "angle_deg": round(angle, 6),
                "jitter_deg": round(jitter, 6),
                "radial_index": index,
                "radius_px": radial_radius,
            },
        )

    return candidates


def _polygon_ring(geometry: Mapping[str, Any]) -> list[list[float]] | None:
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes)) or not coordinates:
        return None
    ring_data = coordinates[0]
    if not isinstance(ring_data, Sequence) or isinstance(ring_data, (str, bytes)):
        return None
    ring: list[list[float]] = []
    for point in ring_data:
        coords = _coordinates(point)
        if coords is None:
            return None
        ring.append(coords)
    if len(ring) < 4:
        return None
    if ring[0][:2] != ring[-1][:2]:
        ring.append(list(ring[0]))
    unique_xy = {(point[0], point[1]) for point in ring[:-1]}
    if len(unique_xy) < 3:
        return None
    return ring


def _ring_area(ring: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for left, right in zip(ring, ring[1:]):
        total += (left[0] * right[1]) - (right[0] * left[1])
    return total * 0.5


def _polygon_centroid(ring: Sequence[Sequence[float]], area: float) -> list[float]:
    cx = 0.0
    cy = 0.0
    for left, right in zip(ring, ring[1:]):
        cross = (left[0] * right[1]) - (right[0] * left[1])
        cx += (left[0] + right[0]) * cross
        cy += (left[1] + right[1]) * cross
    factor = 1.0 / (6.0 * area)
    return [cx * factor, cy * factor, 0.0]


def _point_in_polygon(x: float, y: float, ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    for left, right in zip(ring, ring[1:]):
        x0, y0 = left[0], left[1]
        x1, y1 = right[0], right[1]
        intersects = (y0 > y) != (y1 > y)
        if intersects:
            x_at_y = ((x1 - x0) * (y - y0) / (y1 - y0)) + x0
            if x < x_at_y:
                inside = not inside
    return inside


def _polygon_visual_center(ring: Sequence[Sequence[float]], fallback: Sequence[float]) -> list[float]:
    xs = [point[0] for point in ring[:-1]]
    ys = [point[1] for point in ring[:-1]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x or min_y == max_y:
        return [fallback[0], fallback[1], 0.0]

    best: tuple[float, float, float] | None = None
    steps = 12
    for ix in range(steps):
        x = min_x + ((ix + 0.5) * (max_x - min_x) / steps)
        for iy in range(steps):
            y = min_y + ((iy + 0.5) * (max_y - min_y) / steps)
            if not _point_in_polygon(x, y, ring):
                continue
            distance = min((x - point[0]) ** 2 + (y - point[1]) ** 2 for point in ring[:-1])
            candidate = (distance, x, y)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return [fallback[0], fallback[1], 0.0]
    return [best[1], best[2], 0.0]


def _polygon_label_candidates(
    *,
    label_id: str,
    geometry: Mapping[str, Any],
    score: float,
    ordering_key: str,
    terrain_sample: Mapping[str, Any],
) -> tuple[LabelCandidate, list[LabelCandidate]] | None:
    ring = _polygon_ring(geometry)
    if ring is None:
        return None
    area = _ring_area(ring)
    if abs(area) < 1.0e-9:
        return None

    centroid = _polygon_centroid(ring, area)
    centroid_inside = _point_in_polygon(centroid[0], centroid[1], ring)
    visual_center = _polygon_visual_center(ring, centroid)

    centroid_candidate = LabelCandidate(
        candidate_id=f"{label_id}:centroid",
        candidate_type="centroid",
        anchor=centroid,
        score=score,
        bounds=[centroid[0], centroid[1], centroid[0], centroid[1]],
        terrain_sample=terrain_sample,
        details={"area": abs(area), "inside_polygon": centroid_inside},
        ordering_key=f"{ordering_key}:00:centroid",
    )
    visual_candidate = LabelCandidate(
        candidate_id=f"{label_id}:visual-center",
        candidate_type="visual_center",
        anchor=visual_center,
        score=score - 0.001,
        bounds=[visual_center[0], visual_center[1], visual_center[0], visual_center[1]],
        terrain_sample=terrain_sample,
        details={"area": abs(area), "fallback_for": "centroid"},
        ordering_key=f"{ordering_key}:01:visual-center",
    )
    selected = centroid_candidate if centroid_inside else visual_candidate
    return selected, [centroid_candidate, visual_candidate]


def _label_sort_key(record: Mapping[str, Any], fallback_key: str) -> tuple[str, str, str]:
    label_id = str(record.get("id", fallback_key))
    text = str(record.get("text", ""))
    geometry = record.get("geometry") if isinstance(record.get("geometry"), Mapping) else {}
    geometry_type = str(geometry.get("type", record.get("geometry_type", "")))
    return (label_id, geometry_type, text)


@dataclass
class LabelCandidate:
    candidate_id: str
    candidate_type: str
    anchor: Sequence[float]
    score: float = 0.0
    bounds: Sequence[float] | None = None
    terrain_sample: Mapping[str, Any] | None = None
    details: Mapping[str, Any] | None = None
    ordering_key: str | None = None

    def __post_init__(self) -> None:
        self.anchor = tuple(float(value) for value in self.anchor)
        self.bounds = tuple(float(value) for value in self.bounds) if self.bounds is not None else None
        self.terrain_sample = _json_safe(dict(self.terrain_sample or {}))
        self.details = _json_safe(dict(self.details or {}))
        self.ordering_key = self.ordering_key or self.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "anchor": list(self.anchor),
            "score": float(self.score),
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "terrain_sample": dict(self.terrain_sample or {}),
            "details": dict(self.details or {}),
            "ordering_key": self.ordering_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelCandidate":
        return cls(
            candidate_id=str(data["candidate_id"]),
            candidate_type=str(data["candidate_type"]),
            anchor=data["anchor"],
            score=float(data.get("score", 0.0)),
            bounds=data.get("bounds"),
            terrain_sample=data.get("terrain_sample") or {},
            details=data.get("details") or {},
            ordering_key=data.get("ordering_key"),
        )


@dataclass
class AcceptedLabel:
    label_id: str
    source_id: str
    text: str
    geometry_type: str
    candidate: LabelCandidate | Mapping[str, Any]
    candidates: Sequence[LabelCandidate | Mapping[str, Any]] = field(default_factory=tuple)
    priority_class: str = "default"
    screen_bounds: Sequence[float] | None = None
    world_bounds: Sequence[float] | None = None
    typography: Mapping[str, Any] | None = None
    glyphs: Sequence[str] = field(default_factory=tuple)
    ordering_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, LabelCandidate):
            self.candidate = LabelCandidate.from_dict(self.candidate)
        candidate_items = self.candidates or (self.candidate,)
        self.candidates = sorted(
            (
                candidate if isinstance(candidate, LabelCandidate) else LabelCandidate.from_dict(candidate)
                for candidate in candidate_items
            ),
            key=lambda candidate: candidate.ordering_key or candidate.candidate_id,
        )
        if not any(candidate.candidate_id == self.candidate.candidate_id for candidate in self.candidates):
            self.candidates.insert(0, self.candidate)
        self.screen_bounds = (
            tuple(float(value) for value in self.screen_bounds) if self.screen_bounds is not None else None
        )
        self.world_bounds = (
            tuple(float(value) for value in self.world_bounds) if self.world_bounds is not None else None
        )
        self.typography = _json_safe(dict(self.typography or {}))
        self.glyphs = tuple(str(glyph) for glyph in self.glyphs)
        self.ordering_key = self.ordering_key or self.label_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "source_id": self.source_id,
            "text": self.text,
            "geometry_type": self.geometry_type,
            "candidate": self.candidate.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "priority_class": self.priority_class,
            "screen_bounds": list(self.screen_bounds) if self.screen_bounds is not None else None,
            "world_bounds": list(self.world_bounds) if self.world_bounds is not None else None,
            "typography": dict(self.typography or {}),
            "glyphs": list(self.glyphs),
            "ordering_key": self.ordering_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AcceptedLabel":
        return cls(
            label_id=str(data["label_id"]),
            source_id=str(data["source_id"]),
            text=str(data["text"]),
            geometry_type=str(data["geometry_type"]),
            candidate=data["candidate"],
            candidates=data.get("candidates") or (data["candidate"],),
            priority_class=str(data.get("priority_class", "default")),
            screen_bounds=data.get("screen_bounds"),
            world_bounds=data.get("world_bounds"),
            typography=data.get("typography") or {},
            glyphs=data.get("glyphs") or (),
            ordering_key=data.get("ordering_key"),
        )


@dataclass
class RejectedLabel:
    label_id: str
    source_id: str
    reason: str
    candidate_id: str | None = None
    diagnostic_refs: Sequence[str] = field(default_factory=tuple)
    ordering_key: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.reason not in REJECTION_REASONS:
            raise ValueError(f"Unknown label rejection reason: {self.reason!r}")
        self.diagnostic_refs = tuple(str(ref) for ref in self.diagnostic_refs)
        self.details = _json_safe(dict(self.details or {}))
        self.ordering_key = self.ordering_key or f"{self.label_id}:{self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "diagnostic_refs": list(self.diagnostic_refs),
            "ordering_key": self.ordering_key,
            "details": dict(self.details or {}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RejectedLabel":
        return cls(
            label_id=str(data["label_id"]),
            source_id=str(data["source_id"]),
            reason=str(data["reason"]),
            candidate_id=data.get("candidate_id"),
            diagnostic_refs=data.get("diagnostic_refs") or (),
            ordering_key=data.get("ordering_key"),
            details=data.get("details") or {},
        )


@dataclass
class KeepoutRegion:
    region_id: str
    kind: str
    bounds: Sequence[float]
    priority: int = 0

    def __post_init__(self) -> None:
        self.bounds = tuple(float(value) for value in self.bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "bounds": list(self.bounds),
            "priority": int(self.priority),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KeepoutRegion":
        return cls(
            region_id=str(data["region_id"]),
            kind=str(data["kind"]),
            bounds=data["bounds"],
            priority=int(data.get("priority", 0)),
        )


@dataclass
class PriorityClass:
    name: str
    rank: int = 0
    tie_break_policy: str = "stable_ordering_key"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rank": int(self.rank),
            "tie_break_policy": self.tie_break_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorityClass":
        return cls(
            name=str(data["name"]),
            rank=int(data.get("rank", 0)),
            tie_break_policy=str(data.get("tie_break_policy", "stable_ordering_key")),
        )


@dataclass
class LabelPlan:
    accepted: Sequence[AcceptedLabel | Mapping[str, Any]]
    rejected: Sequence[RejectedLabel | Mapping[str, Any]]
    diagnostics: Sequence[Diagnostic | Mapping[str, Any]] = field(default_factory=tuple)
    bounds: Mapping[str, Any] | None = None
    seed: int = 0
    payload_version: int = PAYLOAD_VERSION

    def __post_init__(self) -> None:
        self.accepted = sorted(
            (
                label if isinstance(label, AcceptedLabel) else AcceptedLabel.from_dict(label)
                for label in self.accepted
            ),
            key=lambda label: label.ordering_key or label.label_id,
        )
        self.rejected = sorted(
            (
                label if isinstance(label, RejectedLabel) else RejectedLabel.from_dict(label)
                for label in self.rejected
            ),
            key=lambda label: label.ordering_key or label.label_id,
        )
        self.diagnostics = sorted(
            (
                diagnostic
                if isinstance(diagnostic, Diagnostic)
                else Diagnostic.from_dict(diagnostic)
                for diagnostic in self.diagnostics
            ),
            key=lambda diagnostic: diagnostic.sort_key(),
        )
        self.bounds = _json_safe(dict(self.bounds or {"screen": None, "world": None}))
        self.seed = int(self.seed)
        self.payload_version = int(self.payload_version)

    @classmethod
    def compile(
        cls,
        *,
        labels: Any,
        camera: Any,
        viewport: Any,
        terrain: Any | None = None,
        keepouts: Sequence[KeepoutRegion | Mapping[str, Any]] = (),
        priority_rules: Sequence[PriorityClass | Mapping[str, Any]] | None = None,
        typography: Mapping[str, Any] | None = None,
        glyph_atlas: Any | None = None,
        seed: int = 0,
    ) -> "LabelPlan":
        del camera
        viewport_size = _viewport_size(viewport)
        keepout_payload = [
            region.to_dict() if isinstance(region, KeepoutRegion) else KeepoutRegion.from_dict(region).to_dict()
            for region in keepouts
        ]
        priority_payload = [
            item.to_dict() if isinstance(item, PriorityClass) else PriorityClass.from_dict(item).to_dict()
            for item in (priority_rules or ())
        ]
        priority_ranks = {str(item["name"]): int(item["rank"]) for item in priority_payload}
        atlas_glyphs = _glyph_set(glyph_atlas)
        accepted: list[AcceptedLabel] = []
        rejected: list[RejectedLabel] = []
        diagnostics: list[Diagnostic] = []
        missing_by_label: dict[str, list[str]] = {}

        records = sorted(_iter_label_records(labels), key=lambda item: _label_sort_key(item[1], item[0]))
        for fallback_key, record in records:
            label_id = str(record.get("id", fallback_key))
            source_id = str(record.get("source_id", label_id))
            text = str(record.get("text", ""))
            ordering_key = f"{label_id}:{source_id}:{_stable_json(record)}"

            if not text.strip():
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="empty_text",
                        ordering_key=ordering_key,
                    )
                )
                continue

            missing = sorted({char for char in text if atlas_glyphs is not None and char not in atlas_glyphs})
            if missing:
                missing_by_label[label_id] = missing
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="missing_glyph",
                        diagnostic_refs=["missing_glyphs"],
                        ordering_key=ordering_key,
                        details={"missing_glyphs": missing},
                    )
                )
                continue

            geometry = record.get("geometry") if isinstance(record.get("geometry"), Mapping) else {}
            geometry_type = str(geometry.get("type", record.get("geometry_type", "Point")))
            geometry_type_key = geometry_type.lower()
            terrain_sample = _terrain_sample(record, terrain, label_id)

            if geometry_type_key == "point":
                coords = _coordinates(geometry.get("coordinates", record.get("position", record.get("world_pos"))))
                if coords is None:
                    rejected.append(
                        RejectedLabel(
                            label_id=label_id,
                            source_id=source_id,
                            reason="invalid_geometry",
                            ordering_key=ordering_key,
                        )
                    )
                    continue

                terrain_sample = _terrain_sample(record, terrain, label_id, coords)
                if terrain_sample.get("visible") is not False and "elevation" in terrain_sample:
                    coords = [coords[0], coords[1], _number(terrain_sample["elevation"], default=coords[2])]

                x, y, z = coords
                screen_bounds = [x, y, x, y]
                world_bounds = [x, y, z, x, y, z]
                candidates = _point_label_candidates(
                    label_id=label_id,
                    coords=coords,
                    score=_priority_score(record, priority_ranks),
                    ordering_key=ordering_key,
                    record=record,
                    seed=int(seed),
                    terrain_sample=terrain_sample,
                )
                candidate = candidates[0]
            elif geometry_type_key == "polygon":
                polygon_candidates = _polygon_label_candidates(
                    label_id=label_id,
                    geometry=geometry,
                    score=_priority_score(record, priority_ranks),
                    ordering_key=ordering_key,
                    terrain_sample=terrain_sample,
                )
                if polygon_candidates is None:
                    rejected.append(
                        RejectedLabel(
                            label_id=label_id,
                            source_id=source_id,
                            reason="invalid_geometry",
                            ordering_key=ordering_key,
                        )
                    )
                    continue
                candidate, candidates = polygon_candidates
                x, y, z = candidate.anchor
                screen_bounds = list(candidate.bounds or [x, y, x, y])
                world_bounds = [x, y, z, x, y, z]
            else:
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="unsupported_geometry_type",
                        ordering_key=ordering_key,
                        details={"geometry_type": geometry_type},
                    )
                )
                continue

            if viewport_size is not None:
                width, height = viewport_size
                if x < 0.0 or y < 0.0 or x > width or y > height:
                    rejected.append(
                        RejectedLabel(
                            label_id=label_id,
                            source_id=source_id,
                            reason="outside_view",
                            ordering_key=ordering_key,
                            details={"viewport": [width, height]},
                        )
                    )
                    continue

            if terrain_sample.get("visible") is False:
                diagnostic_refs = ["label_rejection_summary"]
                if terrain_sample.get("unavailable") is True:
                    diagnostics.append(
                        placeholder_fallback_diagnostic(
                            "terrain_sampler",
                            layer_id="labels",
                            object_id=label_id,
                        )
                    )
                    diagnostic_refs.append("placeholder_fallback")
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="terrain_occluded",
                        candidate_id=candidate.candidate_id,
                        diagnostic_refs=diagnostic_refs,
                        ordering_key=ordering_key,
                        details={"terrain_sample": terrain_sample},
                    )
                )
                continue

            keepout_hit = next(
                (
                    keepout
                    for keepout in keepout_payload
                    if _rects_intersect(screen_bounds, keepout.get("bounds"))
                ),
                None,
            )
            if keepout_hit is not None:
                rejected.append(
                    RejectedLabel(
                        label_id=label_id,
                        source_id=source_id,
                        reason="keepout_region",
                        candidate_id=candidate.candidate_id,
                        ordering_key=ordering_key,
                        details={
                            "keepout_bounds": keepout_hit["bounds"],
                            "keepout_kind": keepout_hit["kind"],
                            "keepout_region_id": keepout_hit["region_id"],
                        },
                    )
                )
                continue

            accepted.append(
                AcceptedLabel(
                    label_id=label_id,
                    source_id=source_id,
                    text=text,
                    geometry_type=geometry_type,
                    candidate=candidate,
                    candidates=candidates,
                    priority_class=str(record.get("priority_class", "default")),
                    screen_bounds=screen_bounds,
                    world_bounds=world_bounds,
                    typography=dict(typography or record.get("typography") or {}),
                    glyphs=list(text),
                    ordering_key=ordering_key,
                )
            )

        accepted, collision_rejections = _resolve_label_collisions(accepted)
        rejected.extend(collision_rejections)

        for label_id, glyphs in sorted(missing_by_label.items()):
            diagnostics.append(missing_glyphs_diagnostic(glyphs, layer_id="labels", object_id=label_id))

        if rejected:
            counts: dict[str, int] = {}
            for item in rejected:
                counts[item.reason] = counts.get(item.reason, 0) + 1
            diagnostics.append(label_rejection_summary_diagnostic(counts, layer_id="labels"))

        bounds = _plan_bounds(accepted)
        bounds["keepouts"] = sorted(keepout_payload, key=lambda item: (item["priority"], item["kind"], item["region_id"]))
        bounds["priority_rules"] = sorted(priority_payload, key=lambda item: (item["rank"], item["name"]))
        return cls(
            accepted=accepted,
            rejected=rejected,
            diagnostics=diagnostics,
            bounds=bounds,
            seed=seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_version": self.payload_version,
            "seed": self.seed,
            "accepted": [label.to_dict() for label in self.accepted],
            "rejected": [label.to_dict() for label in self.rejected],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "bounds": _json_safe(dict(self.bounds or {})),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelPlan":
        return cls(
            accepted=data.get("accepted") or (),
            rejected=data.get("rejected") or (),
            diagnostics=data.get("diagnostics") or (),
            bounds=data.get("bounds") or {},
            seed=int(data.get("seed", 0)),
            payload_version=int(data.get("payload_version", PAYLOAD_VERSION)),
        )

    def _payload_with_backend(
        self,
        *,
        kind: str,
        backend: str | None,
        supported_backends: set[str],
    ) -> dict[str, Any]:
        backend_name = backend or "label_plan"
        payload = self.to_dict()
        payload["kind"] = kind
        payload["backend"] = backend_name
        payload["supported"] = backend_name in supported_backends
        if not payload["supported"]:
            payload["diagnostics"] = [
                *payload["diagnostics"],
                placeholder_fallback_diagnostic(
                    f"{kind}:{backend_name}",
                    layer_id="labels",
                ).to_dict(),
            ]
        return payload

    def to_render_payload(self, *, backend: str | None = None) -> dict[str, Any]:
        return self._payload_with_backend(
            kind="label_plan_render_payload",
            backend=backend,
            supported_backends={"default", "label_plan", "software"},
        )

    def to_export_payload(self, *, backend: str | None = None) -> dict[str, Any]:
        return self._payload_with_backend(
            kind="label_plan_export_payload",
            backend=backend,
            supported_backends={"default", "json", "label_plan"},
        )


def _plan_bounds(accepted: Sequence[AcceptedLabel]) -> dict[str, Any]:
    if not accepted:
        return {"screen": None, "world": None}

    screen_values = [label.screen_bounds for label in accepted if label.screen_bounds is not None]
    world_values = [label.world_bounds for label in accepted if label.world_bounds is not None]
    screen = None
    world = None
    if screen_values:
        screen = [
            min(bounds[0] for bounds in screen_values),
            min(bounds[1] for bounds in screen_values),
            max(bounds[2] for bounds in screen_values),
            max(bounds[3] for bounds in screen_values),
        ]
    if world_values:
        world = [
            min(bounds[0] for bounds in world_values),
            min(bounds[1] for bounds in world_values),
            min(bounds[2] for bounds in world_values),
            max(bounds[3] for bounds in world_values),
            max(bounds[4] for bounds in world_values),
            max(bounds[5] for bounds in world_values),
        ]
    return {"screen": screen, "world": world}


def _resolve_label_collisions(
    accepted: Sequence[AcceptedLabel],
) -> tuple[list[AcceptedLabel], list[RejectedLabel]]:
    winners: list[AcceptedLabel] = []
    rejected: list[RejectedLabel] = []
    solve_order = sorted(
        accepted,
        key=lambda label: (
            -float(label.candidate.score),
            label.ordering_key or label.label_id,
            label.label_id,
        ),
    )

    for label in solve_order:
        winner = next(
            (
                accepted_label
                for accepted_label in winners
                if _rects_intersect(label.screen_bounds, accepted_label.screen_bounds)
            ),
            None,
        )
        if winner is None:
            winners.append(label)
            continue

        label_score = float(label.candidate.score)
        winner_score = float(winner.candidate.score)
        reason = "priority_lost" if label_score < winner_score else "collision"
        rejected.append(
            RejectedLabel(
                label_id=label.label_id,
                source_id=label.source_id,
                reason=reason,
                candidate_id=label.candidate.candidate_id,
                diagnostic_refs=["label_rejection_summary"],
                ordering_key=label.ordering_key,
                details={
                    "collides_with": winner.label_id,
                    "candidate_bounds": list(label.screen_bounds or ()),
                    "winner_bounds": list(winner.screen_bounds or ()),
                    "candidate_priority": label_score,
                    "candidate_priority_class": label.priority_class,
                    "winner_priority": winner_score,
                    "winner_priority_class": winner.priority_class,
                },
            )
        )

    return winners, rejected


__all__ = [
    "AcceptedLabel",
    "KeepoutRegion",
    "LabelCandidate",
    "LabelPlan",
    "PAYLOAD_VERSION",
    "PriorityClass",
    "REJECTION_REASONS",
    "RejectedLabel",
]
