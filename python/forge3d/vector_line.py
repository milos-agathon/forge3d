"""Bounded local line ribbons for the native viewer overlay renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .diagnostics import Diagnostic, unsupported_style_field_diagnostic
from .terrain_params import PrimitiveType, VectorOverlayConfig, VectorVertex
from .viewer_contract import WorldPosition, vector_overlay_vertices, world_position

__all__ = ["VectorLineLayer", "VectorLineStyleError"]


class VectorLineStyleError(ValueError):
    """Unsupported local line styling, with existing structured diagnostics."""

    def __init__(self, diagnostics: Sequence[Diagnostic]):
        self.diagnostics = tuple(diagnostics)
        super().__init__("; ".join(
            f"{d.code}: {d.details.get('fields', [])}" for d in diagnostics
        ))


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


@dataclass(frozen=True)
class VectorLineLayer:
    """An open polyline in absolute viewer world ``(X, display Y, Z)``.

    Width is the full horizontal ribbon width; halo is its border on each
    side, both in world units (metres for metric terrain). Caps are ``butt``
    or ``square`` (default ``butt``); joins are ``bevel`` or ``miter``
    (default ``miter``). Miter length is clamped to four times the respective
    stroke/halo half-width, following the existing renderer contract. Closed,
    reversing, or folded ribbons are rejected.

    Native drape bilinearly samples terrain at each generated vertex and adds
    ``(height - domain_min) * z_scale + z_offset`` to input display Y.
    Without drape, only z_offset is added to Y. Interior triangles interpolate
    these vertex heights; add path points for curved terrain while allowing
    room for the width and halo at corners. Dense points near tight turns can
    fold the offset edges and are rejected. The renderer clamps samples outside
    the terrain. This is not MapScene routing.
    """

    name: str
    points: Sequence[WorldPosition]
    width: float = 2.0
    color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    halo: float = 0.0
    halo_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    cap: Literal["butt", "square"] = "butt"
    join: Literal["bevel", "miter"] = "miter"
    drape: bool = True
    z_offset: float = 0.5
    opacity: float = 1.0
    feature_id: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty")
        points = tuple(tuple(world_position(p, name="line point")) for p in self.points)
        if len(points) < 2:
            raise ValueError("line requires at least two points")
        object.__setattr__(self, "points", points)
        for name in ("width", "halo", "z_offset", "opacity"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        # VectorOverlayConfig's existing lower bound for world-unit ribbons.
        if self.width < 0.1:
            raise ValueError("width must be >= 0.1 world units")
        if self.halo < 0.0:
            raise ValueError("halo must be >= 0")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in [0, 1]")
        if not isinstance(self.drape, bool):
            raise ValueError("drape must be boolean")
        for field, supported in (
            ("cap", ("butt", "square")), ("join", ("bevel", "miter"))
        ):
            if getattr(self, field) not in supported:
                raise VectorLineStyleError([unsupported_style_field_diagnostic(
                    self.name, [f"line-{field}={getattr(self, field)!r}"], section="layout"
                )])
        for name in ("color", "halo_color"):
            color = getattr(self, name)
            if len(color) != 4:
                raise ValueError(f"{name} must have four RGBA components")
            row = vector_overlay_vertices([[0, 0, 0, *color, self.feature_id]])[0]
            object.__setattr__(self, name, tuple(row[3:7]))
            object.__setattr__(self, "feature_id", row[7])

    @classmethod
    def from_style(
        cls, layer: Any, points: Sequence[WorldPosition], *,
        properties: Mapping[str, Any] | None = None, zoom: float = 10.0,
        **options: Any,
    ) -> VectorLineLayer:
        """Translate one visible local line layer, interpreting width in world units.

        Evaluates existing color/number expressions against properties and zoom.
        Dashes, round caps/joins and every unsupported paint/layout field raise
        VectorLineStyleError before geometry or IPC. Halo/drape/offset are helper
        options, not extensions to the serialized Mapbox style schema. Defaults
        are width 1 world unit, butt caps and miter joins clamped at ratio 4,
        following the renderer rather than full Mapbox style parity.
        """
        from .style import (
            StyleLayer, evaluate_color_expr, evaluate_number_expr, parse_style,
        )

        if not isinstance(layer, StyleLayer):
            layer = parse_style({"version": 8, "layers": [layer]}).layers[0]
        if layer.layer_type != "line":
            from .diagnostics import unsupported_style_layer_type_diagnostic
            raise VectorLineStyleError([
                unsupported_style_layer_type_diagnostic(layer.id, layer.layer_type)
            ])
        diagnostics = []
        paint_fields = set(layer.unsupported_paint_fields)
        if layer.paint.line_dasharray is not None:
            paint_fields.add("line-dasharray")
        # Properties recognized by the parser but belonging to other layer types.
        for name, value in vars(layer.paint).items():
            if value is not None and name not in (
                "line_color", "line_width", "line_opacity", "line_dasharray"
            ):
                paint_fields.add(name.replace("_", "-"))
        layout_fields = set(layer.unsupported_layout_fields)
        for name, value in vars(layer.layout).items():
            if value is not None and name not in ("visibility", "line_cap", "line_join"):
                layout_fields.add(name.replace("_", "-"))
        for name, allowed in (
            ("line_cap", ("butt", "square")), ("line_join", ("miter", "bevel"))
        ):
            value = getattr(layer.layout, name)
            if value is not None and value not in allowed:
                layout_fields.add(f"{name.replace('_', '-')}={value!r}")
        for section, fields in (("paint", paint_fields), ("layout", layout_fields)):
            if fields:
                diagnostics.append(unsupported_style_field_diagnostic(
                    layer.id, sorted(fields), section=section
                ))
        if diagnostics:
            raise VectorLineStyleError(diagnostics)
        props = dict(properties or {})
        zoom = _finite(zoom, "zoom")
        if not layer.is_visible() or not layer.in_zoom_range(zoom) or not layer.matches_filter(props):
            raise ValueError("line style is not visible or does not match this feature/zoom")

        def number(value: Any, default: float, name: str) -> float:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite numeric data")
            result = default if value is None else evaluate_number_expr(value, props, zoom)
            if result is None:
                raise ValueError(f"cannot evaluate {name}")
            return _finite(result, name)

        color = (
            (0.0, 0.0, 0.0, 1.0) if layer.paint.line_color is None
            else evaluate_color_expr(layer.paint.line_color, props, zoom)
        )
        if color is None:
            raise ValueError("cannot evaluate line-color")
        translated = dict(
            name=layer.id, points=points,
            width=number(layer.paint.line_width, 1.0, "line-width"), color=color,
            opacity=number(layer.paint.line_opacity, 1.0, "line-opacity"),
            cap=layer.layout.line_cap or "butt", join=layer.layout.line_join or "miter",
        )
        overlap = options.keys() & translated.keys()
        if overlap:
            raise ValueError(f"style values cannot be overridden: {sorted(overlap)}")
        return cls(**translated, **options)

    def to_overlay_config(self) -> VectorOverlayConfig:
        """Build a triangle ribbon and disjoint halo border; no caller mesh work."""
        directions = []
        for a, b in zip(self.points, self.points[1:]):
            dx, dz = b[0] - a[0], b[2] - a[2]
            length = math.hypot(dx, dz)
            if not math.isfinite(length) or length == 0.0:
                raise ValueError("line segments must have finite nonzero horizontal length")
            directions.append((dx / length, dz / length))
        if self.points[0][::2] == self.points[-1][::2]:
            raise ValueError("closed lines are unsupported")

        def stations(
            radius: float, end_border: float,
        ) -> list[tuple[WorldPosition, WorldPosition]]:
            result = []
            for i, point in enumerate(self.points):
                if i in (0, len(self.points) - 1):
                    direction = directions[0 if i == 0 else -1]
                    extension = radius if self.cap == "square" else end_border
                    sign = -1.0 if i == 0 else 1.0
                    center = (
                        point[0] + sign * extension * direction[0],
                        point[2] + sign * extension * direction[1],
                    )
                    normal = (-direction[1] * radius, direction[0] * radius)
                    offsets = [(normal, (-normal[0], -normal[1]))]
                else:
                    before, after = directions[i - 1], directions[i]
                    dot = before[0] * after[0] + before[1] * after[1]
                    denominator = 1.0 + dot
                    if denominator <= 0.0:
                        raise ValueError("reversing line joins are unsupported")
                    miter = (
                        -(before[1] + after[1]) * radius / denominator,
                        (before[0] + after[0]) * radius / denominator,
                    )
                    if self.join == "miter":
                        # src/vector/line_helpers.rs clamps the miter at ratio 4.
                        length = math.hypot(*miter)
                        if length > radius * 4.0:
                            scale = radius * 4.0 / length
                            miter = (miter[0] * scale, miter[1] * scale)
                    opposite = (-miter[0], -miter[1])
                    center = (point[0], point[2])
                    turn = _cross(before, after)
                    if self.join == "miter" or turn == 0.0:
                        offsets = [(miter, opposite)]
                    elif turn > 0.0:
                        offsets = [
                            (miter, (d[1] * radius, -d[0] * radius))
                            for d in (before, after)
                        ]
                    else:
                        offsets = [
                            ((-d[1] * radius, d[0] * radius), opposite)
                            for d in (before, after)
                        ]
                for left, right in offsets:
                    y = point[1] + (0.0 if self.drape else self.z_offset)
                    result.append(tuple(
                        (center[0] + v[0], y, center[1] + v[1])
                        for v in (left, right)
                    ))
            return result

        inner = stations(self.width / 2.0, 0.0)
        outer = stations(self.width / 2.0 + self.halo, self.halo) if self.halo else inner
        # Reject crossings/folds rather than submit an ambiguous overlapped ribbon.
        boundary = [s[0] for s in outer] + [s[1] for s in reversed(outer)]
        boundary = [p for i, p in enumerate(boundary) if p != boundary[i - 1]]
        edges = list(zip(boundary, boundary[1:] + boundary[:1]))

        def orientation(a: WorldPosition, b: WorldPosition, c: WorldPosition) -> float:
            return _cross(
                (b[0] - a[0], b[2] - a[2]), (c[0] - a[0], c[2] - a[2])
            )

        def on_segment(a: WorldPosition, b: WorldPosition, p: WorldPosition) -> bool:
            return (
                orientation(a, b, p) == 0
                and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
                and min(a[2], b[2]) <= p[2] <= max(a[2], b[2])
            )

        for i, (a, b) in enumerate(edges):
            for j in range(i + 2, len(edges)):
                if i == 0 and j == len(edges) - 1:
                    continue
                c, d = edges[j]
                crossing = (
                    orientation(a, b, c) * orientation(a, b, d) < 0
                    and orientation(c, d, a) * orientation(c, d, b) < 0
                )
                touching = any((
                    on_segment(a, b, c), on_segment(a, b, d),
                    on_segment(c, d, a), on_segment(c, d, b),
                ))
                if crossing or touching:
                    raise ValueError("line width/halo produces a crossing or folded ribbon")

        vertices: list[VectorVertex] = []
        indices: list[int] = []

        def triangle(
            a: WorldPosition, b: WorldPosition, c: WorldPosition,
            color: tuple[float, float, float, float],
        ) -> None:
            for point in (a, b, c):
                world_position(point, name="generated line vertex")
            area = orientation(a, b, c)
            if not math.isfinite(area):
                raise ValueError("line geometry must be finite")
            if area == 0.0:
                return
            start = len(vertices)
            vertices.extend(VectorVertex(*p, *color, self.feature_id) for p in (a, b, c))
            indices.extend((start, start + 1, start + 2))

        for (al, ar), (bl, br) in zip(inner, inner[1:]):
            triangle(al, ar, bl, self.color)
            triangle(ar, br, bl, self.color)
        if self.halo:
            core_border = [s[0] for s in inner] + [s[1] for s in reversed(inner)]
            halo_border = [s[0] for s in outer] + [s[1] for s in reversed(outer)]
            for i in range(len(core_border)):
                j = (i + 1) % len(core_border)
                triangle(core_border[i], halo_border[i], halo_border[j], self.halo_color)
                triangle(core_border[i], halo_border[j], core_border[j], self.halo_color)
        if not indices:
            raise ValueError("line geometry has no representable area")
        return VectorOverlayConfig(
            name=self.name, vertices=vertices, indices=indices,
            primitive=PrimitiveType.TRIANGLES, drape=self.drape,
            drape_offset=self.z_offset if self.drape else 0.0,
            opacity=self.opacity, line_width=self.width,
        )
