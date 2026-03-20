from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .geometry import MeshBuffers, _mesh_to_py


def _as_float32_2d(array: np.ndarray, expected_cols: int, *, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_cols:
        raise ValueError(f"{name} must have shape (N, {expected_cols})")
    return np.ascontiguousarray(arr)


def _as_color(color: Sequence[float]) -> tuple[float, float, float, float]:
    if len(color) != 4:
        raise ValueError("color must contain 4 float components")
    values = tuple(float(component) for component in color)
    if not all(np.isfinite(component) for component in values):
        raise ValueError("color must contain only finite float components")
    return values  # type: ignore[return-value]


def _positive_finite_or_none(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float when provided")
    return value


def _validate_lod_distances(levels: Sequence["TerrainScatterLevel"]) -> None:
    previous_max_distance = 0.0
    for index, level in enumerate(levels):
        max_distance = _positive_finite_or_none(
            level.max_distance,
            name=f"levels[{index}].max_distance",
        )
        if max_distance is None:
            if index != len(levels) - 1:
                raise ValueError("only the final LOD level may omit max_distance")
            continue
        if max_distance <= previous_max_distance:
            raise ValueError("LOD max_distance values must be strictly increasing")
        previous_max_distance = max_distance


def bilinear_sample(field: np.ndarray, row: float, col: float) -> float:
    arr = np.asarray(field, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("bilinear_sample field must be a 2D array")

    max_row = max(arr.shape[0] - 1, 0)
    max_col = max(arr.shape[1] - 1, 0)
    row = float(np.clip(row, 0.0, max_row))
    col = float(np.clip(col, 0.0, max_col))

    row0 = int(np.floor(row))
    col0 = int(np.floor(col))
    row1 = min(row0 + 1, max_row)
    col1 = min(col0 + 1, max_col)

    tr = row - row0
    tc = col - col0

    v00 = float(arr[row0, col0])
    v01 = float(arr[row0, col1])
    v10 = float(arr[row1, col0])
    v11 = float(arr[row1, col1])

    top = (1.0 - tc) * v00 + tc * v01
    bottom = (1.0 - tc) * v10 + tc * v11
    return float((1.0 - tr) * top + tr * bottom)


def make_transform_row_major(
    translation: Sequence[float],
    *,
    yaw_deg: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    if len(translation) != 3:
        raise ValueError("translation must contain 3 float components")
    tx, ty, tz = (float(value) for value in translation)
    scale = float(scale)
    yaw = np.deg2rad(float(yaw_deg))
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return np.asarray(
        [
            [cos_yaw * scale, 0.0, sin_yaw * scale, tx],
            [0.0, scale, 0.0, ty],
            [-sin_yaw * scale, 0.0, cos_yaw * scale, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ).reshape(16)


@dataclass(frozen=True)
class TerrainScatterFilters:
    min_slope_deg: float | None = None
    max_slope_deg: float | None = None
    min_elevation: float | None = None
    max_elevation: float | None = None


class TerrainScatterSource:
    def __init__(
        self,
        heightmap: np.ndarray,
        *,
        z_scale: float = 1.0,
        terrain_width: float | None = None,
    ) -> None:
        heights = np.asarray(heightmap, dtype=np.float32)
        if heights.ndim != 2:
            raise ValueError("heightmap must be a 2D float32 array")
        if heights.size == 0:
            raise ValueError("heightmap must not be empty")

        finite = heights[np.isfinite(heights)]
        if finite.size == 0:
            raise ValueError("heightmap must contain at least one finite sample")
        fill_value = float(finite.min())
        heights = np.where(np.isfinite(heights), heights, fill_value).astype(np.float32)

        self.heightmap = np.ascontiguousarray(heights)
        self.z_scale = float(z_scale)
        if not np.isfinite(self.z_scale) or self.z_scale <= 0.0:
            raise ValueError("z_scale must be a positive finite float")
        self.height = int(self.heightmap.shape[0])
        self.width = int(self.heightmap.shape[1])
        self.terrain_width = float(terrain_width) if terrain_width is not None else float(max(self.heightmap.shape))
        if not np.isfinite(self.terrain_width) or self.terrain_width <= 0.0:
            raise ValueError("terrain_width must be a positive finite float")
        self.min_height = float(np.min(self.heightmap))
        self.max_height = float(np.max(self.heightmap))

        self._scaled_heights = np.ascontiguousarray((self.heightmap - self.min_height) * self.z_scale)
        x_step = self.terrain_width / max(self.width - 1, 1)
        z_step = self.terrain_width / max(self.height - 1, 1)
        dz_dz, dz_dx = np.gradient(self._scaled_heights, z_step, x_step)
        slope = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dz * dz_dz)))
        self._slope_degrees = np.ascontiguousarray(slope.astype(np.float32))

    @property
    def slope_degrees(self) -> np.ndarray:
        return self._slope_degrees

    @property
    def normalized_elevation(self) -> np.ndarray:
        span = max(self.max_height - self.min_height, 1e-6)
        return np.ascontiguousarray(((self.heightmap - self.min_height) / span).astype(np.float32))

    def contract_to_pixel(self, x: float, z: float) -> tuple[float, float]:
        row = (float(z) / max(self.terrain_width, 1e-6)) * max(self.height - 1, 1)
        col = (float(x) / max(self.terrain_width, 1e-6)) * max(self.width - 1, 1)
        return row, col

    def pixel_to_contract_xz(self, row: float, col: float) -> tuple[float, float]:
        x = (float(col) / max(self.width - 1, 1)) * self.terrain_width
        z = (float(row) / max(self.height - 1, 1)) * self.terrain_width
        return (float(x), float(z))

    def pixel_to_contract(self, row: float, col: float) -> tuple[float, float, float]:
        x, z = self.pixel_to_contract_xz(row, col)
        y = bilinear_sample(self._scaled_heights, row, col)
        return (float(x), float(y), float(z))

    def sample_height(self, row: float, col: float) -> float:
        return bilinear_sample(self.heightmap, row, col)

    def sample_scaled_height(self, row: float, col: float) -> float:
        return bilinear_sample(self._scaled_heights, row, col)

    def sample_slope_degrees(self, row: float, col: float) -> float:
        return bilinear_sample(self._slope_degrees, row, col)


def viewer_orbit_radius(
    source_or_width: TerrainScatterSource | float,
    *,
    scale: float = 1.9,
    minimum: float = 5.0,
) -> float:
    """Return a terrain-viewer orbit radius in terrain-width units."""
    if isinstance(source_or_width, TerrainScatterSource):
        terrain_width = source_or_width.terrain_width
    else:
        terrain_width = float(source_or_width)

    scale = float(scale)
    minimum = float(minimum)
    if not np.isfinite(terrain_width) or terrain_width <= 0.0:
        raise ValueError("terrain_width must be a positive finite float")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive finite float")
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum must be a non-negative finite float")
    return float(max(terrain_width * scale, minimum))


@dataclass(frozen=True)
class TerrainScatterLevel:
    mesh: MeshBuffers
    max_distance: float | None = None


@dataclass
class TerrainScatterBatch:
    levels: Sequence[TerrainScatterLevel]
    transforms: np.ndarray
    name: str | None = None
    color: Sequence[float] = (0.85, 0.85, 0.85, 1.0)
    max_draw_distance: float | None = None

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("TerrainScatterBatch requires at least one LOD level")
        self.transforms = _as_float32_2d(self.transforms, 16, name="transforms")
        if not np.isfinite(self.transforms).all():
            raise ValueError("transforms must contain only finite values")
        self.color = _as_color(self.color)
        self.max_draw_distance = _positive_finite_or_none(
            self.max_draw_distance,
            name="max_draw_distance",
        )
        _validate_lod_distances(self.levels)
        if self.transforms.shape[0] == 0:
            raise ValueError("TerrainScatterBatch requires at least one transform")

    @property
    def instance_count(self) -> int:
        return int(self.transforms.shape[0])

    def to_native_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "color": tuple(self.color),
            "max_draw_distance": self.max_draw_distance,
            "transforms": self.transforms,
            "levels": [
                {
                    "mesh": _mesh_to_py(level.mesh),
                    "max_distance": level.max_distance,
                }
                for level in self.levels
            ],
        }

    def to_viewer_payload(self) -> dict[str, Any]:
        levels: list[dict[str, Any]] = []
        for level in self.levels:
            indices = np.asarray(level.mesh.indices, dtype=np.uint32).reshape(-1)
            levels.append(
                {
                    "positions": _as_float32_2d(level.mesh.positions, 3, name="mesh.positions").tolist(),
                    "normals": _as_float32_2d(level.mesh.normals, 3, name="mesh.normals").tolist()
                    if np.asarray(level.mesh.normals).size
                    else [],
                    "indices": np.ascontiguousarray(indices).tolist(),
                    "max_distance": level.max_distance,
                }
            )

        return {
            "name": self.name,
            "color": list(self.color),
            "max_draw_distance": self.max_draw_distance,
            "transforms": self.transforms.tolist(),
            "levels": levels,
        }


def _passes_filters(
    source: TerrainScatterSource,
    row: float,
    col: float,
    filters: TerrainScatterFilters,
) -> bool:
    slope = source.sample_slope_degrees(row, col)
    height = source.sample_height(row, col)

    if filters.min_slope_deg is not None and slope < float(filters.min_slope_deg):
        return False
    if filters.max_slope_deg is not None and slope > float(filters.max_slope_deg):
        return False
    if filters.min_elevation is not None and height < float(filters.min_elevation):
        return False
    if filters.max_elevation is not None and height > float(filters.max_elevation):
        return False
    return True


def _randomized_transform(
    source: TerrainScatterSource,
    row: float,
    col: float,
    *,
    rng: np.random.Generator,
    yaw_range_deg: tuple[float, float],
    scale_range: tuple[float, float],
) -> np.ndarray:
    position = source.pixel_to_contract(row, col)
    yaw = float(rng.uniform(float(yaw_range_deg[0]), float(yaw_range_deg[1])))
    scale = float(rng.uniform(float(scale_range[0]), float(scale_range[1])))
    return make_transform_row_major(position, yaw_deg=yaw, scale=scale)


def _normalize_scale_range(scale_range: tuple[float, float]) -> tuple[float, float]:
    lo = float(scale_range[0])
    hi = float(scale_range[1])
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("scale_range values must be positive")
    if hi < lo:
        raise ValueError("scale_range must be ordered (min, max)")
    return (lo, hi)


def _normalize_spacing(spacing: float) -> float:
    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    return spacing


def _normalize_jitter(jitter: float) -> float:
    jitter = float(jitter)
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative")
    return jitter


def _normalize_edge_margin(source: TerrainScatterSource, edge_margin: float) -> float:
    edge_margin = float(edge_margin)
    if not np.isfinite(edge_margin) or edge_margin < 0.0:
        raise ValueError("edge_margin must be a non-negative finite float")
    max_margin = max(source.terrain_width * 0.5 - 1e-6, 0.0)
    if edge_margin > max_margin:
        raise ValueError("edge_margin must be smaller than half the terrain width")
    return edge_margin


def _inside_edge_margin(
    source: TerrainScatterSource,
    row: float,
    col: float,
    edge_margin: float,
) -> bool:
    if edge_margin <= 0.0:
        return True
    x, z = source.pixel_to_contract_xz(row, col)
    max_coord = source.terrain_width - edge_margin
    return edge_margin <= x <= max_coord and edge_margin <= z <= max_coord


def _validate_density_mask(density_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(density_mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("density_mask must be a 2D array")
    if arr.size == 0:
        raise ValueError("density_mask must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("density_mask must contain only finite values")
    return np.ascontiguousarray(arr)


def _zero_transforms_error(
    name: str,
    source: TerrainScatterSource,
    *,
    detail: str,
) -> ValueError:
    return ValueError(
        f"{name} generated zero accepted transforms on {source.width}x{source.height} terrain; "
        f"{detail}. Loosen filters, reduce edge_margin, or adjust spacing/density."
    )


def _sample_mask(mask: np.ndarray, source: TerrainScatterSource, row: float, col: float) -> float:
    mask_row = (float(row) / max(source.height - 1, 1)) * max(mask.shape[0] - 1, 1)
    mask_col = (float(col) / max(source.width - 1, 1)) * max(mask.shape[1] - 1, 1)
    return float(np.clip(bilinear_sample(mask, mask_row, mask_col), 0.0, 1.0))


def seeded_random_transforms(
    source: TerrainScatterSource,
    *,
    count: int,
    seed: int,
    filters: TerrainScatterFilters | None = None,
    yaw_range_deg: tuple[float, float] = (0.0, 360.0),
    scale_range: tuple[float, float] = (1.0, 1.0),
    edge_margin: float = 0.0,
    max_attempts: int | None = None,
) -> np.ndarray:
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")

    filters = filters or TerrainScatterFilters()
    scale_range = _normalize_scale_range(scale_range)
    edge_margin = _normalize_edge_margin(source, edge_margin)
    rng = np.random.default_rng(int(seed))
    limit = int(max_attempts) if max_attempts is not None else max(count * 10, count + 1)
    if limit <= 0:
        raise ValueError("max_attempts must be positive when provided")

    transforms: list[np.ndarray] = []
    attempts = 0
    while len(transforms) < count and attempts < limit:
        row = float(rng.uniform(0.0, max(source.height - 1, 1)))
        col = float(rng.uniform(0.0, max(source.width - 1, 1)))
        attempts += 1
        if not _passes_filters(source, row, col, filters):
            continue
        if not _inside_edge_margin(source, row, col, edge_margin):
            continue
        transforms.append(
            _randomized_transform(
                source,
                row,
                col,
                rng=rng,
                yaw_range_deg=yaw_range_deg,
                scale_range=scale_range,
            )
        )

    if not transforms:
        raise _zero_transforms_error(
            "seeded_random_transforms",
            source,
            detail=f"accepted 0 of {attempts} attempts",
        )

    if len(transforms) < count:
        raise ValueError(
            f"seeded_random_transforms accepted only {len(transforms)} of {count} requested "
            f"transforms after {attempts} attempts on {source.width}x{source.height} terrain. "
            f"Loosen filters, reduce edge_margin, or increase max_attempts."
        )

    return np.ascontiguousarray(np.vstack(transforms).astype(np.float32))


def grid_jitter_transforms(
    source: TerrainScatterSource,
    *,
    spacing: float,
    seed: int,
    jitter: float = 0.5,
    filters: TerrainScatterFilters | None = None,
    yaw_range_deg: tuple[float, float] = (0.0, 360.0),
    scale_range: tuple[float, float] = (1.0, 1.0),
    edge_margin: float = 0.0,
) -> np.ndarray:
    spacing = _normalize_spacing(spacing)
    jitter = _normalize_jitter(jitter)
    filters = filters or TerrainScatterFilters()
    scale_range = _normalize_scale_range(scale_range)
    edge_margin = _normalize_edge_margin(source, edge_margin)
    rng = np.random.default_rng(int(seed))
    half_jitter = spacing * 0.5 * jitter
    cells = int(np.ceil(source.terrain_width / spacing))

    transforms: list[np.ndarray] = []
    for z_index in range(cells):
        for x_index in range(cells):
            x = (x_index + 0.5) * spacing
            z = (z_index + 0.5) * spacing
            if x > source.terrain_width or z > source.terrain_width:
                continue
            x = float(np.clip(x + rng.uniform(-half_jitter, half_jitter), 0.0, source.terrain_width))
            z = float(np.clip(z + rng.uniform(-half_jitter, half_jitter), 0.0, source.terrain_width))
            row, col = source.contract_to_pixel(x, z)
            if not _inside_edge_margin(source, row, col, edge_margin):
                continue
            if not _passes_filters(source, row, col, filters):
                continue
            transforms.append(
                _randomized_transform(
                    source,
                    row,
                    col,
                    rng=rng,
                    yaw_range_deg=yaw_range_deg,
                    scale_range=scale_range,
                )
            )

    if not transforms:
        raise _zero_transforms_error(
            "grid_jitter_transforms",
            source,
            detail=f"accepted 0 of {cells * cells} grid cells at spacing={spacing:g}",
        )

    return np.ascontiguousarray(np.vstack(transforms).astype(np.float32))


def mask_density_transforms(
    source: TerrainScatterSource,
    density_mask: np.ndarray,
    *,
    spacing: float,
    seed: int,
    jitter: float = 0.5,
    density_scale: float = 1.0,
    filters: TerrainScatterFilters | None = None,
    yaw_range_deg: tuple[float, float] = (0.0, 360.0),
    scale_range: tuple[float, float] = (1.0, 1.0),
    edge_margin: float = 0.0,
) -> np.ndarray:
    spacing = _normalize_spacing(spacing)
    jitter = _normalize_jitter(jitter)
    density_scale = float(density_scale)
    if density_scale < 0.0:
        raise ValueError("density_scale must be non-negative")

    density_mask = _validate_density_mask(density_mask)
    filters = filters or TerrainScatterFilters()
    scale_range = _normalize_scale_range(scale_range)
    edge_margin = _normalize_edge_margin(source, edge_margin)
    rng = np.random.default_rng(int(seed))
    half_jitter = spacing * 0.5 * jitter
    cells = int(np.ceil(source.terrain_width / spacing))

    transforms: list[np.ndarray] = []
    for z_index in range(cells):
        for x_index in range(cells):
            x = (x_index + 0.5) * spacing
            z = (z_index + 0.5) * spacing
            if x > source.terrain_width or z > source.terrain_width:
                continue
            x = float(np.clip(x + rng.uniform(-half_jitter, half_jitter), 0.0, source.terrain_width))
            z = float(np.clip(z + rng.uniform(-half_jitter, half_jitter), 0.0, source.terrain_width))
            row, col = source.contract_to_pixel(x, z)
            if not _inside_edge_margin(source, row, col, edge_margin):
                continue
            density = np.clip(_sample_mask(density_mask, source, row, col) * density_scale, 0.0, 1.0)
            if float(rng.random()) > float(density):
                continue
            if not _passes_filters(source, row, col, filters):
                continue
            transforms.append(
                _randomized_transform(
                    source,
                    row,
                    col,
                    rng=rng,
                    yaw_range_deg=yaw_range_deg,
                    scale_range=scale_range,
                )
            )

    if not transforms:
        raise _zero_transforms_error(
            "mask_density_transforms",
            source,
            detail=f"accepted 0 of {cells * cells} grid cells at spacing={spacing:g}",
        )

    return np.ascontiguousarray(np.vstack(transforms).astype(np.float32))


def serialize_batches_for_native(batches: Iterable[TerrainScatterBatch]) -> list[dict[str, Any]]:
    return [batch.to_native_dict() for batch in batches]


def serialize_batches_for_viewer(batches: Iterable[TerrainScatterBatch]) -> list[dict[str, Any]]:
    return [batch.to_viewer_payload() for batch in batches]


def apply_to_renderer(renderer: Any, batches: Iterable[TerrainScatterBatch]) -> None:
    renderer.set_scatter_batches(serialize_batches_for_native(batches))


def clear_renderer(renderer: Any) -> None:
    renderer.clear_scatter_batches()


def apply_to_viewer(viewer: Any, batches: Iterable[TerrainScatterBatch]) -> None:
    payload = serialize_batches_for_viewer(batches)
    if hasattr(viewer, "set_terrain_scatter"):
        viewer.set_terrain_scatter(payload)
        return
    viewer.send_ipc({"cmd": "set_terrain_scatter", "batches": payload})


def clear_viewer(viewer: Any) -> None:
    if hasattr(viewer, "clear_terrain_scatter"):
        viewer.clear_terrain_scatter()
        return
    viewer.send_ipc({"cmd": "clear_terrain_scatter"})


__all__ = [
    "TerrainScatterBatch",
    "TerrainScatterFilters",
    "TerrainScatterLevel",
    "TerrainScatterSource",
    "apply_to_renderer",
    "apply_to_viewer",
    "bilinear_sample",
    "clear_renderer",
    "clear_viewer",
    "grid_jitter_transforms",
    "make_transform_row_major",
    "mask_density_transforms",
    "seeded_random_transforms",
    "serialize_batches_for_native",
    "serialize_batches_for_viewer",
    "viewer_orbit_radius",
]
