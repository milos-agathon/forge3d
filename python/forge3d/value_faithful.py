"""Bind a prepared count grid to statistical relief and a linked publication."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import product
from math import atan2, degrees
import json
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np

from . import gis
from ._png import load_png_rgba, save_png
from .map_plate import BBox, MapPlate, MapPlateConfig
from .map_scene import (
    OrbitCamera, _build_mapscene_terrain_params, _load_native_heightmap,
    _shared_terrain_render_context,
)
from .north_arrow import NorthArrow, NorthArrowConfig
from .publication import (
    PublicationExtent, PublicationRecipe, PublicationSource, _LEGEND_TOLERANCE,
    _check_frame_palette, _drawn_map_rect, _legend_block, _publication_scene,
    _read_publication_source, _text_block,
)
from .raster_style import CategoricalRasterStyle, RasterHeightSurfaceStyle, _finite, _rgba

__all__ = ["ValueFaithfulRasterBinding"]

# TerrainScene draws a fixed 512 x 512 procedural mesh (core.rs, upload.rs).
_MESH_INTERVALS = 512 - 1
_SAMPLES_PER_CELL = 3


@dataclass(frozen=True)
class ValueFaithfulRasterBinding:
    """A prepared count field, its display measurement and its publication extent.

    The single-band GeoTIFF must declare measurement_field and
    measurement_units tags. Counts are nonnegative integers. Density divides
    by the full cell area in the projected metre grid, expressed in km2.
    Colour edges start at zero and end at an inclusive maximum. Zero has its
    own key. The other bins are left-closed except for the first positive bin.
    Missing GeoTIFF nodata and the unmasked withheld code are both holes in
    relief and have separate keys in the north-up comparison inset.
    """

    source: PublicationSource
    extent: PublicationExtent
    measurement_type: str
    height_scale: float
    colour_edges: Sequence[float]
    palette: Sequence[Sequence[int]]
    missing_code: int
    withheld_code: int

    def __post_init__(self) -> None:
        if self.measurement_type not in ("count", "density"):
            raise ValueError("measurement_type must be 'count' or 'density'")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.extent.page_id) or not self.extent.title.strip():
            raise ValueError("Publication extent needs a safe ID and nonempty title")
        if any(type(code) is not int or code >= 0 for code in (self.missing_code, self.withheld_code)):
            raise ValueError("Missing and withheld codes must be negative integers")
        if self.missing_code == self.withheld_code:
            raise ValueError("Missing and withheld codes must differ")
        scale = _finite(self.height_scale, "height_scale")
        if scale <= 0:
            raise ValueError("height_scale must be positive")
        edges = tuple(_finite(x, "colour edge") for x in self.colour_edges)
        if len(edges) < 2 or edges[0] != 0 or any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("Colour edges must increase from zero to a finite maximum")
        maximum_height = edges[-1] * scale
        # The native linear geometry shader floors its denominator at 1e-6.
        if not 1e-6 <= maximum_height <= np.finfo(np.float32).max:
            raise ValueError("Declared height domain is outside the native linear geometry range")
        palette = tuple(_rgba(colour) for colour in self.palette)
        # Zero, positive intervals, withheld and missing, in legend order.
        if len(palette) != len(edges) + 2 or any(c[3] != 255 for c in palette):
            raise ValueError("Supply opaque colours for zero, each positive interval, withheld and missing")
        colours = np.asarray((*palette, (0, 0, 0, 255)), np.int16)
        for i, colour in enumerate(colours[:-1]):
            if np.any(np.max(np.abs(colours[i + 1:] - colour), axis=1) <= _LEGEND_TOLERANCE):
                raise ValueError("Legend keys must differ from each other and from the empty frame background")
        object.__setattr__(self, "height_scale", scale)
        object.__setattr__(self, "colour_edges", edges)
        object.__setattr__(self, "palette", palette)

    def to_dict(self) -> dict:
        """Return a JSON-ready authoring declaration. The GeoTIFF stays external."""
        return {**asdict(self), "source": self.source.description(),
                "colour_edges": list(self.colour_edges), "palette": [list(c) for c in self.palette]}

    @classmethod
    def from_dict(cls, data: Mapping) -> ValueFaithfulRasterBinding:
        values = dict(data)
        values["source"] = PublicationSource(**values["source"])
        extent = dict(values["extent"])
        extent["bounds"] = BBox(**extent["bounds"])
        values["extent"] = PublicationExtent(**extent)
        return cls(**values)

    def _resolve(self) -> _BoundRaster:
        field = self.source.provenance.get("measurement_field")
        if not isinstance(field, str) or not field.strip():
            raise ValueError("Source provenance must declare measurement_field")
        raw, valid, grid, resolution, description = _read_publication_source(
            self.source, require_scale=False, expected_tags={
                "measurement_field": field, "measurement_units": self.source.units,
            })
        if resolution[0] != resolution[1]:
            raise ValueError("Statistical relief requires square source cells")
        if description["nodata"] != [self.missing_code]:
            raise ValueError("Prepared GeoTIFF nodata must match missing_code")
        bounds = self.extent.bounds
        window = PublicationRecipe._window(bounds, grid, resolution, raw.shape)
        raw, valid = raw[window], valid[window]
        if raw.shape[0] != raw.shape[1] or raw.shape[0] < 2:
            raise ValueError("The native statistical mesh requires a square extent of at least two cells per side")
        _check_mesh_registration(raw.shape[0])
        if not np.array_equal(valid, raw != self.missing_code):
            raise ValueError("Prepared cells must use the declared missing code")
        withheld = raw == self.withheld_code
        known = valid & ~withheld
        counts = raw[known]
        if not counts.size:
            raise ValueError("The selected extent has no known counts for a relief surface")
        if (np.any(counts < 0) or np.any(counts != np.floor(counts))
                or np.any(counts > 2**53)):
            raise ValueError("Known cells must be nonnegative integers exactly representable in float64")

        cell_area_m2 = resolution[0] * resolution[1]
        area_km2 = cell_area_m2 / 1_000_000
        if not np.isfinite(area_km2) or area_km2 <= 0:
            raise ValueError("Source cell area is not representable in km2")
        values = np.full(raw.shape, np.nan, np.float64)
        values[known] = counts
        units = f"{self.source.units} / cell"
        if self.measurement_type == "density":
            values[known] /= area_km2
            units = f"{self.source.units} / km2"
        edges = self.colour_edges
        if not np.isfinite(values[known]).all() or np.any(values[known] > edges[-1]):
            raise ValueError("Known measurements lie outside the declared colour domain")
        # The upper edge was checked above. The native classifier requires
        # a bin boundary, even when there is only one positive interval.
        classified = gis.classify_raster(values, bins=edges[:-1], valid_mask=known)
        classes = classified["array"][0].astype(np.int32)
        classes[known & (values == 0)] = 1
        classes[withheld] = len(edges) + 1
        classes[~valid] = len(edges) + 2

        labels = ["0"]
        for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
            lower = ">0" if i == 0 else _number(lo)
            upper = _number(hi) if i == len(edges) - 2 else f"<{_number(hi)}"
            labels.append(f"{lower} to {upper}")
        labels.extend(("Withheld / unavailable", self.source.missing_label))
        legend_style = CategoricalRasterStyle(dict(enumerate(self.palette, 1)),
                                             dict(enumerate(labels, 1)), units=units)
        height_style = RasterHeightSurfaceStyle(self.height_scale, units)
        # Validate the renderable height range before producing any publication.
        heights = height_style.apply(values).heights
        if np.any((values[known] > 0) & (heights[known] == 0)):
            raise ValueError("Positive statistical heights underflow float32")
        contributors = int(known.sum())
        summary = {
            "known_count_sum": sum(map(int, counts)),
            "known_cells": contributors,
            "known_zero_cells": int(np.count_nonzero(counts == 0)),
            "withheld_cells": int(withheld.sum()),
            "missing_cells": int((~valid).sum()),
            "total_cells": int(raw.size),
            "known_fraction": contributors / raw.size,
            "cell_area_m2": cell_area_m2,
            "known_area_m2": contributors * cell_area_m2,
            "extent_area_m2": raw.size * cell_area_m2,
        }
        return _BoundRaster(self, description, values, classes, legend_style, height_style, summary)

    def render_series(self, cameras: Mapping[str, OrbitCamera], directory: str | Path,
                      *, config: MapPlateConfig | None = None) -> list[dict]:
        """Render fresh native perspectives with one registered comparison inset.

        Each camera ID becomes a PNG and JSON receipt. All frames share the
        selected extent, full-cell denominator, transform and colour keys.
        MapPlate's existing Pro gate applies.
        """
        return self._resolve().render_series(cameras, Path(directory), config)


def _number(value: float) -> str:
    return np.format_float_positional(value, unique=True, trim="-")


def _height_reference(edges, scale):
    # Use declared class boundaries so the reference stays fixed across views.
    return [{"value": value, "height_m": value * scale}
            for value in dict.fromkeys((0., edges[len(edges) // 2], edges[-1]))]


def _check_mesh_registration(cells: int) -> None:
    # A geographic centre is inside a native mesh quad. Both bracketing
    # vertices must sample within that cell's constant-height texel plateau.
    i = np.arange(cells, dtype=np.int64)
    centres = (2 * i + 1) * _MESH_INTERVALS
    denominator = 2 * cells
    lower = centres // denominator
    upper = (centres + denominator - 1) // denominator
    last_texel = _SAMPLES_PER_CELL * cells - 1
    if (np.any(lower * last_texel <= _SAMPLES_PER_CELL * i * _MESH_INTERVALS)
            or np.any(upper * last_texel >= (_SAMPLES_PER_CELL * (i + 1) - 1) * _MESH_INTERVALS)):
        raise ValueError("Source cells are too fine for the native statistical mesh to preserve their centre heights")


@dataclass
class _BoundRaster:
    binding: ValueFaithfulRasterBinding
    description: dict
    values: np.ndarray
    classes: np.ndarray
    legend_style: CategoricalRasterStyle
    height_style: RasterHeightSurfaceStyle
    summary: dict

    def scene(self, camera, size, *, aovs=()):
        # Both unavailable states are masked by a zero-alpha thematic texel.
        # TerrainSource preserves the matching invalid statistical heights.
        known = np.isfinite(self.values)
        classes = np.where(known, self.classes, -1)
        palette = {i: c for i, c in self.legend_style.palette.items() if i <= len(self.binding.colour_edges)}
        style = CategoricalRasterStyle(palette, nodata=-1, units=self.legend_style.units)
        # Repeat values to keep mesh vertices bracketing each cell centre on
        # its own plateau. Nearest thematic sampling still uses source cells.
        heights_input = self.values.repeat(_SAMPLES_PER_CELL, axis=0).repeat(_SAMPLES_PER_CELL, axis=1)
        thematic_input = classes.repeat(_SAMPLES_PER_CELL, axis=0).repeat(_SAMPLES_PER_CELL, axis=1)
        high = float(np.float32(self.binding.colour_edges[-1] * self.binding.height_scale))
        # Native mesh coordinates centre heights on the domain midpoint.
        # Keep the author's camera relative to statistical zero across edits.
        x, y, z = camera.target
        native_camera = replace(camera, target=(x, y, z - high / 2))
        scene = _publication_scene(thematic_input, style, self.binding.extent.bounds, size, self.description,
                                   values=heights_input, height_style=self.height_style,
                                   camera=native_camera, camera_mode="mesh:zup", aovs=aovs)
        scene.recipe.terrain.metadata["statistical_height_domain"] = [0.0, high]
        return scene

    def _check_camera(self, scene, size):
        recipe = scene.recipe
        low, high = recipe.terrain.metadata["statistical_height_domain"]
        params = _build_mapscene_terrain_params(
            recipe, _load_native_heightmap(recipe.terrain), size, projection_only=True,
        )
        if params is None or not callable(getattr(params, "project_terrain_points", None)):
            raise RuntimeError("Native statistical camera projection is unavailable")
        corners = [(u, v, h, 0.0) for (u, v), h in product(product((0., 1.), repeat=2), (low, high))]
        width, height = size
        # The whole declared volume lies inside a convex view frustum if
        # each corner lies inside. Use the renderer's actual camera and clip.
        for point in params.project_terrain_points(corners):
            if point is None or not (0 < point[0] < width and 0 < point[1] < height and 0 < point[2] < 1):
                raise ValueError("Camera clips the statistical relief domain; frame the full extent and height range")
        # Grid north at the centre of the reference plane. Project through
        # the same camera, using northward-decreasing raster rows.
        centre, north = params.project_terrain_points([(0.5, 0.5, low, 0.), (0.5, 0., low, 0.)])
        dx, dy = north[0] - centre[0], north[1] - centre[1]
        return None if dx == dy == 0 else degrees(atan2(dx, -dy))

    def inset_scene(self, size):
        return _publication_scene(self.classes, self.legend_style, self.binding.extent.bounds,
                                  size, self.description)

    def render_series(self, cameras, directory, config):
        cameras = tuple(cameras.items())
        if (not cameras or any(not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key, _ in cameras)
                or len({key.casefold() for key, _ in cameras}) != len(cameras)):
            raise ValueError("Publication requires nonempty, distinct, safe frame IDs")
        if any(not isinstance(camera, OrbitCamera) for _, camera in cameras):
            raise TypeError("Statistical relief requires OrbitCamera views")
        cfg = config or MapPlateConfig(width=1600, height=1200, margin=(100, 480, 200, 50))
        plate = MapPlate(cfg)
        top, right, bottom, left = cfg.margin
        size = cfg.width - right - left, cfg.height - top - bottom
        map_rect = _drawn_map_rect(cfg, size)
        title = f"{self.binding.extent.title} / {self.binding.source.name} / {self.binding.source.year}"
        from ._map_scene_render import _text_outline_metrics
        title_size = _text_outline_metrics(title, 38)
        if title_size[0] > cfg.width - 30 or title_size[1] > top - 30:
            raise ValueError("Publication title cannot fit")
        plate.add_title(title, font_size=38)
        caption = self._caption()
        legend = self.legend_style.legend()
        legend["colour_edges"] = list(self.binding.colour_edges)
        legend["height_transform"] = {"factor": self.binding.height_scale, "output_units": "m",
                                      "kind": "linear_statistical_height"}
        legend["height_reference"] = _height_reference(self.binding.colour_edges, self.binding.height_scale)
        sidebar = np.full((size[1], right - 30, 4), 255, np.uint8)
        y = _legend_block(sidebar, legend) + 12
        width_km = _number(self.binding.extent.bounds.width / 1000)
        y += _text_block(sidebar, f"Same {width_km} km grid / north up", (15, y), right - 60, size[1] - y, 22) + 8
        inset_side = min(right - 60, size[1] - y)
        if inset_side < max(64, *self.values.shape) or min(size) < 64:
            raise ValueError("Publication needs room for native viewports and at least one inset pixel per cell")
        inset_size = (inset_side, inset_side)
        inset_x = cfg.width - right + (right - 10 - sidebar.shape[1]) // 2 + 15
        inset_rect = [inset_x, top + y, inset_x + inset_side, top + y + inset_side]
        footer = np.full((bottom, cfg.width, 4), cfg.background, np.uint8)
        _text_block(footer, caption, (left, 10), cfg.width - left - 30, bottom - 20, 22)
        directory.mkdir(parents=True, exist_ok=True)
        receipts = []
        with tempfile.TemporaryDirectory(prefix="forge3d-relief-", dir=directory) as temp, _shared_terrain_render_context():
            temp = Path(temp)
            inset, inset_render = PublicationRecipe._render(
                self.inset_scene(inset_size), temp / "inset.png", cache=None, style=self.legend_style)
            sidebar[y:y + inset_side, 15:15 + inset_side] = inset
            plate.add_legend(sidebar)
            colours = [*self.legend_style.palette.values(), (0, 0, 0, 255)]
            for key, camera in cameras:
                scene = self.scene(camera, size)
                rotation = self._check_camera(scene, size)
                scene.render(str(temp / "relief.png"))
                relief = load_png_rgba(temp / "relief.png")
                deviation = _check_frame_palette(relief, colours)
                # C06's opaque albedo encodes empty pixels as black. Replace
                # the validated background only; black cannot be a data key.
                empty = np.all(relief[..., :3] == 0, axis=-1)
                if empty.all():
                    raise ValueError("Native statistical relief rendered no visible data cells")
                relief[empty] = (22, 31, 40, 255)
                plate.set_map_region(relief, self.binding.extent.bounds)
                image = plate.compose()
                image[-bottom:] = footer
                # Keep the changing orientation cue in the heading margin,
                # clear of the checked relief viewport and the inset.
                if rotation is not None:
                    arrow = NorthArrow(NorthArrowConfig(size=48, rotation_deg=rotation,
                                                       show_n_label=False)).render()
                    if top < arrow.shape[0] or left + arrow.shape[1] >= (cfg.width - title_size[0] - 22) // 2:
                        raise ValueError("Publication heading has no room for the grid north arrow")
                    arrow_top = top - arrow.shape[0]
                    plate._blit(image, arrow, left, arrow_top)
                    _text_block(image, "Grid N", (left + arrow.shape[1] + 4, arrow_top + 18),
                                (cfg.width - title_size[0] - 22) // 2 - left - arrow.shape[1] - 8,
                                arrow.shape[0] - 18, 18)
                png = directory / f"{key}.png"
                receipt = {
                    "frame_id": key, "binding": self.binding.to_dict(), "source": self.description,
                    "coverage": self.summary, "legend": legend, "caption": caption,
                    "camera": camera.to_dict(), "grid_north_rotation_deg": rotation,
                    "map_rect": map_rect, "inset_rect": inset_rect,
                    "inset_render": inset_render, "main_render": {
                        "backend": scene.last_render_backend, "metadata": scene.last_render_metadata,
                        "max_legend_deviation_levels": deviation,
                    },
                    "output": {"path": str(png), "size": [cfg.width, cfg.height],
                               "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()},
                }
                save_png(temp / "page.png", image)
                (temp / "page.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf8")
                (temp / "page.png").replace(png)
                (temp / "page.json").replace(png.with_suffix(".json"))
                receipts.append(receipt)
        return receipts

    def _caption(self):
        source = self.binding.source
        c = self.summary
        scale = _number(self.binding.height_scale)
        width_km = _number(self.binding.extent.bounds.width / 1000)
        reference = _height_reference(self.binding.colour_edges, self.binding.height_scale)
        # A compact reference for the encoded statistical heights, not a
        # ruler in the perspective image where scale changes with depth.
        examples = ", ".join(f"{_number(p['value'])} = {format(p['height_m'], '.6g')} m" for p in reference)
        return (
            f"{source.name} | {source.year} | {source.edition}\n"
            f"Statistical height = value x {scale} m per ({self.legend_style.units}). Edges interpolate.\n"
            f"Height key ({self.legend_style.units})  {examples}.\n"
            f"In this {width_km} km square, published counts sum to {c['known_count_sum']:,} {source.units} across "
            f"{c['known_cells']:,} of {c['total_cells']:,} cells. "
            f"Withheld {c['withheld_cells']:,} / no record {c['missing_cells']:,}. "
            "The sum covers published cells only. The inset shows unavailable cells."
        )
