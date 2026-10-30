"""Linked, flat plan-view PNG publications using MapScene and MapPlate.

No image is accepted from the caller. Every map goes through MapScene.render,
including unchanged inputs (whose reuse is validated by its native cache), and
its frame is the native unlit albedo, so displayed colours are legend colours.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from . import gis
from ._png import load_png_rgba, save_png
from .map_plate import BBox, MapPlate, MapPlateConfig
from .map_scene import LightingPreset, MapScene, OrbitCamera, OutputSpec, RasterOverlay, TerrainSource
from .raster_style import CategoricalRasterStyle, RasterStyleResult
from .scale_bar import ScaleBar, ScaleBarConfig

__all__ = ["PublicationSource", "PublicationExtent", "PublicationRecipe"]


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _bbox(b: BBox) -> None:
    if not np.isfinite([b.west, b.south, b.east, b.north]).all() or b.width <= 0 or b.height <= 0:
        raise ValueError("Publication bounds must be finite with positive area; no antimeridian wrapping")


@dataclass(frozen=True)
class PublicationSource:
    """A single-band GeoTIFF and its declared observation provenance.

    Nodata is defined by the file mask/nodata value. It is never a zero count.
    Year/edition describe this file, not a request to relabel another edition.
    The caller must supply the corresponding dataset when changing year.
    """
    path: str | Path
    name: str
    year: str
    edition: str
    uri: str
    units: str
    missing_label: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def description(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        for key in ("name", "year", "edition", "uri", "units", "missing_label"):
            if not isinstance(result[key], str) or not result[key].strip():
                raise ValueError(f"Publication source {key} must be nonempty text")
        return result


@dataclass(frozen=True)
class PublicationExtent:
    """An exact, source-grid-aligned extent; its ID is a safe PNG basename."""
    page_id: str
    title: str
    bounds: BBox


@dataclass
class PublicationRecipe:
    """One linked template, an explicit native classification and a locator.

    Only north-up, flat screen-mode maps in projected metre CRSs supported by
    the existing CRS helper are qualified. Extents must follow source cell
    edges: no hidden clipping, resampling, camera-footprint approximation or
    perspective scale. PNG is display RGBA8; no press-colour/vector claim.
    """
    source: PublicationSource
    breaks: Sequence[float]
    palette: Sequence[Sequence[int]]
    locator_bounds: BBox
    config: MapPlateConfig = field(default_factory=lambda: MapPlateConfig(
        width=1600, height=1200, margin=(100, 460, 170, 50)))
    missing_color: Sequence[int] = (222, 225, 224, 255)
    right: bool = False
    camera_mode: str = "screen"

    def _read(self):
        from pyproj import CRS, Geod  # Require the geodesic backend; no approximate fallback.
        from .crs import transform_coords

        description = self.source.description()
        path = Path(self.source.path)
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = gis.read_raster(path, masked=True)
        info = loaded["info"]
        values = np.asarray(loaded["array"])
        if values.ndim != 3 or values.shape[0] != 1:
            raise ValueError("Publication requires a single-band GeoTIFF")
        values = values[0]
        valid = np.asarray(loaded["mask"], dtype=bool).reshape(values.shape) & np.isfinite(values)
        authority = info.get("crs_authority")
        crs = f"{authority['name']}:{authority['code']}" if authority else info.get("crs_wkt")
        if not crs or not info.get("transform"):
            raise ValueError("Publication requires explicit CRS and affine registration")
        axes = CRS.from_user_input(crs)
        if not axes.is_projected or any(a.unit_conversion_factor != 1 for a in axes.axis_info):
            raise ValueError("Publication requires a projected CRS with metre axes")
        a, b, x, d, e, y = info["transform"]
        if b != 0 or d != 0 or a <= 0 or e >= 0:
            raise ValueError("Publication requires a north-up grid; rotated/skewed grids are unsupported")
        bounds = BBox(x, y + e * values.shape[0], x + a * values.shape[1], y, crs)
        _bbox(bounds)
        # Preflight the same transform/backend used by ScaleBar, which otherwise
        # has a legacy approximate fallback. Unsupported CRSs fail here.
        transform_coords(np.array([[bounds.west, bounds.center_lat],
                                   [bounds.east, bounds.center_lat]]), crs, "EPSG:4326")
        Geod(ellps="WGS84")
        description.update(sha256=source_sha, grid=asdict(bounds), transform=list(info["transform"]))
        return values.copy(), valid.copy(), bounds, (a, -e), description

    @staticmethod
    def _window(bounds, grid, resolution, shape):
        _bbox(bounds)
        if bounds.crs != grid.crs:
            raise ValueError("Publication extents must use the source CRS")
        dx, dy = resolution
        indices = ((bounds.west - grid.west) / dx, (grid.north - bounds.north) / dy,
                   (bounds.east - grid.west) / dx, (grid.north - bounds.south) / dy)
        # Edges are compared after float64 subtraction and division. Allow only
        # that representational rounding: a few ULPs of the largest coordinate,
        # expressed in cells. Any larger offset is a genuinely misaligned extent.
        largest = max(abs(v) for v in (grid.west, grid.south, grid.east, grid.north,
                                       bounds.west, bounds.south, bounds.east, bounds.north))
        ulps = 4 * float(np.spacing(largest))
        tolerance = (ulps / dx, ulps / dy, ulps / dx, ulps / dy)
        if max(tolerance) >= 0.5:
            raise ValueError("Publication grid is too fine for its coordinates to identify cell edges")
        if any(abs(v - round(v)) > t for v, t in zip(indices, tolerance)):
            raise ValueError("Publication extents must follow source cell edges")
        c0, r0, c1, r1 = map(round, indices)
        if not (0 <= c0 < c1 <= shape[1] and 0 <= r0 < r1 <= shape[0]):
            raise ValueError("Publication extent is outside the source grid")
        return slice(r0, r1), slice(c0, c1)

    def _classify(self, values, valid) -> tuple[np.ndarray, CategoricalRasterStyle, RasterStyleResult, list]:
        if not isinstance(self.right, bool):
            raise ValueError("right must be boolean")
        # Classification and bin validation belong to the existing native GIS API.
        classified = gis.classify_raster(values, bins=self.breaks, right=self.right, valid_mask=valid)
        table = classified["class_table"]
        if len(self.palette) != len(table) - 1:
            raise ValueError("Publication palette must contain one colour per classified interval")
        labels = {0: self.source.missing_label}
        for row in table[1:]:
            lo, hi = row["left"], row["right"]
            number = lambda v: np.format_float_positional(float(v), unique=True, trim="-")
            if lo is None:
                label = f"{'<=' if self.right else '<'} {number(hi)}" if hi is not None else "All valid values"
            elif hi is None:
                label = f"{number(lo)} {'<' if self.right else '<='} value"
            else:
                label = f"{number(lo)} {'<' if self.right else '<='} value {'<=' if self.right else '<'} {number(hi)}"
            labels[row["class_id"]] = label
        style = CategoricalRasterStyle({0: self.missing_color, **dict(enumerate(self.palette, 1))},
                                       labels=labels, units=self.source.units)
        # The published frame is opaque. A transparent key could not be shown.
        if any(rgba[3] != 255 for rgba in style.palette.values()):
            raise ValueError("Publication palette and missing colour must be fully opaque")
        classes = np.asarray(classified["array"])[0]
        result = style.apply(classes)
        return classes, style, result, table

    @staticmethod
    def _size(bounds, available):
        factor = min(available[0] / bounds.width, available[1] / bounds.height)
        return max(1, round(bounds.width * factor)), max(1, round(bounds.height * factor))

    def _scene(self, classes, style, bounds, size, description):
        return MapScene(
            terrain=TerrainSource(data=np.zeros(classes.shape, np.float32), crs=bounds.crs,
                metadata={"bounds": [bounds.west, bounds.south, bounds.east, bounds.north],
                          "resolution": [bounds.width / classes.shape[1], bounds.height / classes.shape[0]]}),
            layers=[RasterOverlay("publication", data=classes, style=style, crs=bounds.crs,
                                  metadata={"source": description})],
            camera=OrbitCamera(),
            # The lit beauty frame is shaded and tone-mapped, so its colours do
            # not match the legend. The frame itself is the native unlit albedo.
            lighting=LightingPreset(settings={"unlit_albedo": True, "cli_params": {"camera_mode": "screen"}}),
            output=OutputSpec(width=size[0], height=size[1]))

    @staticmethod
    def _render(scene, path, *, cache, style):
        scene.render(str(path), cache=cache)
        pixels = load_png_rgba(path)
        if pixels.shape != (scene.recipe.output.height, scene.recipe.output.width, 4):
            raise ValueError("Native publication frame dimensions do not match the requested viewport")
        # Fresh or reused, every drawn colour must be a legend colour, up to the
        # 8-bit rounding of the native half-float albedo; otherwise the legend lies.
        palette = np.array(list(style.palette.values()), np.int16)
        colours = np.unique(pixels.reshape(-1, 4), axis=0).astype(np.int16)
        deviation = int(np.abs(colours[:, None] - palette[None]).max(axis=2).min(axis=1).max())
        if deviation > 1:
            raise ValueError(f"Native map colours differ from the legend by {deviation} levels")
        return pixels, {"backend": scene.last_render_backend, "metadata": scene.last_render_metadata,
                        "map_pixels": "native unlit albedo frame, sRGB-encoded RGBA8",
                        "max_legend_deviation_levels": deviation}

    def render_series(self, pages: Sequence[PublicationExtent], directory: str | Path,
                      *, cache: str | Path | None = None) -> list[dict[str, Any]]:
        """Regenerate linked PNGs and JSON receipts. Never trust an existing PNG.

        ``cache`` is passed to MapScene's validated cache, not a publication
        image cache. Without it every main/locator map renders fresh. A failed
        render raises; a previous output is never returned as the new result.
        Source files must not be modified concurrently during this call.
        """
        if self.camera_mode != "screen":
            raise ValueError("Unsupported publication camera footprint: only flat screen plan views are qualified")
        pages = tuple(pages)
        if not pages or len({p.page_id for p in pages}) != len(pages):
            raise ValueError("Publication requires nonempty, distinct page IDs")
        for page in pages:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", page.page_id) or not page.title.strip():
                raise ValueError("Publication page needs a safe basename and nonempty title")
            from ._map_scene_render import _text_outline_metrics
            title_size = _text_outline_metrics(f"{page.title} / {self.source.year}", 38)
            if title_size[0] > self.config.width - 30 or title_size[1] > self.config.margin[0] - 30:
                raise ValueError("Publication title cannot fit; increase the publication size/margins")
        values, valid, grid, resolution, description = self._read()
        locator_window = self._window(self.locator_bounds, grid, resolution, values.shape)
        windows = [self._window(p.bounds, grid, resolution, values.shape) for p in pages]
        for page in pages:
            if not (self.locator_bounds.west <= page.bounds.west < page.bounds.east <= self.locator_bounds.east
                    and self.locator_bounds.south <= page.bounds.south < page.bounds.north <= self.locator_bounds.north):
                raise ValueError("Locator must contain the full main-view footprint")
        classes, style, styled, table = self._classify(values, valid)
        cfg = self.config
        top, right, bottom, left = cfg.margin
        map_area = cfg.width - right - left, cfg.height - top - bottom
        if min(*map_area, right - 40, bottom - 30, top - 30) <= 0:
            raise ValueError("Publication margins leave no room for its contents")
        source_id = _identity(description)
        classification_id = _identity({"source": source_id, "style": style.to_dict(),
                                       "breaks": list(self.breaks), "right": self.right})
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        receipts = []
        # Temporary frames cannot collide with, or stand in for, final outputs.
        with tempfile.TemporaryDirectory(prefix="forge3d-publication-", dir=output) as temporary:
            temp = Path(temporary)
            locator_size = self._size(self.locator_bounds, (right - 60, map_area[1] // 2))
            locator_scene = self._scene(classes[locator_window], style, self.locator_bounds,
                                        locator_size, description)
            locator, locator_render = self._render(locator_scene, temp / "locator.png", cache=cache, style=style)
            locator_id = _identity(locator_scene.to_dict())
            for page, window in zip(pages, windows):
                size = self._size(page.bounds, map_area)
                scene = self._scene(classes[window], style, page.bounds, size, description)
                main, rendered = self._render(scene, temp / "main.png", cache=cache, style=style)
                footprint = [[page.bounds.west, page.bounds.north], [page.bounds.east, page.bounds.north],
                             [page.bounds.east, page.bounds.south], [page.bounds.west, page.bounds.south]]
                # Screen mode maps the complete UV rectangle to the viewport.
                # This is its actual footprint, not an oblique-camera bbox.
                pixel_footprint = [[(x - self.locator_bounds.west) / self.locator_bounds.width * locator_size[0],
                                    (self.locator_bounds.north - y) / self.locator_bounds.height * locator_size[1]]
                                   for x, y in footprint]
                mpp = ScaleBar.compute_meters_per_pixel(page.bounds, size[0])
                if not np.isfinite(mpp) or mpp <= 0:
                    raise ValueError("Publication ground scale is unavailable for this extent")
                caption = (f"{self.source.name} | {self.source.year} | {self.source.edition}\n"
                           f"{self.source.uri}\n"
                           "Flat plan view / grid north up. Scale: WGS84 ground distance along the map centre line; varies across the map.")
                plate = MapPlate(cfg)
                plate.set_map_region(main, page.bounds)
                plate.add_title(f"{page.title} / {self.source.year}", font_size=38)
                sidebar = _sidebar(styled.legend, locator, pixel_footprint, right - 30, map_area[1])
                plate.add_legend(sidebar)
                scale_bar = ScaleBar(mpp, ScaleBarConfig(width_px=250, height_px=30,
                                         font_size=20, padding=12))
                scale_image = scale_bar.render()
                _, scale_label, _ = scale_bar.render_geometry()
                border = np.flatnonzero(np.all(scale_image[scale_bar.config.padding, :, :3] ==
                                               scale_bar.config.border_color[:3], axis=1))
                scale_span = int(border[-1] - border[0] + 1)  # Pixel-edge span, not centre-to-centre.
                # MapPlate centres the map inside its region and anchors furniture
                # to the region, not the image. Place the bar on the drawn map,
                # whose pixels must be the unresampled frame the scale describes.
                map_rect = _drawn_map_rect(cfg, size)
                sh, sw = scale_image.shape[:2]
                scale_rect = [map_rect[0] + 20, map_rect[3] - 20 - sh, map_rect[0] + 20 + sw, map_rect[3] - 20]
                if scale_rect[2] > map_rect[2] or scale_rect[1] < map_rect[1]:
                    raise ValueError("Publication scale bar cannot fit on the map; increase the publication size")
                image = plate.compose()
                plate._blit(image, scale_image, scale_rect[0], scale_rect[1])
                _text_block(image, caption, (left, cfg.height - bottom + 20), cfg.width - left - 30,
                            bottom - 30, 20)
                main_id = _identity(scene.to_dict())
                dependencies = {
                    "source": {"id": source_id, "depends_on": ["file bytes", "provenance", "registration"]},
                    "classification": {"id": classification_id, "depends_on": [source_id, "breaks", "palette", "units", "missing rules"]},
                    "map": {"id": main_id, "depends_on": [classification_id, "extent", "viewport", "scene"]},
                    "legend": {"id": _identity(styled.legend), "depends_on": ["classification semantics", "palette", "units", "missing rules"]},
                    "caption": {"id": _identity(caption), "depends_on": ["source name", "edition", "year", "URI", "view qualification"]},
                    "locator": {"id": _identity([locator_id, footprint]), "depends_on": [locator_id, "actual screen footprint"]},
                    "scale": {"id": _identity([asdict(page.bounds), size, mpp]), "depends_on": ["extent", "display viewport", "CRS", "WGS84"]},
                }
                png = output / f"{page.page_id}.png"
                receipt = {"page_id": page.page_id, "title": page.title, "source": description,
                           "dependencies": dependencies, "legend": styled.legend, "classification": table,
                           "caption": caption, "footprint": {"kind": "flat_screen_viewport", "crs": grid.crs,
                           "coordinates": footprint, "locator_pixels": pixel_footprint},
                           "scale": {"meters_per_pixel": mpp, "applies_to": "horizontal map centre line",
                                     "bar_label": scale_label, "bar_pixel_span": scale_span,
                                     "bar_rect": scale_rect},
                           "map_rect": map_rect,
                           "main_viewport": list(size), "locator_viewport": list(locator_size),
                           "main_render": rendered, "locator_render": locator_render,
                           "output": {"path": str(png), "size": [cfg.width, cfg.height], "format": "RGBA8 PNG",
                                      "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()},
                           "limitations": ["Flat north-up plan views only", "Grid-aligned extents only",
                               "One classified source; file mask is the missing-data rule",
                               "No perspective scale, moving frames, vector text or press-colour qualification"]}
                # Compose and validate everything before replacing this page.
                save_png(temp / "page.png", image)
                (temp / "page.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf8")
                (temp / "page.png").replace(png)
                (temp / "page.json").replace(output / f"{page.page_id}.json")
                receipts.append(receipt)
        return receipts


def _drawn_map_rect(cfg, size):
    """Return MapPlate's drawn map rectangle [x0, y0, x1, y1]; reject resampling."""
    top, right, bottom, left = cfg.margin
    rw, rh = cfg.width - right - left, cfg.height - top - bottom
    scale = min(rw / size[0], rh / size[1])
    if (max(1, int(size[0] * scale)), max(1, int(size[1] * scale))) != tuple(size):
        raise ValueError("Publication map would be resampled by the plate; scale would not apply")
    x, y = left + (rw - size[0]) // 2, top + (rh - size[1]) // 2
    return [x, y, x + size[0], y + size[1]]


def _text_block(image, text, origin, width, height, font_size):
    """Fit measured LITTERA text; reject overflow instead of clipping it."""
    from ._map_scene_render import _draw_text, _text_outline_metrics
    x, y = origin
    line_height = int(np.ceil(font_size * 1.5))
    lines = []
    for paragraph in text.splitlines():
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if _text_outline_metrics(candidate, font_size)[0] > width:
                if not line or _text_outline_metrics(word, font_size)[0] > width:
                    raise ValueError("Publication text cannot fit; increase the publication size/margins")
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    if len(lines) * line_height > height:
        raise ValueError("Publication text cannot fit; increase the publication size/margins")
    for i, line in enumerate(lines):
        _draw_text(image, line, (x, y + i * line_height), color=(35, 44, 48, 255),
                   halo=(0, 0, 0, 0), halo_width_px=0, font_size=font_size)
    return len(lines) * line_height


def _sidebar(legend, locator, footprint, width, height):
    canvas = np.full((height, width, 4), 255, np.uint8)
    y = _text_block(canvas, legend["units"], (15, 0), width - 30, height, 25) + 15
    for entry in legend["entries"]:
        canvas[y:y + 24, 15:45] = entry["rgba"]
        used = _text_block(canvas, entry["label"], (60, y - 2), width - 75, height - y, 22)
        y += max(30, used) + 10
    y += 16
    y += _text_block(canvas, "Locator / grid north up", (15, y), width - 30, height - y, 22) + 8
    h, w = locator.shape[:2]
    if y + h > height or w + 30 > width:
        raise ValueError("Publication legend and locator cannot fit; increase the publication size/margins")
    canvas[y:y + h, 15:15 + w] = locator
    # Pixel-edge coordinates follow the exact same affine mapping as screen UVs.
    # Pillow is already an optional MapPlate dependency; only the footprint is drawn here.
    from PIL import Image, ImageDraw
    image = Image.fromarray(canvas)
    points = [(15 + x, y + v) for x, v in footprint]
    ImageDraw.Draw(image).line(points + points[:1], fill=(18, 33, 43, 255), width=3)
    return np.asarray(image)
