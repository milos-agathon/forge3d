"""Linked plan-view and flat-terrain perspective PNG publications.

No image is accepted from the caller. Every map goes through MapScene.render,
including unchanged inputs (whose reuse is validated by its native cache), and
its frame is the native unlit albedo, so displayed colours are legend colours.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
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
from .observation_time import ObservationSelection, ObservationTrack

__all__ = ["PublicationSource", "PublicationExtent", "PublicationRecipe",
           "PublicationInset", "PublicationLayout", "PublicationSubject", "publication_output_support"]
_LEGEND_TOLERANCE = 1  # RGBA8 levels allowed by the native albedo check.


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _bbox(b: BBox) -> None:
    if not np.isfinite([b.west, b.south, b.east, b.north]).all() or b.width <= 0 or b.height <= 0:
        raise ValueError("Publication bounds must be finite with positive area; no antimeridian wrapping")


def _check_source_metadata(path: Path, year: str) -> None:
    """Validate embedded GDAL metadata without decoding or transforming pixels."""
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    from xml.etree import ElementTree

    # Native GIS reads the TIFF itself. It cannot consume external PAM overrides.
    if Path(str(path) + ".aux.xml").exists():
        raise ValueError("Publication requires a self-contained GeoTIFF; metadata sidecars are unsupported")
    with path.open("rb") as stream:
        header = stream.read(8)
        if header[2:4] in (b"+\0", b"\0+"):  # BigTIFF's header is 16 bytes.
            header += stream.read(8)
        directory = ImageFileDirectory_v2(header)
        stream.seek(directory.next)
        directory.load(stream)
        metadata = directory.get(42112)  # GDAL_METADATA TIFF tag.
    if metadata is None:
        return
    try:
        items = ElementTree.fromstring(metadata).findall("Item")
    except (ElementTree.ParseError, TypeError) as exc:
        raise ValueError("Publication GeoTIFF metadata is unreadable") from exc
    for item in items:
        role = item.get("role", "").lower()
        if role in {"scale", "offset"}:
            try:
                value = float(item.text or "")
            except ValueError as exc:
                raise ValueError("Publication GeoTIFF scale/offset must be identity") from exc
            if value != (1 if role == "scale" else 0):
                raise ValueError("Publication GeoTIFF scale/offset must be identity; decode values before publishing")
        if item.get("name") == "observation_year" and (item.text or "").strip() != year.strip():
            raise ValueError("Publication source year differs from the GeoTIFF observation_year tag")


@dataclass(frozen=True)
class PublicationSource:
    """A single-band GeoTIFF and its declared observation provenance.

    Nodata is defined by the file mask/nodata value. It is never a zero count.
    Year/edition describe this file, not a request to relabel another edition.
    The caller must supply the corresponding dataset when changing year.
    An embedded observation_year tag, when present, must match the declaration;
    absent tags leave year provenance with the caller. Scale/offset must be identity.
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
    camera: OrbitCamera | None = None


@dataclass(frozen=True)
class PublicationInset:
    """A linked source-grid view in an absolute page rectangle (pixel edges).

    Reserve space for this rectangle in the layout margins. Maps are rendered
    at their displayed size, with the same source, classification and missing key.
    """
    inset_id: str
    title: str
    bounds: BBox
    rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class PublicationLayout:
    """Reusable page geometry with linked maps, independent of source edition."""
    config: MapPlateConfig
    insets: Sequence[PublicationInset] = ()


@dataclass(frozen=True)
class PublicationSubject:
    """An explicit dataset and encoding for a subject-driven series."""
    subject_id: str
    title: str
    source: PublicationSource
    breaks: Sequence[float]
    palette: Sequence[Sequence[int]]


def publication_output_support() -> dict[str, Any]:
    """Route-specific capabilities; these declarations are not inspection receipts."""
    return {
        "publication_png": {
            "formats": ["PNG"], "resolution": "Exact configured pixel dimensions; maps rendered at displayed size",
            "colour": "sRGB-encoded RGBA8; no ICC/press proof", "editable_text": False,
            "pdf": False, "svg": False, "print_colour": False,
            "losses": ["Flattened text and maps", "No georeferencing in PNG", "No physical-print qualification"],
        },
        "map_plate_png": {
            "formats": ["PNG"], "resolution": "Configured pixels; input maps may be resized nearest-neighbour",
            "colour": "Caller-supplied RGBA8; no colour conversion", "editable_text": False,
            "pdf": False, "svg": False, "print_colour": False,
            "losses": ["Flattened text and maps", "Resampling may alter line widths"],
        },
        "map_plate_jpeg": {
            "formats": ["JPEG"], "resolution": "Configured pixels; input maps may be resized nearest-neighbour",
            "colour": "RGB8; alpha discarded; lossy compression", "editable_text": False,
            "pdf": False, "svg": False, "print_colour": False,
            "losses": ["Flattened text and maps", "Alpha discarded", "Lossy colours and edges"],
        },
    }


@dataclass
class PublicationRecipe:
    """A linked layout and native classification in supported projected metre CRSs.

    Extents follow source cell edges. ``screen`` is a north-up plan view;
    ``flat-perspective`` requires a square extent and explicit page camera,
    with every corner ray hitting the actual flat terrain. Perspective scale
    is omitted. PNG is display RGBA8, without press-colour/vector qualification.
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
    layout: PublicationLayout | None = None

    def render_flyover(self, frames: Sequence[PublicationExtent], directory: str | Path,
                       *, layout: PublicationLayout, base_seed: int,
                       cache: str | Path | None = None) -> dict[str, Any]:
        """Publish an explicit camera sequence through CHRONOS, with linked bookends.

        Frames are supplied by the camera-authoring workflow; this method does
        not interpolate cameras or select observations. Every page consumes
        its actual CHRONOS frame, and title/end are the first/last final pages.
        """
        from ._publication_flyover import _render_publication_flyover
        return _render_publication_flyover(self, frames, directory, layout=layout,
                                          base_seed=base_seed, cache=cache)

    def render_times(self, track: ObservationTrack, presentation_times: Sequence[datetime | str],
                     pages: Sequence[PublicationExtent], directory: str | Path,
                     *, cache: str | Path | None = None) -> list[dict[str, Any]]:
        """Consume C04 selections, including visible missing pages and age captions.

        Each observation payload must contain a PublicationSource description
        under ``source``. Selection/validity/hold rules belong exclusively to C04.
        Stale series folders/pages are refused before writing. The selection
        table is replaced atomically after successful rendering; individual
        files are atomic, but the whole series is not a transaction.
        """
        times, pages = tuple(presentation_times), tuple(pages)
        if not times:
            raise ValueError("A time series requires declared presentation times")
        _distinct_ids([p.page_id for p in pages])
        selected = track.select(times)
        sources = [None if s.record is None else PublicationSource(**s.record.to_dict()["payload"]["source"])
                   for s in selected]
        for selection, source in zip(selected, sources):
            if source is not None:
                self._check_selection_source(source, selection)
        output = Path(directory)
        table = track.selection_table(times)
        table["outputs"] = [{"directory": f"time-{i}", "page_ids": [p.page_id for p in pages]}
                            for i in range(len(selected))]
        expected = {row["directory"] for row in table["outputs"]}
        expected_files = {f"{p.page_id}.{ext}" for p in pages for ext in ("png", "json")}
        stale = [p for p in output.glob("time-*") if p.name not in expected]
        for name in expected:
            stale.extend(p for p in (output / name).glob("*")
                         if p.suffix.lower() in {".png", ".json"} and p.name not in expected_files)
        if stale:
            raise ValueError("Stale time-series outputs are not in the selection table; use a fresh directory: "
                             + ", ".join(str(p) for p in sorted(stale)))
        output.mkdir(parents=True, exist_ok=True)
        receipts = []
        for index, (selection, source) in enumerate(zip(selected, sources)):
            destination = output / f"time-{index}"
            if source is None:
                receipts.extend(self._missing_pages(pages, destination, selection))
            else:
                recipe = replace(self, source=source)
                receipts.extend(recipe.render_series(pages, destination, cache=cache, selection=selection))
        with tempfile.TemporaryDirectory(prefix=".selection-", dir=output) as staging:
            table_path = Path(staging) / "selection-table.json"
            table_path.write_text(json.dumps(table, indent=2, allow_nan=False), encoding="utf8")
            table_path.replace(output / "selection-table.json")
        return receipts

    @staticmethod
    def _check_selection_source(source, selection):
        if (selection.record is None or
                selection.record.to_dict()["payload"].get("source") != source.description()):
            raise ValueError("Publication source must be the C04 selected record's source")
        if source.year.strip() != selection.record.observation_year:
            raise ValueError("Publication source year must match the observation year in its declared timezone")

    def _missing_pages(self, pages, directory, selection):
        cfg = self.layout.config if self.layout is not None else self.config
        directory.mkdir(parents=True, exist_ok=True)
        receipts = []
        for page in pages:
            # Deliberately no map, legend or source: this is a missing-observation
            # notice, never a placeholder native frame or relabelled observation.
            image = MapPlate(cfg).compose()
            _text_block(image, page.title, (30, 30), cfg.width - 60, cfg.height - 60, 38)
            _text_block(image, selection.caption, (30, cfg.margin[0]), cfg.width - 60,
                        cfg.height - cfg.margin[0] - 30, 25)
            png = directory / f"{page.page_id}.png"
            receipt = {"page_id": page.page_id, "source": None, "selection": selection.to_dict(),
                       "caption": selection.caption, "map_rendered": False,
                       "output_support": publication_output_support()["publication_png"],
                       "output": {"path": str(png), "size": [cfg.width, cfg.height], "format": "RGBA8 PNG",
                                  "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()}}
            with tempfile.TemporaryDirectory(prefix=".missing-", dir=directory) as staging:
                temp = Path(staging)
                save_png(temp / "page.png", image)
                (temp / "page.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf8")
                (temp / "page.png").replace(png)
                (temp / "page.json").replace(png.with_suffix(".json"))
            receipts.append(receipt)
        return receipts

    def render_bookends(self, first: PublicationExtent, last: PublicationExtent, directory: str | Path,
                        *, layout: PublicationLayout, cache: str | Path | None = None) -> list[dict[str, Any]]:
        """Render matching title/end PNGs at a declared final frame layout.

        Both use one source/style snapshot and the same linked inset layout.
        This does not encode a movie or claim inspection of its moving frames.
        """
        recipe = replace(self, layout=layout)
        return recipe.render_series(
            [replace(first, page_id="title"), replace(last, page_id="end")], directory, cache=cache)

    def render_subjects(self, subjects: Sequence[PublicationSubject], pages: Sequence[PublicationExtent],
                        directory: str | Path, *, cache: str | Path | None = None) -> list[dict[str, Any]]:
        """Apply the same layout/extents to declared subjects; no implicit reclassification."""
        subjects, pages = tuple(subjects), tuple(pages)
        _distinct_ids([s.subject_id for s in subjects])
        if any(not s.title.strip() for s in subjects):
            raise ValueError("Publication subjects require nonempty titles")
        receipts = []
        for subject in subjects:
            recipe = replace(self, source=subject.source, breaks=subject.breaks, palette=subject.palette)
            selected = [replace(p, title=f"{subject.title} / {p.title}") for p in pages]
            receipts.extend(recipe.render_series(selected, Path(directory) / subject.subject_id, cache=cache))
        return receipts

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
        _check_source_metadata(path, self.source.year)
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
        colours = np.array(list(style.palette.values()), np.int16)
        for index, colour in enumerate(colours[:-1]):
            if np.any(np.max(np.abs(colours[index + 1:] - colour), axis=1) <= _LEGEND_TOLERANCE):
                raise ValueError("Publication legend keys must be distinguishable beyond the frame colour tolerance")
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

    def _page_scene(self, classes, style, page, size, description):
        scene = self._scene(classes, style, page.bounds, size, description)
        uv = None
        if self.camera_mode == "flat-perspective":
            if page.camera is None or page.bounds.width != page.bounds.height:
                raise ValueError("Flat perspective requires an explicit camera and square ground extent")
            from ._map_scene_projection import flat_ground_footprint_uv
            scene.recipe.camera = page.camera
            scene.recipe.terrain.metadata = {**scene.recipe.terrain.metadata,
                "width": classes.shape[1], "height": classes.shape[0]}
            scene.recipe.lighting.settings = {"unlit_albedo": True, "cli_params": {"camera_mode": "mesh:zup"}}
            uv = flat_ground_footprint_uv(scene.recipe)
        return scene, uv

    @staticmethod
    def _render(scene, path, *, cache, style):
        scene.render(str(path), cache=cache)
        pixels = load_png_rgba(path)
        deviation = PublicationRecipe._check_pixels(scene, pixels, style)
        return pixels, {"backend": scene.last_render_backend, "metadata": scene.last_render_metadata,
                        "map_pixels": "native unlit albedo frame, sRGB-encoded RGBA8",
                        "max_legend_deviation_levels": deviation}

    @staticmethod
    def _check_pixels(scene, pixels, style):
        if pixels.shape != (scene.recipe.output.height, scene.recipe.output.width, 4):
            raise ValueError("Native publication frame dimensions do not match the requested viewport")
        # Fresh or reused, every drawn colour must be a legend colour, up to the
        # 8-bit rounding of the native half-float albedo; otherwise the legend lies.
        palette = np.array(list(style.palette.values()), np.int16)
        colours = np.unique(pixels.reshape(-1, 4), axis=0).astype(np.int16)
        deviation = int(np.abs(colours[:, None] - palette[None]).max(axis=2).min(axis=1).max())
        if deviation > _LEGEND_TOLERANCE:
            raise ValueError(f"Native map colours differ from the legend by {deviation} levels")
        return deviation

    def render_series(self, pages: Sequence[PublicationExtent], directory: str | Path,
                      *, cache: str | Path | None = None,
                      selection: ObservationSelection | None = None) -> list[dict[str, Any]]:
        """Regenerate linked PNGs and JSON receipts. Never trust an existing PNG.

        ``cache`` is passed to MapScene's validated cache, not a publication
        image cache. Without it every main/locator map renders fresh. A failed
        render raises; a previous output is never returned as the new result.
        Source files must not be modified concurrently during this call.
        """
        if self.camera_mode not in {"screen", "flat-perspective"}:
            raise ValueError("Unsupported publication camera footprint: use screen or flat-perspective")
        if selection is not None:
            self._check_selection_source(self.source, selection)
        if self.layout is not None:
            # Resolve reusable geometry without mutating the caller's recipe.
            return replace(self, config=self.layout.config, layout=None)._render_layout_series(
                pages, directory, cache=cache, insets=tuple(self.layout.insets), selection=selection)
        return self._render_layout_series(pages, directory, cache=cache, insets=(), selection=selection)

    def _render_layout_series(self, pages, directory, *, cache, insets, selection, flyover_frames=None):
        pages = tuple(pages)
        if not pages or len({p.page_id.casefold() for p in pages}) != len(pages):
            raise ValueError("Publication requires nonempty, distinct page IDs")
        for page in pages:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", page.page_id) or not page.title.strip():
                raise ValueError("Publication page needs a safe basename and nonempty title")
            from ._map_scene_render import _text_outline_metrics
            title_size = _text_outline_metrics(f"{page.title} / {self.source.year}", 38)
            if title_size[0] > self.config.width - 30 or title_size[1] > self.config.margin[0] - 30:
                raise ValueError("Publication title cannot fit; increase the publication size/margins")
            if self.camera_mode == "flat-perspective" and (page.camera is None or page.bounds.width != page.bounds.height):
                raise ValueError("Flat perspective requires an explicit camera and square ground extent")
            if self.camera_mode == "screen" and page.camera is not None:
                raise ValueError("A publication camera requires flat-perspective mode")
        values, valid, grid, resolution, description = self._read()
        locator_window = self._window(self.locator_bounds, grid, resolution, values.shape)
        windows = [self._window(p.bounds, grid, resolution, values.shape) for p in pages]
        inset_windows = [self._window(i.bounds, grid, resolution, values.shape) for i in insets]
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
        if insets:
            _distinct_ids([i.inset_id for i in insets])
            occupied = [(left, top, cfg.width - right, cfg.height - bottom),
                        (cfg.width - right, top, cfg.width, cfg.height - bottom),
                        (0, 0, cfg.width, top), (0, cfg.height - bottom, cfg.width, cfg.height)]
            for inset in insets:
                r = inset.rect
                if (len(r) != 4 or any(not isinstance(v, int) for v in r)
                        or not 0 <= r[0] < r[2] <= cfg.width or not 0 <= r[1] < r[3] <= cfg.height):
                    raise ValueError("Inset rectangle must have integer pixel edges inside the page")
                if any(r[0] < b[2] and b[0] < r[2] and r[1] < b[3] and b[1] < r[3] for b in occupied):
                    raise ValueError("Inset rectangle overlaps map, furniture or another inset; reserve layout margin space")
                if not inset.title.strip():
                    raise ValueError("Inset title must be nonempty")
                occupied.append(r)
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
            linked_insets = []
            for inset, window in zip(insets, inset_windows):
                x0, y0, x1, y1 = inset.rect
                panel = np.full((y1 - y0, x1 - x0, 4), 255, np.uint8)
                heading = _text_block(panel, inset.title, (0, 0), x1 - x0, y1 - y0, 22)
                available = (x1 - x0, y1 - y0 - heading)
                if min(available) <= 0:
                    raise ValueError("Inset rectangle leaves no space for its map")
                inset_size = self._size(inset.bounds, available)
                inset_scene = self._scene(classes[window], style, inset.bounds, inset_size, description)
                pixels, render = self._render(inset_scene, temp / "inset.png", cache=cache, style=style)
                iy = heading + (available[1] - inset_size[1]) // 2
                ix = (available[0] - inset_size[0]) // 2
                panel[iy:iy + inset_size[1], ix:ix + inset_size[0]] = pixels
                linked_insets.append((inset, panel, {
                    "inset_id": inset.inset_id, "title": inset.title, "bounds": asdict(inset.bounds),
                    "rect": list(inset.rect), "map_rect": [x0 + ix, y0 + iy, x0 + ix + inset_size[0], y0 + iy + inset_size[1]],
                    "id": _identity([inset_scene.to_dict(), asdict(inset)]), "render": render,
                    "scale": "omitted", "source_id": source_id, "classification_id": classification_id,
                }))
            for page, window in zip(pages, windows):
                size = self._size(page.bounds, map_area)
                scene, uv = self._page_scene(classes[window], style, page, size, description)
                if flyover_frames is None:
                    main, rendered = self._render(scene, temp / "main.png", cache=cache, style=style)
                else:
                    frame = flyover_frames[page.page_id]
                    if frame["scene_id"] != _identity(scene.to_dict()):
                        raise ValueError("Publication scene changed after its CHRONOS frame was rendered")
                    main = load_png_rgba(frame["path"])
                    if hashlib.sha256(main.tobytes()).hexdigest() != frame["pixel_hash"]:
                        raise ValueError("CHRONOS frame pixels differ from their retained manifest")
                    deviation = self._check_pixels(scene, main, style)
                    rendered = {"backend": "chronos_mapscene", "metadata": frame["binding"],
                                "map_pixels": "CHRONOS native unlit albedo frame, sRGB-encoded RGBA8",
                                "max_legend_deviation_levels": deviation}
                footprint = [[page.bounds.west, page.bounds.north], [page.bounds.east, page.bounds.north],
                             [page.bounds.east, page.bounds.south], [page.bounds.west, page.bounds.south]]
                if self.camera_mode == "flat-perspective":
                    footprint = [[page.bounds.west + u * page.bounds.width,
                                  page.bounds.north - v * page.bounds.height] for u, v in uv]
                # Screen mode maps the complete UV rectangle to the viewport.
                # This is its actual footprint, not an oblique-camera bbox.
                pixel_footprint = [[(x - self.locator_bounds.west) / self.locator_bounds.width * locator_size[0],
                                    (self.locator_bounds.north - y) / self.locator_bounds.height * locator_size[1]]
                                   for x, y in footprint]
                mpp = ScaleBar.compute_meters_per_pixel(page.bounds, size[0]) if self.camera_mode == "screen" else None
                if mpp is not None and (not np.isfinite(mpp) or mpp <= 0):
                    raise ValueError("Publication ground scale is unavailable for this extent")
                caption = (f"{self.source.name} | {self.source.year} | {self.source.edition}\n"
                           f"{self.source.uri}\n"
                           "Flat plan view / grid north up. Scale: WGS84 ground distance along the map centre line; varies across the map.")
                if self.camera_mode == "flat-perspective":
                    caption = (f"{self.source.name} | {self.source.year} | {self.source.edition}\n"
                               f"{self.source.uri}\nFlat-terrain perspective. Scale omitted: ground distance per pixel varies across the view.")
                if selection is not None:
                    caption += "\n" + selection.caption
                plate = MapPlate(cfg)
                plate.set_map_region(main, page.bounds)
                plate.add_title(f"{page.title} / {self.source.year}", font_size=38)
                sidebar, locator_rect = _sidebar(styled.legend, locator, pixel_footprint, right - 30, map_area[1])
                plate.add_legend(sidebar)
                map_rect = _drawn_map_rect(cfg, size)
                scale_label = scale_span = scale_rect = None
                if self.camera_mode == "screen":
                    scale_bar = ScaleBar(mpp, ScaleBarConfig(width_px=250, height_px=30,
                                             font_size=20, padding=12))
                    scale_image = scale_bar.render()
                    _, scale_label, _ = scale_bar.render_geometry()
                    border = np.flatnonzero(np.all(scale_image[scale_bar.config.padding, :, :3] ==
                                                   scale_bar.config.border_color[:3], axis=1))
                    scale_span = int(border[-1] - border[0] + 1)  # Pixel-edge span, not centre-to-centre.
                    number, unit = scale_label.split()
                    if unit not in {"m", "km"}:
                        raise ValueError("Publication scale bar label has unsupported units")
                    labelled_metres = float(number) * {"m": 1, "km": 1000}[unit]
                    if (not np.isfinite(labelled_metres) or labelled_metres <= 0
                            or abs(labelled_metres - scale_span * mpp) > mpp):
                        raise ValueError("Publication scale bar label differs from its drawn distance by more than one pixel")
                    # MapPlate centres the map inside its region and anchors furniture
                    # to the region, not the image. Place the bar on the drawn map,
                    # whose pixels must be the unresampled frame the scale describes.
                    sh, sw = scale_image.shape[:2]
                    scale_rect = [map_rect[0] + 20, map_rect[3] - 20 - sh, map_rect[0] + 20 + sw, map_rect[3] - 20]
                    if scale_rect[2] > map_rect[2] or scale_rect[1] < map_rect[1]:
                        raise ValueError("Publication scale bar cannot fit on the map; increase the publication size")
                image = plate.compose()
                for inset, panel, _ in linked_insets:
                    plate._blit(image, panel, inset.rect[0], inset.rect[1])
                if self.camera_mode == "screen":
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
                if linked_insets:
                    dependencies["insets"] = {"id": _identity([r["id"] for _, _, r in linked_insets]),
                                              "depends_on": [source_id, classification_id, "layout inset views"]}
                png = output / f"{page.page_id}.png"
                receipt = {"page_id": page.page_id, "title": page.title, "source": description,
                           "dependencies": dependencies, "legend": styled.legend, "classification": table,
                           "caption": caption, "footprint": {"kind": "flat_screen_viewport", "crs": grid.crs,
                           "coordinates": footprint, "locator_pixels": pixel_footprint},
                           "scale": {"meters_per_pixel": mpp, "applies_to": "horizontal map centre line",
                                     "bar_label": scale_label, "bar_pixel_span": scale_span,
                                     "bar_rect": scale_rect},
                           "map_rect": map_rect,
                           "map_bounds": asdict(page.bounds),
                           "main_viewport": list(size), "locator_viewport": list(locator_size),
                           "locator_rect": [cfg.width - right + 10 + locator_rect[0], top + locator_rect[1],
                                            cfg.width - right + 10 + locator_rect[2], top + locator_rect[3]],
                           "main_render": rendered, "locator_render": locator_render,
                           "insets": [r for _, _, r in linked_insets],
                           "output_support": publication_output_support()["publication_png"],
                           "output": {"path": str(png), "size": [cfg.width, cfg.height], "format": "RGBA8 PNG",
                                      "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()},
                           "limitations": ["Flat north-up plan views only", "Grid-aligned extents only",
                               "One classified source; file mask is the missing-data rule",
                               "No perspective scale, moving frames, vector text or press-colour qualification"]}
                if selection is not None:
                    receipt["selection"] = selection.to_dict()
                if flyover_frames is not None:
                    receipt["flyover"] = flyover_frames[page.page_id]["binding"]
                if self.camera_mode == "flat-perspective":
                    receipt["footprint"]["kind"] = "flat_terrain_perspective_ground_intersection"
                    receipt["dependencies"]["locator"]["depends_on"] = [locator_id, "actual camera ground footprint"]
                    receipt["scale"] = {"omitted": True, "reason": "Perspective ground distance per pixel varies across the view"}
                    receipt["dependencies"]["scale"] = {"id": _identity(receipt["scale"]), "depends_on": ["view qualification"]}
                    receipt["camera"] = page.camera.to_dict()
                    receipt["limitations"] = ["Flat terrain only; square ground extent", "Fully covered perspective viewport only",
                        "No effects, offline sampling or clipmap", "Scale omitted", "No vector text or press-colour qualification"]
                # Compose and validate everything before replacing this page.
                save_png(temp / "page.png", image)
                (temp / "page.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf8")
                (temp / "page.png").replace(png)
                (temp / "page.json").replace(output / f"{page.page_id}.json")
                receipts.append(receipt)
        return receipts


def _distinct_ids(values):
    if (not values or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in values)
            or len({v.casefold() for v in values}) != len(values)):
        raise ValueError("Publication requires nonempty, distinct safe IDs")


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
    return np.asarray(image), [15, y, 15 + w, y + h]
