"""Compile fixed-camera observation maps for MapScene, CHRONOS and C06."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ._canonical_json import canonical_json_bytes
from .alignment import _affine_from_metadata, _affine_to_metadata, _require_rasterio
from .map_scene import (
    MapFurnitureLayer, MapScene, RasterOverlay, SceneRecipe, VectorOverlay,
    _terrain_alignment_grid,
)
from .observation_time import ObservationRecord, ObservationSelection, ObservationTrack, _stamp, _utc
from .publication import PublicationSource, _check_observation_source

if TYPE_CHECKING:
    from .chronos import FlythroughManifest

__all__ = ["ObservationRasterStyle", "ObservationEvents", "compile_observation_map",
           "CompiledObservationMap"]

_RASTER_ID = "observation-raster"
_EVENT_ID = "observation-events"
_METADATA_KEY = "observation_map"


@dataclass(frozen=True)
class ObservationRasterStyle:
    """A fixed linear stretch from physical units to grayscale or RGB."""

    domain: tuple[float, float]

    def __post_init__(self) -> None:
        lo, hi = self.domain
        if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(hi - lo) and lo < hi):
            raise ValueError("Observation raster domain must be finite and increasing")
        object.__setattr__(self, "domain", (float(lo), float(hi)))


@dataclass(frozen=True)
class ObservationEvents:
    """Prepared inline GeoJSON observations and a trailing presentation window.

    Each record's payload contains a Feature and source provenance. The CRS
    must already match the map. Equal acquisition times are allowed for events.
    """

    records: Sequence[ObservationRecord]
    lookback: timedelta
    crs: str
    label: str
    style: Mapping[str, Any]

    def __post_init__(self) -> None:
        records = ObservationTrack(self.records, "no-data").records
        if self.lookback <= timedelta(0) or not self.crs or not self.label.strip():
            raise ValueError("Observation events require a positive lookback, CRS and label")
        for record in records:
            payload = record.to_dict()["payload"]
            if (not isinstance(payload.get("feature"), dict)
                    or payload["feature"].get("type") != "Feature"
                    or not isinstance(payload["feature"].get("geometry"), dict)
                    or not isinstance(payload.get("source"), dict)
                    or not payload["source"]):
                raise ValueError("Each vector event requires an inline GeoJSON Feature and source provenance")
        object.__setattr__(self, "records", records)


@dataclass(frozen=True)
class CompiledObservationMap:
    """Frozen recipes shared by sequential execution, indexed access and replay."""

    _recipes: tuple[bytes, ...]
    _table: bytes

    def frame(self, index: int) -> MapScene:
        if not 0 <= index < len(self._recipes):
            raise IndexError(f"Observation map frame {index} is out of range")
        scene = MapScene(MapScene._recipe_from_dict(json.loads(self._recipes[index])))
        scene.compile_plan()
        return scene

    def selection_table(self) -> dict[str, Any]:
        return json.loads(self._table)

    def render(self, directory: str | Path, *, base_seed: int, samples: int = 1,
               frames: Sequence[int] | None = None, cache: str | Path | None = None) -> FlythroughManifest:
        """Render the chosen original frame indices through CHRONOS."""
        from .chronos import render_flythrough

        indices = tuple(range(len(self._recipes))) if frames is None else tuple(frames)
        if len(set(indices)) != len(indices):
            raise ValueError("Observation map frame indices must be distinct")
        return render_flythrough({i: self.frame(i) for i in indices}, base_seed=base_seed,
                                 samples=samples, out_dir=directory, cache=cache)


def _freeze_raster(source: PublicationSource, grid, style, directory: Path) -> dict[str, Any]:
    rasterio = _require_rasterio()
    path = Path(source.path)
    source_bytes = path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    expected_hash = source.provenance.get("sha256")
    if expected_hash is not None and source_hash != expected_hash:
        raise ValueError(f"Observation source hash differs from its declaration: {path}")
    with rasterio.open(path) as original:
        if len(original.files) != 1:
            raise ValueError("Prepared observation GeoTIFF must embed its mask and metadata")
    with rasterio.MemoryFile(source_bytes) as memory, memory.open() as dataset:
        transform = _affine_from_metadata(grid["transform"])
        if (dataset.driver != "GTiff" or dataset.count not in (1, 3) or dataset.crs != grid["crs"]
                or dataset.shape != (grid["height"], grid["width"])
                or dataset.transform != transform):
            raise ValueError(f"Observation GeoTIFF must contain one or three bands on the exact terrain grid: {path}")
        data = dataset.read()
        valid = (dataset.read_masks() > 0).all(axis=0)
        scales = tuple(dataset.scales)
        offsets = tuple(dataset.offsets)
        if not all(math.isfinite(n) for n in (*scales, *offsets)):
            raise ValueError("Observation scale and offset must be finite")
        values = data[:, valid].astype(np.float64) * np.asarray(scales)[:, None] + np.asarray(offsets)[:, None]
        if not np.isfinite(values).all():
            raise ValueError("Valid observation pixels must be finite")
        lo, hi = style.domain
        # Colour bytes under the internal invalid mask carry no observations.
        rgb = np.zeros(data.shape, np.uint8)
        rgb[:, valid] = np.rint(np.clip((values - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        if dataset.count == 1:
            rgb = np.repeat(rgb, 3, axis=0)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.MemoryFile() as snapshot:
            with snapshot.open(driver="GTiff", width=dataset.width, height=dataset.height,
                               count=3, dtype="uint8", crs=dataset.crs, transform=transform,
                               photometric="RGB", compress="deflate") as output:
                output.write(rgb)
                output.write_mask(valid)
            snapshot_bytes = snapshot.read()
        provenance = {
            "source": source.description(), "sha256": source_hash,
            "crs": dataset.crs.to_string(), "geotransform": _affine_to_metadata(transform),
            "shape": list(dataset.shape), "scale": scales, "offset": offsets,
            "valid_pixels": int(valid.sum()), "excluded_pixels": int(valid.size - valid.sum()),
        }
    digest = hashlib.sha256(snapshot_bytes).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = (directory / f"{digest}.tif").resolve()
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=directory) as stage:
            scratch = Path(stage) / target.name
            scratch.write_bytes(snapshot_bytes)
            scratch.replace(target)
    if target.read_bytes() != snapshot_bytes:
        raise ValueError(f"Observation raster snapshot was modified: {target}")
    return dict(provenance, snapshot_path=str(target), snapshot_sha256=digest)


def _event_window(events: ObservationEvents, time: datetime, grid):
    end = _utc(time)
    start = end - events.lookback
    selected = sorted(
        (r for r in events.records if start < _utc(r.observed_at) <= end
         and _utc(r.valid_from) <= end < _utc(r.valid_until)),
        key=lambda r: (_utc(r.observed_at), r.record_id),
    )
    rows = [r.to_dict() for r in selected]
    layer = None
    if rows:
        preparation_grid = {
            "width": grid["width"], "height": grid["height"],
            "geotransform": _affine_to_metadata(_affine_from_metadata(grid["transform"])),
        }
        layer = VectorOverlay(_EVENT_ID, features=[row["payload"]["feature"] for row in rows],
                              crs=events.crs, style=events.style,
                              metadata={"preparation_grid": preparation_grid})
    count = len(rows)
    unit = "event" if count == 1 else "events"
    summary = f"{events.label} | {count} reported {unit}"
    if count == 0:
        summary += " (coverage unknown)"
    caption = f"{summary} | ({_stamp(start)}, {_stamp(end)}]"
    return layer, {"start_exclusive": _stamp(start), "end_inclusive": _stamp(end),
                   "records": rows, "caption": caption}


def _raster_caption(selection: ObservationSelection, raster: dict | None) -> str:
    if raster is None:
        return "Missing raster | No acquisition selected"
    detail = selection.caption.split(" | ", 1)[1]
    age = selection.age_seconds
    age_display = (f"{age / 86400:.2f} days" if age >= 86400 else
                   f"{age / 3600:.2f} hours" if age >= 3600 else
                   f"{age:.6g} seconds")
    detail = detail.split(" | Age ", 1)[0] + f" | Age {age_display}"
    coverage = " | All pixels excluded" if not raster["valid_pixels"] else ""
    return f"{selection.record.record_id} | {detail}{coverage}"


def compile_observation_map(
    recipe: SceneRecipe,
    raster_track: ObservationTrack,
    presentation_times: Sequence[datetime | str],
    *,
    raster_style: ObservationRasterStyle,
    snapshot_dir: str | Path,
    events: ObservationEvents | None = None,
    interventions: Sequence[str] = (),
) -> CompiledObservationMap:
    """Compile selections, fixed styling, visible captions and source identities.

    The raster record uses C06's source description under payload["source"].
    Its prepared grayscale or RGB GeoTIFF stores scale/offset and an internal QA mask.
    Frames carry immutable styled snapshots. Recompile to author a replacement.
    """
    times = tuple(presentation_times)
    if not times or recipe.output is None or recipe.terrain.path is None:
        raise ValueError("Observation maps require presentation times, output and a file-backed terrain grid")
    if recipe.pass_inputs:
        raise ValueError("Observation maps require a self-contained MapScene recipe")
    from .map_scene import _mapscene_native_effects
    if "taa" in _mapscene_native_effects(recipe):
        raise ValueError("Observation maps require independent frames without temporal antialiasing")
    reserved = {_RASTER_ID, _EVENT_ID}
    if any(getattr(layer, "layer_id", None) in reserved or
           isinstance(layer, RasterOverlay) and layer.style is not None for layer in recipe.layers):
        raise ValueError("Base scene cannot use observation layer IDs or mix thematic and RGB overlays")
    rasterio = _require_rasterio()
    base = MapScene(recipe).recipe
    crs = base.target_crs or base.terrain.crs
    grid = _terrain_alignment_grid(base.terrain, target_crs=crs)
    if grid is None:
        raise ValueError("Observation maps require a georeferenced terrain grid")
    crs = rasterio.crs.CRS.from_user_input(grid["crs"])
    if events is not None and crs != rasterio.crs.CRS.from_user_input(events.crs):
        raise ValueError("Prepared vector observations must use the terrain CRS")
    selections = raster_track.select(times)
    sources = {}
    recipes, rows = [], []
    for index, selection in enumerate(selections):
        raster_layer = raster = None
        if selection.record is not None:
            record = selection.record
            if record.record_id not in sources:
                source = PublicationSource(**record.to_dict()["payload"]["source"])
                _check_observation_source(source, selection)
                sources[record.record_id] = _freeze_raster(source, grid, raster_style, Path(snapshot_dir))
            raster = sources[record.record_id]
            raster_layer = RasterOverlay(
                _RASTER_ID, path=raster["snapshot_path"], crs=raster["crs"],
                metadata={"observation_snapshot_sha256": raster["snapshot_sha256"]},
            )
        event_layer, window = (None, None) if events is None else _event_window(
            events, selection.presentation_time, grid)
        captions = [f"Time {_stamp(selection.presentation_time)}", _raster_caption(selection, raster)]
        if window is not None:
            captions.append(window["caption"])
        row = {"frame_index": index, "raster_selection": selection.to_dict(), "raster": raster,
               "vector_window": window, "rgb_domain": raster_style.domain, "captions": captions}
        furniture = base.map_furniture or MapFurnitureLayer()
        frame = MapScene(replace(
            base, layers=(*base.layers, *(layer for layer in (raster_layer, event_layer) if layer is not None)),
            map_furniture=replace(furniture, metadata=dict(furniture.metadata or {}, **{_METADATA_KEY: row})),
        ))
        frame.compile_plan()
        recipes.append(canonical_json_bytes(frame.recipe.to_dict(), error_context="observation recipe"))
        rows.append(row)
    table = {"raster_policy": raster_track.policy, "raster_interval": "[valid_from, valid_until)",
             "event_interval": "(presentation - lookback, presentation] intersected with validity",
             "event_lookback_seconds": None if events is None else events.lookback.total_seconds(),
             "interventions": list(interventions), "frames": rows}
    return CompiledObservationMap(tuple(recipes), canonical_json_bytes(table, error_context="observation table"))
