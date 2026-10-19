"""Opt-in, single-extent Unreal Landscape interchange (no editor automation).

The encoding and component rules follow Epic's Landscape Technical Guide.
Package reconstruction is testable without Unreal; editor import is a distinct
acceptance step and is never certified by writing these files.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ._png import encode_png
from .preparation import PreparedScene, PreparationGrid, _fail, _json, _layer_raster, _require_rasterio, _sha

__all__ = ["LandscapeLayout", "export_landscape", "verify_landscape_import"]

_TECHNICAL_GUIDE = "https://dev.epicgames.com/documentation/en-us/unreal-engine/landscape-technical-guide-in-unreal-engine"
_IMPORT_GUIDE = "https://dev.epicgames.com/documentation/en-us/unreal-engine/importing-and-exporting-landscape-heightmaps-in-unreal-engine"


@dataclass(frozen=True)
class LandscapeLayout:
    quads_per_section: int
    sections_per_component: int
    components_x: int
    components_y: int

    def __post_init__(self) -> None:
        for value in (self.quads_per_section, self.sections_per_component, self.components_x, self.components_y):
            if isinstance(value, bool) or int(value) != value:
                _fail("invalid_landscape_layout", "Layout values must be integers.", "Specify integer section sizes and component counts.")
        # A section has a power-of-two number of vertices, up to 256.
        if self.quads_per_section not in {7, 15, 31, 63, 127, 255}:
            _fail("invalid_landscape_layout", "Unsupported section size.", "Use 7, 15, 31, 63, 127 or 255 quads per section (Epic component contract).")
        if self.sections_per_component not in {1, 4}:
            _fail("invalid_landscape_layout", "Components require one or four sections.", "Use 1 or 4 sections per component.")
        for count in (self.components_x, self.components_y):
            if isinstance(count, bool) or int(count) != count or count <= 0:
                _fail("invalid_landscape_layout", "Component counts must be positive integers.", "Specify the desired component grid.")

    @property
    def shape(self) -> tuple[int, int]:
        quads = self.quads_per_section * (2 if self.sections_per_component == 4 else 1)
        return (int(self.components_y) * quads + 1, int(self.components_x) * quads + 1)


def export_landscape(
    prepared: PreparedScene,
    directory: str | Path,
    *,
    layout: LandscapeLayout,
    origin: Sequence[float],
    categorical_source: str,
    material_layers: Mapping[int, str],
    void_fill_m: float | None = None,
) -> Path:
    """Write height.png (uint16), per-class weightmaps and explicit settings.

    ``origin`` is a declared (easting, northing, height-in-metres) geographic
    anchor in the terrain's projected CRS/datum. Horizontal CRS units are read
    from the CRS. No exaggeration, padding, clipping, tiling or new resampling
    occurs here; select the exact grid in the preparation recipe first.
    """
    terrain = next(layer for layer in prepared.layers.values() if layer.kind == "elevation")
    grid, terrain_data, terrain_valid = _layer_raster(terrain)
    if terrain_data.shape != layout.shape:
        _fail("landscape_dimensions", f"Layout requires {layout.shape}; terrain is {terrain_data.shape}.",
              "Prepare an explicit target grid of the required dimensions. Crop/pad/resample only in the recorded preparation recipe.")
    crs = _require_rasterio().crs.CRS.from_user_input(grid.crs)
    if not crs.is_projected:
        _fail("landscape_projected_crs", "Landscape requires a local projected grid.", "Reproject to a suitable declared projected CRS before export.")
    _, horizontal_to_m = crs.linear_units_factor
    gt = grid.geotransform
    if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
        _fail("landscape_grid_orientation", "Only north-up east-positive grids are supported.", "Prepare a north-up projected target grid; rotated/sheared/reversed transforms are not guessed.")
    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (3,) or not np.all(np.isfinite(origin_array)):
        _fail("landscape_origin", "A finite three-coordinate origin is required.", "Declare easting/northing in CRS units and height in metres in the measured datum.")
    heights = np.array(terrain_data, dtype=np.float64, copy=True)
    validity = np.asarray(terrain_valid, dtype=bool)
    if not np.all(validity):
        if void_fill_m is None or not math.isfinite(void_fill_m):
            _fail("landscape_voids", "Landscape cannot encode missing heights.", "Declare void_fill_m explicitly; validity.png retains the measured/filled distinction.")
        heights[~validity] = void_fill_m
    if not np.all(np.isfinite(heights)):
        _fail("landscape_nonfinite_height", "Heights must be finite after the void policy.", "Correct invalid heights or declare a finite void fill.")
    low, high = float(heights.min()), float(heights.max())
    if low == high:
        # Flat data uses the engine's documented default Z scale of 100.
        samples = np.full(heights.shape, 32768, dtype=np.uint16)
        step_m = 1.0 / 128.0
        intercept_m = low - 32768 * step_m
    else:
        step_m = (high - low) / 65535.0
        intercept_m = low
        if not math.isfinite(step_m) or step_m <= 0:
            _fail("landscape_height_range", "The source range cannot be encoded in float64.", "Prepare a representable physical height range.")
        samples = np.rint((heights - low) / step_m).clip(0, 65535).astype(np.uint16)
    reconstructed = intercept_m + samples.astype(np.float64) * step_m
    numeric_error = float(np.max(np.abs(reconstructed - heights)))
    # Account for arithmetic precision separately from quantization. This is
    # derived from the actual magnitude/float64 operations, not a product cap.
    floating_bound = float(np.spacing(max(abs(low), abs(high), abs(intercept_m)))) * 4
    quantization_bound = step_m / 2
    if numeric_error > quantization_bound + floating_bound:
        _fail("landscape_reconstruction", "Encoding exceeds its derived quantization bound.", "Choose a numerically representable height range.")
    category = prepared.layers.get(categorical_source)
    if category is None or category.kind != "categorical":
        _fail("landscape_category_grid", "Categorical data must share the exact terrain grid.", "Prepare the categorical source on the same target grid using nearest sampling.")
    category_grid, category_data, category_valid = _layer_raster(category)
    if not category_grid.same_as(grid) or category_data.shape != heights.shape:
        _fail("landscape_category_grid", "Categorical data must share the exact terrain grid.", "Prepare the categorical source on the same target grid using nearest sampling.")
    ids = set(int(v) for v in np.unique(category_data[category_valid]))
    normalized = {int(key): value for key, value in material_layers.items()}
    if ids != set(normalized):
        _fail("landscape_material_mapping", f"Classes {sorted(ids)} differ from mapped IDs {sorted(normalized)}.", "Provide one named Unreal material layer for every valid category and remove unused mappings.")
    names = list(normalized.values())
    if (any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in names)
            or len(set(n.casefold() for n in names)) != len(names)):
        _fail("landscape_material_names", "Layer names must be unique Unreal identifiers.", "Use distinct letter/underscore names followed by letters, digits or underscores.")
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "height.png").write_bytes(encode_png(samples))
    (root / "validity.png").write_bytes(encode_png(validity.astype(np.uint8) * 255))
    (root / "category_validity.png").write_bytes(encode_png(category_valid.astype(np.uint8) * 255))
    weightmaps = []
    for class_id, name in sorted(normalized.items()):
        weight = ((category_data == class_id) & category_valid).astype(np.uint8) * 255
        filename = f"layer_{name}.png"
        (root / filename).write_bytes(encode_png(weight))
        weightmaps.append({"class_id": class_id, "material_layer": name, "file": filename,
                           "layer_info": "non_weight_blended", "valid_class_vertices": int(np.count_nonzero(weight))})
    first_x, first_y = grid.xy(0, 0)
    scale_x, scale_y = gt[1] * horizontal_to_m * 100, -gt[5] * horizontal_to_m * 100
    scale_z = step_m * 128 * 100
    first_cm = [(first_x - origin_array[0]) * horizontal_to_m * 100,
                -(first_y - origin_array[1]) * horizontal_to_m * 100,
                (intercept_m + 32768 * step_m - origin_array[2]) * 100]
    settings = {
        "heightmap_file": "height.png", "width": grid.width, "height": grid.height,
        "quads_per_section": layout.quads_per_section, "sections_per_component": layout.sections_per_component,
        "components_x": layout.components_x, "components_y": layout.components_y,
        "scale_cm": [scale_x, scale_y, scale_z], "rotation_degrees": [0, 0, 0],
        "vertex_0_0_base_location_cm": first_cm, "flip_y": False,
        "enable_edit_layers": True, "world_partition": False,
        "section_base_rule": "ActorXY = vertex_0_0_base_location_cm.XY - minimum_component_SectionBaseXY * scale_cm.XY",
    }
    manifest = {
        "schema_version": 1, "destination": "unreal_landscape", "documentation_version": "5.8",
        "qualified_editor_versions": [], "editor_import_verified": False,
        "geographic": {"crs": grid.crs, "horizontal_units_to_m": horizontal_to_m,
                       "vertical_reference": prepared.recipe.vertical_reference, "vertical_units": "m",
                       "origin": origin_array.tolist(), "local_axes": ["east", "south", "up"],
                       "local_units": "cm", "grid": grid.to_dict(),
                       "sample_registration": "source pixel centres become Landscape vertices",
                       "world_to_local": "X=(E-originE)*horizontal_units_to_m*100; Y=-(N-originN)*horizontal_units_to_m*100; Z=(H-originH)*100",
                       "local_to_world": "E=originE+X/(100*horizontal_units_to_m); N=originN-Y/(100*horizontal_units_to_m); H=originH+Z/100"},
        "encoding": {"format": "16-bit grayscale PNG", "sample_min": 0, "sample_max": 65535,
                     "intercept_m": intercept_m, "step_m": step_m, "source_range_m": [low, high],
                     "reconstruction": "height_m = intercept_m + uint16_sample * step_m",
                     "unreal_reconstruction": "local_Z_cm = base_location_Z_cm + (uint16_sample - 32768) / 128 * scale_Z",
                     "quantization_bound_m": quantization_bound, "floating_bound_m": floating_bound,
                     "observed_max_error_m": numeric_error},
        "import_settings": settings, "weightmaps": weightmaps,
        "void_policy": {"fill_m": void_fill_m, "filled_vertices": int(np.count_nonzero(~validity)),
                        "validity_file": "validity.png", "category_validity_file": "category_validity.png"},
        "grid_operations": {"crop": "none", "padding": "none", "resampling": "none at export; see preparation report",
                            "seams": "single extent; shared component-edge vertices come from one height image"},
        "provenance": prepared.report,
        "losses": ["No editor version has been qualified by package creation.",
                   "Missing heights are filled only under the declared policy; validity does not create Landscape holes.",
                   "Weightmaps preserve class IDs through named binary masks; material graphs and appearance require editor assignment.",
                   "No World Partition, tiled import, geographic datum conversion or automatic Landscape creation is performed."],
        "sources": [_TECHNICAL_GUIDE, _IMPORT_GUIDE],
    }
    manifest["files"] = {p.name: _sha(p) for p in sorted(root.glob("*.png"))}
    (root / "landscape.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    (root / "IMPORT.md").write_text(_instructions(manifest), encoding="utf-8")
    return root


def _instructions(manifest: Mapping[str, Any]) -> str:
    settings = manifest["import_settings"]
    return ("# Landscape manual import\n\n"
            "Package encoding follows Unreal 5.8 documentation. No editor build is qualified yet.\n\n"
            "1. Open a non-World-Partition level. Enter Landscape mode, Import from File, and select height.png.\n"
            "2. Enter the exact width, height, section/component counts and scale below. Do not fit, resize, smooth or flip the image. Enable Edit Layers.\n"
            "3. Assign a Landscape material with the exact named layers below. For categorical meaning, use non-weight-blended Layer Info assets and matching material layer sampling (not interpolated class numbers). Import each named 8-bit mask.\n"
            "4. Create the Landscape. Inspect the minimum component SectionBaseX/SectionBaseY. Set actor XY using section_base_rule below, actor Z to vertex_0_0_base_location_cm[2], and rotation to zero. The actor pivot is not necessarily vertex (0,0).\n"
            "5. Compare independent geographic checkpoints, the first/last vertices and heights with landscape.json. Verify class masks in Layer Debug mode. Preserve category_validity.png: an unmapped vertex has no measured class.\n"
            "6. Test sculpting/painting on a separate edit layer, undo it, and record editor/plugin versions, manual steps and material assignment. Inspect the intended still and camera path for banding, cracks, class edges, material scale and LOD changes.\n"
            "7. Export measured world-space vertex positions in row-major (height,width,XYZ cm) form and use verify_landscape_import with a tolerance derived from the measurement method. A successful file decode alone is not editor acceptance.\n\n"
            "Exact settings:\n\n```json\n" + _json(settings) + "\n```\n\nMaterial layer bindings:\n\n```json\n"
            + _json(manifest["weightmaps"]) + "\n```\n\nHeight decoding and provenance: landscape.json.\n\n"
            + f"[Epic technical guide]({_TECHNICAL_GUIDE}) and [import guide]({_IMPORT_GUIDE}).\n")


def verify_landscape_import(
    directory: str | Path,
    imported_positions_cm: np.ndarray,
    *,
    editor_version: str,
    measurement_tolerance_m: float,
) -> dict[str, Any]:
    """Compare independently measured editor vertices; never infer their origin.

    This returns geographic/height evidence only. Material, editability, visual
    and camera-path checks remain separate. The caller records the actual editor
    build and a tolerance derived from its measurement procedure.
    """
    root = Path(directory).resolve()
    manifest = json.loads((root / "landscape.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("destination") != "unreal_landscape":
        _fail("unsupported_landscape_package", "Unrecognized package.", "Use Landscape package schema 1.")
    if not editor_version.strip() or not math.isfinite(measurement_tolerance_m) or measurement_tolerance_m < 0:
        _fail("invalid_editor_evidence", "Editor build and nonnegative finite measurement tolerance are required.", "Record the actual editor and derive tolerance from the measurement method.")
    for filename, expected in manifest["files"].items():
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha(path) != expected:
            _fail("landscape_asset_integrity", filename, "Restore the exported package before verification.")
    # The display PNG reader intentionally converts 16-bit data to RGBA8.
    # Use GDAL's independent lossless decoder for numeric reconstruction.
    with _require_rasterio().open(root / "height.png") as dataset:
        samples = dataset.read(1)
        if samples.dtype != np.uint16 or dataset.count != 1:
            _fail("landscape_height_encoding", "Height image is not single-band uint16.", "Restore the original exported heightmap.")
    grid = PreparationGrid(**manifest["geographic"]["grid"])
    positions = np.asarray(imported_positions_cm, dtype=np.float64)
    if positions.shape != (grid.height, grid.width, 3) or not np.all(np.isfinite(positions)):
        _fail("invalid_editor_samples", "Expected finite height x width x XYZ vertex positions.", "Measure each vertex in source image row/column order in local-world centimetres.")
    geographic = manifest["geographic"]
    origin = geographic["origin"]
    rows, cols = np.indices(samples.shape)
    gt = grid.geotransform
    expected_x = (gt[0] + (cols + .5) * gt[1] - origin[0]) * geographic["horizontal_units_to_m"]
    expected_y = -(gt[3] + (rows + .5) * gt[5] - origin[1]) * geographic["horizontal_units_to_m"]
    expected_z = manifest["encoding"]["intercept_m"] + samples.astype(np.float64) * manifest["encoding"]["step_m"] - origin[2]
    error_xy = np.hypot(positions[..., 0] / 100 - expected_x, positions[..., 1] / 100 - expected_y)
    error_z = np.abs(positions[..., 2] / 100 - expected_z)
    maximum_xy, maximum_z = float(error_xy.max()), float(error_z.max())
    if maximum_xy > measurement_tolerance_m or maximum_z > measurement_tolerance_m:
        _fail("landscape_editor_mismatch", f"XY error {maximum_xy} m; Z error {maximum_z} m.", "Correct spacing, axis orientation, section-base placement and Z scale before accepting the import.")
    return {"editor_version": editor_version, "measurement_tolerance_m": measurement_tolerance_m,
            "measured_vertices": grid.height * grid.width, "max_xy_error_m": maximum_xy,
            "max_z_error_m": maximum_z, "quantization_bound_m": manifest["encoding"]["quantization_bound_m"],
            "material_verified": False, "editability_verified": False, "visual_verified": False}
