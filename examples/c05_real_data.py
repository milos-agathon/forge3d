"""Recorded Rotterdam and Grindelwald C05 demonstrations, without an editor.

Acquire sources with c05_sources.py first. Staging is shared by both tools;
preparation, revision, diagnostics and Landscape checks use the public API.
Nothing in this example claims an Unreal import or absolute survey accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import time

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_origin
import pyogrio
import shapely
from PIL import Image

from _import_shim import ensure_repo_import
ensure_repo_import()
from forge3d import (PreparationGrid, PreparationSource, PreparationRecipe,
                     PreparationError, PreparedScene, prepare_scene,
                     LandscapeLayout, export_landscape)
from forge3d.preparation import _sha


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def peak_memory() -> int:
    import psutil
    return psutil.Process().memory_info().peak_wset


def implementation_identity() -> dict:
    """Hash the actual local implementation alongside each executed result."""
    repo = Path(__file__).resolve().parents[1]
    files = [repo / "python/forge3d" / name for name in
             ("preparation.py", "landscape.py", "map_scene.py", "alignment.py", "crs.py", "__init__.py")]
    files += [repo / "examples" / name for name in
              ("prepare_scene.py", "c05_real_data.py", "c05_sources.py", "c05_render.py", "c05_qgis.py", "c05_qgis_3d.py", "c05_report.py")]
    return {"code_sha256": {str(path): _sha(path) for path in files},
            "scope": "Local Python implementation at execution; native runtime identity is recorded separately."}


def geotiff(path, values, valid, crs, transform, **tags):
    values = np.asarray(values)
    bands = values[None] if values.ndim == 2 else np.moveaxis(values, -1, 0)
    with rasterio.open(path, "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1],
                       count=len(bands), dtype=bands.dtype, crs=crs, transform=transform,
                       compress="deflate") as ds:
        ds.write(bands)
        ds.write_mask(np.asarray(valid, dtype=np.uint8) * 255)
        ds.update_tags(AREA_OR_POINT="Area", **tags)


def stage(root: Path, cbs_zip: Path) -> None:
    started = time.perf_counter()
    common = root / "common"
    common.mkdir(exist_ok=False)
    sources = json.loads((root / "sources/sources.json").read_text(encoding="utf-8"))
    for item in sources["sources"]:
        if _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Source identity mismatch: {item['path']}")
    # Cell IDs, not a floating point rasterizer, determine the count placement.
    uri = "/vsizip/" + cbs_zip.as_posix() + "/cbs_vk100_2024_v2.gpkg"
    cells = pyogrio.read_dataframe(uri, bbox=(88000, 430000, 100000, 442000),
                                    columns=["crs28992res100m", "aantal_inwoners"])
    raw = np.zeros((120, 120), dtype=np.int32)  # absent values have no meaning; state and mask govern validity
    present = np.zeros(raw.shape, dtype=bool)
    selected = []
    for index, row in cells.iterrows():
        match = re.fullmatch(r"E(\d+)N(\d+)", row.crs28992res100m)
        if match is None:
            raise ValueError("Unexpected CBS cell identifier")
        x, y = (int(v) * 100 for v in match.groups())
        if tuple(row.geometry.bounds) != (x, y, x + 100, y + 100):
            raise ValueError("Cell code and independent polygon bounds disagree")
        if not (88000 <= x < 100000 and 430000 <= y < 442000):
            continue  # bbox queries also return cells merely touching the extent
        r, c = (441900 - y) // 100, (x - 88000) // 100
        if present[r, c]:
            raise ValueError("Duplicate CBS cell")
        raw[r, c], present[r, c] = row.aantal_inwoners, True
        selected.append(index)
    valid = present & (raw >= 0)
    states = np.where(valid, 2, np.where(present, 1, 0)).astype(np.uint8)
    geotiff(common / "rotterdam-cbs-2024-v2.tif", raw, valid, "EPSG:28992",
            from_origin(88000, 442000, 100, 100), units="people/cell")
    np.save(common / "cbs-present.npy", present, allow_pickle=False)
    np.save(common / "cbs-disclosure.npy", present & ~valid, allow_pickle=False)
    geotiff(common / "rotterdam-population-state.tif", states, np.ones(raw.shape, bool), "EPSG:28992",
            from_origin(88000, 442000, 100, 100), state_codes="0=absent;1=published suppressed code;2=published count")
    cells.loc[selected].to_file(common / "rotterdam-cbs.geojson", driver="GeoJSON")
    cbs = {"archive": str(cbs_zip), "sha256": _sha(cbs_zip), "edition": "2024 v2",
           "archive_date": "2026-08-27", "correction_published": "2026-08-28",
           "crs": "EPSG:28992", "cell_metres": 100, "source_fields": list(cells.columns),
           "selected_features": len(selected), "present_cells": int(present.sum()),
           "published_count_cells": int(valid.sum()), "disclosed_cells": int((present & ~valid).sum()),
           "absent_cells": int((~present).sum()), "sum_of_published_rounded_values": int(raw[valid].sum()),
           "rounding": "published values rounded to multiples of five; not a complete regional population total",
           "negative_codes": {str(v): int((raw[present] == v).sum()) for v in np.unique(raw[present & ~valid])},
            "missing_meaning": "Absent feature is state 0; a present negative source code is state 1 (suppressed/0..4 or unknown per CBS); published count is state 2. Masked absent payload 0 is not a population estimate.",
            "state_codes": {"0": "absent feature", "1": "present suppressed/unknown value", "2": "published rounded count"},
           "independent_cell_code_geometry_check": "all selected features exactly agree",
           "documentation": ["https://geodata.cbs.nl/files/Vierkant_statistiek/2024/",
                             "https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/2-technische-gegevens"]}

    tiles = [r for r in sources["sources"] if r.get("edition") == "2025"]
    datasets = [rasterio.open(item["path"]) for item in tiles]
    try:
        heights, transform = merge(datasets, bounds=(2640000, 1156928, 2648072, 1165000), res=2)
        nodata = datasets[0].nodata
    finally:
        for ds in datasets:
            ds.close()
    good = np.isfinite(heights[0]) & (heights[0] != nodata)
    geotiff(common / "grindelwald-alti-2025-2m.tif", heights[0], good, "EPSG:2056", transform,
            units="m", vertical_reference="LN02")
    tlm_file = next(Path(r["path"]) for r in sources["sources"] if str(r["path"]).endswith(".shp"))
    cover = pyogrio.read_dataframe(tlm_file, bbox=(2640000, 1156928, 2648072, 1165000))
    source_crs = cover.crs.to_wkt()
    # This 2D display uses measured polygon footprints; their Z is not terrain.
    cover.geometry = cover.geometry.map(shapely.force_2d)
    cover = cover.set_crs(2056, allow_override=True)
    cover.to_file(common / "grindelwald-tlm.geojson", driver="GeoJSON")
    names = sorted(cover.OBJEKTART.unique())
    categories = {i + 1: name for i, name in enumerate(names)}
    shape, affine = (1009, 1009), from_origin(2640000, 1165000, 8, 8)
    ids, overlaps = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    for class_id, name in categories.items():
        mask = rasterize([(g, 1) for g in cover.loc[cover.OBJEKTART == name].geometry],
                         out_shape=shape, transform=affine, fill=0, dtype="uint8") != 0
        ids[mask] = class_id
        overlaps += mask
    valid_cover = overlaps == 1
    geotiff(common / "grindelwald-tlm-classes.tif", ids, valid_cover, "EPSG:2056", affine)
    np.save(common / "tlm-overlap-classes.npy", overlaps, allow_pickle=False)
    # Independent shapely centre-in-polygon check for every covered sample.
    # rasterio geometry_mask/rasterize is deliberately not reused for this check.
    checked = 0
    for class_id, name in categories.items():
        rr, cc = np.where((ids == class_id) & valid_cover)
        points = shapely.points(2640000 + (cc + .5) * 8, 1165000 - (rr + .5) * 8)
        union = shapely.union_all(cover.loc[cover.OBJEKTART == name].geometry.values)
        if not np.all(shapely.covers(union, points)):
            raise AssertionError("TLM raster class centre falls outside measured class footprint")
        checked += len(points)
    tlm = {"edition": "2026-02", "source_crs_wkt": source_crs,
           "operation": "Separate LV95 horizontal CRS; discard polygon Z only for 2D footprints. No height conversion.",
           "features": len(cover), "classes": categories, "survey_years": cover.HERKUNFT_J.value_counts().to_dict(),
           "class_cells": {str(i): int(((ids == i) & valid_cover).sum()) for i in categories},
           "unknown_cells": int((overlaps == 0).sum()), "ambiguous_class_cells_masked": int((overlaps > 1).sum()),
           "shapely_independent_centre_checks": checked,
           "dates": "per-feature original, revision and source dates retained in GeoJSON; release year is not survey year"}
    record(common / "provenance.json", {"sources_manifest": str(root / "sources/sources.json"),
           "cbs": cbs, "tlm": tlm, "swiss_mosaic": {"tiles": len(tiles), "shape": list(heights[0].shape),
           "valid_cells": int(good.sum()), "operation": "crop and mosaic at original 2 m spacing, no interpolation"},
           "files": {p.name: _sha(p) for p in common.iterdir() if p.is_file()},
           "shared_staging_seconds": time.perf_counter() - started, "peak_working_set_bytes": peak_memory()})
    print("Shared source staging complete", flush=True)


def configuration(root: Path, case: str, revision: bool) -> PreparationRecipe:
    common = root / "common"
    meta = json.loads((common / "provenance.json").read_text(encoding="utf-8"))
    if case == "rotterdam":
        grid = PreparationGrid("EPSG:28992", 1100 if revision else 1200, 1100 if revision else 1200,
                               (88500 if revision else 88000, 10, 0, 441500 if revision else 442000, 0, -10))
        source = PreparationSource("terrain", root / "sources/rotterdam-ahn4-10m.tif", "elevation",
                                   units="m", vertical_reference="NAP", metadata=next(
                                       s for s in json.loads((root / "sources/sources.json").read_text(encoding="utf-8"))["sources"] if s.get("edition") == "AHN4"))
        state = PreparationSource("population_state", common / "rotterdam-population-state.tif", "categorical",
                                  metadata={"state_codes": meta["cbs"]["state_codes"], "count_source_id": "population"})
        data = PreparationSource("population", common / "rotterdam-cbs-2024-v2.tif", "count", units="people/cell",
                                 metadata={**meta["cbs"], "state_source_id": "population_state"})
        # Fixed reusable range: previously unseen counts need no per-value lookup.
        # Values outside the range use the endpoint colour; analysis is unchanged.
        styles = {"population_state": {"classes": {"0": [193, 192, 179, 255], "1": [125, 88, 148, 255], "2": [0, 0, 0, 0]}},
                  "population": {"range": [0, 2010], "scale": "log1p", "colors": [[246, 226, 178, 255], [146, 30, 56, 255]]}}
        datum = "NAP"
    else:
        grid = PreparationGrid("EPSG:2056", 883 if revision else 1009, 883 if revision else 1009,
                               (2640504 if revision else 2640000, 8, 0, 1164496 if revision else 1165000, 0, -8))
        source = PreparationSource("terrain", common / "grindelwald-alti-2025-2m.tif", "elevation", units="m",
                                   vertical_reference="LN02", metadata={"edition": "2025", "product": "swissALTI3D", "survey_date": "per-STAC tile metadata"})
        data = PreparationSource("landcover", common / "grindelwald-tlm-classes.tif", "categorical", metadata=meta["tlm"])
        # Fixed cartographic colors, not synthesized surface measurements.
        colors = {"Fels": [156, 151, 139, 255], "Fels locker": [179, 171, 153, 255], "Felsbloecke locker": [189, 179, 154, 255],
                  "Feuchtgebiet": [102, 157, 152, 255], "Fliessgewaesser": [65, 135, 179, 255], "Gehoelzflaeche": [106, 142, 93, 255],
                  "Gebueschwald": [91, 125, 88, 255], "Gletscher": [213, 238, 239, 255], "Lockergestein": [195, 179, 148, 255],
                  "Lockergestein locker": [206, 191, 163, 255], "Schneefeld Toteis": [240, 245, 241, 255],
                  "Stehende Gewaesser": [57, 121, 167, 255], "Wald": [57, 102, 72, 255], "Wald offen": [118, 155, 102, 255]}
        styles = {"landcover": {"classes": {str(i): colors[name] for i, name in meta["tlm"]["classes"].items()}}}
        datum = "LN02"
    # Same-CRS integer/half-integer lattice: exact arithmetic permits zero tolerance.
    with rasterio.open(source.path) as ds:
        xy = ds.transform * (0.5, 0.5)
    return PreparationRecipe(grid, [source, state, data] if case == "rotterdam" else [source, data], datum,
        checkpoints=[{"source_id": "terrain", "source_pixel": [0, 0], "expected_target_xy": list(xy), "tolerance": 0}],
        display_styles=styles)


def package_check(prepared: PreparedScene, directory: Path) -> dict:
    m = json.loads((directory / "landscape.json").read_text(encoding="utf-8"))
    q = np.array(Image.open(directory / "height.png"), dtype=np.float64)
    original = prepared.layers["terrain"].data
    decoded = m["encoding"]["intercept_m"] + q * m["encoding"]["step_m"]
    error = float(np.abs(decoded - original).max())
    bound = m["encoding"]["quantization_bound_m"] + m["encoding"]["floating_bound_m"]
    if error > bound:
        raise AssertionError("Independent PNG decoder exceeds quantization bound")
    category = prepared.layers["landcover"]
    summed = np.zeros(q.shape, dtype=np.uint16)
    for layer in m["weightmaps"]:
        raw = np.array(Image.open(directory / layer["file"]))
        expected = (category.data == layer["class_id"]) & category.valid_mask
        if not np.array_equal(raw, expected.astype(np.uint8) * 255):
            raise AssertionError("Material mask changed class semantics")
        summed += raw != 0
    if not np.array_equal(summed, category.valid_mask.astype(np.uint16)):
        raise AssertionError("Unknown or overlapping material assignment")
    grid = prepared.recipe.target_grid
    settings = m["import_settings"]
    origin = m["geographic"]["origin"]
    first = settings["vertex_0_0_base_location_cm"]
    # Reconstruct horizontal coordinates solely from destination settings.
    cols, rows = np.meshgrid(np.arange(grid.width), np.arange(grid.height))
    east = origin[0] + (first[0] + cols * settings["scale_cm"][0]) / 100
    north = origin[1] - (first[1] + rows * settings["scale_cm"][1]) / 100
    expected_e = grid.geotransform[0] + (cols + .5) * grid.geotransform[1]
    expected_n = grid.geotransform[3] + (rows + .5) * grid.geotransform[5]
    if not (np.array_equal(east, expected_e) and np.array_equal(north, expected_n)):
        raise AssertionError("Landscape origin/axes/pixel centres changed coordinates")
    return {"independent_decoder": "Pillow", "vertices": q.size, "max_height_error_m": error,
            "derived_bound_m": bound, "all_xy_exact": True, "all_class_masks_exact": True,
            "editor_import_verified": False, "shared_edges": "one image, no tiled seams; actual editor LOD unrun"}


def prepare(root: Path, case: str, revision: bool) -> None:
    started = time.perf_counter()
    directory = root / case / ("revision" if revision else "initial")
    directory.mkdir(parents=True, exist_ok=False)
    recipe = configuration(root, case, revision)
    recipe.save(directory / "recipe.json")
    prepared = prepare_scene(PreparationRecipe.load(directory / "recipe.json"))
    prepared.save(directory / "prepared")
    prepared.coverage_preview(directory / "coverage.html")
    restored = PreparedScene.load(directory / "prepared")
    for source_id, layer in prepared.layers.items():
        other = restored.layers[source_id]
        if not (np.array_equal(layer.data, other.data, equal_nan=True) and np.array_equal(layer.valid_mask, other.valid_mask)):
            raise AssertionError("Saved preparation changed analysis arrays")
    assets = prepared.viewer_assets(directory / "viewer-assets", terrain_void_fill=0)
    # Use the exact same drape in the QGIS comparison. Unknown categories and
    # heights get a declared cartographic hatch, never an invented measurement.
    texture = np.full((recipe.target_grid.height, recipe.target_grid.width, 4), [193, 192, 179, 255], np.uint8)
    display_valid = np.zeros(texture.shape[:2], dtype=bool)
    for overlay in assets["overlays"]:
        rgba = np.array(Image.open(overlay["path"]).convert("RGBA"))
        selected = rgba[..., 3] != 0
        texture[selected] = rgba[selected]
        display_valid |= selected
    rr, cc = np.indices(texture.shape[:2])
    unknown = ~display_valid
    terrain_valid = prepared.layers["terrain"].valid_mask
    texture[unknown & ((rr + cc) % 8 == 0)] = [145, 145, 138, 255]
    texture[~terrain_valid] = [72, 111, 137, 255]
    texture[(~terrain_valid) & ((rr - cc) % 8 == 0)] = [192, 212, 220, 255]
    Image.fromarray(texture).save(directory / "display.png")
    geotiff(directory / "display.tif", texture, np.ones(texture.shape[:2], bool), recipe.target_grid.crs, recipe.target_grid.transform)
    evidence = {"case": case, "revision": revision, "preparation_seconds": time.perf_counter() - started,
                "implementation": implementation_identity(),
                "peak_working_set_bytes": peak_memory(), "source_hashes": {s.source_id: _sha(Path(s.path)) for s in recipe.sources},
                "reuse_arrays_exact": True, "grid": recipe.target_grid.to_dict(),
                "terrain_range_m": [float(prepared.layers["terrain"].data[terrain_valid].min()), float(prepared.layers["terrain"].data[terrain_valid].max())],
                "render_void_fill_m": 0, "void_hatch": "blue diagonal; fill is display only, not measured water height",
                "unknown_category_hatch": "gray diagonal", "display_range": recipe.display_styles}
    if case == "rotterdam":
        raw = prepared.layers["population"]
        with rasterio.open(root / "common/rotterdam-cbs-2024-v2.tif") as ds:
            if not (np.array_equal(ds.read(1), raw.data) and np.array_equal(ds.read_masks(1) != 0, raw.valid_mask)):
                raise AssertionError("Native CBS values or masks changed")
        evidence["native_count_grid_preserved_exactly"] = True
        state = prepared.layers["population_state"]
        original = np.load(root / "common/cbs-present.npy")
        disclosed = np.load(root / "common/cbs-disclosure.npy")
        expected = np.where(original, np.where(disclosed, 1, 2), 0).astype(np.uint8)
        # The categorical state display is nearest sampled; independently expand
        # the native cells, then crop the revised extent by its recorded grid.
        expanded = np.repeat(np.repeat(expected, 10, axis=0), 10, axis=1)
        if revision: expanded = expanded[50:-50, 50:-50]
        if not np.array_equal(state.data, expanded): raise AssertionError("CBS state distinctions changed")
        evidence["population_states_exact"] = {str(v): int((state.data == v).sum()) for v in (0, 1, 2)}
    else:
        classes = prepared.layers["landcover"]
        names = classes.metadata["declared"]["classes"] if "declared" in classes.metadata else json.loads((root / "common/provenance.json").read_text(encoding="utf-8"))["tlm"]["classes"]
        ids = set(int(v) for v in np.unique(classes.data[classes.valid_mask]))
        mapping = {i: "TLM_" + re.sub("[^A-Za-z0-9_]", "_", names[str(i)]) for i in ids}
        export_landscape(prepared, directory / "landscape", layout=LandscapeLayout(63, 1, 14 if revision else 16, 14 if revision else 16),
                         origin=(2640000, 1165000, 0), categorical_source="landcover", material_layers=mapping)
        evidence["landscape"] = package_check(prepared, directory / "landscape")
    record(directory / "checks.json", evidence)
    print(case, "revision" if revision else "initial", json.dumps(evidence), flush=True)


def diagnostic_controls(root: Path) -> None:
    evidence = []
    for case in ("rotterdam", "grindelwald"):
        recipe = configuration(root, case, False)
        broken = recipe.to_dict()
        broken["sources"][0]["vertical_reference"] = None
        try:
            PreparationRecipe.from_dict(broken)
        except PreparationError as exc:
            if exc.diagnostic.code != "missing_vertical_reference":
                raise
            evidence.append({"case": case, "injected": "missing vertical datum", "diagnostic": vars(exc.diagnostic),
                             "correction": "Set the one source vertical_reference field to the documented datum", "recovery": "initial recipe already successfully executed"})
        else:
            raise AssertionError("Missing datum accepted")
        try:
            prepare_scene(replace(recipe, vertical_reference="ellipsoidal"))
        except PreparationError as exc:
            if exc.diagnostic.code != "unsupported_vertical_transform":
                raise
            evidence.append({"case": case, "injected": f"{recipe.vertical_reference} to ellipsoidal", "diagnostic": vars(exc.diagnostic)})
        else:
            raise AssertionError("Unsupported datum conversion accepted")
        # Deliberate source geotransform shift of one original pixel. Keep the
        # original independent pixel checkpoint, so relabeling is detected.
        source = recipe.sources[0]
        shifted = root / case / "shifted-control.tif"
        with rasterio.open(source.path) as ds:
            gt = list(ds.transform.to_gdal())
            gt[0] += gt[1]
            values, good, crs = ds.read(1), ds.read_masks(1) != 0, ds.crs
        geotiff(shifted, values, good, crs, rasterio.Affine.from_gdal(*gt), vertical_reference=recipe.vertical_reference)
        try:
            prepare_scene(replace(recipe, sources=[replace(source, path=shifted), *recipe.sources[1:]]))
        except PreparationError as exc:
            if exc.diagnostic.code != "checkpoint_failed":
                raise
            evidence.append({"case": case, "injected": "source shifted east by one original pixel", "shift_m": gt[1], "diagnostic": vars(exc.diagnostic)})
        else:
            raise AssertionError("Shifted geotransform control accepted")
    record(root / "diagnostic-controls.json", {"controls": evidence})
    print("Metadata, unsupported-datum and shifted-transform controls passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stage", "prepare", "controls"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--case", choices=["rotterdam", "grindelwald"], default="rotterdam")
    parser.add_argument("--revision", action="store_true")
    parser.add_argument("--cbs-zip", type=Path, default=Path("data/cbs_vk100/2026-cbs_vk100_2024_v2.zip"))
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.stage == "stage":
        stage(root, args.cbs_zip.resolve())
    elif args.stage == "prepare":
        prepare(root, args.case, args.revision)
    else:
        diagnostic_controls(root)
