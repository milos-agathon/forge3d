"""CBS Amsterdam/Rotterdam linked publications and independent revision checks.

Requires the datasets extra (real rasterio, pyproj, pyogrio), a native renderer
and MapPlate's existing Pro entitlement. Downloads only official CBS archives.
Run with the matching Forge3D Python source installed; see the C06 guide.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

# Import the real package before the repository's partial test stub can shadow it.
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, transform_bounds
from rasterio.vrt import WarpedVRT
import numpy as np
import pyogrio
from pyproj import Transformer, Geod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _import_shim  # noqa: E402,F401
import forge3d as f3d  # noqa: E402
from forge3d.publication import PublicationSource, PublicationExtent, PublicationRecipe
from forge3d.map_plate import BBox

ARCHIVES = {"2024": "2026-cbs_vk100_2024_v2.zip", "2023": "2026-cbs_vk100_2023_vol.zip"}
RD_BOUNDS = (84000, 425000, 136000, 497000)
REVISED_BREAKS = [25, 100, 250, 500]
PALETTE = [(255, 243, 204, 255), (249, 200, 118, 255), (234, 139, 76, 255),
           (193, 71, 61, 255), (115, 31, 58, 255)]
SEMANTICS = "https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/2-technische-gegevens"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, year, archive):
    """Retain source geometry/value evidence, then explicitly resample to UTM."""
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://geodata.cbs.nl/files/Vierkant_statistiek/{year}/{ARCHIVES[year]}"
    if archive is None:
        archive = root / ARCHIVES[year]
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
    archive = archive.resolve()
    member = "cbs_vk100_2024_v2.gpkg" if year == "2024" else "cbs_vk100_2023_vol.gpkg"
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != [member]:
            raise ValueError("Unexpected CBS archive members")
    transformed = transform_bounds("EPSG:28992", "EPSG:32631", *RD_BOUNDS)
    w, s = np.floor(np.array(transformed[:2]) / 100).astype(int) * 100
    e, n = np.ceil(np.array(transformed[2:]) / 100).astype(int) * 100
    # Read every RD cell that any destination cell of the UTM rectangle can
    # sample (densified edges, one-cell pad). Nodata in the prepared grid then
    # only means CBS absent/suppressed, never "outside the loaded subset".
    read = transform_bounds("EPSG:32631", "EPSG:28992", int(w), int(s), int(e), int(n), densify_pts=101)
    west, south = (np.floor(np.array(read[:2]) / 100).astype(int) * 100 - 100).tolist()
    east, north = (np.ceil(np.array(read[2:]) / 100).astype(int) * 100 + 100).tolist()
    cells = pyogrio.read_dataframe("/vsizip/" + archive.as_posix() + "/" + member,
        bbox=(west, south, east, north), columns=["crs28992res100m", "aantal_inwoners"])
    if cells.crs.to_epsg() != 28992:
        raise ValueError("Unexpected source CRS")
    codes = cells.crs28992res100m.str.extract(r"^E(\d+)N(\d+)$").astype(int).to_numpy() * 100
    geometric = cells.geometry.bounds.to_numpy()
    expected = np.column_stack([codes, codes + 100])
    np.testing.assert_array_equal(geometric, expected)
    inside = ((codes[:, 0] >= west) & (codes[:, 0] < east) &
              (codes[:, 1] >= south) & (codes[:, 1] < north))
    codes, counts = codes[inside], cells.aantal_inwoners.to_numpy()[inside]
    rows, cols = ((north - 100 - codes[:, 1]) // 100), ((codes[:, 0] - west) // 100)
    flat = rows * ((east - west) // 100) + cols
    if len(np.unique(flat)) != len(flat):
        raise ValueError("Duplicate CBS grid cells")
    raw = np.full(((north - south) // 100, (east - west) // 100), -9999, np.float32)
    present = np.zeros(raw.shape, bool)
    present[rows, cols] = True
    raw[rows, cols] = counts
    valid = present & np.isfinite(raw) & (raw >= 0)
    negative = {str(int(n)): int(np.count_nonzero(raw[present] == n)) for n in np.unique(raw[present & ~valid])}
    raw[~valid] = -9999
    original = root / f"cbs-{year}-rd.tif"
    with rasterio.open(original, "w", driver="GTiff", width=raw.shape[1], height=raw.shape[0],
        count=1, dtype="float32", crs="EPSG:28992", transform=from_origin(west, north, 100, 100), nodata=-9999) as dst:
        dst.write(raw, 1)
    transform = from_origin(int(w), int(n), 100, 100)
    utm = np.full(((n - s) // 100, (e - w) // 100), -9999, np.float32)
    # Rasterio.reproject hard-codes a 0.125 source-pixel approximation.
    # WarpedVRT exposes the tolerance, but this runtime rejects exact zero.
    # Use the representable source-coordinate spacing, expressed in pixels;
    # this is derived from the actual input, not an invented error allowance.
    tolerance = float(np.spacing(np.float64(max(abs(v) for v in (west, south, east, north)))) / 100)
    with rasterio.open(original) as src:
        with WarpedVRT(src, crs="EPSG:32631", transform=transform, width=utm.shape[1], height=utm.shape[0],
                       nodata=-9999, resampling=Resampling.nearest, tolerance=tolerance) as vrt:
            utm = vrt.read(1)
    # Independent inverse projection of every destination centre proves which
    # original cell's value was used; it is not a comparison with the renderer.
    rr, cc = np.indices(utm.shape)
    xx, yy = Transformer.from_crs(32631, 28992, always_xy=True).transform(w + (cc + .5) * 100, n - (rr + .5) * 100)
    sc, sr = np.floor((xx - west) / 100).astype(int), np.floor((north - yy) / 100).astype(int)
    expected_utm = np.full(utm.shape, -9999, np.float32)
    covered = (sr >= 0) & (sr < raw.shape[0]) & (sc >= 0) & (sc < raw.shape[1])
    if not covered.all():
        raise AssertionError(f"Prepared grid samples outside the loaded RD cells: {int((~covered).sum())}")
    expected_utm[covered] = raw[sr[covered], sc[covered]]
    mismatches = int(np.count_nonzero(expected_utm != utm))
    if mismatches:
        raise AssertionError(f"Nearest resampling differs from independently projected cell centres: {mismatches}")
    path = root / f"cbs-{year}-utm31.tif"
    with rasterio.open(path, "w", driver="GTiff", width=utm.shape[1], height=utm.shape[0], count=1,
        dtype="float32", crs="EPSG:32631", transform=transform, nodata=-9999) as dst:
        dst.write(utm, 1)
        dst.update_tags(observation_year=year, source_archive_sha256=sha(archive))
    record = {"archive": str(archive), "archive_url": url, "archive_sha256": sha(archive),
              "archive_member": member, "accessed": "2026-10-29", "semantics": SEMANTICS,
              "source_field": "aantal_inwoners", "cell_id": "crs28992res100m", "source_crs": "EPSG:28992",
              "rd_subset": list(RD_BOUNDS), "rd_read_window": [west, south, east, north],
              "utm_cells_outside_read_window": 0, "source_cell_size_metres": 100,
              "selected_features": len(codes), "published_cells": int(valid.sum()),
              "absent_cells": int((~present).sum()), "negative_source_codes": negative,
              "geometry_vs_cell_code_mismatches": 0, "original_subset": str(original),
              "original_subset_sha256": sha(original), "prepared_sha256": sha(path),
              "transform": "RD New to UTM 31N / EPSG:32631; 100 m destination cells; nearest neighbour; no aggregation",
              "value_meaning": "Published rounded people per original 100 m CBS cell; resampled display values, not destination-cell totals or a complete population total",
              "missing_rule": "Absent cells and present -99997 values become file nodata -9999; CBS defines -99997 as 0-4 / secret / not present and identical to unpublished cells; combined key, never a published count",
              "independent_resampling_mismatches": mismatches,
              "warp_tolerance_source_pixels": tolerance,
              "warp_tolerance_basis": "float64 spacing at the largest original RD coordinate / 100 m source pixel",
              "versions": {"rasterio": rasterio.__version__, "GDAL": rasterio.__gdal_version__,
                           "pyogrio": pyogrio.__version__}}
    (root / f"cbs-{year}-provenance.json").write_text(json.dumps(record, indent=2), encoding="utf8")
    return PublicationSource(path, "CBS population grid", year,
        "2024 v2 / corrected archive" if year == "2024" else "2023 vol / complete archive",
        url, "People / source 100 m cell", "0-4 people or not published (CBS)", record), BBox(int(w), int(s), int(e), int(n), "EPSG:32631")


def dependency_ids(receipt):
    return {key: value["id"] for key, value in receipt["dependencies"].items()}


def _inside(inner, outer):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def displayed_classes(receipt, bounds, source_path, breaks):
    """Saved page pixels vs NumPy classification of the GeoTIFF, per cell centre."""
    from PIL import Image
    image = np.asarray(Image.open(receipt["output"]["path"]).convert("RGBA")).astype(int)
    x0, y0, x1, y1 = receipt["map_rect"]
    with rasterio.open(source_path) as ds:
        window = ds.window(bounds.west, bounds.south, bounds.east, bounds.north)
        values = ds.read(1, window=window)
        valid = values != ds.nodata
    rows, cols = values.shape
    ys = y0 + ((np.arange(rows) + .5) * (y1 - y0) / rows).astype(int)
    xs = x0 + ((np.arange(cols) + .5) * (x1 - x0) / cols).astype(int)
    shown = image[np.ix_(ys, xs)][..., :3]
    classes = np.where(valid, np.searchsorted(breaks, values, side="right") + 1, 0)
    palette = {e["class_id"]: e["rgba"][:3] for e in receipt["legend"]["entries"]}
    expected = np.array([palette[k] for k in range(len(palette))])[classes]
    bx0, by0, bx1, by1 = receipt["scale"]["bar_rect"]
    under_bar = (ys[:, None] >= by0) & (ys[:, None] < by1) & (xs[None] >= bx0) & (xs[None] < bx1)
    deviation = np.abs(shown - expected).max(axis=2)[~under_bar]
    assert deviation.max() <= 1, int(deviation.max())
    return {"cells_checked": int(deviation.size), "max_deviation_levels": int(deviation.max()),
            "classes_present": sorted(int(k) for k in np.unique(classes))}


def independent(receipt, bounds, locator):
    t = Transformer.from_crs(32631, 4326, always_xy=True)
    lon, lat = t.transform([bounds.west, bounds.east], [bounds.center_lat] * 2)
    independent_distance = Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])[2]
    drawn_distance = receipt["scale"]["meters_per_pixel"] * receipt["main_viewport"][0]
    label_value, label_unit = receipt["scale"]["bar_label"].split()
    labelled_metres = float(label_value) * {"km": 1000, "m": 1}[label_unit]
    independent_mpp = independent_distance / receipt["main_viewport"][0]
    bar_ground_metres = receipt["scale"]["bar_pixel_span"] * independent_mpp
    # Existing ScaleBar truncates its length to whole pixels. One pixel is the
    # quantization contract, not a newly selected geographic error tolerance.
    assert abs(labelled_metres - bar_ground_metres) < independent_mpp
    xy = np.array(receipt["footprint"]["coordinates"])
    expected = (xy - [locator.west, locator.north]) / [locator.width, -locator.height] * receipt["locator_viewport"]
    np.testing.assert_array_equal(expected, receipt["footprint"]["locator_pixels"])
    # Registration proof for every rendered thematic cell centre is performed
    # by the calibration frame below, separate from these affine calculations.
    return {"independent_ground_width_m": independent_distance, "recipe_ground_width_m": drawn_distance,
            "difference_m": drawn_distance - independent_distance,
            "difference_in_output_pixels": (drawn_distance - independent_distance) / receipt["scale"]["meters_per_pixel"],
            "scale_label_metres": labelled_metres, "drawn_bar_ground_metres": bar_ground_metres,
            "bar_rounding_error_metres": labelled_metres - bar_ground_metres,
            "bar_rounding_contract": "less than one output pixel of ground distance",
            "locator_pixel_coordinate_mismatches": 0}


def calibration(root):
    """A real asymmetric native frame proves screen registration, not a bbox assumption."""
    import forge3d.map_scene as ms
    classes = np.arange(12, dtype=np.uint16).reshape(3, 4)
    colors = {i: (int(i * 19), int(255 - i * 13), int(40 + i * 15), 255) for i in range(12)}
    style = f3d.CategoricalRasterStyle(colors)
    scene = f3d.MapScene(terrain=f3d.TerrainSource(data=np.zeros((3, 4), np.float32), crs="EPSG:32631",
        metadata={"source_id": "c06-asymmetric-screen-registration", "resolution": [100, 100]}),
        layers=[f3d.RasterOverlay("registration", data=classes, style=style, crs="EPSG:32631")],
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=400, height=300, aovs=("albedo",)))
    path = root / "registration.png"
    frames = []
    original = ms._render_terrain_renderer_result
    def observe(*args, **kwargs):
        frame = original(*args, **kwargs)
        if frame is not None:
            frames.append(frame)
        return frame
    ms._render_terrain_renderer_result = observe
    try:
        scene.render(str(path))
    finally:
        ms._render_terrain_renderer_result = original
    albedo = np.asarray(frames[-1].aov_frame.albedo())
    # Read the actual native albedo AOV; spatial rank survives its half-float
    # storage, and the colour check uses that storage format's own precision.
    centres = albedo[np.ix_([50, 150, 250], [50, 150, 250, 350])][..., 0]
    order = np.argsort(centres, axis=None).tolist()
    if order != list(range(12)):
        raise AssertionError(f"Native screen registration is not north-up: {centres}")
    actual = albedo[np.ix_([50, 150, 250], [50, 150, 250, 350])][..., :3]
    srgb = np.array(list(colors.values())).reshape(3, 4, 4)[..., :3] / 255
    expected = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    error = float(np.max(np.abs(actual - expected)))
    storage_ulp = float(np.max(np.spacing(expected.astype(np.float16)).astype(np.float32)))
    # Identify each native colour against the independently decoded palette.
    # Record conversion error, without inventing a colour-accuracy tolerance.
    distances = np.sum((actual.reshape(12, 1, 3) - expected.reshape(1, 12, 3)) ** 2, axis=2)
    palette_ids = np.argmin(distances, axis=1).tolist()
    assert palette_ids == list(range(12)), palette_ids
    return {"backend": scene.last_render_backend, "cell_centres_checked": 12,
            "palette_max_linear_error": error, "rgba16float_ulp": storage_ulp,
            "identified_palette_entries": palette_ids,
            "colour_accuracy": "Measured conversion error only; no exact-colour or half-float-ULP qualification claimed",
            "asymmetric_cell_order": order, "image": str(path), "metadata": scene.last_render_metadata}


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source, locator = prepare(root / "sources", "2024", args.cbs_2024)
    older, older_locator = prepare(root / "sources", "2023", args.cbs_2023)
    assert locator == older_locator
    centre = Transformer.from_crs(4326, 32631, always_xy=True)
    pages = []
    for city, lon, lat in (("Amsterdam", 4.90, 52.37), ("Rotterdam", 4.48, 51.92)):
        cx, cy = (round(v / 100) * 100 for v in centre.transform(lon, lat))
        pages.append(PublicationExtent(city.lower(), city,
            BBox(cx - 8000, cy - 8000, cx + 8000, cy + 8000, locator.crs)))
    recipe = PublicationRecipe(source, [50, 150, 300, 600], PALETTE, locator)
    cache = root / "native-cache"
    cache_initially_empty = not cache.exists()
    base = recipe.render_series(pages, root / "base", cache=cache)
    same = recipe.render_series(pages, root / "unchanged", cache=cache)
    breaks = replace(recipe, breaks=REVISED_BREAKS).render_series(pages, root / "breaks", cache=cache)
    moved_pages = [replace(pages[0], bounds=replace(pages[0].bounds,
        west=pages[0].bounds.west + 1000, east=pages[0].bounds.east - 1000)), pages[1]]
    extent = recipe.render_series(moved_pages, root / "extent", cache=cache)
    year = replace(recipe, source=older).render_series(pages, root / "year-2023", cache=cache)
    evidence = {"base_revision": "37159353f618f76e6020f6221b2a475e7969013a",
                "root_revision_preserved": "7c81644ad7d7180dd51d59d93b2180878eec5131",
                "interpreter": sys.version, "native": str(f3d._forge3d.__file__),
                "native_sha256": sha(Path(f3d._forge3d.__file__)), "forge3d": f3d.__version__,
                "adapter": f3d.device_probe(None), "registration": calibration(root), "pages": {}}
    for i, page in enumerate(pages):
        b, s, c, e, y = (items[i] for items in (base, same, breaks, extent, year))
        for receipt in (b, s, c, e, y):
            for render in (receipt["main_render"], receipt["locator_render"]):
                assert render["metadata"]["bypass_reason"] is None, render["metadata"]
                assert render["max_legend_deviation_levels"] <= 1
        # Reuse comes only from unchanged validated frames; every revision misses.
        hits = {name: (r["main_render"]["metadata"]["cache_hit"], r["locator_render"]["metadata"]["cache_hit"])
                for name, r in (("base", b), ("unchanged", s), ("breaks", c), ("extent", e), ("year-2023", y))}
        assert hits["unchanged"] == (True, True), hits
        assert hits["breaks"] == (False, False) and hits["year-2023"] == (False, False), hits
        assert hits["extent"] == (i == 1, True), hits  # The locator's own scene did not change.
        if cache_initially_empty:
            assert hits["base"] == (False, False), hits  # The locator renders once per series.
        assert dependency_ids(b) == dependency_ids(s)
        assert b["output"]["pixel_sha256"] == s["output"]["pixel_sha256"]
        for key in ("source", "caption", "scale"):
            assert dependency_ids(b)[key] == dependency_ids(c)[key]
        for key in ("legend", "scale"):
            assert dependency_ids(b)[key] == dependency_ids(y)[key]
        for key in ("map", "legend", "locator", "classification"):
            assert dependency_ids(b)[key] != dependency_ids(c)[key]
        for key in ("map", "locator", "caption", "source"):
            assert dependency_ids(b)[key] != dependency_ids(y)[key]
        assert b["source"]["sha256"] != y["source"]["sha256"]
        assert b["output"]["pixel_sha256"] != y["output"]["pixel_sha256"]
        assert b["output"]["pixel_sha256"] != c["output"]["pixel_sha256"]
        if i == 0:
            for key in ("map", "locator", "scale"):
                assert dependency_ids(b)[key] != dependency_ids(e)[key]
            for key in ("source", "classification", "legend", "caption"):
                assert dependency_ids(b)[key] == dependency_ids(e)[key]
        else:
            assert dependency_ids(b) == dependency_ids(e)
            assert b["output"]["pixel_sha256"] == e["output"]["pixel_sha256"]
        evidence["pages"][page.page_id] = {"independent": independent(b, page.bounds, locator),
            "displayed_classes": {"base": displayed_classes(b, page.bounds, source.path, recipe.breaks),
                                  "breaks": displayed_classes(c, page.bounds, source.path, REVISED_BREAKS),
                                  "year-2023": displayed_classes(y, page.bounds, older.path, recipe.breaks)},
            "unchanged_page_identical": b["output"]["pixel_sha256"] == s["output"]["pixel_sha256"],
            "cache_hits_main_locator": hits,
            "max_legend_deviation_levels": {name: max(r["main_render"]["max_legend_deviation_levels"],
                                                      r["locator_render"]["max_legend_deviation_levels"])
                                            for name, r in (("base", b), ("unchanged", s), ("breaks", c),
                                                            ("extent", e), ("year-2023", y))},
            "scale_bar_inside_map": all(_inside(r["scale"]["bar_rect"], r["map_rect"]) for r in (b, c, e, y)),
            "base": str(root / "base" / (page.page_id + ".json")),
            "class_break_revision": str(root / "breaks" / (page.page_id + ".json")),
            "extent_revision": str(root / "extent" / (page.page_id + ".json")),
            "year_revision": str(root / "year-2023" / (page.page_id + ".json"))}
    evidence["cache_initially_empty"] = cache_initially_empty
    (root / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf8")
    print(json.dumps({"output": str(root), "pages": evidence["pages"], "adapter": evidence["adapter"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cbs-2024", type=Path)
    parser.add_argument("--cbs-2023", type=Path)
    run(parser.parse_args())
