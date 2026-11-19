"""Compare executed Monterey receipts with QGIS and independently inspect rendered pixels."""

import argparse
import csv
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio

from monterey import write_json

# Existing MENSURA GeodTest tolerances (inverse length and direct position).
DISTANCE_M = 1e-8
COORDINATE_DEG = 1e-9
ACCENT = (142, 43, 25)
# A line or marker drawn through an exact position must cover that position
# at the quantized output pixel (allow floor and nearest integer).
PIXEL_RADIUS = 1


def image(path):
    with Image.open(path) as png:
        return np.array(png.convert("RGB"))


def pixel_hit(pixels, x, y, *, antialiased=False):
    col, row = round(x), round(y)
    patch = pixels[row - PIXEL_RADIUS:row + PIXEL_RADIUS + 1,
                   col - PIXEL_RADIUS:col + PIXEL_RADIUS + 1]
    if antialiased:
        # Qt composites trace red over a white halo. Count pixels with at
        # least 50% red coverage, using the midpoint in each RGB channel.
        return bool(np.any(np.all(patch <= (np.array(ACCENT) + 255) / 2, axis=-1)
                           & (patch[..., 0] > patch[..., 1])))
    return bool(np.any(np.all(patch == ACCENT, axis=-1)))


def verify_run(root, name):
    forge, qgis = root / name, root / f"qgis-{name}"
    result = json.loads((forge / "samples.json").read_text())
    layout = json.loads((forge / "layout.json").read_text())
    other_metadata = json.loads((qgis / "qgis.json").read_text())
    timing = json.loads((forge / "timing.json").read_text())
    for file, checksum in layout["sha256"].items():
        assert sha256((forge / file).read_bytes()).hexdigest() == checksum, file
    assert result["source"]["sha256"] == other_metadata["source_sha256"]
    assert json.loads((forge / "recipe.json").read_text()) == json.loads((qgis / "recipe.json").read_text())
    with (qgis / "samples.csv").open(newline="") as stream:
        other_rows = list(csv.DictReader(stream))
    rows = result["samples"]
    assert len(rows) == len(other_rows)
    problems = []
    distances, positions, elevations = [], [], []
    direct_provider_differences = []
    for row, other in zip(rows, other_rows):
        assert row["index"] == int(other["index"])
        distances.append(abs(row["distance_m"] - float(other["distance_m"])))
        positions.append(max(abs(row[key] - float(other[key])) for key in ("longitude", "latitude")))
        qgis_z = float(other["elevation_m"]) if other["elevation_m"] else None
        if (row["status"] != other["status"] or row["elevation_m"] != qgis_z
                or row["row"] != int(other["row"]) or row["column"] != int(other["column"])):
            problems.append(row["index"])
        if row["elevation_m"] is not None and qgis_z is not None:
            elevations.append(abs(row["elevation_m"] - qgis_z))
        direct = float(other["provider_direct_elevation_m"]) if other["provider_direct_elevation_m"] else None
        if direct != qgis_z:
            direct_provider_differences.append({"index": row["index"], "lon_lat": [row["longitude"], row["latitude"]],
                                                "direct_m": direct, "shared_rule_m": qgis_z})
    assert max(distances) < DISTANCE_M and max(positions) < COORDINATE_DEG
    assert not problems, f"Containing-cell height/validity mismatch at {problems}"

    for key, value in other_metadata["axes"].items():
        assert layout["profile"][key] == value
    map_pixels = image(forge / "map.png")
    qgis_map = image(qgis / "map.png")
    plate, qgis_plate = image(forge / "section.png"), image(qgis / "section.png")
    assert plate.shape == qgis_plate.shape == (1130, 1500, 3)
    assert map_pixels.shape == qgis_map.shape == (624, 936, 3)
    mismatched_pixels = np.any(map_pixels != qgis_map, axis=-1)

    # Compare every map pixel to NOAA's cropped raw cells, using the known
    # 3 Ãƒâ€” 3 on-screen footprint of each source pixel. No layout transform used.
    legend = {e["class_id"]: e["rgba"][:3] for e in layout["legend"]["entries"]}
    palette = np.array([legend[i] for i in sorted(legend)], dtype=np.uint8)
    with rasterio.open(root / "subset/surface.tif") as ds:
        data = ds.read(1, masked=True)
        classes = np.where(np.ma.getmaskarray(data), 0, np.digitize(data.data, layout["breaks_m"]) + 1)
        expected = palette[classes].repeat(3, axis=0).repeat(3, axis=1)
        assert np.array_equal(map_pixels, expected), "Forge3D raster is shifted or misclassified"
        assert np.array_equal(qgis_map[1::3, 1::3], palette[classes]), "QGIS shifted cell centres"
        # QGIS ties can differ at viewport pixel edges. At every screen pixel
        # its color must still match a source cell within one output pixel.
        padded = np.pad(expected, ((1, 1), (1, 1), (0, 0)), mode="edge")
        matches = np.zeros(mismatched_pixels.shape, dtype=bool)
        for y in range(3):
            for x in range(3):
                matches |= np.all(qgis_map == padded[y:y + 624, x:x + 936], axis=-1)
        assert matches.all(), "QGIS map differs beyond one display pixel"
        coords = []
        for row in rows:
            col_px, row_px = (~ds.transform) * (row["longitude"], row["latitude"])
            x, y = (45 + 3 * col_px - 0.5, 114 + 3 * row_px - 0.5)
            assert pixel_hit(plate, x, y) and pixel_hit(qgis_plate, x, y, antialiased=True), row["index"]
            coords.append([x, y])

    with rasterio.open(root / "subset/surface_sid.tif") as sid:
        ids = sid.read(1)
        source_ids = Counter(int(ids[r["row"], r["column"]]) for r in rows if r["status"] == "valid")
    anchors = []
    for anchor in layout["annotations"]:
        i = anchor["sample_index"]
        row = rows[i]
        x, y = coords[i]
        assert math.dist(anchor["map_pixel"], (x, y)) < PIXEL_RADIUS
        px = 134 + 1320 * row["distance_m"] / other_metadata["axes"]["distance_m"][1]
        low, high = other_metadata["axes"]["elevation_m"]
        py = 1010 - 168 * (row["elevation_m"] - low) / (high - low)
        assert math.dist(anchor["profile_pixel"], (px, py)) < PIXEL_RADIUS
        assert pixel_hit(plate, px, py) and pixel_hit(qgis_plate, px, py, antialiased=True), anchor
        anchors.append({**anchor, "distance_m": row["distance_m"], "elevation_m": row["elevation_m"]})
    return {
        "samples": len(rows), "distance_m": rows[-1]["distance_m"],
        "spacing_m": result["interval_m"], "status_counts": dict(Counter(r["status"] for r in rows)),
        "max_difference_m": max(distances), "max_coordinate_difference_deg": max(positions),
        "max_elevation_difference_m": max(elevations), "registered_background_pixels": len(map_pixels) * map_pixels.shape[1],
        "map_trace_samples_checked": len(rows), "anchors": anchors, "sample_source_ids": dict(source_ids),
        "qgis_differing_map_pixels": int(mismatched_pixels.sum()), "qgis_source_centres_checked": int(classes.size),
        "qgis_direct_provider_differences": direct_provider_differences, "forge3d": timing, "qgis": other_metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    before = json.loads((root / "original/recipe.json").read_text())
    after = json.loads((root / "revised/recipe.json").read_text())
    assert set(before) == set(after)
    assert [key for key in before if before[key] != after[key]] == ["end"]
    report = {name: verify_run(root, name) for name in ("original", "revised")}
    report["revision"] = {
        "before": before["end"], "after": after["end"], "endpoint_edits": 1,
        "manual_trace_profile_or_annotation_edits": 0,
    }
    write_json(root / "comparison.json", report)


if __name__ == "__main__":
    main()
