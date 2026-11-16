"""Amsterdam's corrected CBS 2024 RD New counts, relief plate and flyover."""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import pyogrio
import rasterio
from rasterio.transform import from_origin

from forge3d import OrbitCamera, PublicationExtent, PublicationSource, ValueFaithfulRasterBinding
from forge3d.map_plate import BBox

ARCHIVE = "2026-cbs_vk100_2024_v2.zip"
MEMBER = "cbs_vk100_2024_v2.gpkg"
URI = f"https://geodata.cbs.nl/files/Vierkant_statistiek/2024/{ARCHIVE}"
CBS = "https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024"
GRID = (117000, 482000, 127000, 492000)
EXTENT = BBox(118000, 483000, 126000, 491000, "EPSG:28992")
CELL = 100
MISSING, WITHHELD = -9999, -99997
FIELDS = {"aantal_inwoners": "CBS resident population", "aantal_mannen": "CBS male residents"}
EDGES = (0, 50, 150, 300, 600, 2500)
PALETTE = (
    (248, 237, 204, 255),  # Published zero
    (248, 210, 143, 255), (239, 163, 91, 255), (218, 112, 68, 255),
    (176, 63, 65, 255), (107, 35, 61, 255),
    (122, 109, 157, 255), (211, 218, 222, 255),
)
FPS = 24
DURATION_SECONDS = 5


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(archive, directory):
    west, south, east, north = GRID
    shape = ((north - south) // CELL, (east - west) // CELL)
    directory.mkdir(parents=True, exist_ok=True)
    gpkg = directory / MEMBER
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != [MEMBER]:
            raise ValueError("Unexpected CBS edition archive members")
        # Publish the complete member in one step so an existing reader never
        # sees a partly extracted GeoPackage during regeneration.
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            staged = Path(staging) / MEMBER
            with bundle.open(MEMBER) as source, staged.open("wb") as target:
                shutil.copyfileobj(source, target)
            staged.replace(gpkg)
    cells = pyogrio.read_dataframe(gpkg, bbox=GRID, columns=["crs28992res100m", *FIELDS])
    if cells.crs.to_epsg() != 28992:
        raise ValueError("Unexpected CBS grid CRS")
    xy = cells.crs28992res100m.str.extract(r"^E(\d+)N(\d+)$").astype(int).to_numpy() * CELL
    np.testing.assert_array_equal(cells.geometry.bounds.to_numpy(), np.c_[xy, xy + CELL])
    np.testing.assert_array_equal(cells.geometry.area.to_numpy(), CELL ** 2)
    inside = ((xy[:, 0] >= west) & (xy[:, 0] < east) & (xy[:, 1] >= south) & (xy[:, 1] < north))
    x, y = xy[inside].T
    rows, cols = (north - CELL - y) // CELL, (x - west) // CELL
    if np.unique(np.ravel_multi_index((rows, cols), shape)).size != len(rows):
        raise ValueError("Duplicate CBS cells")
    provenance = {
        "archive": archive.name, "archive_sha256": sha(archive), "member": MEMBER,
        "member_sha256": sha(gpkg), "crs": "EPSG:28992", "grid": list(GRID),
        "cell_size_m": CELL, "cell_id": "crs28992res100m", "technical_specification": f"{CBS}/2-technische-gegevens",
        "publication_rules": f"{CBS}/4-beschrijving-cijfers",
        "suppression": {"-99997": "0-4 / confidential / absent, no known count",
                        "-99995": "publication deferred; unexpected for these fields"},
        "prepared_nodata": MISSING, "edition_note": "Published values are rounded to multiples of five.",
        "observation_date": "2024-01-01", "resampling": "none",
    }
    fields = {}
    for field, label in FIELDS.items():
        counts = cells[field].to_numpy()[inside]
        if np.any(~np.isfinite(counts)) or np.any((counts < 0) & (counts != WITHHELD)):
            raise ValueError(f"Unexpected CBS missing or suppression code in {field}")
        data = np.full(shape, MISSING, np.int32)
        data[rows, cols] = counts
        path = directory / f"{field}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=shape[1], height=shape[0],
                           count=1, dtype=data.dtype, crs=cells.crs, nodata=MISSING,
                           transform=from_origin(west, north, CELL, CELL)) as tif:
            tif.write(data, 1)
            tif.update_tags(observation_year="2024", measurement_field=field, measurement_units="people")
        fields[field] = PublicationSource(
            path, label, "2024", "corrected v2 / 100 m", URI, "people", "No source record",
            {**provenance, "measurement_field": field, "prepared_sha256": sha(path)},
        )
    return fields, gpkg


def check_page(page, gpkg):
    """Independent SQL source and pixel oracle, including missing cell positions."""
    from PIL import Image

    source = page["source"]
    field = source["provenance"]["measurement_field"]
    if field not in FIELDS:
        raise ValueError("Unexpected source field")
    bounds = page["binding"]["extent"]["bounds"]
    west, south, east, north = (bounds[key] for key in ("west", "south", "east", "north"))
    rows, cols = int((north - south) / CELL), int((east - west) / CELL)
    data = np.full((rows, cols), MISSING, np.int32)
    with closing(sqlite3.connect(gpkg)) as db:
        query = (
            f'SELECT r.minx,r.miny,r.maxx,r.maxy,t."{field}" '
            'FROM rtree_cbs_vk100_2024_geom r JOIN cbs_vk100_2024 t ON r.id=t.fid '
            'WHERE r.minx>=? AND r.minx<? AND r.miny>=? AND r.miny<?'
        )
        for x, y, x1, y1, value in db.execute(query, (west, east, south, north)):
            assert (x1 - x) * (y1 - y) == 10_000
            data[int((north - y1) / CELL), int((x - west) / CELL)] = value
    cov = page["coverage"]
    published = data >= 0
    assert cov["known_count_sum"] == int(data[published].sum(dtype=np.int64))
    assert cov["known_cells"] == int(published.sum())
    assert cov["known_zero_cells"] == int((data == 0).sum())
    assert cov["withheld_cells"] == int((data == WITHHELD).sum())
    assert cov["missing_cells"] == int((data == MISSING).sum())
    assert cov["cell_area_m2"] == 10_000 and cov["known_area_m2"] == 10_000 * int(published.sum())
    assert cov["extent_area_m2"] == (east - west) * (north - south)
    edges = page["legend"]["colour_edges"]
    values = data if page["binding"]["measurement_type"] == "count" else data.astype(np.float64) * 100
    expected = np.searchsorted(edges[1:-1], values, side="right") + 1
    expected[data == 0] = 0
    expected[data == WITHHELD] = len(edges)
    expected[data == MISSING] = len(edges) + 1
    palette = np.array(page["binding"]["palette"], dtype=np.int16)
    image = np.asarray(Image.open(page["output"]["path"]).convert("RGBA"))
    ix0, iy0, ix1, iy1 = page["inset_rect"]
    xs = ix0 + ((np.arange(cols) + .5) * (ix1 - ix0) / cols).astype(int)
    ys = iy0 + ((np.arange(rows) + .5) * (iy1 - iy0) / rows).astype(int)
    delta = np.abs(image[np.ix_(ys, xs)].astype(np.int16) - palette[expected])
    # Same existing one-level RGBA8 allowance as C06's native albedo test.
    assert delta.max() <= 1, int(delta.max())
    assert page["main_render"]["backend"] == page["inset_render"]["backend"] == "gpu_terrain"
    return {"source_cells_checked": int(data.size), "inset_max_palette_deviation": int(delta.max()),
            "known_min": int(data[published].min()), "known_max": int(data[published].max())}


def camera(extent, t):
    return OrbitCamera(target=(0, 0, 310), distance=extent.width * (1.14 - .025 * t),
                       azimuth_deg=-107 + 34 * t, elevation_deg=21 + 7 * t, fov_deg=76)


def run(archive, output, *, plate_only=False):
    output.mkdir(parents=True, exist_ok=True)
    fields, gpkg = prepare(archive, output / "source")
    page = PublicationExtent("amsterdam", "Amsterdam", EXTENT)
    binding = ValueFaithfulRasterBinding(fields["aantal_inwoners"], page, "count", 1.1,
                                         EDGES, PALETTE, MISSING, WITHHELD)
    shifted = replace(page, bounds=replace(EXTENT, west=118500, east=126500))
    plate_camera = camera(EXTENT, .5)
    variants = {
        "plate": binding,
        "field": replace(binding, source=fields["aantal_mannen"]),
        "extent": replace(binding, extent=shifted),
        "transform": replace(binding, height_scale=.55),
        "density": replace(binding, measurement_type="density", height_scale=.011,
                           colour_edges=tuple(x * 100 for x in EDGES)),
    }
    checks = {}
    for name, variant in variants.items():
        start = time.perf_counter()
        restored = ValueFaithfulRasterBinding.from_dict(json.loads(json.dumps(variant.to_dict(), allow_nan=False)))
        assert restored.to_dict() == variant.to_dict()
        receipt, = restored.render_series({name: plate_camera}, output)
        elapsed = time.perf_counter() - start
        checks[name] = {**check_page(receipt, gpkg), "coverage": receipt["coverage"],
                        "source_field": receipt["source"]["provenance"]["measurement_field"],
                        "domain": receipt["legend"]["colour_edges"],
                        "height_transform": receipt["legend"]["height_transform"],
                        "elapsed_seconds_including_render": elapsed}
    interventions = {
        "field": ["Select the prepared male residents field, with its source description and unit tags."],
        "extent": ["Move the grid window five cells east."],
        "transform": ["Change the single height factor from 1.1 to 0.55."],
        "density": ["Choose density and express the same domain and height factor per km2."],
        "manual_edits_to_marks_or_furniture": [],
    }
    evidence = {"source": fields["aantal_inwoners"].description(), "checks": checks,
                "interventions": interventions}
    if not plate_only:
        frames = {f"frame_{i:04}": camera(EXTENT, i / (FPS * DURATION_SECONDS - 1))
                  for i in range(FPS * DURATION_SECONDS)}
        directory = output / "frames"
        receipts = binding.render_series(frames, directory)
        assert all(row["legend"] == receipts[0]["legend"] and row["coverage"] == receipts[0]["coverage"]
                   for row in receipts)
        indices = (0, len(frames) // 2, len(frames) - 1)
        for name, i in zip(("opening", "midpoint", "closing"), indices):
            shutil.copyfile(receipts[i]["output"]["path"], output / f"{name}.png")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to encode the flyover")
        movie = output / "amsterdam.mp4"
        subprocess.run([ffmpeg, "-y", "-v", "error", "-framerate", str(FPS),
                        "-i", str(directory / "frame_%04d.png"), "-frames:v", str(len(frames)),
                        "-c:v", "libx264rgb", "-crf", "0", "-movflags", "+faststart", str(movie)], check=True)
        evidence["movie"] = {"frames": len(frames), "fps": FPS, "sha256": sha(movie),
                             "stills": {name: receipts[i]["output"]["pixel_sha256"]
                                        for name, i in zip(("opening", "midpoint", "closing"), indices)}}
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plate-only", action="store_true", help="Render the plate and revisions without the movie")
    args = parser.parse_args()
    run(args.archive.resolve(), args.output.resolve(), plate_only=args.plate_only)
