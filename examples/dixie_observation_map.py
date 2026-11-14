"""Prepare and render the Dixie Fire observation-map example from frozen public downloads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

# Import the real rasterio before selecting any source-tree Python package.
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

CRS = "EPSG:32610"
BOUNDS = (630000, 4434000, 678000, 4472400)
WIDTH, HEIGHT = 1200, 960
CELL = 30  # Landsat source sample spacing in metres
FROZEN_SOURCES = Path(__file__).resolve().parents[1] / "docs/guides/dixie_observation_sources.json"
DOMAIN = [0.0, 0.35]  # authored common display stretch in unitless reflectance
FIRMS = "viirs-snpp_2021_United_States.csv"
FIRMS_URL = f"https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/2021/{FIRMS}"
DEM = ["Copernicus_DSM_COG_10_N40_00_W122_00_DEM", "Copernicus_DSM_COG_10_N40_00_W121_00_DEM"]
ITEMS = [
    "LC08_L2SP_044032_20210711_02_T1", "LC08_L2SP_044032_20210727_02_T2",
    "LC08_L2SP_044032_20210812_02_T1", "LC08_L2SP_044032_20210828_02_T1",
]
# One second on screen per authored snapshot. The calendar clock is irregular.
ORIGINAL_TIMES = [
    "2021-07-11T18:45:11Z", "2021-07-11T18:45:11.493695Z", "2021-07-14T19:00:00Z",
    "2021-07-20T20:00:00Z", "2021-07-27T18:45:18.288170Z", "2021-08-05T20:00:00Z",
    "2021-08-12T18:45:24.905913Z",
]
REVISED_TIMES = [
    *ORIGINAL_TIMES[:4], "2021-07-28T20:00:00Z", "2021-08-06T20:00:00Z",
    "2021-08-28T18:45:29.398002Z", "2021-08-28T18:45:29.398003Z",
]
POINT_COLOR = "#ff683d"
# Six source pixels leave a visible centre after the fixed publication inset.
POINT_WIDTH = 6
TITLE = "Dixie Fire 2021  |  Landsat 8 surface reflectance"
LEGEND = [
    f"RGB stretch {DOMAIN[0]:.2f} to {DOMAIN[1]:.2f} reflectance",
    {"label": "Terrain context / reflectance unavailable", "color": "#929ca4"},
    {"label": "VIIRS active-fire report (point)", "color": POINT_COLOR},
]


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(io.DEFAULT_BUFFER_SIZE):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stamp(value):
    return value.isoformat().replace("+00:00", "Z")


def clock(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def prepare(raw, output):
    """Retain raw DN, physical units, QA exclusions and exact georeferencing."""
    left, bottom, right, top = BOUNDS
    shape = ((top - bottom) // CELL, (right - left) // CELL)
    transform = from_origin(left, top, CELL, CELL)
    frozen = read_json(FROZEN_SOURCES)
    source_index = {r["payload"]["source"]["provenance"]["catalog_id"]: r["payload"]["source"]["provenance"]
                    for r in frozen["rasters"]}
    profile = dict(driver="GTiff", width=shape[1], height=shape[0], crs=CRS,
                   transform=transform, compress="deflate")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in ITEMS:
        catalog = raw / "landsat" / f"{item}.json"
        if digest(catalog) != source_index[item]["stac_sha256"]:
            raise ValueError(f"Landsat catalog bytes differ from the frozen item {item}")
        stac = read_json(catalog)
        assets, arrays, band_metadata = {}, {}, []
        for band in ("red", "green", "blue", "qa_pixel", "qa_radsat"):
            asset = stac["assets"][band]
            url = asset["href"].split("?")[0]
            path = raw / "landsat" / url.split("/")[-1]
            aligned = np.zeros(shape, np.uint16)
            source_hash = digest(path)
            if source_hash != source_index[item]["bands"][band]["sha256"]:
                raise ValueError(f"Landsat bytes differ from the frozen {item} {band} asset")
            with rasterio.open(path) as src:
                reproject(rasterio.band(src, 1), aligned, dst_transform=transform,
                          dst_crs=CRS, resampling=Resampling.nearest)
                assets[band] = dict(uri=url, sha256=source_hash, crs=src.crs.to_string(),
                                    geotransform=src.transform.to_gdal())
            arrays[band] = aligned
            if band in ("red", "green", "blue"):
                band_metadata.append(asset["raster:bands"][0])
        rgb = np.stack([arrays[k] for k in ("red", "green", "blue")])
        # USGS Collection 2 Landsat 8 QA_PIXEL bits 0–5. Exclude all
        # QA_RADSAT flags conservatively, including terrain occlusion.
        mask = ((arrays["qa_pixel"] & 63) == 0) & (arrays["qa_radsat"] == 0) & (rgb != 0).all(axis=0)
        destination = (output / f"{item}.tif").resolve()
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
            destination, "w", **profile, count=3, dtype="uint16", photometric="RGB"
        ) as dst:
            dst.write(rgb)
            dst.write_mask(mask)
            dst.scales = tuple(band["scale"] for band in band_metadata)
            dst.offsets = tuple(band["offset"] for band in band_metadata)
            dst.descriptions = ("SR_B4 red", "SR_B3 green", "SR_B2 blue")
        acq = clock(stac["properties"]["datetime"])
        product_id = assets["red"]["uri"].rsplit("/", 2)[1]
        records.append(dict(record_id=product_id, observed_at=stamp(acq), valid_from=stamp(acq),
                            valid_until=stamp(acq + timedelta(days=2)), payload=dict(source=dict(
            path=str(destination), name="Landsat 8", year=str(acq.year), edition=product_id,
            uri=f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2/items/{item}",
            units="unitless surface reflectance", missing_label="QA excluded / unavailable",
            provenance=dict(sha256=digest(destination), stac_sha256=digest(catalog), bands=assets,
                            qa_pixel_excluded_bits=list(range(6)), qa_radsat_required=0,
                            resampling="nearest", catalog_id=item)))))

    elevations = np.full(shape, np.nan, np.float32)
    dem_tiles = []
    dem_index = {entry["id"]: entry for entry in frozen["terrain_source"]["tiles"]}
    for tile in DEM:
        catalog = raw / "dem" / f"{tile}.json"
        catalog_hash = digest(catalog)
        if catalog_hash != dem_index[tile]["catalog_sha256"]:
            raise ValueError(f"DEM catalog bytes differ from the frozen tile {tile}")
        url = read_json(catalog)["assets"]["data"]["href"].split("?")[0]
        path = raw / "dem" / url.split("/")[-1]
        source_hash = digest(path)
        if source_hash != dem_index[tile]["sha256"]:
            raise ValueError(f"DEM bytes differ from the frozen tile {tile}")
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, 1), elevations, dst_transform=transform, dst_crs=CRS,
                      dst_nodata=np.nan, resampling=Resampling.bilinear, init_dest_nodata=False)
            dem_tiles.append(dict(id=tile, uri=url, sha256=source_hash, catalog_sha256=catalog_hash,
                                  crs=src.crs.to_string(), geotransform=src.transform.to_gdal()))
    if not np.isfinite(elevations).all():
        raise ValueError("The terrain context must cover the full fixed extent")
    terrain = (output / "terrain.tif").resolve()
    with rasterio.open(terrain, "w", **profile, count=1, dtype="float32") as dst:
        dst.write(elevations, 1)
    dy, dx = np.gradient(elevations, CELL)
    illumination = (1 + (dx + dy) / np.sqrt(1 + dx**2 + dy**2)) / 2
    context = (output / "terrain-context.tif").resolve()
    colors = np.array([104, 113, 123])[:, None, None] + 65 * np.clip(illumination, 0, 1)
    with rasterio.open(context, "w", **profile, count=3, dtype="uint8", photometric="RGB") as dst:
        dst.write(np.rint(colors).astype(np.uint8))
    event_file = raw / FIRMS
    viirs_source = dict(id=FIRMS, uri=FIRMS_URL, sha256=digest(event_file), input_crs="EPSG:4326",
                        prepared_crs=CRS, interval_utc=["2021-07-10T00:00:00Z", "2021-08-29T00:00:00Z"])
    if viirs_source["sha256"] != frozen["viirs_source"]["sha256"]:
        raise ValueError("VIIRS bytes differ from the frozen FIRMS archive")
    project = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    features = []
    with event_file.open(newline="", encoding="utf-8") as source:
        for line, detection in enumerate(csv.DictReader(source), 2):
            date = detection["acq_date"]
            if not "2021-07-10" <= date < "2021-08-29":
                continue
            x, y = project.transform(float(detection["longitude"]), float(detection["latitude"]))
            if not (left <= x < right and bottom <= y < top):
                continue
            hhmm = detection["acq_time"].zfill(4)
            acq = f"{date}T{hhmm[:2]}:{hhmm[2:]}:00Z"
            features.append(dict(record_id=f"S-NPP:{line}", observed_at=acq, valid_from=acq,
                                 valid_until=stamp(clock(acq) + timedelta(days=1)), payload=dict(
                source=viirs_source, feature=dict(type="Feature", id=f"S-NPP:{line}",
                geometry=dict(type="Point", coordinates=[x, y]), properties=detection))))
    if len(features) != frozen["events_in_frozen_geographic_and_date_subset"]:
        raise ValueError("Prepared VIIRS subset differs from the frozen study area and interval")
    write_json(output / "input.json", dict(
        crs=CRS, bounds=BOUNDS, size=[WIDTH, HEIGHT], terrain=str(terrain), context=str(context),
        terrain_source=dict(tiles=dem_tiles, prepared_sha256=digest(terrain), context_sha256=digest(context)),
        rasters=records, events=features, viirs_source=viirs_source, rgb_domain=DOMAIN,
        title=TITLE, legend=LEGEND, point_color=POINT_COLOR, point_width=POINT_WIDTH,
        lookback_seconds=86400, policy="hold", screen_seconds_per_frame=1,
        variants=dict(original=dict(times=ORIGINAL_TIMES, rasters=ITEMS[:3], interventions=[]),
                      revised=dict(times=REVISED_TIMES, rasters=[*ITEMS[:2], ITEMS[3]], interventions=[
                          "Replace the 12 August acquisition with 28 August.",
                          "Retime the 27 July and 5 August presentations to 28 July and 6 August.",
                          "Inspect the final acquisition immediately before and at its acquisition instant.",
                      ]))))


def write_outputs(directory, count, seconds):
    """Dated comparison plate and hard-cut movie from the matching stills."""
    start = time.perf_counter()
    paths = [directory / f"frame_{i:08d}.png" for i in range(count)]
    with Image.open(paths[0]) as first:
        width, height = first.size
    plate = Image.new("RGB", (width * 2, height * ((count + 1) // 2)), "#1b2021")
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            plate.paste(image, (width * (index % 2), height * (index // 2)))
    plate.save(directory / "comparison-plate.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-framerate", f"1/{seconds}", "-i", str(directory / "frame_%08d.png"),
                    "-frames:v", str(count), "-c:v", "libx264rgb", "-crf", "0",
                    "-pix_fmt", "rgb24", "-movflags", "+faststart", str(directory / "observations.mp4")],
                   check=True)
    return time.perf_counter() - start


def render(prepared, output, variant):
    from forge3d import render_execution_report
    from forge3d.map_scene import LightingPreset, MapFurnitureLayer, OrbitCamera, OutputSpec
    from forge3d.map_scene import RasterOverlay, SceneRecipe, TerrainSource
    from forge3d.observation_map import ObservationEvents, ObservationRasterStyle, compile_observation_map
    from forge3d.observation_time import ObservationRecord, ObservationTrack

    data = read_json(prepared / "input.json")
    case = data["variants"][variant]
    started = time.perf_counter()
    track = ObservationTrack([ObservationRecord(**r) for r in data["rasters"]
                              if r["payload"]["source"]["provenance"]["catalog_id"] in case["rasters"]],
                             data["policy"])
    style = {"layers": [{"id": "viirs", "type": "line",
                         "paint": {"line-color": data["point_color"], "line-width": data["point_width"]}}]}
    points = ObservationEvents([ObservationRecord(**r) for r in data["events"]],
                               timedelta(seconds=data["lookback_seconds"]), data["crs"], "VIIRS", style)
    scene = SceneRecipe(
        terrain=TerrainSource(path=data["terrain"], crs=data["crs"]),
        camera=OrbitCamera(),
        lighting=LightingPreset(settings={"unlit_albedo": True, "hue_variation_strength": 0,
                                          "cli_params": {"camera_mode": "screen"}}),
        layers=[RasterOverlay("terrain-context", path=data["context"], crs=data["crs"])],
        target_crs=data["crs"], output=OutputSpec(*data["size"]),
        map_furniture=MapFurnitureLayer(title=data["title"], legend={"entries": data["legend"]}),
    )
    compiled = compile_observation_map(scene, track, case["times"], raster_style=ObservationRasterStyle(
        tuple(data["rgb_domain"])), snapshot_dir=prepared / "snapshots", events=points,
        interventions=case["interventions"])
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    manifest = compiled.render(output, base_seed=0)
    render_seconds = time.perf_counter() - started
    table = compiled.selection_table()
    table["viirs_source"] = data["viirs_source"]
    table["terrain_source"] = data["terrain_source"]
    table["screen_seconds_per_frame"] = data["screen_seconds_per_frame"]
    write_json(output / "time-selection.json", table)
    index = len(case["times"]) - 1
    random = compiled.render(output / "random", base_seed=0, frames=[index])
    assert random.frame_record(index)["compiled_frame"] == manifest.frame_record(index)["compiled_frame"]
    replay = manifest.load(output / "flythrough_manifest.json").replay_observation_frame(
        index, output / "replay.png")
    write_json(output / "native-execution.json", json.loads(render_execution_report()))
    encode_seconds = write_outputs(output, len(case["times"]), data["screen_seconds_per_frame"])
    write_json(output / "run.json", dict(compile_seconds=compile_seconds, render_seconds=render_seconds,
                                       plate_and_encode_seconds=encode_seconds, replay_pixel_hash=replay["pixel_hash"],
                                       random_frame=index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "render"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("original", "revised"), default="original")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output)
    else:
        render(args.source, args.output, args.variant)
