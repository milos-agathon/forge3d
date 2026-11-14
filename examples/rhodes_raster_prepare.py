"""Prepare the pinned Rhodes Sentinel-2 observations and GLO-30 terrain."""

from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument(
    "--fetch", action="store_true", help="retrieve pinned public assets"
)
args = parser.parse_args()
root = args.directory.resolve()
sources = root / "sources"
prepared = root / "prepared"
prepared.mkdir(exist_ok=True)
crs = "EPSG:32635"
width = height = 1800
transform = from_origin(570000, 3998000, 10, 10)
bounds = rasterio.transform.array_bounds(height, width, transform)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, data, nodata=None):
    p = prepared / (name + ".tif")
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
    ) as ds:
        ds.write(data, 1)
    return digest(p)


source_lock = json.loads(
    Path(__file__).with_name("rhodes_raster_sources.json").read_text()
)


def acquire(path, url, sha):
    if not path.is_file() and args.fetch:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url) as source, temporary.open("wb") as output:
            import shutil

            shutil.copyfileobj(source, output)
        if digest(temporary) != sha:
            raise ValueError(f"Download SHA-256 mismatch: {path.name}")
        temporary.replace(path)
    if digest(path) != sha:
        raise ValueError(f"Source SHA-256 mismatch: {path.name}")


records = []
for pinned in source_lock["records"]:
    folder = sources / pinned["source_id"]
    acquire(folder / "item.json", pinned["metadata_url"], pinned["metadata_sha256"])
    acquire(
        folder / "product-metadata.xml",
        pinned["product_metadata_url"],
        pinned["product_metadata_sha256"],
    )
    item = json.loads((folder / "item.json").read_bytes())
    meta = item["properties"]
    xml = ET.parse(folder / "product-metadata.xml")
    assert xml.findtext(".//BOA_QUANTIFICATION_VALUE") == "10000"
    offsets = {e.attrib["band_id"]: e.text for e in xml.findall(".//BOA_ADD_OFFSET")}
    assert offsets["3"] == offsets["7"] == "-1000"
    assert item["id"] == pinned["source_id"]
    assert meta["earthsearch:boa_offset_applied"] is True
    raw = {}
    for band, filename in [("red", "B04.tif"), ("nir", "B08.tif"), ("scl", "SCL.tif")]:
        asset = pinned["assets"][band]
        acquire(folder / filename, asset["url"], asset["sha256"])
        with rasterio.open(folder / filename) as ds:
            assert ds.crs.to_epsg() == 32635 and ds.nodata == 0
            win = from_bounds(*bounds, ds.transform)
            assert win == win.round_offsets().round_lengths(), win
            raw[band] = ds.read(
                1, window=win, out_shape=(height, width), resampling=Resampling.nearest
            )
    valid = np.isin(raw["scl"], [4, 5]) & (raw["red"] != 0) & (raw["nir"] != 0)
    red = raw["red"].astype(np.float32) * np.float32(0.0001)
    nir = raw["nir"].astype(np.float32) * np.float32(0.0001)
    ndvi = np.full((height, width), np.nan, dtype=np.float32)
    np.divide(nir - red, nir + red, out=ndvi, where=valid)
    day = meta["datetime"][:10]
    records.append(
        {
            "source_id": item["id"],
            "observed_at": meta["datetime"],
            "product_uri": meta["s2:product_uri"],
            "processing_baseline": meta["s2:processing_baseline"],
            "metadata_sha256": digest(folder / "item.json"),
            "product_metadata_sha256": digest(folder / "product-metadata.xml"),
            "assets": {
                k: {
                    "url": item["assets"][k]["href"],
                    "sha256": digest(folder / f),
                    "stac_raster_bands": item["assets"][k]["raster:bands"],
                }
                for k, f in [("red", "B04.tif"), ("nir", "B08.tif"), ("scl", "SCL.tif")]
            },
            "ndvi": write(day, ndvi, np.nan),
            "valid": int(valid.sum()),
        }
    )
for record in records[1:]:
    dates = [records[0]["observed_at"][:10], record["observed_at"][:10]]
    with (
        rasterio.open(prepared / (dates[0] + ".tif")) as a,
        rasterio.open(prepared / (dates[1] + ".tif")) as b,
    ):
        mask = (np.isfinite(a.read(1)) & np.isfinite(b.read(1))).astype(np.uint8)
    record["joint_mask_sha256"] = write(dates[1] + "-mask", mask)
dem = np.full((height, width), np.nan, dtype=np.float32)
terrain = []
for tile in source_lock["terrain"]:
    source = sources / tile["name"]
    acquire(source, tile["url"], tile["sha256"])
    with rasterio.open(source) as ds:
        reproject(
            rasterio.band(ds, 1),
            dem,
            dst_crs=crs,
            dst_transform=transform,
            dst_nodata=np.nan,
            init_dest_nodata=False,
            resampling=Resampling.bilinear,
        )
    terrain.append({"name": source.name, "sha256": digest(source)})
assert np.isfinite(dem).all(), np.isfinite(dem).sum()
report = {
    "grid": {
        "crs": crs,
        "width": width,
        "height": height,
        "geotransform": transform.to_gdal(),
    },
    "input_decoding": "COG DN has BOA offset applied; multiply by 0.0001 in float32. SCL 4/5 only by nearest; DN 0 is nodata; native 10m bands use a registered pixel window.",
    "records": records,
    "terrain": terrain,
    "terrain_vertical_reference": "EGM2008 orthometric metres",
    "terrain_sha256": write("terrain", dem, np.nan),
}
(prepared / "sources.json").write_text(json.dumps(report, indent=2))
print([(r["source_id"], r["valid"]) for r in records], (dem.min(), dem.max()))
