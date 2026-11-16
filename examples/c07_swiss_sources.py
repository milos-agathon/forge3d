"""Prepare the swisstopo inputs for the C07 example (rasterio and pyogrio)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

CATALOG = "https://data.geo.admin.ch/api/stac/v0.9/collections"
TLM_EDITION = "swisstlm3d_2026-02"
BOUNDS = (2639000, 1156000, 2649000, 1166000)
# Each of the renderer's 512 vertices per axis samples its matching texel.
TEXELS = 512
LAYERS = ("TLM_NAME_PKT", "TLM_SIEDLUNGSNAME_ZENTRUM", "TLM_EISENBAHN")
TLM_MEMBERS = tuple(f"swissTLM3D_{layer}{ext}" for layer in LAYERS
                    for ext in (".cpg", ".dbf", ".prj", ".shp", ".shx"))


def get_json(url):
    with urlopen(url) as response:
        return json.load(response)


def download(url, path):
    if not path.is_file():
        with urlopen(url) as response:
            path.write_bytes(response.read())
    return path


class RemoteZip(io.RawIOBase):
    """Read only the selected ZIP members through HTTP byte ranges."""

    def __init__(self, url):
        self.url = url
        with urlopen(Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.pos + size, self.size)
        if end <= self.pos:
            return b""
        byte_range = f"bytes={self.pos}-{end - 1}"
        with urlopen(Request(self.url, headers={"Range": byte_range})) as response:
            if response.status != 206:
                raise RuntimeError(f"swisstopo did not honor {byte_range}")
            data = response.read()
        if len(data) != end - self.pos:
            raise RuntimeError(f"incomplete swisstopo download for {byte_range}")
        self.pos = end
        return data


def terrain_assets():
    to_wgs84 = Transformer.from_crs(2056, 4326, always_xy=True)
    bbox = to_wgs84.transform_bounds(*BOUNDS)
    url = f"{CATALOG}/ch.swisstopo.swissalti3d/items?" + urlencode(
        {"bbox": ",".join(map(str, bbox)), "limit": 100}
    )
    west, south, east, north = (value // 1000 for value in BOUNDS)
    tiles = {f"{x}-{y}" for x in range(west, east) for y in range(south, north)}
    latest = {}
    while url:
        page = get_json(url)
        for item in page["features"]:
            _, edition, tile = item["id"].split("_")
            if tile in tiles and (tile not in latest or edition > latest[tile][0]):
                latest[tile] = (edition, item["assets"][item["id"] + "_2_2056_5728.tif"]["href"])
        url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)
    if tiles != latest.keys():
        raise RuntimeError(f"missing swissALTI3D tiles {sorted(tiles - latest.keys())}")
    return [latest[tile][1] for tile in sorted(tiles)]


def _tlm_hashes(cache, *, snapshot=None):
    hashes = {}
    for name in TLM_MEMBERS:
        path = cache / name
        if not path.is_file():
            raise ValueError(f"missing TLM archive member: {name}")
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        if snapshot is not None:
            (snapshot / name).write_bytes(data)
    return hashes


def _extract_tlm(archive, cache):
    with RemoteZip(archive) as remote, ZipFile(remote) as source:
        seen = set()
        for name in source.namelist():
            leaf = name.replace("\\", "/").rsplit("/", 1)[-1]
            if leaf in TLM_MEMBERS:
                if leaf in seen:
                    raise ValueError(f"duplicate TLM archive member: {leaf}")
                (cache / leaf).write_bytes(source.read(name))
                seen.add(leaf)


def _prepare(out, staging):
    receipt_path = out / "sources.json"
    cache = out / "sources"
    staged_cache = staging / "sources"
    staged_cache.mkdir()
    archive = (f"https://data.geo.admin.ch/ch.swisstopo.swisstlm3d/{TLM_EDITION}/"
               f"{TLM_EDITION}_2056_5728.shp.zip")
    # Reuse exactly the same editions on subsequent runs.
    receipt = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else None
    if receipt is not None:
        if (receipt.get("tlm_edition") != TLM_EDITION or receipt.get("tlm") != archive
                or receipt.get("tlm_layers") != list(LAYERS)):
            raise ValueError("TLM edition, archive or layers differ from the pinned source receipt")
        pins = receipt.get("tlm_members_sha256")
        if (not isinstance(pins, dict) or pins.keys() != set(TLM_MEMBERS)
                or not receipt.get("features_sha256") or not receipt.get("terrain_sha256")):
            raise ValueError("source receipt lacks TLM member or derived input pins; prepare in a fresh directory")
        tlm_cache = staging / "verified-tlm"
        tlm_cache.mkdir()
        # Derive features from the exact bytes whose pins we check.
        tlm_hashes = _tlm_hashes(cache, snapshot=tlm_cache)
        for name, checksum in tlm_hashes.items():
            if checksum != pins[name]:
                raise ValueError(f"cached TLM member differs from the source receipt: {name}")
        locked_hashes = {entry["url"]: entry["sha256"] for entry in receipt["dem"]}
        urls = [entry["url"] for entry in receipt["dem"]]
    else:
        if (cache.exists()
                or any((out / name).exists() for name in ("terrain.npy", "features.json"))):
            raise ValueError("existing Swiss inputs without a source receipt; prepare in a fresh directory")
        tlm_cache = staged_cache
        _extract_tlm(archive, tlm_cache)
        tlm_hashes = _tlm_hashes(tlm_cache)
        locked_hashes = {}
        urls = terrain_assets()

    terrain = np.full((TEXELS, TEXELS), np.nan, dtype=np.float32)
    transform = from_bounds(*BOUNDS, TEXELS, TEXELS)
    dem_receipt = []
    for url in urls:
        file = cache / url.rsplit("/", 1)[1]
        if not file.is_file():
            file = download(url, staged_cache / file.name)
        checksum = hashlib.sha256(file.read_bytes()).hexdigest()
        if url in locked_hashes and checksum != locked_hashes[url]:
            raise ValueError(f"swissALTI3D tile differs from the source receipt: {file.name}")
        with rasterio.open(file) as tile:
            if tile.crs.to_epsg() != 2056 or tuple(tile.res) != (2, 2):
                raise ValueError(f"unexpected grid for {file.name}")
            reproject(
                rasterio.band(tile, 1), terrain, src_nodata=tile.nodata, dst_crs=tile.crs,
                dst_transform=transform, resampling=Resampling.bilinear,
                init_dest_nodata=False, dst_nodata=np.nan,
            )
        dem_receipt.append({"url": url, "sha256": checksum})
    if not np.isfinite(terrain).all():
        raise ValueError("swissALTI3D mosaic has missing samples")
    terrain_path = staging / "terrain.npy"
    np.save(terrain_path, terrain, allow_pickle=False)
    terrain_hash = hashlib.sha256(terrain_path.read_bytes()).hexdigest()

    features = {}
    for layer in LAYERS:
        frame = pyogrio.read_dataframe(
            tlm_cache / ("swissTLM3D_" + layer + ".shp"), bbox=BOUNDS, encoding="UTF-8",
            columns=["UUID", "NAME", "OBJEKTART"],
        )
        if [part.to_epsg() for part in frame.crs.sub_crs_list] != [2056, 5728]:
            raise ValueError(f"unexpected CRS for {layer}")
        features[layer] = list(frame.iterfeatures(drop_id=True))
    feature_bytes = json.dumps(features, ensure_ascii=False).encode("utf-8")
    feature_hash = hashlib.sha256(feature_bytes).hexdigest()
    if receipt is not None:
        if feature_hash != receipt["features_sha256"]:
            raise ValueError("derived TLM features differ from the source receipt")
        if terrain_hash != receipt["terrain_sha256"]:
            raise ValueError("derived terrain differs from the source receipt")
    (staging / "features.json").write_bytes(feature_bytes)
    if receipt is None:
        (staging / receipt_path.name).write_text(json.dumps({
            "dem": dem_receipt, "tlm": archive, "tlm_edition": TLM_EDITION, "tlm_layers": LAYERS,
            "tlm_members_sha256": tlm_hashes,
            "bounds_wsen": BOUNDS, "crs": "EPSG:2056", "vertical": "LN02", "units": "metres",
            "texture_shape": terrain.shape,
            "texture_affine": tuple(transform)[:6],
            "row_order": "north-first",
            "mesh_bounds_wsen": (
                BOUNDS[0] + transform.a / 2, BOUNDS[1] - transform.e / 2,
                BOUNDS[2] - transform.a / 2, BOUNDS[3] + transform.e / 2,
            ),
            "terrain_sha256": terrain_hash, "features_sha256": feature_hash,
        }, indent=2) + "\n", encoding="utf-8")

    # Publish only after the inputs and derived products match their pins.
    if receipt is None:
        # Claim the new cache without replacing another preparation's inputs.
        staged_cache.rename(cache)
    else:
        for file in staged_cache.iterdir():
            file.replace(cache / file.name)
    for name in ("terrain.npy", "features.json"):
        (staging / name).replace(out / name)
    if receipt is None:
        (staging / receipt_path.name).replace(receipt_path)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as directory:
        _prepare(out, Path(directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    prepare(parser.parse_args().out)
