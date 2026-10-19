"""Acquire pinned public source subsets for the C05 demonstrations.

This is example staging, not a new Forge3D provider adapter. Run from the repo
root. The local preparation API consumes only the resulting recorded assets.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import struct
import urllib.parse
import urllib.request
import zipfile
import zlib

import rasterio  # real package before the checkout shim
from _import_shim import ensure_repo_import
ensure_repo_import()
from forge3d.preparation import _sha


def read_url(url: str, headers: dict[str, str] | None = None) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as response:
        return response.read()


def json_url(url: str, path: Path) -> dict:
    if not path.exists():
        data = read_url(url)
        json.loads(data)
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_bytes(data)
        partial.replace(path)
    return json.loads(path.read_text(encoding="utf-8"))


def download(url: str, path: Path) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            expected = response.headers.get("Content-Length")
            while block := response.read(1024 * 1024):
                output.write(block)
        if expected and partial.stat().st_size != int(expected):
            raise ValueError(f"Incomplete download: {url}")
        partial.replace(path)
    return {"url": url, "path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


class RangeReader(io.RawIOBase):
    """Seekable ZIP metadata reader; refuses a server ignoring byte ranges."""
    def __init__(self, url: str):
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0:
            return b""
        end = min(self.size, self.position + size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise RuntimeError("Archive server did not honor the byte range")
            data = response.read()
        self.position += len(data)
        return data


def extract_member(reader: RangeReader, info: zipfile.ZipInfo, path: Path) -> dict:
    if not path.exists():
        reader.seek(info.header_offset)
        header = reader.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid ZIP local header")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        offset = info.header_offset + 30 + name_length + extra_length
        request = urllib.request.Request(reader.url, headers={"Range": f"bytes={offset}-{offset + info.compress_size - 1}"})
        decompressor = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
        if info.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
            raise ValueError("Unsupported ZIP member compression")
        crc, count = 0, 0
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            if response.status != 206:
                raise RuntimeError("Archive server ignored the member byte range")
            while block := response.read(1024 * 1024):
                data = decompressor.decompress(block) if decompressor else block
                output.write(data)
                crc, count = zlib.crc32(data, crc), count + len(data)
            if decompressor:
                data = decompressor.flush()
                output.write(data)
                crc, count = zlib.crc32(data, crc), count + len(data)
        if count != info.file_size or crc & 0xFFFFFFFF != info.CRC:
            raise ValueError(f"ZIP integrity failure: {info.filename}")
        partial.replace(path)
    else:
        crc = 0
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                crc = zlib.crc32(block, crc)
        if path.stat().st_size != info.file_size or crc & 0xFFFFFFFF != info.CRC:
            raise ValueError(f"Cached ZIP member is incomplete or corrupt: {path}")
    return {"archive_url": reader.url, "archive_etag": reader.etag, "member": info.filename,
            "member_crc32": info.CRC, "path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


def ahn_metadata(root: Path) -> dict:
    import xml.etree.ElementTree as ET
    records = {}
    for name, request in [("ahn-description.xml", "DescribeCoverage&coverage=dtm_05m"),
                          ("ahn-capabilities.xml", "GetCapabilities")]:
        records[name] = download("https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=1.0.0&request=" + request, root / name)
    description = ET.parse(root / "ahn-description.xml")
    default = description.find(".//{http://www.opengis.net/wcs}supportedInterpolations").get("default")
    if default != "nearest neighbor": raise ValueError("AHN default sampling changed; review before reusing the pinned request")
    return {"product_resolution_m": 0.5, "product_gridding": "Squared IDW of classified ground returns",
            "acquisition_years": [2020, 2021, 2022], "local_flight_date": "not resolved by this service metadata",
            "wcs_default_interpolation": default,
            "operation": "Official WCS subset at 10 m spacing; request omits interpolation, so the service-declared nearest-neighbour default applies to the 0.5 m product.",
            "service_metadata": records, "acquisition_documentation": "https://www.ahn.nl/ahn-4"}


def acquire(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    previous = root / "sources.json"
    if previous.exists():
        for item in json.loads(previous.read_text(encoding="utf-8"))["sources"]:
            if _sha(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Pinned source changed; restore it before acquisition: {item['path']}")
    manifest = {"access_date": "2026-10-16", "sources": []}
    params = {"service": "WCS", "version": "1.0.0", "request": "GetCoverage", "coverage": "dtm_05m",
              "crs": "EPSG:28992", "bbox": "88000,430000,100000,442000", "resx": 10, "resy": 10, "format": "GEOTIFF"}
    ahn_url = "https://service.pdok.nl/rws/ahn/wcs/v1_0?" + urllib.parse.urlencode(params)
    ahn = download(ahn_url, root / "rotterdam-ahn4-10m.tif")
    with rasterio.open(ahn["path"]) as dataset:
        if dataset.crs.to_epsg() != 28992 or dataset.count != 1:
            raise ValueError("AHN service returned an unexpected grid")
        ahn["grid"] = {"shape": dataset.shape, "geotransform": dataset.transform.to_gdal()}
    ahn.update({"edition": "AHN4", "vertical_reference": "NAP", "units": "m", "request": params,
                **ahn_metadata(root),
                "documentation": "https://www.ahn.nl/dataroom"})
    manifest["sources"].append(ahn)

    # Same LV95 area covers Grindelwald and the Eiger. The 1009 x 1009,
    # 8 m target has 16 x 16 components of 63 quads for Landscape export.
    from pyproj import Transformer
    geographic = Transformer.from_crs(2056, 4326, always_xy=True)
    west, south = geographic.transform(2640000, 1156928)
    east, north = geographic.transform(2648072, 1165000)
    bbox = f"{west},{south},{east},{north}"
    url = "https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swissalti3d/items?" + urllib.parse.urlencode({"bbox": bbox})
    items, page = [], 0
    while url:
        payload = json_url(url, root / f"alti-catalogue-{page}.json")
        items.extend(payload["features"])
        url = next((link["href"] for link in payload["links"] if link["rel"] == "next"), None)
        page += 1
    # Use one recorded source edition consistently across the full extent.
    for item in items:
        if item["id"].split("_")[1] != "2025":
            continue
        for name, asset in item["assets"].items():
            if not name.endswith("_2_2056_5728.tif"):
                continue
            record = download(asset["href"], root / "swissalti3d-2025" / name)
            record.update({"stac_id": item["id"], "stac_properties": item["properties"], "edition": "2025",
                           "vertical_reference": "LN02", "horizontal_crs": "EPSG:2056"})
            manifest["sources"].append(record)
    if not any(s.get("edition") == "2025" for s in manifest["sources"]):
        raise ValueError("No 2025 swissALTI3D tiles in the declared area")

    tlm = json_url("https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swisstlm3d/items", root / "tlm-catalogue.json")
    item = next(i for i in tlm["features"] if i["id"] == "swisstlm3d_2026-02")
    asset = next(a for key, a in item["assets"].items() if key.endswith(".shp.zip"))
    reader = RangeReader(asset["href"])
    with zipfile.ZipFile(reader) as archive:
        stems = []
        for info in archive.infolist():
            if "TLM_BODENBEDECKUNG" in info.filename and info.filename.endswith(".shp"):
                with archive.open(info) as stream:
                    header = stream.read(100)
                xmin, ymin, xmax, ymax = struct.unpack_from("<4d", header, 36)
                if xmin <= 2648072 and xmax >= 2640000 and ymin <= 1165000 and ymax >= 1156928:
                    stems.append(info.filename[:-4])
        for info in archive.infolist():
            if any(info.filename.rsplit(".", 1)[0] == stem for stem in stems) and Path(info.filename).suffix.lower() in {".shp", ".shx", ".dbf", ".prj", ".cpg"}:
                record = extract_member(reader, info, root / "swisstlm3d-2026" / Path(info.filename).name)
                record.update({"stac_id": item["id"], "stac_properties": item["properties"], "edition": "2026-02"})
                manifest["sources"].append(record)
                print("Acquired", info.filename, flush=True)
    (root / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Recorded {len(manifest['sources'])} source files in {root}")


def acquire_registration_water(root: Path) -> None:
    """Independent BRT footprints used to inspect Rotterdam river alignment."""
    root.mkdir(parents=True, exist_ok=True)
    previous = root / "river-source.json"
    if previous.exists():
        for page in json.loads(previous.read_text(encoding="utf-8"))["pages"]:
            if _sha(Path(page["path"])) != page["sha256"]:
                raise ValueError(f"Pinned water-source page changed: {page['path']}")
    base = "https://api.pdok.nl/kadaster/brt-top10nl/ogc/v1/collections/waterdeel_vlak"
    params = {"bbox": "88000,430000,100000,442000", "f": "json",
              "bbox-crs": "http://www.opengis.net/def/crs/EPSG/0/28992",
              "crs": "http://www.opengis.net/def/crs/EPSG/0/28992"}
    url = base + "/items?" + urllib.parse.urlencode(params)
    features, pages = [], []
    while url:
        path = root / f"brt-water-page-{len(pages)}.json"
        data = json_url(url, path)
        features.extend(data["features"])
        pages.append({"url": url, "path": str(path.resolve()), "sha256": _sha(path)})
        url = next((link["href"] for link in data.get("links", []) if link["rel"] == "next"), None)
    output = root / "rotterdam-water.geojson"
    output.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:28992"}}, "features": features}), encoding="utf-8")
    (root / "river-source.json").write_text(json.dumps({"collection": base, "pages": pages, "features": len(features),
        "horizontal_crs": "EPSG:28992", "path": str(output.resolve()), "sha256": _sha(output),
        "meaning": "Independent BRT water footprints and original feature dates; AHN voids are not asserted to be water"}, indent=2), encoding="utf-8")
    print(f"Recorded {len(features)} water footprints from {len(pages)} complete service pages", flush=True)


def acquire_swiss_reference(root: Path) -> None:
    """Pin an independent named-summit coordinate, not a DEM-derived point."""
    previous = root / "swiss-eiger-position-source.json"
    if previous.exists():
        pinned = json.loads(previous.read_text(encoding="utf-8"))
        if _sha(Path(pinned["path"])) != pinned["sha256"]:
            raise ValueError("Pinned Swiss position reference changed")
    params = {"layer": "ch.swisstopo.swissnames3d", "searchText": "Eiger", "searchField": "name",
              "contains": "false", "returnGeometry": "true", "sr": 2056}
    source = download("https://api3.geo.admin.ch/rest/services/api/MapServer/find?" + urllib.parse.urlencode(params), root / "swiss-eiger-position.json")
    results = json.loads(Path(source["path"]).read_text(encoding="utf-8"))["results"]
    unique = {r["featureId"]: r for r in results if r["attributes"].get("name") == "Eiger"
              and r["attributes"].get("objektart") == "Alpiner Gipfel"}
    if len(unique) != 1:
        raise ValueError("The Swiss service did not return one unambiguous Eiger summit")
    source.update({"feature_id": next(iter(unique)), "crs": "EPSG:2056", "layer": params["layer"],
                   "meaning": "Independent swissNAMES3D named summit; not an ALTI3D sample or a surveyed ground-control certificate"})
    previous.write_text(json.dumps(source, indent=2), encoding="utf-8")
    print("Recorded independent Eiger position", source["sha256"], flush=True)


def acquire_cbs(root: Path) -> None:
    previous = root / "cbs-source.json"
    if previous.exists():
        pinned = json.loads(previous.read_text(encoding="utf-8"))
        if _sha(Path(pinned["path"])) != pinned["sha256"]:
            raise ValueError("Pinned CBS archive changed; restore the recorded file")
    record = download("https://geodata.cbs.nl/files/Vierkant_statistiek/2024/2026-cbs_vk100_2024_v2.zip", root / "cbs_vk100_2024_v2.zip")
    with zipfile.ZipFile(record["path"]) as archive:
        if archive.namelist() != ["cbs_vk100_2024_v2.gpkg"]:
            raise ValueError("Unexpected corrected CBS archive members")
    record.update({"edition": "2024 v2", "correction_published": "2026-08-28",
                   "index": "https://geodata.cbs.nl/files/Vierkant_statistiek/2024/"})
    (root / "cbs-source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Recorded official corrected CBS archive", record["sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--registration-water-only", action="store_true")
    selection.add_argument("--cbs-only", action="store_true")
    selection.add_argument("--swiss-reference-only", action="store_true")
    args = parser.parse_args()
    if args.registration_water_only:
        acquire_registration_water(args.directory.resolve())
    elif args.cbs_only:
        acquire_cbs(args.directory.resolve())
    elif args.swiss_reference_only:
        acquire_swiss_reference(args.directory.resolve())
    else:
        acquire(args.directory.resolve())
        acquire_cbs(args.directory.resolve())
        acquire_registration_water(args.directory.resolve())
        acquire_swiss_reference(args.directory.resolve())
