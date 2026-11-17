"""Prepare the pinned NOAA subset, or render one Monterey transect recipe."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import urlopen

BASE = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/"
TILES = {
    "surface": ("15s_surface_elev_gtif", "4ff26df2ad67546a8c6d94e54c31f7080c80f87f1bf16f2f07e64624ec373432"),
    "surface_sid": ("15s_surface_sid_gtif", "996f9d17a36fd9482dca09e75c6e77ccbf78580d2765f830a7f683f73975d0b2"),
    "geoid": ("15s_geoid_gtif", "3c434a66cfb169bc6241db494634298d8e17ef202275752d5e7cb75c825d4470"),
}
SOURCE = "NOAA ETOPO 2022 v1 / 15 arc-sec surface"
VERTICAL = "EPSG:3855"
WINDOW = (2952, 1904, 312, 208)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_memory():
    if __import__("os").name != "nt":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if __import__("sys").platform == "darwin" else value * 1024

    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD),
                    ("peak_working_set", ctypes.c_size_t),
                    ("other", ctypes.c_size_t * 7)]

    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    if not info(current(), ctypes.byref(counters), counters.size):
        raise ctypes.WinError()
    return counters.peak_working_set


def prepare(directory):
    import rasterio
    from rasterio.windows import Window

    window = Window(*WINDOW)
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    subsets = directory / "subset"
    subsets.mkdir()
    sources = {}
    for role, (folder, expected) in TILES.items():
        name = f"ETOPO_2022_v1_15s_N45W135_{role}.tif"
        url, path = f"{BASE}{folder}/{name}", inputs / name
        if not path.exists():
            with urlopen(url) as response, path.with_suffix(".download").open("wb") as staged:
                shutil.copyfileobj(response, staged)
            staged_path = path.with_suffix(".download")
            if sha(staged_path) != expected:
                raise ValueError(f"Downloaded tile bytes differ from the pinned tile: {name}")
            staged_path.rename(path)
        if sha(path) != expected:
            raise ValueError(f"Tile bytes differ from the pinned tile: {name}")
        target = subsets / f"{role}.tif"
        with rasterio.open(path) as src:
            values = src.read(1, window=window, masked=True)
            transform = src.window_transform(window)
            with rasterio.open(target, "w", driver="GTiff", width=int(window.width),
                               height=int(window.height), count=1, dtype=src.dtypes[0],
                               crs=src.crs, transform=transform, nodata=src.nodata) as dst:
                dst.write(values.data, 1)
                dst.update_tags(**src.tags())
                dst.set_band_unit(1, src.units[0])
            sources[role] = {
                "upstream_url": url, "upstream_sha256": expected, "subset_sha256": sha(target),
                "pixel_sha256": hashlib.sha256(values.data.astype(values.dtype.newbyteorder("<")).tobytes()).hexdigest(),
                "crs": src.crs.to_string(), "nodata": src.nodata, "unit_tag": src.units[0],
                "transform": tuple(transform)[:6], "technical_tags": src.tags(),
                "valid_pixels": int(values.count()), "value_range": [float(values.min()), float(values.max())],
            }
            if role == "surface_sid":
                sources[role]["cell_counts_by_source_id"] = dict(Counter(map(int, values.compressed())))
    guide = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/docs/1.2%20ETOPO%202022%20User%20Guide.pdf"
    write_json(subsets / "source.json", {
        "product": SOURCE, "doi": "https://doi.org/10.25921/fd45-gt74", "user_guide": guide,
        "window_col_row_width_height": WINDOW, "vertical_crs": VERTICAL,
        "vertical_reference": "EGM2008 orthometric height, positive up, metres",
        "geoid_use": "Companion only. Heights remain EGM2008; no geoid offset applied.",
        "source_id_use": "Companion categorical provenance, unresampled. Its metre tag is inherited.",
        "source_id_names": {"4": "NOAA Regional DEMs", "5": "GMRT v4.0", "9": "Copernicus DEM 30m", "10": "FABDEM"},
        "software": {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__}, "layers": sources,
    })


def render(subset, recipe_path, output):
    from forge3d.measured_sections import Transect, render_section, sample_transect

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    raster = subset / "surface.tif"
    if sha(raster) != json.loads((subset / "source.json").read_text())["layers"]["surface"]["subset_sha256"]:
        raise ValueError("Subset no longer matches the preparation receipt")
    start = time.perf_counter()
    section = sample_transect(raster, Transect(recipe["start"], recipe["end"], recipe["spacing_m"]),
                              source_id=SOURCE, vertical_crs=VERTICAL)
    draw_start = time.perf_counter()
    render_section(section, raster, output, title=recipe["title"])
    end = time.perf_counter()
    shutil.copyfile(recipe_path, output / "recipe.json")
    write_json(output / "timing.json", {
        "sample_seconds": draw_start - start, "render_and_export_seconds": end - draw_start,
        "peak_process_working_set_bytes": peak_memory(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("directory", type=Path)
    build = commands.add_parser("render")
    for name in ("subset", "recipe", "output"):
        build.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.directory)
    else:
        render(args.subset, args.recipe, args.output)


if __name__ == "__main__":
    main()
