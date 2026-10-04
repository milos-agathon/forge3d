"""Step 1 of 3: download the Bryce Canyon flyover data into ./data.

  Bryce_Canyon.tif     USGS 1 m lidar, the 3800 x 3800 m flight tile
  context_dem_2m.tif   12 288 m square at 2 m: USGS 3DEP 1/3" with the lidar tile blended in
  naip_tile_1m.png     NAIP imagery at 1 m over the flight tile
  naip_context_4m.png  NAIP imagery at 4 m over the context square

Files that already exist are kept, so an interrupted run can simply be restarted.
Everything is in UTM zone 12N (EPSG:26912).
"""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

DATA = Path(__file__).resolve().parent / "data"
TNM = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation"
# UT Southern QL1 2018 fully covers the tile (the newer Kane 2020 tile has gaps there).
LIDAR_URL = f"{TNM}/1m/Projects/UT_Southern_QL1_2018/TIFF/USGS_one_meter_x39y417_UT_Southern_QL1_2018.tif"
DEP_URL = f"{TNM}/13/TIFF/current/n38w113/USGS_13_n38w113.tif"
NAIP_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
CRS = "EPSG:26912"

# Flight tile: the source pixels from corner (395998, 4165574), read 1:1 without resampling.
# The grid is labelled half a metre east/north of that corner (as in forge3d's Bryce_Canyon.tif
# sample); the imagery uses the same labels, so terrain and imagery stay registered.
TILE, TILE_LEFT, TILE_TOP = 3800, 395998.5, 4165574.5
CTX, CTX_RES = 12288.0, 2.0
CTX_LEFT, CTX_TOP = TILE_LEFT + TILE / 2 - CTX / 2, TILE_TOP - TILE / 2 + CTX / 2
FEATHER_PX = 30  # 60 m blend from lidar to 3DEP at the tile edge


def save_tif(path: Path, values: np.ndarray, left: float, top: float, res: float) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path.name}: source data has gaps")
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1,
                       dtype="float32", crs=CRS, transform=from_origin(left, top, res, res), nodata=np.nan,
                       compress="deflate", tiled=True) as dst:
        dst.write(values.astype(np.float32), 1)
    tmp.replace(path)
    print(f"wrote {path.name}")


def lidar_tile() -> np.ndarray:
    with rasterio.open("/vsicurl/" + LIDAR_URL) as src:
        col, row = ~src.transform * (TILE_LEFT - 0.5, TILE_TOP - 0.5)
        return src.read(1, window=Window(round(col), round(row), TILE, TILE)).astype(np.float32)


def context_dem(lidar: np.ndarray) -> np.ndarray:
    n = int(CTX / CTX_RES)
    dem = np.full((n, n), np.nan, dtype=np.float32)
    with rasterio.open("/vsicurl/" + DEP_URL) as src:
        reproject(rasterio.band(src, 1), dem, dst_transform=from_origin(CTX_LEFT, CTX_TOP, CTX_RES, CTX_RES),
                  dst_crs=CRS, src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    lidar2 = lidar.reshape(TILE // 2, 2, TILE // 2, 2).mean(axis=(1, 3))
    r0, c0 = round((CTX_TOP - TILE_TOP) / CTX_RES), round((TILE_LEFT - CTX_LEFT) / CTX_RES)
    edge = np.minimum.outer(*(2 * [np.minimum(np.arange(TILE // 2), np.arange(TILE // 2)[::-1])]))
    w = np.clip(edge / FEATHER_PX, 0.0, 1.0).astype(np.float32)
    region = dem[r0:r0 + TILE // 2, c0:c0 + TILE // 2]
    region[:] = w * lidar2 + (1.0 - w) * region
    return dem


def naip(path: Path, left: float, top: float, size_m: float, px: int) -> None:
    bbox = (left, top - size_m, left + size_m, top)
    query = urllib.parse.urlencode({
        "bbox": ",".join(map(str, bbox)), "bboxSR": 26912, "imageSR": 26912, "size": f"{px},{px}",
        "format": "png32", "pixelType": "U8", "bandIds": "0,1,2", "transparent": "true",
        "interpolation": "RSP_BilinearInterpolation", "f": "json"})
    with urllib.request.urlopen(f"{NAIP_URL}?{query}") as response:
        meta = json.load(response)
    extent = meta.get("extent", {})
    if "error" in meta or tuple(extent.get(k) for k in ("xmin", "ymin", "xmax", "ymax")) != bbox:
        raise RuntimeError(f"{path.name}: unexpected NAIP export response: {meta}")
    with urllib.request.urlopen(meta["href"]) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGBA")
    if image.size != (px, px) or np.asarray(image.getchannel("A")).min() < 255:
        raise RuntimeError(f"{path.name}: NAIP image is the wrong size or has gaps")
    tmp = path.with_suffix(".part.png")
    image.convert("RGB").save(tmp)
    tmp.replace(path)
    print(f"wrote {path.name}")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    tile_path, ctx_path = DATA / "Bryce_Canyon.tif", DATA / "context_dem_2m.tif"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        if not tile_path.exists():
            save_tif(tile_path, lidar_tile(), TILE_LEFT, TILE_TOP, 1.0)
        if not ctx_path.exists():
            with rasterio.open(tile_path) as src:
                save_tif(ctx_path, context_dem(src.read(1)), CTX_LEFT, CTX_TOP, CTX_RES)
    for name, left, top, size_m, px in [("naip_tile_1m.png", TILE_LEFT, TILE_TOP, TILE, 3800),
                                        ("naip_context_4m.png", CTX_LEFT, CTX_TOP, CTX, 3072)]:
        if not (DATA / name).exists():
            naip(DATA / name, left, top, size_m, px)


if __name__ == "__main__":
    main()
