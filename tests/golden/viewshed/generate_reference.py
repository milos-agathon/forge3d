"""Regenerate the offline HELIOS WhiteboxTools reference raster."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import forge3d
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds as window_from_bounds
import shapefile


BOUNDS = (7.0, 46.4, 8.0, 47.2)
SIZE = 64
OBSERVER_CELL = (55, 49)
OBSERVER_HEIGHT_M = 8_000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whitebox-tools", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("whitebox_switzerland_64.png"),
    )
    args = parser.parse_args()
    root = Path(__file__).parents[3]
    source = root / "assets" / "tif" / "switzerland_dem.tif"

    with rasterio.open(source) as dataset:
        orthometric = dataset.read(
            1,
            window=window_from_bounds(*BOUNDS, dataset.transform),
            out_shape=(SIZE, SIZE),
            resampling=Resampling.bilinear,
            masked=True,
        )
    if orthometric.count() != orthometric.size:
        raise RuntimeError("HELIOS source crop unexpectedly contains nodata")
    orthometric = np.asarray(orthometric, dtype=np.float32)
    ellipsoidal = forge3d.dem_orthometric_to_ellipsoidal(orthometric, BOUNDS)
    source_transform = from_bounds(*BOUNDS, SIZE, SIZE)
    target_transform, target_width, target_height = calculate_default_transform(
        "EPSG:4326", "EPSG:2056", SIZE, SIZE, *BOUNDS
    )
    projected = np.empty((target_height, target_width), dtype=np.float32)
    reproject(
        ellipsoidal,
        projected,
        src_transform=source_transform,
        src_crs="EPSG:4326",
        dst_transform=target_transform,
        dst_crs="EPSG:2056",
        resampling=Resampling.bilinear,
    )

    row, column = OBSERVER_CELL
    latitude = BOUNDS[3] - (row + 0.5) * (BOUNDS[3] - BOUNDS[1]) / SIZE
    longitude = BOUNDS[0] + (column + 0.5) * (BOUNDS[2] - BOUNDS[0]) / SIZE
    station_x, station_y = rasterio.warp.transform(
        "EPSG:4326", "EPSG:2056", [longitude], [latitude]
    )

    with tempfile.TemporaryDirectory(prefix="forge3d-helios-whitebox-") as directory:
        work = Path(directory)
        with rasterio.open(
            work / "dem.tif",
            "w",
            driver="GTiff",
            width=target_width,
            height=target_height,
            count=1,
            dtype="float32",
            crs="EPSG:2056",
            transform=target_transform,
        ) as dataset:
            dataset.write(projected, 1)
        writer = shapefile.Writer(str(work / "station"))
        writer.field("id", "N")
        writer.point(station_x[0], station_y[0])
        writer.record(1)
        writer.close()
        (work / "station.prj").write_text(
            rasterio.crs.CRS.from_epsg(2056).to_wkt(), encoding="utf-8"
        )
        command = [
            str(args.whitebox_tools.resolve()),
            "--run=Viewshed",
            f"--wd={work}",
            "--dem=dem.tif",
            "--stations=station.shp",
            "--output=whitebox.tif",
            f"--height={OBSERVER_HEIGHT_M}",
            "--compress_rasters=False",
        ]
        print(subprocess.list2cmdline(command))
        subprocess.run(command, check=True)
        with rasterio.open(work / "whitebox.tif") as dataset:
            reference = np.zeros((SIZE, SIZE), dtype=np.float32)
            reproject(
                dataset.read(1),
                reference,
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=source_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
            )

    gray = (reference > 0).astype(np.uint8) * 255
    rgba = np.dstack((gray, gray, gray, np.full_like(gray, 255)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forge3d.numpy_to_png(args.output, rgba)


if __name__ == "__main__":
    main()
