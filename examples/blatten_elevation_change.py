"""Compare the official swisstopo 19 and 30 May 2025 DSMs on their native grid."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from forge3d import elevation_change
from forge3d.raster_registration import geotiff_registration

# LV95 edges of the meadow chosen from both quick orthophotos before differencing.
CONTROL = (2628180, 1140814, 2628194, 1140825)
HASHES = ("6b420277bfc5404d708f0bea54d3c032d0864b0d2bcd93ccea693dbecffe9d6c",
          "6bcc84392634b44ecebff90485968912e68b0c65d035968c4609ee0a8b707b22")


def checksum(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def exact_window(dataset, bounds):
    win = from_bounds(*bounds, transform=dataset.transform)
    if any(v != int(v) for v in win.flatten()):
        raise ValueError("crop would require resampling")
    return Window(*map(int, win.flatten()))


def demonstrate(may19: Path, may30: Path, output: Path, confidence: float):
    for path, expected in zip((may19, may30), HASHES):
        if checksum(path) != expected:
            raise ValueError(f"unexpected DSM checksum for {path}")
    with TemporaryDirectory() as tmp, rasterio.open(may19) as pre, rasterio.open(may30) as post:
        if pre.crs != post.crs or geotiff_registration(pre) != geotiff_registration(post):
            raise ValueError("incompatible source CRS or registration")
        window = exact_window(post, pre.bounds)
        if (window.width, window.height) != (pre.width, pre.height) or post.window_transform(window) != pre.transform:
            raise ValueError("survey grids differ within the proposed crop")
        profile = dict(driver="GTiff", count=1, transform=pre.transform, crs=pre.crs,
                       width=pre.width, height=pre.height, tiled=True, compress="deflate")
        crop = Path(tmp) / "may30.tif"
        control = Path(tmp) / "stable.tif"
        with rasterio.open(crop, "w", **profile, dtype=post.dtypes[0], nodata=post.nodata) as dst:
            for _, part in dst.block_windows(1):
                src = Window(window.col_off + part.col_off, window.row_off + part.row_off,
                             part.width, part.height)
                dst.write(post.read(1, window=src, masked=True).filled(post.nodata), 1, window=part)
            dst.update_tags(AREA_OR_POINT=geotiff_registration(post).title())
        patch = exact_window(pre, CONTROL)
        with rasterio.open(control, "w", **profile, dtype="uint8") as dst:
            dst.write(np.ones((patch.height, patch.width), dtype="uint8"), 1, window=patch)
        source = ("swisstopo DSMs · difference spread estimated on a shared meadow control\n"
                  "Local grassland calibration extrapolated across the footprint.")
        return elevation_change(
            may19, crop, output, confidence=confidence,
            vertical_reference1="LN02", vertical_reference2="LN02", stable_ground=control,
            title="Blatten  |  Surface change, 19–30 May 2025", error_source=source,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("may19", type=Path)
    parser.add_argument("may30", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confidence", type=float, required=True)
    args = parser.parse_args()
    print(demonstrate(args.may19, args.may30, args.output, args.confidence))
