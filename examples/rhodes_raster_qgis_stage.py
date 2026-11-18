"""Stage the matched QGIS camera cases for the existing C05 capture route."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

from rhodes_raster_recipe import CAMERAS, COLORS


def stage(root):
    prepared = root / "prepared"
    grid = json.loads((prepared / "sources.json").read_text())["grid"]
    palette = np.asarray([(0, 0, 0, 0), *COLORS], dtype=np.uint8)
    times = {}
    for stem in ("2023-08-07-0", "2023-08-17-2"):
        started = time.perf_counter()
        date = stem[:10]
        target = root / "qgis-3d" / stem
        qgis = target / "qgis"
        qgis.mkdir(parents=True, exist_ok=True)
        (target / "prepared").mkdir(exist_ok=True)
        for name in ("2023-07-18", date, date + "-mask"):
            shutil.copyfile(
                prepared / f"{name}.tif", target / "prepared" / f"{name}.tif"
            )
        for suffix in ("values", "classes"):
            name = f"{stem}-{suffix}.tif"
            shutil.copyfile(root / "qgis" / name, qgis / name)
        shutil.copyfile(root / "qgis" / f"{stem}.qgz", qgis / "scene.qgz")
        shutil.copyfile(prepared / "terrain.tif", qgis / "terrain.tif")
        shutil.copyfile(root / "gpu" / f"{stem}.recipe.json", target / "recipe.json")
        with rasterio.open(qgis / "terrain.tif") as dem:
            profile = dem.profile
        profile.pop("nodata", None)
        profile.update(count=4, dtype="uint8", photometric="RGB")
        classes = np.load(root / "qgis" / f"{stem}-classes.npy")
        with rasterio.open(qgis / "display.tif", "w", **profile) as display:
            display.write(np.moveaxis(palette[classes], -1, 0))
            display.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
        frames = [
            dict(frame=i, theta_deg=tilt, phi_deg=azimuth, radius=21000, fov_deg=58)
            for i, (azimuth, tilt) in enumerate(CAMERAS)
        ]
        (target / "camera-path.json").write_text(
            json.dumps({"grid": grid, "fps": 12, "frames": frames})
        )
        times[stem] = time.perf_counter() - started
    (root / "qgis-3d" / "setup-cost.json").write_text(json.dumps(times, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    stage(args.directory.resolve())
