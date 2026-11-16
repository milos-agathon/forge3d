"""Regenerate real CBS flat-terrain oblique pages and matching title/end PNGs.

Usage: python examples/c06_oblique_publication.py PREPARED_C06_DIRECTORY OUTPUT
Requires ordinary production MapPlate entitlement; no test license is enabled.
"""
import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys

import rasterio  # Resolve real Rasterio before the repository import shim.
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _import_shim  # noqa: E402,F401
from forge3d.map_plate import BBox, MapPlateConfig
from forge3d.map_scene import OrbitCamera
from forge3d.publication import PublicationSource, PublicationExtent, PublicationRecipe, PublicationLayout


def run(prepared, output):
    inputs = [json.loads((prepared / "base" / f"{city}.json").read_text())
              for city in ("amsterdam", "rotterdam")]
    base = inputs[0]
    source = PublicationSource(**{f.name: base["source"][f.name] for f in fields(PublicationSource)})
    if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != base["source"]["sha256"]:
        raise ValueError("Prepared CBS source changed; rerun preparation")
    grid = BBox(**base["source"]["grid"])
    camera = OrbitCamera(distance=8000, elevation_deg=30, azimuth_deg=270, fov_deg=35)
    pages = []
    for receipt in inputs:
        if receipt["source"] != base["source"]:
            raise ValueError("Prepared pages must share one source")
        xy = np.asarray(receipt["footprint"]["coordinates"])
        bounds = BBox(float(xy[:, 0].min()), float(xy[:, 1].min()),
                      float(xy[:, 0].max()), float(xy[:, 1].max()), grid.crs)
        pages.append(PublicationExtent(receipt["page_id"], receipt["title"], bounds, camera))
    breaks = [row["right"] for row in base["classification"][1:] if row["right"] is not None]
    palette = [row["rgba"] for row in base["legend"]["entries"][1:]]
    recipe = PublicationRecipe(source, breaks, palette, grid, camera_mode="flat-perspective",
                               missing_color=base["legend"]["entries"][0]["rgba"])
    layout = PublicationLayout(MapPlateConfig(width=1920, height=1080, margin=(100, 460, 170, 50)))
    result = {}
    for revision, values in [("base", breaks), ("breaks", [25, 100, 250, 500])]:
        current = replace(recipe, breaks=values)
        result[revision] = {"pages": current.render_series(pages, output / "oblique" / revision),
                            "bookends": current.render_bookends(*pages, output / "bookends" / revision, layout=layout)}
    (output / "oblique-bookends.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.prepared, args.output)
