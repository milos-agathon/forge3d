"""Render a canonical two-shot plan over a terrain-attached summit fixture.

Run with rebuilt bindings and a freshly built interactive_viewer. No download,
signing, or synthetic frame fallback is used. The viewer render_animation route
is outside the offscreen render-certificate scope.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
# Import before the repository shim: python/rasterio is only a test stub.
import rasterio
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from forge3d.cinema import FramingSpec, Shot, Subject, plan
from forge3d.terrain_scatter import TerrainScatterSource
from forge3d.viewer import open_viewer_async


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--size", type=int, default=720)
    args = parser.parse_args()
    if args.fps <= 0 or args.size <= 0:
        parser.error("fps and size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    scene = json.loads((Path(__file__).resolve().parents[1] / "tests/data/cinema_scenes/terrain_attached_mountain.json").read_text())
    fields = scene["shot"]
    fields["subject"] = Subject(**fields["subject"])
    fields["framing"] = FramingSpec(**fields["framing"])
    reveal = Shot(**fields)
    orbit = replace(reveal, kind="orbit", start_eye=reveal.end_eye,
                    end_eye=(200.0, reveal.end_eye[1], 300.0), bearing_deg=(0.0, 90.0), duration_s=30.0)
    heights = np.asarray(scene["heightmap"], dtype=np.float32)
    source = TerrainScatterSource(heights, terrain_width=scene["terrain_width"])
    result = plan([reveal, orbit], source)
    (args.output / "shotplan.json").write_text(result.to_json(), encoding="utf-8")
    (args.output / "validation.json").write_text(json.dumps(result.validation, indent=2), encoding="utf-8")
    dem = args.output / "summit.tif"
    # North-up GeoTIFF maps to the viewer's local positive-X/positive-Z footprint.
    with rasterio.open(dem, "w", driver="GTiff", width=source.width, height=source.height,
                       count=1, dtype="float32", transform=Affine(source.terrain_width/source.width, 0, 0,
                       0, -source.terrain_width/source.height, 0)) as dst:
        dst.write(source._scaled_heights, 1)
    viewer = open_viewer_async(width=args.size, height=args.size, title="AUTOCHIRON reveal + orbit")
    try:
        viewer.load_terrain(dem)
        # The PBR viewer uses (height-min_height)*z_scale in world units.
        # Its legacy simple preview normalizes height to pixel width instead.
        viewer.send_ipc({"cmd": "set_terrain_pbr", "enabled": True})
        center = np.mean(reveal.subject.bounds, axis=0)
        viewer.add_label("Summit", (float(center[0]), reveal.subject.bounds[1][1], float(center[2])))
        viewer.render_animation(result.bake(), args.output / "frames", fps=args.fps,
                                width=args.size, height=args.size)
        (args.output / "viewer-stats.json").write_text(json.dumps(viewer.get_stats(), indent=2), encoding="utf-8")
    finally:
        viewer.close()
    print(json.dumps({"plan": str(args.output / "shotplan.json"),
                      "frames": len(list((args.output / "frames").glob("frame_*.png")))}, sort_keys=True))


if __name__ == "__main__":
    main()
