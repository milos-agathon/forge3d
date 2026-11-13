"""Publish a C07 camera path using the source of an executed oblique receipt.

Requires normal production MapPlate entitlement. Camera duration is unrelated
to observation selection; the source observation stays fixed throughout.
"""
import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import rasterio  # Resolve the real package before the repository import shim.
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _import_shim  # noqa: E402,F401
from forge3d.animation import CameraAnimation
from forge3d.map_plate import BBox, MapPlateConfig
from forge3d.map_scene import OrbitCamera
from forge3d.publication import PublicationExtent, PublicationLayout, PublicationRecipe, PublicationSource


def run(receipt_path, output, *, frame_count, duration, end_azimuth, base_seed):
    if frame_count < 2 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("A moving camera sequence requires at least two frames and a positive duration")
    receipt = json.loads(Path(receipt_path).read_text())
    source = PublicationSource(**{f.name: receipt["source"][f.name] for f in fields(PublicationSource)})
    if hashlib.sha256(Path(source.path).read_bytes()).hexdigest() != receipt["source"]["sha256"]:
        raise ValueError("Source changed since the reference publication")
    camera = OrbitCamera(**{f.name:receipt["camera"][f.name] for f in fields(OrbitCamera)})
    animation = CameraAnimation()
    for seconds, azimuth in ((0, camera.azimuth_deg), (duration, end_azimuth)):
        animation.add_keyframe(time=seconds, phi=azimuth, theta=camera.elevation_deg,
                               radius=camera.distance, fov=camera.fov_deg, target=tuple(camera.target))
    seconds = np.linspace(0, duration, frame_count).tolist()
    frames = []
    for i, instant in enumerate(seconds):
        state = animation.evaluate(instant)
        frames.append(PublicationExtent(f"frame-{i:05d}", receipt["title"], BBox(**receipt["map_bounds"]),
            OrbitCamera(distance=state.radius, elevation_deg=state.theta_deg, azimuth_deg=state.phi_deg,
                        fov_deg=state.fov_deg, target=state.target)))
    breaks = [r["right"] for r in receipt["classification"][1:] if r["right"] is not None]
    recipe = PublicationRecipe(source, breaks, [r["rgba"] for r in receipt["legend"]["entries"][1:]],
        BBox(**receipt["source"]["grid"]), camera_mode="flat-perspective",
        missing_color=receipt["legend"]["entries"][0]["rgba"])
    layout = PublicationLayout(MapPlateConfig(width=1920, height=1080, margin=(100,460,170,50)))
    results = {}
    for revision, thresholds in (("base", breaks), ("breaks", [25,100,250,500])):
        start = time.perf_counter()
        result = replace(recipe, breaks=thresholds).render_flyover(frames, output / revision,
            layout=layout, base_seed=base_seed)
        results[revision] = {"elapsed_seconds": time.perf_counter()-start,
                             "manifest": f"{revision}/publication-flyover.json",
                             "frames": len(result["frames"])}
    workload = {"camera_author": "forge3d.animation.CameraAnimation", "camera_seconds": seconds,
        "duration_seconds": duration, "first_azimuth": camera.azimuth_deg, "last_azimuth": end_azimuth,
        "source_sha256": receipt["source"]["sha256"], "observation": receipt["source"]["year"],
        "results": results, "production_license": "required by the normal MapPlate export gate"}
    (output / "workload.json").write_text(json.dumps(workload, indent=2), encoding="utf8")
    return workload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--end-azimuth", type=float, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    args = parser.parse_args()
    run(args.receipt, args.output, frame_count=args.frames, duration=args.duration,
        end_azimuth=args.end_azimuth, base_seed=args.base_seed)
