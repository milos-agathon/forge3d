"""Render the baked two-shot consumer camera on physical viewer hardware.

This viewer frame is outside offscreen certificate scope. No signing occurs.
The output records the actual consumer pose and rendered command revision.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("FORGE3D_TEST_INSTALLED_WHEEL") != "1":
    sys.path.insert(0, str(ROOT / "python"))
from forge3d.cinema import FramingSpec, Shot, Subject, plan
from forge3d.terrain_scatter import TerrainScatterSource
from forge3d.viewer import open_viewer_async, _find_viewer_binary
from forge3d import _forge3d


def render(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    fixture = ROOT / "tests/data/cinema_scenes/terrain_attached_mountain.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    fields = dict(data["shot"])
    fields["subject"] = Subject(**fields["subject"])
    fields["framing"] = FramingSpec(**fields["framing"])
    reveal = Shot(**fields)
    orbit = replace(reveal, kind="orbit", start_eye=None, end_eye=None,
                    bearing_deg=(0., 90.), duration_s=30.)
    source = TerrainScatterSource(np.asarray(data["heightmap"], np.float32),
                                  terrain_width=data["terrain_width"])
    solved = plan([reveal, orbit], source)
    frame_time = reveal.duration_s + orbit.duration_s / 2
    state = solved.bake().evaluate(frame_time)
    assert state is not None and state.target is not None
    dem = output / "summit.tif"
    with rasterio.open(dem, "w", driver="GTiff", width=source.width, height=source.height,
                       count=1, dtype="float32", transform=Affine(source.terrain_width/source.width,
                       0, 0, 0, -source.terrain_width/source.height, 0)) as dst:
        dst.write(source._scaled_heights, 1)
    viewer = open_viewer_async(width=256, height=256, title="AUTOCHIRON rendered-shot candidate")
    try:
        viewer.load_terrain(dem)
        viewer.send_ipc({"cmd": "set_terrain_pbr", "enabled": True})
        viewer.set_orbit_camera(phi_deg=state.phi_deg, theta_deg=state.theta_deg,
                                radius=state.radius, fov_deg=state.fov_deg, target=state.target)
        camera_revision = int(viewer.get_stats()["applied_command_revision"])
        viewer.snapshot(output / "frame_a.png", width=256, height=256)
        first_stats = viewer.get_stats()
        viewer.snapshot(output / "frame_b.png", width=256, height=256)
        second_stats = viewer.get_stats()
    finally:
        viewer.close()
    assert first_stats["rendered_frame_revision"] >= camera_revision
    assert second_stats["rendered_frame_revision"] >= camera_revision
    assert first_stats["adapter_device_type"] in ("DiscreteGpu", "IntegratedGpu"), first_stats
    assert first_stats["adapter_name"] and first_stats["adapter_backend"]
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in ("frame_a.png", "frame_b.png")}
    assert hashes["frame_a.png"] == hashes["frame_b.png"], hashes
    (output / "shotplan.json").write_text(solved.to_json(), encoding="utf-8")
    metadata = {"schema": "forge3d.autochiron.viewer-golden/1", "frame_time": frame_time,
                "viewer_binary_sha256": hashlib.sha256(Path(_find_viewer_binary()).read_bytes()).hexdigest(),
                "native_binding_sha256": hashlib.sha256(Path(_forge3d.__file__).read_bytes()).hexdigest(),
                "keyframes_sha256": hashlib.sha256(solved.keyframes_json.encode()).hexdigest(),
                "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "consumer_pose": {"phi": state.phi_deg, "theta": state.theta_deg,
                                  "radius": state.radius, "fov": state.fov_deg, "target": state.target},
                "camera_command_revision": camera_revision, "first_stats": first_stats,
                "second_stats": second_stats, "png_sha256": hashes["frame_a.png"],
                "certificate_scope": "Viewer snapshot excluded; no offscreen attestation or signing"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"png_sha256": metadata["png_sha256"], "frame_time": frame_time,
                      "adapter": first_stats["adapter_name"], "backend": first_stats["adapter_backend"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args().output)
