"""Physical D03 viewer evidence through the public helper; no acceptance fallback."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge3d import VectorLineLayer
from forge3d.viewer import open_viewer_async

pytestmark = [
    pytest.mark.interactive_viewer,
    pytest.mark.skipif(os.environ.get("RUN_M06_VIEWER_CI") != "1", reason="required M-06 hardware lane only"),
]


def _reference_vertices(config, heights):
    """Independent bilinear samples, rounded at the renderer's f32 boundary."""
    rows = [v.to_array() for v in config.vertices]
    height, width = heights.shape
    for row in rows:
        u = np.float32(np.clip(row[0] / width, 0, 1)) * np.float32(width - 1)
        v = np.float32(np.clip(row[2] / height, 0, 1)) * np.float32(height - 1)
        x0, y0 = int(np.floor(u)), int(np.floor(v))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        fx, fy = u - np.float32(x0), v - np.float32(y0)
        h0 = heights[y0, x0] * (np.float32(1) - fx) + heights[y0, x1] * fx
        h1 = heights[y1, x0] * (np.float32(1) - fx) + heights[y1, x1] * fx
        sampled = h0 * (np.float32(1) - fy) + h1 * fy
        row[1] += float((sampled - heights.min()) + np.float32(config.drape_offset))
    return rows


def test_public_two_segment_terrain_drape_and_offset(tmp_path):
    from tests.test_m06_full_geospatial_viewer import ARTIFACT_DIR

    yy, xx = np.mgrid[:128, :128]
    heights = (100 + .02 * xx + .04 * yy + .0001 * xx * yy).astype(np.float32)
    terrain = tmp_path / "d03-asymmetric.tif"
    Image.fromarray(heights).save(terrain)
    line = VectorLineLayer(
        "road", [(25, 0, 25), (70, 0, 25), (100, 0, 95)], width=6,
        halo=2, color=(1, 0, 0, 1), halo_color=(0, 0, 1, 1),
        cap="square", join="bevel", drape=True, z_offset=.5, feature_id=31,
    )
    viewer = open_viewer_async(width=640, height=400)
    evidence = {}
    try:
        viewer.load_terrain(terrain)
        viewer.send_ipc({"cmd": "set_terrain_pbr", "sky": {"enabled": False}})
        viewer.send_ipc({"cmd": "set_terrain", "sun_intensity": 0, "ambient": 1})
        viewer.set_orbit_camera(90, 52, 180, fov_deg=50, target=(64, 4, 64))
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        viewer.snapshot(ARTIFACT_DIR / "d03-terrain.png", width=640, height=400)
        baseline = np.asarray(Image.open(ARTIFACT_DIR / "d03-terrain.png").convert("RGB"))
        identity = {key: value for key, value in viewer.get_stats().items() if key.startswith("adapter_")}
        assert identity["adapter_device_type"].lower() == "discretegpu", identity
        assert identity["adapter_backend"].lower() == "vulkan", identity
        evidence["adapter"] = identity
        binary = Path(os.environ["FORGE3D_VIEWER_BINARY"])
        evidence["viewer_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        evidence["terrain_sha256"] = hashlib.sha256(terrain.read_bytes()).hexdigest()
        (ARTIFACT_DIR / terrain.name).write_bytes(terrain.read_bytes())
        evidence["fixture"] = {"size": [128, 128], "frame": [640, 400],
                               "domain": [float(heights.min()), float(heights.max())],
                               "lighting": "ambient=1, sun_intensity=0; sky disabled"}
        captures = []
        for offset in (.5, 8.5):
            current = replace(line, z_offset=offset)
            overlay_id = viewer.add_vector_line(current)
            capture = ARTIFACT_DIR / f"d03-offset-{offset}.png"
            viewer.snapshot(capture, width=640, height=400)
            frame = np.asarray(Image.open(capture).convert("RGB"))
            changed = (frame != baseline).any(axis=2)
            counts = {}
            for name, channel in (("stroke", 0), ("halo", 2)):
                other = [c for c in range(3) if c != channel]
                mask = changed & (frame[..., channel] > frame[..., other].sum(axis=2))
                pixels = np.argwhere(mask)
                assert len(pixels) > 0, f"no physical {name} pixels"
                counts[name] = len(pixels)
            viewer.send_ipc({"cmd": "remove_vector_overlay", "id": overlay_id})
            config = current.to_overlay_config()
            reference = viewer.add_vector_overlay(
                "independent-drape-reference", _reference_vertices(config, heights),
                config.indices, primitive="triangles", drape=False, drape_offset=0,
                depth_bias=config.depth_bias, line_width=config.line_width,
            )
            reference_path = ARTIFACT_DIR / f"d03-reference-{offset}.png"
            viewer.snapshot(reference_path, width=640, height=400)
            expected = np.asarray(Image.open(reference_path).convert("RGB"))
            mismatched_pixels = int((frame != expected).any(axis=2).sum())
            assert mismatched_pixels == 0, f"native drape differs at {mismatched_pixels} pixels"
            captures.append({"offset": offset, "pixels": counts,
                             "reference_mismatched_pixels": mismatched_pixels,
                             "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            overlay_id = reference
            viewer.send_ipc({"cmd": "remove_vector_overlay", "id": overlay_id})
        evidence.update(captures=captures, vertices=line.to_overlay_config().vertex_count,
                        triangles=line.to_overlay_config().index_count // 3)
        assert captures[0]["rgb_sha256"] != captures[1]["rgb_sha256"]
    finally:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "d03-measurements.json").write_text(json.dumps(evidence, indent=2), encoding="utf8")
        viewer.close()
