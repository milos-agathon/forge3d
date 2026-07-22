"""Real GPU 600-frame ANAMNESIS acceptance workload.

This is imported by the opt-in pytest gate and can also be executed directly.
It uses the production TerrainRenderer for every cold frame. The only CPU step
is a tiny deterministic label composite, which makes the changed label visible
in exactly one frame and therefore gives the expected three-pass invalidation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

import forge3d as f3d
from forge3d.anamnesis import render_sequence
from forge3d.determinism import (
    _canonical_params_config,
    canonical_heightmap,
    write_canonical_hdr,
)


_GLYPHS = {
    " ": (0, 0, 0, 0, 0),
    "N": (0x7F, 0x02, 0x04, 0x08, 0x7F),
    "O": (0x3E, 0x41, 0x41, 0x41, 0x3E),
    "d": (0x18, 0x24, 0x24, 0x12, 0x7F),
    "e": (0x38, 0x54, 0x54, 0x54, 0x18),
    "i": (0, 0x44, 0x7D, 0x40, 0),
    "l": (0, 0x41, 0x7F, 0x40, 0),
    "m": (0x7C, 0x04, 0x18, 0x04, 0x78),
    "s": (0x48, 0x54, 0x54, 0x54, 0x24),
    "t": (0x04, 0x3F, 0x44, 0x40, 0x20),
    "u": (0x3C, 0x40, 0x40, 0x20, 0x7C),
    "w": (0x3C, 0x40, 0x30, 0x40, 0x3C),
}


def _composite_label(rgba: np.ndarray, text: str) -> np.ndarray:
    output = np.ascontiguousarray(rgba, dtype=np.uint8).copy()
    height, width, _ = output.shape
    origin_x, origin_y = 1, 1
    for character_index, character in enumerate(text):
        columns = _GLYPHS.get(character, _GLYPHS[" "])
        for x, column in enumerate(columns):
            pixel_x = origin_x + character_index * 6 + x
            if pixel_x >= width:
                break
            for y in range(7):
                pixel_y = origin_y + y
                if pixel_y < height and column & (1 << y):
                    output[pixel_y, pixel_x] = (255, 255, 255, 255)
    return output


def _recipe(text: str, *, backend: str, height_sha256: str, hdr_sha256: str) -> dict[str, Any]:
    return {
        "terrain": {
            "height_sha256": height_sha256,
            "material_set": "terrain_default/v1",
            "environment_sha256": hdr_sha256,
            "z_scale": 1.0,
        },
        "camera": {
            "path": "anamnesis-real-terrain-flythrough/v1",
            "phi_start": 20.0,
            "phi_step": 0.25,
            "theta": 45.0,
        },
        "layers": [
            {
                "kind": "label",
                "id": "summit",
                "text": text,
                "visible_frames": [250],
            }
        ],
        "anamnesis_state": {"backend": backend, "seed": 0xA11A17},
        "output": {"width": 64, "height": 64, "samples": 1, "format": "rgba8"},
    }


def run_acceptance(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    height_sha256 = hashlib.sha256(heightmap.tobytes()).hexdigest()
    hdr_path = root / "environment.hdr"
    write_canonical_hdr(str(hdr_path))
    hdr_sha256 = hashlib.sha256(hdr_path.read_bytes()).hexdigest()

    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    env_maps = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    backend = str(f3d.device_probe().get("backend", "unknown")).lower()

    def render_frame(recipe: dict[str, Any], frame: int) -> bytes:
        config = _canonical_params_config()(64, 64)
        config.cam_phi_deg = float(recipe["camera"]["phi_start"]) + frame * float(
            recipe["camera"]["phi_step"]
        )
        config.cam_theta_deg = float(recipe["camera"]["theta"])
        params = f3d.TerrainRenderParams(config)
        rendered = renderer.render_terrain_pbr_pom(
            material_set,
            env_maps,
            params,
            heightmap,
            time_seconds=frame / 60.0,
            cache=None,
        )
        rgba = np.asarray(rendered.to_numpy())
        label = recipe["layers"][0]
        if frame in label["visible_frames"]:
            rgba = _composite_label(rgba, str(label["text"]))
        return np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()

    original = _recipe(
        "Old summit",
        backend=backend,
        height_sha256=height_sha256,
        hdr_sha256=hdr_sha256,
    )
    modified = copy.deepcopy(original)
    modified["layers"][0]["text"] = "New summit"
    fingerprint = hashlib.sha256(
        Path(__file__).read_bytes() + f3d.__version__.encode("ascii")
    ).digest()

    render_sequence(
        original,
        cache=root / "warm",
        render_frame=render_frame,
        render_frame_fingerprint=fingerprint,
        verify_reads=False,
    )
    incremental = render_sequence(
        modified,
        cache=root / "warm",
        render_frame=render_frame,
        render_frame_fingerprint=fingerprint,
        verify_reads=False,
    )
    cold = render_sequence(
        modified,
        cache=root / "cold-modified",
        render_frame=render_frame,
        render_frame_fingerprint=fingerprint,
        verify_reads=False,
    )

    symmetric_difference = sorted(
        set(incremental.predicted_recompute) ^ set(incremental.observed_recompute)
    )
    result = {
        "matching_frames": sum(
            left == right
            for left, right in zip(incremental.frame_hashes, cold.frame_hashes, strict=True)
        ),
        "frame_count": len(cold.frame_hashes),
        "hash_lists_equal": incremental.frame_hashes == cold.frame_hashes,
        "predicted": len(incremental.predicted_recompute),
        "observed": len(incremental.observed_recompute),
        "symmetric_difference": symmetric_difference,
        "pass_labels": sorted({label for _, label in incremental.observed_recompute}),
        "cold_seconds": cold.elapsed_seconds,
        "incremental_seconds": incremental.elapsed_seconds,
        "speedup": cold.elapsed_seconds / incremental.elapsed_seconds,
        "cache_report": {
            "hits": len(incremental.cache_report.hits),
            "misses": len(incremental.cache_report.misses),
            "bytes_read": incremental.cache_report.bytes_read,
            "bytes_written": incremental.cache_report.bytes_written,
            "wall_ms_saved": incremental.cache_report.wall_ms_saved,
            "hit_rate": incremental.cache_report.hit_rate,
        },
        "backend": backend,
    }
    result_path = os.environ.get("FORGE3D_ANAMNESIS_RESULT_PATH")
    if result_path:
        Path(result_path).write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
    if result["matching_frames"] != 600 or not result["hash_lists_equal"]:
        raise AssertionError(f"GPU frame hash mismatch: {result}")
    if result["predicted"] != 3 or result["observed"] != 3 or symmetric_difference:
        raise AssertionError(f"GPU recompute mismatch: {result}")
    if result["pass_labels"] != ["frame.output", "label.compile", "label.composite"]:
        raise AssertionError(f"unexpected GPU recompute labels: {result}")
    if result["speedup"] < 20.0:
        raise AssertionError(f"GPU speedup below 20x: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="forge3d-anamnesis-gpu-acceptance-") as root:
        payload = json.dumps(run_acceptance(root), sort_keys=True)
        if arguments.result is not None:
            arguments.result.write_text(payload + "\n", encoding="utf-8")
        print(payload)
