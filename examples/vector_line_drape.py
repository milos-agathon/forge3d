"""Render a local road ribbon over asymmetric terrain, without hand-built meshes.

Run from the repository: python examples/vector_line_drape.py --output artifacts/road.png
Build the viewer first: cargo build --release --bin interactive_viewer --features async_readback,enable-gpu-instancing
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

import _import_shim  # noqa: F401
from forge3d import VectorLineLayer
from forge3d.viewer import open_viewer_async


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/vector-line-road.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:128, :128]
    heights = (100 + .02 * xx + .04 * yy + .0001 * xx * yy).astype(np.float32)
    road = VectorLineLayer.from_style(
        {"id": "local-road", "type": "line", "paint": {
            "line-color": "#ff0000", "line-width": 6,
        }, "layout": {"line-cap": "square", "line-join": "bevel"}},
        [(25, 0, 25), (70, 0, 25), (100, 0, 95)],
        halo=2, halo_color=(0, 0, 1, 1), drape=True, z_offset=.5, feature_id=31,
    )
    with tempfile.TemporaryDirectory() as directory:
        dem = Path(directory) / "terrain.npy"
        np.save(dem, heights)
        with open_viewer_async(terrain_path=dem, width=640, height=400) as viewer:
            viewer.send_ipc({"cmd": "set_terrain_pbr", "sky": {"enabled": False}})
            viewer.set_orbit_camera(90, 35, 180, fov_deg=50, target=(64, 4, 64))
            overlay_id = viewer.add_vector_line(road)
            viewer.snapshot(args.output.resolve(), width=640, height=400)
            stats = viewer.get_stats()
    print(json.dumps({
        "output": str(args.output), "overlay_id": overlay_id,
        "adapter": {key: value for key, value in stats.items() if key.startswith("adapter_")},
        "vertices": road.to_overlay_config().vertex_count,
    }, indent=2))


if __name__ == "__main__":
    main()
