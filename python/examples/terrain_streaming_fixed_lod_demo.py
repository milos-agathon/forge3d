# E1/E6: Terrain streaming demo for fixed LOD mosaic
# Usage: python python/examples/terrain_streaming_fixed_lod_demo.py

from __future__ import annotations
import os

try:
    from forge3d import TerrainSpike  # requires native module built with --features terrain_spike
except Exception as e:
    raise SystemExit("TerrainSpike not available. Build with: maturin develop -r --features terrain_spike") from e


def main() -> None:
    out = os.path.abspath("terrain_streaming_fixed_lod.png")
    width, height = 1024, 768
    r = TerrainSpike(width, height, 256, "viridis")

    # Enable tiling over a square world; bounds are arbitrary world units
    # Here we align world UV to [0,1] implicitly by using the full grid plane
    r.enable_tiling(-2000.0, -2000.0, 2000.0, 2000.0, cache_capacity=64, max_lod=5)

    # Fixed-LOD mosaic configuration: choose LOD=2 (4x4 tiles across world)
    lod = 2
    tiles_n = 1 << lod
    r.enable_height_mosaic(tile_px=64, tiles_x=tiles_n, tiles_y=tiles_n, fixed_lod=lod, filter_linear=True)

    # Position camera and stream visible tiles at the chosen LOD
    eye = (6.0, 4.0, 6.0)
    target = (0.0, 0.0, 0.0)
    up = (0.0, 1.0, 0.0)
    r.set_camera_look_at(eye, target, up, 45.0, 0.1, 100.0)

    # Convert to direction (normalized look vector)
    cam_dir = (
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    )

    visible = r.stream_tiles_to_height_mosaic_at_lod(
        camera_pos=eye,
        camera_dir=cam_dir,
        lod=lod,
        fov_deg=45.0,
        aspect=float(width) / float(height),
        near=0.1,
        far=200.0,
        max_uploads=16,
    )
    print("Streamed tiles (lod,x,y):", visible)

    # Render the frame
    r.render_png(out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
