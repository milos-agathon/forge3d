# E4: Large DEM stress test
# Usage examples:
#   python python/examples/terrain_e4_stress_test.py --width 1280 --height 720 --world 16384 \
#       --lod 6 --tile-px 64 --mosaic-x 32 --mosaic-y 32 --steps 300 --max-uploads 64 \
#       --final-png terrain_e4_final.png
#
# Notes:
# - This focuses on streaming performance (tile uploads to the GPU mosaic) rather than rendering.
# - We avoid per-frame PNG readbacks; optionally render a single final frame to PNG for verification.
# - Cache hit rate is estimated at Python level from previously seen tile IDs.

from __future__ import annotations
import argparse
import math
import os
import time
import tempfile
from typing import List, Tuple, Set
import csv

import numpy as np

try:
    from forge3d import TerrainSpike  # requires native module built with --features terrain_spike
except Exception as e:
    raise SystemExit(
        "TerrainSpike not available. Build with: maturin develop -r --features terrain_spike"
    ) from e


TileId = Tuple[int, int, int]  # (lod, x, y)


def percentiles(ms: List[float]) -> Tuple[float, float, float, float, float]:
    if not ms:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    a = np.asarray(ms, dtype=np.float64)
    return (
        float(a.min()),
        float(np.percentile(a, 50)),
        float(a.mean()),
        float(np.percentile(a, 95)),
        float(a.max()),
    )


def camera_path(step: int, steps: int, world: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    # Sweep camera across X, small oscillation in Z, fixed Y height
    t = step / max(1, steps - 1)
    x = -0.6 * world + 1.2 * world * t
    z = 0.25 * world * math.sin(2.0 * math.pi * t)
    y = 0.05 * world
    pos = (x, y, z)

    # Look roughly towards world center with slight downward tilt
    target = (0.0, 0.0, 0.0)
    dir_vec = (target[0] - x, target[1] - y, target[2] - z)
    return pos, dir_vec


def main() -> None:
    ap = argparse.ArgumentParser(description="E4: Large DEM streaming stress test")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--world", type=float, default=16384.0, help="World extent (one dimension), world spans [-world/2, +world/2]")
    ap.add_argument("--lod", type=int, default=6, help="Fixed LOD to stream into the height mosaic")
    ap.add_argument("--tile-px", type=int, default=64, help="Tile pixel size for mosaics")
    ap.add_argument("--mosaic-x", type=int, default=32, help="Mosaic tiles in X (height mosaic)")
    ap.add_argument("--mosaic-y", type=int, default=32, help="Mosaic tiles in Y (height mosaic)")
    ap.add_argument("--steps", type=int, default=300, help="Number of frames in sweep")
    ap.add_argument("--max-uploads", type=int, default=64, help="Per-frame upload budget")
    ap.add_argument("--colormap", type=str, default="viridis")
    ap.add_argument("--final-png", type=str, default="", help="Optional path to render final frame PNG")
    ap.add_argument("--mlod-final-png", type=str, default="", help="Optional path to render final multi-LOD frame PNG")
    ap.add_argument("--csv", type=str, default="", help="Optional path to write per-frame CSV metrics")
    ap.add_argument("--overlay", action="store_true", help="Also stress-test overlay streaming")
    ap.add_argument("--overlay-alpha", type=float, default=0.85)
    # Coalescing policy controls
    ap.add_argument("--coalesce-policy", type=str, default="coarse", choices=["coarse", "fine"], help="LOD coalescing policy for height")
    ap.add_argument("--overlay-coalesce-policy", type=str, default=None, help="LOD coalescing policy for overlay (defaults to --coalesce-policy)")
    # Multi-LOD rendering options
    ap.add_argument("--mlod", action="store_true", help="Enable multi-LOD rendering test")
    ap.add_argument(
        "--mlod-pixel-error",
        type=float,
        default=1.5,
        help="Screen-space error budget (pixels) for multi-LOD tile selection",
    )
    ap.add_argument(
        "--mlod-freq",
        type=int,
        default=0,
        help="Render multi-LOD every N frames for profiling (0=disabled)",
    )
    ap.add_argument(
        "--mlod-max-uploads",
        type=int,
        default=64,
        help="Per-frame upload budget for multi-LOD render",
    )
    ap.add_argument("--mlod-morph", type=float, default=1.0, help="LOD morph factor [0..1]")
    ap.add_argument(
        "--mlod-coarse-factor",
        type=float,
        default=1.0,
        help="Coarse sampling quantization factor (>=1)",
    )
    ap.add_argument(
        "--mlod-skirt-depth",
        type=float,
        default=0.0,
        help="Skirt depth in world units (>=0)",
    )
    args = ap.parse_args()

    width, height = int(args.width), int(args.height)
    r = TerrainSpike(width, height, 256, args.colormap)

    half = 0.5 * float(args.world)
    r.enable_tiling(-half, -half, +half, +half, cache_capacity=256, max_lod=max(1, int(args.lod) + 2))
    # E1c/E1e: Enable async tile loader with dedup/backpressure (defaults: res=64, in_flight=32)
    if hasattr(r, 'enable_async_loader'):
        try:
            r.enable_async_loader(tile_resolution=int(args.tile_px), max_in_flight=64, coalesce_policy=str(args.coalesce_policy))
        except Exception as e:
            print("Warning: enable_async_loader failed:", e)

    # Height mosaic (non-fixed LOD so we can use a small atlas)
    r.enable_height_mosaic(tile_px=int(args.tile_px), tiles_x=int(args.mosaic_x), tiles_y=int(args.mosaic_y), fixed_lod=None, filter_linear=True)
    # E1: enable page table if available (for overlay shader storage buffer demo)
    if hasattr(r, 'enable_page_table'):
        try:
            r.enable_page_table()
        except Exception as e:
            print("Warning: enable_page_table failed:", e)

    # Optional overlay mosaic
    if bool(args.overlay):
        try:
            if hasattr(r, 'enable_overlay_mosaic'):
                r.enable_overlay_mosaic(tile_px=int(args.tile_px), tiles_x=int(args.mosaic_x), tiles_y=int(args.mosaic_y), srgb=True, filter_linear=True)
            # E3/E1 parity: enable async overlay loader (RGBA8)
            if hasattr(r, 'enable_async_overlay_loader'):
                try:
                    policy = str(args.overlay_coalesce_policy) if args.overlay_coalesce_policy is not None else str(args.coalesce_policy)
                    r.enable_async_overlay_loader(tile_resolution=int(args.tile_px), max_in_flight=64, coalesce_policy=policy)
                except Exception as e:
                    print("Warning: enable_async_overlay_loader failed:", e)
            if hasattr(r, 'set_overlay_enabled'):
                r.set_overlay_enabled(True)
            if hasattr(r, 'set_overlay_alpha'):
                r.set_overlay_alpha(float(args.overlay_alpha))
        except Exception as e:
            print("Warning: failed to enable overlay mosaic:", e)

    # Optional: multi-LOD tuning
    if bool(args.mlod):
        try:
            if float(args.mlod_morph) != 1.0 or float(args.mlod_coarse_factor) != 1.0:
                r.set_lod_morph(float(args.mlod_morph), float(args.mlod_coarse_factor))
            if float(args.mlod_skirt_depth) != 0.0:
                r.set_skirt_depth(float(args.mlod_skirt_depth))
        except Exception as e:
            print("Warning: failed to set multi-LOD parameters:", e)

    # Tracking sets to estimate cache hit ratio (approximate; does not reflect GPU LRU evictions)
    seen_height: Set[TileId] = set()
    seen_overlay: Set[TileId] = set()

    # Metrics per frame
    stream_ms: List[float] = []
    vis_counts: List[int] = []
    upload_counts: List[int] = []

    overlay_stream_ms: List[float] = []
    overlay_vis_counts: List[int] = []
    overlay_upload_counts: List[int] = []

    # Multi-LOD render timings (includes PNG readback time when enabled)
    mlod_ms: List[float] = []

    aspect = float(width) / float(height)
    fov = 45.0
    near = 0.1
    far = float(args.world) * 0.8

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(os.path.abspath(args.csv), "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame",
            "stream_ms",
            "visible",
            "upload_est",
            "overlay_stream_ms",
            "overlay_visible",
            "overlay_upload_est",
            "mlod_ms",
        ])

    for i in range(int(args.steps)):
        pos, cam_dir = camera_path(i, int(args.steps), float(args.world))

        # Stream height tiles
        t0 = time.perf_counter()
        visible = r.stream_tiles_to_height_mosaic_at_lod(
            camera_pos=pos,
            camera_dir=cam_dir,
            lod=int(args.lod),
            fov_deg=fov,
            aspect=aspect,
            near=near,
            far=far,
            max_uploads=int(args.max_uploads),
        )
        t1 = time.perf_counter()

        vis_counts.append(len(visible))
        # Approximate uploads this frame based on previously unseen tiles
        estimated_uploads = 0
        if int(args.max_uploads) > 0:
            for (lod, x, y) in visible:
                key = (int(lod), int(x), int(y))
                if key not in seen_height:
                    seen_height.add(key)
                    estimated_uploads += 1
                    if estimated_uploads >= int(args.max_uploads):
                        break
        upload_counts.append(estimated_uploads)
        stream_ms.append((t1 - t0) * 1000.0)

        # Optional overlay streaming using same camera
        if bool(args.overlay):
            ovis = []
            t2 = time.perf_counter()
            try:
                if hasattr(r, 'stream_tiles_to_overlay_mosaic_at_lod'):
                    ovis = r.stream_tiles_to_overlay_mosaic_at_lod(
                        camera_pos=pos,
                        camera_dir=cam_dir,
                        lod=int(args.lod),
                        fov_deg=fov,
                        aspect=aspect,
                        near=near,
                        far=far,
                        max_uploads=int(args.max_uploads),
                    )
                else:
                    # Fallback: approximate overlay visibility using height visibility at same LOD
                    ovis = visible
            except Exception as e:
                print("Warning: overlay streaming failed:", e)
                ovis = visible
            t3 = time.perf_counter()

            overlay_vis_counts.append(len(ovis))
            o_up = 0
            if int(args.max_uploads) > 0:
                for (lod, x, y) in ovis:
                    key = (int(lod), int(x), int(y))
                    if key not in seen_overlay:
                        seen_overlay.add(key)
                        o_up += 1
                        if o_up >= int(args.max_uploads):
                            break
            overlay_upload_counts.append(o_up)
            overlay_stream_ms.append((t3 - t2) * 1000.0)

        # Optional multi-LOD render timing every N frames
        if bool(args.mlod) and int(args.mlod_freq) > 0 and (i % int(args.mlod_freq) == 0):
            tmp_path = os.path.join(tempfile.gettempdir(), f"forge3d_mlod_{i:05d}.png")
            tml0 = time.perf_counter()
            mlod_time_ms = None
            try:
                if hasattr(r, 'render_multi_lod_png'):
                    r.render_multi_lod_png(
                        tmp_path,
                        camera_pos=pos,
                        camera_dir=cam_dir,
                        pixel_error=float(args.mlod_pixel_error),
                        fov_deg=fov,
                        aspect=aspect,
                        near=near,
                        far=far,
                        max_uploads=int(args.mlod_max_uploads),
                    )
                elif hasattr(r, 'render_mlod_png'):
                    r.render_mlod_png(
                        tmp_path,
                        camera_pos=pos,
                        camera_dir=cam_dir,
                        pixel_error=float(args.mlod_pixel_error),
                        max_uploads=int(args.mlod_max_uploads),
                    )
                else:
                    # Fallback: basic render only
                    r.render_png(tmp_path)
            except Exception as e:
                print("Warning: multi-LOD render failed:", e)
            tml1 = time.perf_counter()
            mlod_time_ms = (tml1 - tml0) * 1000.0
            mlod_ms.append(mlod_time_ms)
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # CSV per-frame row
        if csv_writer is not None:
            row = [
                i,
                (t1 - t0) * 1000.0,
                len(visible),
                estimated_uploads,
            ]
            if bool(args.overlay):
                row.extend([
                    (t3 - t2) * 1000.0,
                    len(ovis),
                    o_up,
                ])
            else:
                row.extend(["", "", ""])  # overlay fields empty
            row.append(mlod_time_ms if 'mlod_time_ms' in locals() and mlod_time_ms is not None else "")
            csv_writer.writerow(row)

    # Final optional render for visual verification
    if args.final_png:
        pos, cam_dir = camera_path(int(args.steps) - 1, int(args.steps), float(args.world))
        # Set camera before rendering
        target = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        r.set_camera_look_at(pos, target, up, fov, near, far)
        r.render_png(os.path.abspath(args.final_png))
        print("Wrote final PNG:", os.path.abspath(args.final_png))

    # Final optional multi-LOD render
    if bool(args.mlod) and args.mlod_final_png:
        pos, cam_dir = camera_path(int(args.steps) - 1, int(args.steps), float(args.world))
        tml0 = time.perf_counter()
        try:
            if hasattr(r, 'render_multi_lod_png'):
                r.render_multi_lod_png(
                    os.path.abspath(args.mlod_final_png),
                    camera_pos=pos,
                    camera_dir=cam_dir,
                    pixel_error=float(args.mlod_pixel_error),
                    fov_deg=fov,
                    aspect=aspect,
                    near=near,
                    far=far,
                    max_uploads=int(args.mlod_max_uploads),
                )
            elif hasattr(r, 'render_mlod_png'):
                r.render_mlod_png(
                    os.path.abspath(args.mlod_final_png),
                    camera_pos=pos,
                    camera_dir=cam_dir,
                    pixel_error=float(args.mlod_pixel_error),
                    max_uploads=int(args.mlod_max_uploads),
                )
            else:
                r.render_png(os.path.abspath(args.mlod_final_png))
        except Exception as e:
            print("Warning: final multi-LOD render failed:", e)
        tml1 = time.perf_counter()
        print("Wrote final multi-LOD PNG:", os.path.abspath(args.mlod_final_png))
        print(f"Final multi-LOD render time: {(tml1 - tml0) * 1000.0:.3f} ms")

    # Summaries
    mn, p50, mean, p95, mx = percentiles(stream_ms)
    print("\nHeight streaming results:")
    print(f"  frames={len(stream_ms)} lod={int(args.lod)} world={float(args.world):.0f} tile_px={int(args.tile_px)} mosaic={int(args.mosaic_x)}x{int(args.mosaic_y)}")
    print(f"  stream time (ms): min={mn:.3f} p50={p50:.3f} mean={mean:.3f} p95={p95:.3f} max={mx:.3f}")
    if vis_counts:
        vv = np.asarray(vis_counts, dtype=np.int32)
        print(f"  visible tiles/frame: min={vv.min()} p50={int(np.percentile(vv,50))} mean={vv.mean():.2f} p95={int(np.percentile(vv,95))} max={vv.max()}")
    if upload_counts:
        uu = np.asarray(upload_counts, dtype=np.int32)
        print(f"  uploads/frame (approx): min={uu.min()} p50={int(np.percentile(uu,50))} mean={uu.mean():.2f} p95={int(np.percentile(uu,95))} max={uu.max()}")
        total_seen = len(seen_height)
        print(f"  unique tiles seen: {total_seen}")
        # Approximate cache hit rate across frames
        total_vis = int(vv.sum()) if vis_counts else 0
        total_est_uploads = int(uu.sum()) if upload_counts else 0
        total_hits = max(0, total_vis - total_est_uploads)
        hit_rate = (total_hits / total_vis) if total_vis > 0 else 0.0
        print(f"  approx cache hit rate: {hit_rate*100.0:.1f}%")

    if bool(args.overlay) and overlay_stream_ms:
        mn, p50, mean, p95, mx = percentiles(overlay_stream_ms)
        print("\nOverlay streaming results:")
        print(f"  stream time (ms): min={mn:.3f} p50={p50:.3f} mean={mean:.3f} p95={p95:.3f} max={mx:.3f}")
        if overlay_vis_counts:
            vv = np.asarray(overlay_vis_counts, dtype=np.int32)
            print(f"  visible tiles/frame: min={vv.min()} p50={int(np.percentile(vv,50))} mean={vv.mean():.2f} p95={int(np.percentile(vv,95))} max={vv.max()}")
        if overlay_upload_counts:
            uu = np.asarray(overlay_upload_counts, dtype=np.int32)
            print(f"  uploads/frame (approx): min={uu.min()} p50={int(np.percentile(uu,50))} mean={uu.mean():.2f} p95={int(np.percentile(uu,95))} max={uu.max()}")
            total_seen = len(seen_overlay)
            print(f"  unique tiles seen: {total_seen}")
            # Approximate cache hit rate across frames (overlay)
            total_vis = int(vv.sum()) if overlay_vis_counts else 0
            total_est_uploads = int(uu.sum()) if overlay_upload_counts else 0
            total_hits = max(0, total_vis - total_est_uploads)
            hit_rate = (total_hits / total_vis) if total_vis > 0 else 0.0
            print(f"  approx cache hit rate: {hit_rate*100.0:.1f}%")

    if bool(args.mlod) and mlod_ms:
        mn, p50, mean, p95, mx = percentiles(mlod_ms)
        print("\nMulti-LOD render (with PNG readback) results:")
        print(
            f"  frames={len(mlod_ms)} freq={int(args.mlod_freq)} pixel_error={float(args.mlod_pixel_error)} max_uploads={int(args.mlod_max_uploads)}"
        )
        print(f"  render time (ms): min={mn:.3f} p50={p50:.3f} mean={mean:.3f} p95={p95:.3f} max={mx:.3f}")

    if csv_file is not None:
        csv_file.close()

    # Cache stats
    try:
        cstats = r.get_cache_stats()
        print("\nTile cache stats:")
        print(f"  capacity={cstats['capacity']} current_size={cstats['current_size']} memory_usage={cstats['memory_usage_bytes']/ (1024*1024):.1f} MiB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
