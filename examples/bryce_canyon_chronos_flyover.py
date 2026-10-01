"""Deterministic CHRONOS flythrough over the real Bryce Canyon DEM.

A morning flight (06:30-09:30 MDT) around the Bryce amphitheatre: each
frame's UTC clock drives real solar geometry via ``forge3d.sun_position_utc``,
and each frame's camera is a waypoint on a Catmull-Rom keyframe path
(``forge3d.animation.CameraAnimation``). CHRONOS compiles, renders, and
freezes every frame natively — labels, LOD, and camera frozen at compile time
into each frame's canonical payload.

The terrain is the real USGS 1 m DEM ``assets/tif/Bryce_Canyon.tif``
(EPSG:26912), decimated to the render grid and shifted to a local zero datum.

Screen-fixed captions are available via ``--captions``.

``--verify`` re-renders every frame and compares pixel hashes (roughly
doubling run time), then replays one stored compiled frame. For a true
30 fps deliverable, render at the target rate — e.g. ``--frames 178
--fps 30`` yields a ~5.9 s clip of genuinely distinct frames rather than
upconverting a lower-rate render.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from _import_shim import ensure_repo_import

ensure_repo_import()

import forge3d as f3d
from forge3d.animation import CameraAnimation
from forge3d.chronos import FlythroughManifest, render_flythrough

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = _REPO_ROOT / "assets" / "tif" / "Bryce_Canyon.tif"
DEFAULT_FRAMES = 25
DEFAULT_SIZE = (960, 540)
DEFAULT_SAMPLES = 1
DEFAULT_MAX_DIM = 1024
BASE_SEED = 20260621
SITE_LAT, SITE_LON = 37.6283, -112.1677  # Bryce Canyon amphitheatre
FLIGHT_DATE = (2026, 6, 21)
FLIGHT_START_UTC_HOUR, FLIGHT_END_UTC_HOUR = 12.5, 15.5  # 06:30-09:30 MDT (UTC-6)
BRYCE_PALETTE = "3a1f14,8b4a2a,c8763f,e8b071,f6e6cf"  # warm sandstone ramp
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent / "out" / "bryce_canyon_chronos_flyover"
)

# Camera radius must stay inside (terrain surface, clip far plane): the mesh
# camera enters the terrain below ~3600 m and clip_far is max(6000, span*1.5)
# with span = 4000 m, so every evaluated radius must land in this band.
CAMERA_RADIUS_MIN, CAMERA_RADIUS_MAX = 3600.0, 5800.0

# (time, phi_deg, theta_deg, radius_m, fov_deg) — a loop around the
# amphitheatre, dipping lower across the middle of the flight. fov 20 with
# theta capped at 42 keeps the 4 km tile overfilling the frame at every
# azimuth so the dark clear-color corners and the straight DEM edge stay
# cropped out (higher theta or wider fov exposes the far tile corner at
# frame corners).
FLIGHT_KEYFRAMES = (
    (0.0, 288.0, 34.0, 4300.0, 20.0),
    (3.0, 312.0, 38.0, 4100.0, 20.0),
    (6.0, 342.0, 42.0, 3900.0, 20.0),
    (9.0, 12.0, 38.0, 4050.0, 20.0),
    (12.0, 36.0, 34.0, 4250.0, 20.0),
)

# Screen-fixed captions, not terrain-anchored labels: the camera moves every
# frame so geographic labels would drift and lie. Windows barely overlap so at
# most one caption is visible per frame; all share one viewport anchor.
# Opt-in via --captions.
CAPTIONS = (
    ("amphitheatre", "Bryce Amphitheatre", 0.04, 0.24),
    ("silent-city", "The Silent City", 0.24, 0.44),
    ("sunset-point", "Sunset Point rim", 0.44, 0.64),
    ("paria-view", "Paria View", 0.64, 0.84),
    ("rim-drive", "Along the rim", 0.84, 1.01),
)
_CAPTION_GLYPHS = sorted({char for record in CAPTIONS for char in record[1]})


def _round6(value: float) -> float:
    return round(float(value), 6) + 0.0


def load_dem(max_dim: int) -> tuple[np.ndarray, float, float]:
    """Decimated read of the real Bryce Canyon 1 m DEM.

    Mirrors the decimating read in
    ``examples/satellite_timelapse/bryce_canyon_storm_timelapse.py::_load_dem_preview``
    without its PIL overlay/storm machinery. Heights are shifted to a local
    zero datum (finite minimum subtracted). Returns ``(heights, span_x_m,
    span_y_m)``.
    """
    if not DEM_PATH.exists():
        raise FileNotFoundError(f"Bryce Canyon DEM not found: {DEM_PATH}")
    try:
        import rasterio
        from rasterio.enums import Resampling
    except ImportError as exc:
        raise RuntimeError(
            "bryce_canyon_chronos_flyover requires rasterio to read "
            f"{DEM_PATH.name}"
        ) from exc
    with rasterio.open(DEM_PATH) as src:
        scale = min(1.0, float(max_dim) / max(src.width, src.height, 1))
        out_h = max(1, int(round(src.height * scale)))
        out_w = max(1, int(round(src.width * scale)))
        data = src.read(
            1, out_shape=(out_h, out_w), masked=True, resampling=Resampling.bilinear
        )
        span_x = float(abs(src.bounds.right - src.bounds.left))
        span_y = float(abs(src.bounds.top - src.bounds.bottom))
    filled = np.asarray(data.filled(np.nan), dtype=np.float32)
    finite = filled[np.isfinite(filled)]
    if finite.size == 0:
        raise RuntimeError(f"DEM contains no finite samples: {DEM_PATH}")
    floor = float(np.nanmin(filled))
    heights = np.where(np.isfinite(filled), filled, floor).astype(np.float32) - floor
    return np.ascontiguousarray(heights), span_x, span_y


def flight_animation() -> CameraAnimation:
    animation = CameraAnimation()
    for time, phi, theta, radius, fov in FLIGHT_KEYFRAMES:
        animation.add_keyframe(time, phi, theta, radius, fov)
    return animation


def _frame_camera(
    animation: CameraAnimation, frame_index: int, total_frames: int
) -> f3d.OrbitCamera:
    t = animation.duration * frame_index / max(total_frames - 1, 1)
    state = animation.evaluate(t)
    if state is None:
        raise ValueError("flight animation produced no camera state")
    if not (CAMERA_RADIUS_MIN <= state.radius <= CAMERA_RADIUS_MAX):
        raise ValueError(
            f"frame {frame_index}: camera radius {state.radius:.1f} m outside "
            f"[{CAMERA_RADIUS_MIN:.0f}, {CAMERA_RADIUS_MAX:.0f}] m "
            "(terrain surface / clip far plane)"
        )
    return f3d.OrbitCamera(
        target=(0.0, 0.0, 0.0),
        distance=_round6(state.radius),
        azimuth_deg=_round6(state.phi_deg),
        elevation_deg=_round6(state.theta_deg),
        fov_deg=_round6(state.fov_deg),
    )


def frame_clock_utc(frame_index: int, total_frames: int) -> tuple[int, int, int]:
    frac_hour = FLIGHT_START_UTC_HOUR + (
        (FLIGHT_END_UTC_HOUR - FLIGHT_START_UTC_HOUR)
        * frame_index
        / max(total_frames - 1, 1)
    )
    total = int(round(frac_hour * 3600.0))
    hour, rem = divmod(total, 3600)
    minute, second = divmod(rem, 60)
    return hour, minute, second


def _caption_windows(total_frames: int) -> list[tuple[str, str, int, int]]:
    windows = []
    for label_id, text, appear_frac, disappear_frac in CAPTIONS:
        appear = max(1, int(round(appear_frac * (total_frames - 1))))
        disappear = int(round(disappear_frac * (total_frames - 1)))
        if disappear <= appear:
            continue
        windows.append((label_id, text, appear, disappear))
    return windows


def _captions_layer(
    frame_index: int,
    windows: list[tuple[str, str, int, int]],
    size: tuple[int, int],
) -> f3d.LabelLayer | None:
    anchor = [_round6(size[0] * 0.06), _round6(size[1] * 0.86)]
    labels = [
        {
            "id": label_id,
            "text": text,
            "geometry": {"type": "Point", "coordinates": anchor},
        }
        for label_id, text, appear, disappear in windows
        if appear <= frame_index < disappear
    ]
    if not labels:
        return None
    return f3d.LabelLayer(
        layer_id="captions",
        labels=labels,
        glyph_atlas={"glyphs": list(_CAPTION_GLYPHS)},
        occlusion="none",
    )


def build_frame_scene(
    frame_index: int,
    *,
    total_frames: int,
    size: tuple[int, int],
    samples: int,
    heights: np.ndarray,
    span_x: float,
    span_y: float,
    animation: CameraAnimation,
    captions: bool,
) -> f3d.MapScene:
    rows, cols = heights.shape
    hour, minute, second = frame_clock_utc(frame_index, total_frames)
    pos = f3d.sun_position_utc(
        SITE_LAT, SITE_LON, *FLIGHT_DATE, hour, minute, second
    )
    # to_scene_direction() returns the toward-sun vector in MapScene's
    # sun_direction convention (azimuth 0 = +Z/north); no axis negation needed.
    direction = pos.to_scene_direction()
    sun_direction = (
        _round6(direction[0]),
        _round6(direction[1]),
        _round6(direction[2]),
    )
    layer = (
        _captions_layer(frame_index, _caption_windows(total_frames), size)
        if captions
        else None
    )
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=heights,
            crs="EPSG:26912",
            metadata={
                "width": cols,
                "height": rows,
                "resolution": [_round6(span_x / cols), _round6(span_y / rows)],
                "source_id": "bryce-canyon-usgs-1m",
                "extent_m": [_round6(span_x), _round6(span_y)],
            },
            elevation_sampling_available=True,
        ),
        camera=_frame_camera(animation, frame_index, total_frames),
        lighting=f3d.LightingPreset(
            name="outdoorsun",
            sun_direction=sun_direction,
            intensity=_round6(1.05 + 0.35 * math.sin(math.radians(pos.elevation))),
            settings={
                "colormap": BRYCE_PALETTE,
                "exaggeration": 1.75,
                "cli_params": {"camera_mode": "mesh:zup"},
                "chronos_clock_utc": f"{hour:02d}:{minute:02d}:{second:02d}",
                "solar_azimuth_deg": _round6(pos.azimuth),
                "solar_elevation_deg": _round6(pos.elevation),
            },
        ),
        output=f3d.OutputSpec(
            width=size[0], height=size[1], format="png", samples=samples
        ),
        reproducibility_profile=f3d.ReproducibilityProfile(seed=BASE_SEED),
        layers=[layer] if layer is not None else [],
    )


def build_scenes(
    *,
    total_frames: int,
    size: tuple[int, int],
    samples: int,
    max_dim: int,
    captions: bool,
) -> dict[int, f3d.MapScene]:
    heights, span_x, span_y = load_dem(max_dim)
    animation = flight_animation()
    return {
        index: build_frame_scene(
            index,
            total_frames=total_frames,
            size=size,
            samples=samples,
            heights=heights,
            span_x=span_x,
            span_y=span_y,
            animation=animation,
            captions=captions,
        )
        for index in range(total_frames)
    }


def run_example(
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    frames: int = DEFAULT_FRAMES,
    size: tuple[int, int] = DEFAULT_SIZE,
    samples: int = DEFAULT_SAMPLES,
    max_dim: int = DEFAULT_MAX_DIM,
    captions: bool = False,
    verify: bool = False,
) -> FlythroughManifest:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(
        total_frames=frames,
        size=size,
        samples=samples,
        max_dim=max_dim,
        captions=captions,
    )
    manifest = render_flythrough(
        scenes, base_seed=BASE_SEED, samples=samples, out_dir=out_dir
    )
    if verify:
        verify_dir = out_dir / "verify"
        verify_manifest = render_flythrough(
            build_scenes(
                total_frames=frames,
                size=size,
                samples=samples,
                max_dim=max_dim,
                captions=captions,
            ),
            base_seed=BASE_SEED,
            samples=samples,
            out_dir=verify_dir,
        )
        for index in range(frames):
            expected = manifest.frame_record(index)["pixel_hash"]
            actual = verify_manifest.frame_record(index)["pixel_hash"]
            if actual != expected:
                raise RuntimeError(
                    f"CHRONOS determinism check failed at frame {index}: "
                    f"{actual} != {expected}"
                )
        print(f"verify: {frames} frame pixel hashes identical across re-render")
        mid = frames // 2
        replay_path = verify_dir / f"replay_{mid:08d}.png"
        provenance = manifest.replay_frame(scenes[mid], mid, replay_path)
        expected = manifest.frame_record(mid)["pixel_hash"]
        if provenance["pixel_hash"] != expected:
            raise RuntimeError(
                f"CHRONOS replay check failed at frame {mid}: "
                f"{provenance['pixel_hash']} != {expected}"
            )
        print(f"verify: replay of frame {mid} reproduces pixel hash {expected[:16]}")
    return manifest


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    # Read-only over the CHRONOS frames: the PNG bytes are the provenance
    # record, so the encoder must never re-save or filter them.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the flyover MP4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frames_dir / "frame_%08d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            # Tag BT.709 explicitly: untagged yuv420p gets guessed as bt601 by
            # some players, visibly shifting the palette.
            "-vf",
            "scale=out_color_matrix=bt709",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the Bryce Canyon CHRONOS flyover."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument(
        "--size", type=int, nargs=2, default=DEFAULT_SIZE, metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    parser.add_argument("--captions", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    size = (int(args.size[0]), int(args.size[1]))
    manifest = run_example(
        out_dir=args.out_dir,
        frames=args.frames,
        size=size,
        samples=args.samples,
        max_dim=args.max_dim,
        captions=args.captions,
        verify=args.verify,
    )
    video_path = None
    if args.video:
        video_path = args.out_dir / "bryce_canyon_flyover.mp4"
        encode_video(args.out_dir, video_path, args.fps)

    pixel_hashes = [record["pixel_hash"] for record in manifest.frames]
    if args.json:
        print(
            json.dumps(
                {
                    "frames": len(manifest.frames),
                    "base_seed": int(manifest.base_seed),
                    "samples": int(manifest.samples),
                    "manifest": str(args.out_dir / "flythrough_manifest.json"),
                    "pixel_hashes": pixel_hashes,
                    "video": str(video_path) if video_path is not None else None,
                },
                sort_keys=True,
            )
        )
    else:
        print(f"frames: {len(manifest.frames)} (base_seed={manifest.base_seed}, samples={manifest.samples})")
        print(f"out_dir: {args.out_dir}")
        print(f"pixel_hash[0]: {pixel_hashes[0]}")
        print(f"pixel_hash[{len(pixel_hashes) - 1}]: {pixel_hashes[-1]}")
        if video_path is not None:
            print(f"video: {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
