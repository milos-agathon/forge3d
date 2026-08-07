#!/usr/bin/env python3
"""Swiss land cover — Verified Map Capsule recipe.

Reproduces the capsule's reference render on local hardware from the two
hash-pinned forge3d registry datasets, and emits a RenderCertificate for the
final path-traced tile. Verified against forge3d <PIN AT TASK 5>.

The scene, all lighting constants and the light-field -> overlay modulation are
the approved Swiss PT register (examples/swiss_landcover_pt_3d.py), vendored
here verbatim so the capsule is self-contained: that script is not shipped with
the repository and its data-prep half reaches the network. Data prep uses the
pre-baked registry rasters, and the plate is composed by the Swiss viewer's own
composer (examples/swiss_terrain_landcover_viewer.py).
See VERIFY.md for the check procedure.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

CAPSULE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = CAPSULE_DIR.parents[1]          # examples/
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import swiss_terrain_landcover_viewer as swiss          # noqa: E402
from forge3d.path_tracing import hybrid_render_terrain_reference  # noqa: E402
import forge3d as f3d                                   # noqa: E402

OUT_DIR = CAPSULE_DIR / "out"
SNAPSHOT = OUT_DIR / "swiss_landcover.png"
CERT_PATH = OUT_DIR / "local.certificate.json"

# Proven Swiss PT register — the swiss_landcover_pt_3d.py CLI defaults, frozen.
GRID_MAX = 1536
FRAME = 1120
TILES = 2
SPP = 1
MAX_FRAMES = 2048
MIN_FRAMES = 32
VARIANCE_THRESHOLD = 1e-3
SEED = 7
SNAPSHOT_SIZE = (1920, 1080)
SUPERSAMPLE = 2

EXIT_NO_GPU = 2
EXIT_NO_HITS = 3
EXIT_NO_CERT = 4

# --- PT scene parameters (swiss_landcover_pt_3d.py lines 43-53, verbatim) ----
SPAN_X = 100.0
RELIEF_WORLD = 3.2  # world-unit relief; Switzerland's true ratio ~1.3, gently exaggerated
CAMERA_FOV_Y = 8.0  # quasi-orthographic nadir (parallax < ~0.5%)
CAMERA_MARGIN = 1.06
# PT azimuth convention is rotated 180 deg from the viewer's compass
# convention (established on the Romania PT map): 135 here = NW light.
SUN_AZIMUTH = 135.0
SUN_ELEVATION = 28.0  # higher sun = shorter, softer alpine shadows
SUN_INTENSITY = 2.6
ENV_INTENSITY = 0.82  # strong sky fill lifts the shadowed flanks
PT_ALBEDO = (0.62, 0.62, 0.62)

# --- light-field -> overlay modulation (lines 58-67, verbatim) --------------
# The display palette is premixed toward white, so keep the floor a touch
# higher than Romania's; floor + gain <= 1.0 (darken-only, no blowout).
SHADE_LOW_PCT = 2.0
SHADE_HIGH_PCT = 99.5
SHADE_GAMMA = 0.85  # <1 lifts midtones: lighter overall plate
TERRAIN_FLOOR = 0.44
TERRAIN_GAIN = 0.56
# Chromatic light: keep the PT render's color, not just its luminance, so
# sky-lit shadows go cool while sun-lit slopes stay palette-true. The tint
# field is anchored to 1.0 over well-lit pixels; strength blends it in.
LIGHT_TINT_STRENGTH = 1.0
LIGHT_TINT_CLAMP = (0.78, 1.25)


def _pt_args() -> argparse.Namespace:
    return argparse.Namespace(
        spp=SPP, max_frames=MAX_FRAMES, min_frames=MIN_FRAMES,
        variance_threshold=VARIANCE_THRESHOLD, seed=SEED, tiles=TILES,
    )


def _neutral_sky_env(height: int = 32, width: int = 64) -> np.ndarray:
    """swiss_landcover_pt_3d._neutral_sky_env (lines 95-106), verbatim."""
    zenith = np.array([0.62, 0.72, 0.92], dtype=np.float32) * 1.1
    horizon = np.array([1.00, 0.99, 0.96], dtype=np.float32) * 0.85
    ground = np.array([0.28, 0.28, 0.29], dtype=np.float32)
    rows = np.linspace(1.0, -1.0, height, dtype=np.float32)
    env_rows = np.empty((height, 3), dtype=np.float32)
    up = np.clip(rows, 0.0, 1.0)[:, None] ** 0.65
    env_rows[:] = horizon[None, :] * (1.0 - up) + zenith[None, :] * up
    below = rows < 0.0
    down = np.clip(-rows[below], 0.0, 1.0)[:, None] ** 0.5
    env_rows[below] = horizon[None, :] * (1.0 - down) + ground[None, :] * down
    return np.repeat(env_rows[:, None, :], width, axis=1)


def _load_dem_grid(render_dem_path: Path, grid_max: int) -> np.ndarray:
    """swiss_landcover_pt_3d._load_dem_grid (lines 109-127), verbatim."""
    with rasterio.open(render_dem_path) as src:
        dem = src.read(1, masked=True)
    data = dem.filled(np.nan).astype(np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        raise RuntimeError("Render DEM has no finite samples")
    fill = float(np.nanpercentile(data[finite], 1.0))
    data = np.where(finite, data, fill)
    scale = min(1.0, grid_max / float(max(data.shape)))
    if scale < 1.0:
        new_size = (max(2, round(data.shape[1] * scale)), max(2, round(data.shape[0] * scale)))
        data = np.asarray(
            Image.fromarray(data, mode="F").resize(new_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    data = data - data.min()
    data = data / max(float(data.max()), 1e-6)
    return np.ascontiguousarray(data, dtype=np.float32)


def _camera_for_grid(rows: int, cols: int, *, tiles: int = 1, tx: int = 0, ty: int = 0) -> dict:
    """swiss_landcover_pt_3d._camera_for_grid (lines 130-144), verbatim."""
    span_z = SPAN_X * rows / cols
    half_extent = 0.5 * max(SPAN_X, span_z) * CAMERA_MARGIN
    distance = half_extent / math.tan(math.radians(CAMERA_FOV_Y / 2.0))
    tile_half = half_extent / tiles
    center_x = -half_extent + (tx + 0.5) * 2.0 * tile_half
    center_z = -half_extent + (ty + 0.5) * 2.0 * tile_half
    fov_y = math.degrees(2.0 * math.atan(math.tan(math.radians(CAMERA_FOV_Y / 2.0)) / tiles))
    return {
        "origin": (center_x, distance, center_z),
        "look_at": (center_x, 0.0, center_z),
        "up": (0.0, 0.0, -1.0),
        "fov_y": fov_y,
        "exposure": 1.0,
    }


def _pt_pass(dem_grid, frame, camera, args, *, label="full", certificate=False):
    """swiss_landcover_pt_3d._pt_pass (lines 147-177) with one added keyword,
    `certificate`, forwarded to hybrid_render_terrain_reference."""
    rows, cols = dem_grid.shape
    spacing = SPAN_X / (cols - 1)
    out = hybrid_render_terrain_reference(
        dem_grid,
        frame,
        frame,
        camera,
        spacing=(spacing, spacing),
        exaggeration=RELIEF_WORLD,
        albedo=PT_ALBEDO,
        sun_azimuth_deg=SUN_AZIMUTH,
        sun_elevation_deg=SUN_ELEVATION,
        sun_intensity=SUN_INTENSITY,
        env_map=_neutral_sky_env(),
        env_intensity=ENV_INTENSITY,
        spp=int(args.spp),
        max_frames=int(args.max_frames),
        min_frames=int(args.min_frames),
        variance_threshold=float(args.variance_threshold),
        seed=int(args.seed),
        certificate=certificate,
    )
    print(
        f"[PT:{label}] converged={out['converged']} frames={out['frames']} "
        f"variance={out['variance']:.3e} "
        f"peak_host_visible={out.get('peak_host_visible_bytes', 0) / 2**20:.1f} MiB"
    )
    rgba = out["rgba"].astype(np.float32) / 255.0
    hit = np.isfinite(out["depth"])
    rgb = np.where(hit[:, :, None], rgba[:, :, :3], np.nan).astype(np.float32)
    return rgb, hit


def _pt_light_field_certified(dem_grid: np.ndarray, frame: int, args) -> tuple:
    """Port of swiss_landcover_pt_3d._pt_light_field (lines 180-198) with ONE
    change: the final tile's hybrid_render_terrain_reference call passes
    certificate=str(CERT_PATH), so the certificate describes the last
    production pass of this run (engine/WGSL/adapter fields are common to all
    tiles). Any stale certificate is deleted first, so a leftover file from an
    earlier run can never satisfy the post-render check. The no-hits guard is
    the caller's (it maps to EXIT_NO_HITS) rather than a RuntimeError."""
    CERT_PATH.unlink(missing_ok=True)
    rows, cols = dem_grid.shape
    tiles = max(1, int(args.tiles))
    if tiles == 1:
        rgb, hit = _pt_pass(
            dem_grid, frame, _camera_for_grid(rows, cols), args,
            certificate=str(CERT_PATH),
        )
    else:
        rgb = np.full((tiles * frame, tiles * frame, 3), np.nan, dtype=np.float32)
        hit = np.zeros((tiles * frame, tiles * frame), dtype=bool)
        last = (tiles - 1, tiles - 1)
        for ty in range(tiles):
            for tx in range(tiles):
                camera = _camera_for_grid(rows, cols, tiles=tiles, tx=tx, ty=ty)
                tile_rgb, tile_hit = _pt_pass(
                    dem_grid, frame, camera, args, label=f"tile {tx},{ty}",
                    certificate=str(CERT_PATH) if (ty, tx) == last else False,
                )
                rgb[ty * frame : (ty + 1) * frame, tx * frame : (tx + 1) * frame] = tile_rgb
                hit[ty * frame : (ty + 1) * frame, tx * frame : (tx + 1) * frame] = tile_hit
    return rgb, hit


def _shade_on_overlay_grid(rgb_field, hit, overlay_size):
    """swiss_landcover_pt_3d._shade_on_overlay_grid (lines 201-231), verbatim."""
    ys, xs = np.nonzero(hit)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    crop = rgb_field[top:bottom, left:right, :]
    lum = crop @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    finite = np.isfinite(lum)
    lum = np.where(finite, lum, float(np.nanmedian(lum)))
    low = float(np.percentile(lum[finite], SHADE_LOW_PCT))
    high = float(np.percentile(lum[finite], SHADE_HIGH_PCT))
    shade = np.clip((lum - low) / max(high - low, 1e-6), 0.0, 1.0)
    shade = np.power(shade, SHADE_GAMMA, dtype=np.float32)

    # Chromatic tint: per-channel ratio to luminance, anchored so well-lit
    # pixels are neutral (1.0) and sky-lit shadows read relatively cool.
    tint = crop / np.maximum(lum, 1e-4)[:, :, None]
    tint = np.where(finite[:, :, None], tint, 1.0)
    lit = shade > 0.75
    if lit.any():
        anchor = np.median(tint[lit], axis=0)
        tint = tint / np.maximum(anchor[None, None, :], 1e-4)
    tint = np.clip(tint, LIGHT_TINT_CLAMP[0], LIGHT_TINT_CLAMP[1]).astype(np.float32)

    def _resize(field: np.ndarray) -> np.ndarray:
        img = Image.fromarray(field, mode="F").resize(overlay_size, Image.Resampling.BICUBIC)
        return np.asarray(img, dtype=np.float32)

    shade_hi = np.clip(_resize(shade), 0.0, 1.0)
    tint_hi = np.stack([_resize(np.ascontiguousarray(tint[:, :, c])) for c in range(3)], axis=-1)
    tint_hi = np.clip(tint_hi, LIGHT_TINT_CLAMP[0], LIGHT_TINT_CLAMP[1])
    return shade_hi, tint_hi


def _modulate_overlay(overlay_path: Path, shade: np.ndarray, tint: np.ndarray) -> Image.Image:
    """swiss_landcover_pt_3d._modulate_overlay (lines 234-241), verbatim."""
    overlay = np.asarray(Image.open(overlay_path).convert("RGBA"), dtype=np.uint8).copy()
    rgb = overlay[:, :, :3].astype(np.float32) / 255.0
    scale = TERRAIN_FLOOR + TERRAIN_GAIN * shade
    tint_mix = 1.0 + LIGHT_TINT_STRENGTH * (tint - 1.0)
    rgb = np.clip(rgb * scale[:, :, None] * tint_mix, 0.0, 1.0)
    overlay[:, :, :3] = np.round(rgb * 255.0).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGBA")


def _present_classes(classes: np.ndarray) -> list[int]:
    """Legend classes actually present, derived exactly as
    bosnia_terrain_landcover_viewer._build_overlay derives `present`
    (line 576): keep palette order, drop classes with no pixels."""
    return [index for index in range(len(swiss.LANDCOVER_CLASSES)) if np.any(classes == index)]


def _fit_on_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Place the map on a transparent canvas of the plate's aspect ratio. The
    Swiss composer sizes its canvas from the raw image, so this is what makes
    the poster SNAPSHOT_SIZE * SUPERSAMPLE and keeps the final downsample a
    pure scale rather than a distorting resize."""
    scale = min(size[0] / image.width, size[1] / image.height)
    fitted_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    fitted = image.convert("RGBA").resize(fitted_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.alpha_composite(
        fitted, ((size[0] - fitted_size[0]) // 2, (size[1] - fitted_size[1]) // 2)
    )
    return canvas


def _downsample_snapshot(image_path: Path, final_size: tuple[int, int], supersample: int) -> None:
    """romania_terrain_landcover_viewer._downsample_snapshot (lines 327-337)."""
    if supersample <= 1:
        return
    image = Image.open(image_path).convert("RGBA")
    if image.size == final_size:
        return
    image.resize(final_size, Image.Resampling.LANCZOS).save(image_path)


def main() -> int:
    if not f3d.has_gpu():
        print("ERROR: no GPU adapter — this capsule requires the native PT path "
              "(no CPU fallback is offered).", file=sys.stderr)
        return EXIT_NO_GPU
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas_size = (SNAPSHOT_SIZE[0] * SUPERSAMPLE, SNAPSHOT_SIZE[1] * SUPERSAMPLE)

    print("== Fetching pinned datasets (sha256-checked by the registry) ==")
    dem_path = f3d.datasets.fetch_dem("swiss")
    lc_path = f3d.datasets.fetch("swiss-land-cover")

    print("== Building overlay + class mask from pre-baked rasters ==")
    terrain_path = swiss.build_aligned_dem(dem_path)
    classes_tif = swiss.build_landcover_classes(lc_path)
    with rasterio.open(terrain_path) as t:
        target_grid = {"crs": t.crs, "transform": t.transform,
                       "width": t.width, "height": t.height, "nodata": -1}
    classes = np.asarray(
        swiss.resample_raster_to_grid(classes_tif, target_grid,
                                      resampling="mode", dst_nodata=-1)["array"],
        dtype=np.int16,
    )
    classes = swiss.despeckle_landcover_classes(classes)
    present = _present_classes(classes)
    print("Legend classes present: "
          + ", ".join(swiss.LANDCOVER_LABELS[index] for index in present))
    overlay_path = OUT_DIR / "overlay.png"
    Image.fromarray(swiss.classes_to_rgba(classes), "RGBA").save(overlay_path)

    print("== Path tracing the light field (PROMETHEUS, certified final tile) ==")
    dem_grid = _load_dem_grid(terrain_path, GRID_MAX)
    print(f"[PT] grid={dem_grid.shape[1]}x{dem_grid.shape[0]} frame={FRAME}px "
          f"spp={SPP} tiles={TILES}x{TILES}")
    rgb_field, hit = _pt_light_field_certified(dem_grid, FRAME, _pt_args())
    if not np.any(hit):
        print("ERROR: path tracer produced no terrain hits.", file=sys.stderr)
        return EXIT_NO_HITS

    print("== Modulating overlay + composing plate ==")
    with Image.open(overlay_path) as probe:
        overlay_size = probe.size
    shade, tint = _shade_on_overlay_grid(rgb_field, hit, overlay_size)
    raw = _modulate_overlay(overlay_path, shade, tint)
    raw_plate = OUT_DIR / f".{SNAPSHOT.stem}_raw.png"
    tmp = OUT_DIR / f".{SNAPSHOT.stem}_tmp.png"
    _fit_on_canvas(raw, canvas_size).save(raw_plate)
    swiss.compose_snapshot(raw_plate, tmp)
    _downsample_snapshot(tmp, SNAPSHOT_SIZE, SUPERSAMPLE)
    tmp.replace(SNAPSHOT)
    raw_plate.unlink(missing_ok=True)

    if not CERT_PATH.is_file():
        print("ERROR: render completed but no certificate was emitted.", file=sys.stderr)
        return EXIT_NO_CERT
    print(f"Success! Map: {SNAPSHOT}\nCertificate (development-signed): {CERT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
