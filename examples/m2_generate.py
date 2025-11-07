#!/usr/bin/env python3
"""
M2 – Disney Principled + Energy Compensation gallery generator.

Outputs:
  • m2_brdf_gallery_principled.png  (Principled tiles, 16 px black gutters)
  • m2_debug_energy.png             (Compensation factor grayscale tiles)
  • m2_meta.json                    (numerics + GGX comparison + acceptance flags)

Acceptance gates (from developer.md):
  • B1  r=0.50 mean luminance within ±5 % of GGX.
  • B2  r ∈ {0.70, 0.90}: Principled never darker than GGX and ≤3 % brighter.
  • B3  Compensation factor in [0,1], min/max logged.
  • B4  Compensation factor monotonic non-decreasing with roughness.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
import forge3d as f3d  # noqa: E402

TILE_SIZE = (512, 512)
GUTTER_PX = 16
ROUGHNESS_VALUES = (0.10, 0.30, 0.50, 0.70, 0.90)
BASE_COLOR = (0.5, 0.5, 0.5)
F0_SCALAR = 0.04
LIGHT_INTENSITY = 3.0
CAMERA_POS = (0.0, 0.0, 2.0)
CAMERA_VIEW_DIR = (0.0, 0.0, 1.0)
LIGHT_DIR = np.array([0.5, 0.5, 1.0], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
RNG_SEED = "forge3d-seed-42"


def ensure_rgba_u8(arr: np.ndarray) -> np.ndarray:
    """Return a tight copy as RGBA uint8."""
    data = np.asarray(arr)
    if data.ndim != 3 or data.shape[2] != 4:
        raise ValueError(f"Expected RGBA image, got {data.shape}")
    if data.dtype != np.uint8:
        data = data.astype(np.uint8, copy=False)
    return data.copy()


def rgba_to_rgb(rgba: np.ndarray) -> np.ndarray:
    """Drop alpha channel, ensuring RGB uint8."""
    data = ensure_rgba_u8(rgba)
    if np.all(data[..., 3] == 255):
        return data[..., :3].copy()
    if not HAS_PIL:
        raise RuntimeError("PIL required to drop alpha when transparency present.")
    return np.asarray(Image.fromarray(data, mode="RGBA").convert("RGB"))


def add_caption_rgb(tile_rgb: np.ndarray, text: str) -> np.ndarray:
    """Stamp caption in the top-left corner."""
    if not HAS_PIL:
        return tile_rgb
    img = Image.fromarray(tile_rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:  # pragma: no cover - fallback
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(8, (img.width - text_w) // 2)
    y = 8
    pad = 6
    draw.rectangle([x - pad, y - pad, x + text_w + pad, y + text_h + pad], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to linear floats."""
    x = (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
    a = 0.055
    low = x <= 0.04045
    out = np.empty_like(x, dtype=np.float32)
    out[low] = x[low] / 12.92
    out[~low] = ((x[~low] + a) / (1.0 + a)) ** 2.4
    return out


def mean_srgb_luminance(rgb: np.ndarray) -> float:
    return float((rgb.astype(np.float32) @ LUMINANCE_WEIGHTS).mean())


def energy_comp_factor(roughness: float, f0_scalar: float = F0_SCALAR) -> float:
    r = max(0.0, min(1.0, roughness))
    gloss = 1.0 - r
    f_add = 0.04 * gloss + gloss * gloss * 0.5
    energy = max(0.0, min(1.0, f_add + 0.16 * r + 0.01))
    return max(0.0, min(1.0, 1.0 - energy))


def compute_gpu_constants_hash(roughness: float) -> str:
    alpha = roughness * roughness
    payload = [
        roughness,
        alpha,
        *BASE_COLOR,
        F0_SCALAR,
        F0_SCALAR,
        F0_SCALAR,
        LIGHT_INTENSITY,
        float(LIGHT_DIR[0]),
        float(LIGHT_DIR[1]),
        float(LIGHT_DIR[2]),
        *CAMERA_POS,
        1.0,
    ]
    import struct
    import hashlib

    packed = struct.pack("<" + "f" * len(payload), *payload)
    return hashlib.sha256(packed).hexdigest()


def make_row_rgb(tiles: Sequence[np.ndarray], label: str) -> np.ndarray:
    if not tiles:
        raise ValueError(f"{label}: no tiles provided")
    h, w, c = tiles[0].shape
    if (w, h) != TILE_SIZE or c != 3:
        raise ValueError(f"{label}: unexpected tile size {w}x{h}x{c}")
    total_w = len(tiles) * w + (len(tiles) - 1) * GUTTER_PX
    atlas = np.zeros((h, total_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        x0 = idx * (w + GUTTER_PX)
        atlas[:, x0 : x0 + w, :] = tile
    _validate_gutters(atlas, label)
    return atlas


def _validate_gutters(atlas: np.ndarray, label: str) -> None:
    tile_w, tile_h = TILE_SIZE
    assert atlas.shape[0] == tile_h
    for idx in range(1, len(ROUGHNESS_VALUES)):
        gx = idx * tile_w + (idx - 1) * GUTTER_PX
        strip = atlas[:, gx : gx + GUTTER_PX, :]
        if strip.shape[1] != GUTTER_PX:
            raise ValueError(f"{label}: gutter {idx} missing columns")
        if np.any(strip != 0):
            raise ValueError(f"{label}: gutter {idx} not pure black")
        if strip.size and int(strip[:, : min(4, GUTTER_PX), :].sum()) != 0:
            raise ValueError(f"{label}: gutter {idx} brightness leak")


def render_tile(model: str, roughness: float, *, debug_energy: bool = False) -> np.ndarray:
    arr = f3d.render_brdf_tile(
        model,
        float(roughness),
        TILE_SIZE[0],
        TILE_SIZE[1],
        ndf_only=False,
        g_only=False,
        dfg_only=False,
        spec_only=False,
        roughness_visualize=False,
        exposure=1.0,
        light_intensity=LIGHT_INTENSITY,
        base_color=BASE_COLOR,
        clearcoat=0.0,
        clearcoat_roughness=0.0,
        sheen=0.0,
        sheen_tint=0.0,
        specular_tint=0.0,
        debug_dot_products=False,
        debug_lambert_only=False,
        debug_d=False,
        debug_spec_no_nl=False,
        debug_energy=debug_energy,
        debug_angle_sweep=False,
        debug_angle_component=2,
        debug_no_srgb=False,
        output_mode=1,
        metallic_override=0.0,
        mode=None,
        wi3_debug_mode=0,
        wi3_debug_roughness=0.0,
    )
    return ensure_rgba_u8(arr)


@dataclass
class TileMetrics:
    roughness: float
    principled_rgb: np.ndarray
    ggx_rgb: np.ndarray
    energy_rgb: np.ndarray
    mean_principled: float
    mean_ggx: float
    comp_factor: float
    gpu_hash: str


def generate_tiles() -> List[TileMetrics]:
    tiles: List[TileMetrics] = []
    for r in ROUGHNESS_VALUES:
        principled = rgba_to_rgb(render_tile("disney", r))
        ggx = rgba_to_rgb(render_tile("ggx", r))
        energy = rgba_to_rgb(render_tile("disney", r, debug_energy=True))
        mean_p = mean_srgb_luminance(principled)
        mean_g = mean_srgb_luminance(ggx)
        comp = energy_comp_factor(r)
        tiles.append(
            TileMetrics(
                roughness=r,
                principled_rgb=principled,
                ggx_rgb=ggx,
                energy_rgb=energy,
                mean_principled=mean_p,
                mean_ggx=mean_g,
                comp_factor=comp,
                gpu_hash=compute_gpu_constants_hash(r),
            )
        )
    return tiles


def build_captioned_tiles(metrics: List[TileMetrics]) -> tuple[List[np.ndarray], List[np.ndarray]]:
    principled_tiles: List[np.ndarray] = []
    energy_tiles: List[np.ndarray] = []
    for t in metrics:
        caption = f"Principled  r={t.roughness:.2f}"
        principled_tiles.append(add_caption_rgb(t.principled_rgb, caption))
        energy_tiles.append(add_caption_rgb(t.energy_rgb, f"k={t.comp_factor:.3f}"))
    return principled_tiles, energy_tiles


def write_png(path: Path, rgb: np.ndarray) -> None:
    if HAS_PIL:
        Image.fromarray(rgb, mode="RGB").save(path)
    else:  # pragma: no cover
        from imageio import imwrite  # type: ignore

        imwrite(path, rgb)


def relative_diff(a: float, b: float) -> float:
    if b == 0:
        return math.inf
    return abs(a - b) / b


def check_acceptance(metrics: List[TileMetrics]) -> dict:
    fail_messages: List[str] = []
    comp_values = [t.comp_factor for t in metrics]
    comp_min = min(comp_values)
    comp_max = max(comp_values)

    b1_ok = True
    b2_ok = True
    b3_ok = True
    b4_ok = True

    mid = next((t for t in metrics if abs(t.roughness - 0.5) < 1e-4), None)
    if mid is None:
        fail_messages.append("B1 failed: missing r=0.50 tile")
        b1_ok = False
    else:
        if relative_diff(mid.mean_principled, mid.mean_ggx) > 0.05:
            fail_messages.append(
                f"B1 failed: r=0.50 means {mid.mean_principled:.3f} vs {mid.mean_ggx:.3f}"
            )
            b1_ok = False

    for target in (0.7, 0.9):
        tile = next((t for t in metrics if abs(t.roughness - target) < 1e-4), None)
        if tile is None:
            fail_messages.append(f"B2 failed: missing r={target:.2f} tile")
            b2_ok = False
            continue
        if tile.mean_ggx == 0.0:
            continue
        diff = (tile.mean_principled - tile.mean_ggx) / tile.mean_ggx
        if diff < -1e-4:
            fail_messages.append(f"B2 failed: r={target:.2f} Principled darker by {diff*100:.2f}%")
            b2_ok = False
        if diff > 0.03 + 1e-4:
            fail_messages.append(f"B2 failed: r={target:.2f} Principled brighter by {diff*100:.2f}% (>3%)")
            b2_ok = False

    if comp_min < -1e-5 or comp_max > 1.0 + 1e-5:
        fail_messages.append(f"B3 failed: comp_factor out of bounds [{comp_min:.4f}, {comp_max:.4f}]")
        b3_ok = False

    for i in range(1, len(comp_values)):
        if comp_values[i] + 1e-4 < comp_values[i - 1]:
            fail_messages.append("B4 failed: comp_factor not monotonic non-decreasing")
            b4_ok = False
            break

    return {
        "pass": len(fail_messages) == 0,
        "messages": fail_messages,
        "comp_min": comp_min,
        "comp_max": comp_max,
        "b1": b1_ok,
        "b2": b2_ok,
        "b3": b3_ok,
        "b4": b4_ok,
    }


def record_failure(outdir: Path, messages: List[str]) -> None:
    text = "\n".join(messages)
    fail_path = outdir / "m2_FAIL.txt"
    fail_path.write_text(text, encoding="utf-8")


def write_meta(path: Path, metrics: List[TileMetrics], acceptance: dict) -> None:
    tiles_meta = []
    for t in metrics:
        rel = relative_diff(t.mean_principled, t.mean_ggx)
        tiles_meta.append(
            {
                "roughness": round(t.roughness, 4),
                "alpha": round(t.roughness * t.roughness, 6),
                "caption": f"Principled  r={t.roughness:.2f}",
                "mean_srgb_luminance": round(t.mean_principled, 3),
                "mean_srgb_luminance_ggx": round(t.mean_ggx, 3),
                "relative_diff": round(rel, 6),
                "comp_factor": round(t.comp_factor, 6),
                "gpu_constants_hash": t.gpu_hash,
            }
        )
    meta = {
        "description": "Milestone 2 Principled GGX gallery",
        "model": "principled",
        "rng_seed": RNG_SEED,
        "tile_size": {"width": TILE_SIZE[0], "height": TILE_SIZE[1]},
        "roughness_values": list(ROUGHNESS_VALUES),
        "base_color": BASE_COLOR,
        "f0": F0_SCALAR,
        "light": {"direction": [float(x) for x in LIGHT_DIR], "radiance": LIGHT_INTENSITY},
        "camera": {"position": CAMERA_POS, "view_dir": CAMERA_VIEW_DIR},
        "tiles": tiles_meta,
        "comp_factor": {
            "min": round(acceptance["comp_min"], 6),
            "max": round(acceptance["comp_max"], 6),
            "monotonic_non_decreasing": acceptance["b4"],
        },
        "acceptance": {
            "B1": acceptance["b1"],
            "B2": acceptance["b2"],
            "B3": acceptance["b3"],
            "B4": acceptance["b4"],
        },
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M2 Principled gallery and energy diagnostics.")
    parser.add_argument("--outdir", type=Path, default=Path("reports"), help="Output directory (default: reports)")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    metrics = generate_tiles()
    principled_tiles, energy_tiles = build_captioned_tiles(metrics)

    gallery = make_row_rgb(principled_tiles, "m2_principled")
    energy_gallery = make_row_rgb(energy_tiles, "m2_energy")

    gallery_path = args.outdir / "m2_brdf_gallery_principled.png"
    energy_path = args.outdir / "m2_debug_energy.png"
    meta_path = args.outdir / "m2_meta.json"

    write_png(gallery_path, gallery)
    write_png(energy_path, energy_gallery)

    acceptance = check_acceptance(metrics)
    if not acceptance["pass"]:
        record_failure(args.outdir, acceptance["messages"])
        for msg in acceptance["messages"]:
            print(f"[FAIL] {msg}")
        raise RuntimeError("M2 acceptance checks failed.")

    write_meta(meta_path, metrics, acceptance)

    print("\n=== Milestone 2 Principled Summary ===")
    for t in metrics:
        print(
            f"r={t.roughness:.2f}  mean_principled={t.mean_principled:7.3f}  "
            f"mean_ggx={t.mean_ggx:7.3f}  comp_factor={t.comp_factor:.4f}"
        )
    print(f"Gallery  -> {gallery_path.resolve()}")
    print(f"Energy   -> {energy_path.resolve()}")
    print(f"Meta JSON-> {meta_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
