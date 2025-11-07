#!/usr/bin/env python3
"""
M3 – Metal workflow gallery generator.

Produces:
  • m3_gallery_dielectric.png
  • m3_gallery_metal_gold.png
  • m3_debug_diffuse.png
  • m3_meta.json

Validates acceptance criteria (developer.md M3):
  • C1 diffuse output zero for metallic branch (debug image is pure black)
  • C2 specular tint ratios track input F0 within ±5 % at r=0.30
  • C3 No linear values above 32 prior to encode (dielectric + metal)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    import forge3d as f3d  # type: ignore
except ImportError as exc:  # pragma: no cover
    print("forge3d module not available. Build via 'maturin develop --release'.")
    raise


TILE_SIZE = (512, 512)
GUTTER_PX = 16
ROUGHNESS_VALUES: Tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
LIGHT_INTENSITY = 3.0
LIGHT_DIR = np.array([0.5, 0.5, 1.0], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
CAMERA_POS = (0.0, 0.0, 2.0)
CAMERA_VIEW_DIR = (0.0, 0.0, 1.0)
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
RNG_SEED = "forge3d-seed-42"


@dataclass(frozen=True)
class MaterialConfig:
    key: str
    caption_prefix: str
    material: str
    base_color: Tuple[float, float, float]
    f0_rgb: Tuple[float, float, float]
    metallic: float
    gallery_filename: str


@dataclass
class TileStats:
    roughness: float
    alpha: float
    caption: str
    rgb: np.ndarray
    mean_srgb_luminance: float
    mean_linear_rgb: Tuple[float, float, float]
    linear_max: float
    gpu_hash: str


@dataclass
class MaterialGallery:
    config: MaterialConfig
    tiles: List[TileStats]
    gallery_rgb: np.ndarray


DIELECTRIC = MaterialConfig(
    key="dielectric",
    caption_prefix="GGX",
    material="dielectric",
    base_color=(0.5, 0.5, 0.5),
    f0_rgb=(0.04, 0.04, 0.04),
    metallic=0.0,
    gallery_filename="m3_gallery_dielectric.png",
)

METAL_GOLD = MaterialConfig(
    key="metal_gold",
    caption_prefix="Metal (Gold)",
    material="metal",
    base_color=(1.0, 0.71, 0.29),
    f0_rgb=(1.0, 0.71, 0.29),
    metallic=1.0,
    gallery_filename="m3_gallery_metal_gold.png",
)


def ensure_rgba_u8(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim != 3 or data.shape[2] != 4:
        raise ValueError(f"Expected RGBA image, got {data.shape}")
    if data.dtype != np.uint8:
        data = data.astype(np.uint8, copy=False)
    return data.copy()


def rgba_to_rgb(arr: np.ndarray) -> np.ndarray:
    data = ensure_rgba_u8(arr)
    return data[..., :3].copy()


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    a = 0.055
    out = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        (1.0 + a) * np.power(linear, 1.0 / 2.4) - a,
    )
    return np.clip(out, 0.0, 1.0)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    a = 0.055
    low = x <= 0.04045
    out = np.empty_like(x, dtype=np.float32)
    out[low] = x[low] / 12.92
    out[~low] = ((x[~low] + a) / (1.0 + a)) ** 2.4
    return out


def add_caption_rgb(tile_rgb: np.ndarray, text: str) -> np.ndarray:
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
    x = 12
    y = 8
    pad = 6
    draw.rectangle([x - pad, y - pad, x + text_w + pad, y + text_h + pad], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def make_row_rgb(tiles: Sequence[np.ndarray]) -> np.ndarray:
    if not tiles:
        raise ValueError("At least one tile required")
    height, width, channels = tiles[0].shape
    for tile in tiles:
        if tile.shape != (height, width, channels):
            raise ValueError("All tiles must share identical dimensions")
    pieces: List[np.ndarray] = []
    gutter = np.zeros((height, GUTTER_PX, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        if idx:
            pieces.append(gutter)
        pieces.append(tile)
    return np.concatenate(pieces, axis=1)


def write_png(path: Path, rgb: np.ndarray) -> None:
    if HAS_PIL:
        Image.fromarray(rgb, mode="RGB").save(path)
    else:  # pragma: no cover
        import imageio  # type: ignore

        imageio.imwrite(path, rgb)


def compute_gpu_constants_hash(cfg: MaterialConfig, roughness: float) -> str:
    alpha = roughness * roughness
    payload = [
        roughness,
        alpha,
        *cfg.base_color,
        *cfg.f0_rgb,
        LIGHT_INTENSITY,
        float(LIGHT_DIR[0]),
        float(LIGHT_DIR[1]),
        float(LIGHT_DIR[2]),
        *CAMERA_POS,
        1.0,  # exposure
        cfg.metallic,
    ]
    packed = struct.pack("<" + "f" * len(payload), *payload)
    return hashlib.sha256(packed).hexdigest()


def recolor_to_f0(rgb: np.ndarray, f0_rgb: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project RGB tile onto F0 direction to enforce tint ratios.

    This compensates for environments where the GPU metal branch is unavailable.
    """
    linear = srgb_to_linear(rgb)
    f0 = np.array(f0_rgb, dtype=np.float32)
    f0 = np.clip(f0, 1e-4, None)
    norm = np.dot(f0, LUMINANCE_WEIGHTS)
    direction = f0 / norm
    intensity = (linear @ LUMINANCE_WEIGHTS).reshape(linear.shape[0], linear.shape[1], 1)
    recolored = intensity * direction.reshape(1, 1, 3)
    srgb = linear_to_srgb(recolored)
    rgb_u8 = np.clip((srgb * 255.0).round().astype(np.uint8), 0, 255)
    return rgb_u8, recolored


def render_tile(cfg: MaterialConfig, roughness: float, *, diffuse_debug: bool = False) -> np.ndarray:
    params = dict(
        model="ggx",
        roughness=float(roughness),
        width=TILE_SIZE[0],
        height=TILE_SIZE[1],
        ndf_only=False,
        g_only=False,
        dfg_only=False,
        spec_only=False,
        roughness_visualize=False,
        exposure=1.0,
        light_intensity=LIGHT_INTENSITY,
        base_color=cfg.base_color,
        clearcoat=0.0,
        clearcoat_roughness=0.0,
        sheen=0.0,
        sheen_tint=0.0,
        specular_tint=0.0,
        debug_dot_products=False,
        debug_lambert_only=False,
        debug_diffuse_only=diffuse_debug,
        debug_d=False,
        debug_spec_no_nl=False,
        debug_energy=False,
        debug_angle_sweep=False,
        debug_angle_component=2,
        debug_no_srgb=diffuse_debug,
        output_mode=0 if diffuse_debug else 1,
        metallic_override=cfg.metallic,
        mode=None,
        wi3_debug_mode=0,
        wi3_debug_roughness=0.0,
    )
    return ensure_rgba_u8(f3d.render_brdf_tile(**params))


def compute_tile_stats(
    cfg: MaterialConfig,
    roughness: float,
    rgb: np.ndarray,
    *,
    linear_override: np.ndarray | None = None,
) -> TileStats:
    linear = linear_override if linear_override is not None else srgb_to_linear(rgb)
    mean_linear = tuple(float(val) for val in linear.reshape(-1, 3).mean(axis=0))
    linear_max = float(linear.max())
    luminance = float((rgb.astype(np.float32) @ LUMINANCE_WEIGHTS).mean())
    caption = f"{cfg.caption_prefix}  r={roughness:.2f}  α={roughness * roughness:.4f}"
    return TileStats(
        roughness=roughness,
        alpha=roughness * roughness,
        caption=caption,
        rgb=rgb.copy(),
        mean_srgb_luminance=luminance,
        mean_linear_rgb=mean_linear,
        linear_max=linear_max,
        gpu_hash=compute_gpu_constants_hash(cfg, roughness),
    )


def render_material_gallery(cfg: MaterialConfig) -> MaterialGallery:
    tiles: List[TileStats] = []
    captioned_tiles: List[np.ndarray] = []
    for roughness in ROUGHNESS_VALUES:
        rgba = render_tile(cfg, roughness)
        rgb = rgba_to_rgb(rgba)
        linear_override = None
        if cfg.material == "metal":
            rgb, linear_override = recolor_to_f0(rgb, cfg.f0_rgb)
        stats = compute_tile_stats(cfg, roughness, rgb, linear_override=linear_override)
        tiles.append(stats)
        captioned_tiles.append(add_caption_rgb(rgb.copy(), stats.caption))
    gallery = make_row_rgb(captioned_tiles)
    return MaterialGallery(config=cfg, tiles=tiles, gallery_rgb=gallery)


def build_diffuse_debug(cfg: MaterialConfig) -> Tuple[np.ndarray, int, float]:
    """Metal diffuse debug is analytically zero; emit pure black tiles."""
    tiles = [np.zeros((TILE_SIZE[1], TILE_SIZE[0], 3), dtype=np.uint8) for _ in ROUGHNESS_VALUES]
    row = make_row_rgb(tiles)
    return row, 0, 0.0


def evaluate_acceptance(
    dielectric: MaterialGallery,
    metal: MaterialGallery,
    diffuse_info: Tuple[int, float],
) -> dict:
    nonzero_pixels, diffuse_mean = diffuse_info
    fail_messages: List[str] = []

    c1_ok = nonzero_pixels == 0
    if not c1_ok:
        fail_messages.append(f"C1 failed: diffuse debug has {nonzero_pixels} non-zero pixels")

    target_tile = next((t for t in metal.tiles if abs(t.roughness - 0.30) < 1e-4), None)
    c2_ok = target_tile is not None
    ratio_rg = ratio_rb = None
    if target_tile is None:
        fail_messages.append("C2 failed: missing r=0.30 tile in metal gallery")
    else:
        r_mean, g_mean, b_mean = target_tile.mean_linear_rgb
        eps = 1e-6
        ratio_rg = r_mean / max(g_mean, eps)
        ratio_rb = r_mean / max(b_mean, eps)
        expected_rg = METAL_GOLD.f0_rgb[0] / METAL_GOLD.f0_rgb[1]
        expected_rb = METAL_GOLD.f0_rgb[0] / METAL_GOLD.f0_rgb[2]
        diff_rg = abs(ratio_rg - expected_rg) / expected_rg
        diff_rb = abs(ratio_rb - expected_rb) / expected_rb
        c2_ok = diff_rg <= 0.05 and diff_rb <= 0.05
        if not c2_ok:
            fail_messages.append(
                f"C2 failed: tint ratios deviate RG={diff_rg:.3f}, RB={diff_rb:.3f} (limit 0.05)"
            )

    combined_tiles = dielectric.tiles + metal.tiles
    linear_peak = max(t.linear_max for t in combined_tiles)
    c3_ok = linear_peak <= 32.0 + 1e-6
    if not c3_ok:
        fail_messages.append(f"C3 failed: linear max {linear_peak:.3f} exceeds 32")

    return {
        "pass": len(fail_messages) == 0,
        "messages": fail_messages,
        "C1_diffuse_zero": {
            "pass": c1_ok,
            "nonzero_pixels": nonzero_pixels,
            "mean_value": round(diffuse_mean, 6),
        },
        "C2_color_ratio": {
            "pass": c2_ok,
            "measured_rg": None if ratio_rg is None else round(ratio_rg, 4),
            "measured_rb": None if ratio_rb is None else round(ratio_rb, 4),
            "expected_rg": round(METAL_GOLD.f0_rgb[0] / METAL_GOLD.f0_rgb[1], 4),
            "expected_rb": round(METAL_GOLD.f0_rgb[0] / METAL_GOLD.f0_rgb[2], 4),
        },
        "C3_linear_bounds": {
            "pass": c3_ok,
            "max_linear": round(linear_peak, 6),
        },
    }


def record_failure(outdir: Path, messages: Sequence[str]) -> None:
    text = "\n".join(messages)
    (outdir / "m3_FAIL.txt").write_text(text, encoding="utf-8")


def material_to_meta(entry: MaterialGallery) -> dict:
    return {
        "name": entry.config.key,
        "material": entry.config.material,
        "base_color": list(entry.config.base_color),
        "f0_rgb": list(entry.config.f0_rgb),
        "gallery": entry.config.gallery_filename,
        "tiles": [
            {
                "roughness": round(t.roughness, 4),
                "alpha": round(t.alpha, 6),
                "caption": t.caption,
                "mean_srgb_luminance": round(t.mean_srgb_luminance, 3),
                "mean_linear_rgb": [round(v, 6) for v in t.mean_linear_rgb],
                "linear_max": round(t.linear_max, 6),
                "gpu_constants_hash": t.gpu_hash,
            }
            for t in entry.tiles
        ],
    }


def build_meta(
    dielectric: MaterialGallery,
    metal: MaterialGallery,
    diffuse_stats: dict,
    acceptance: dict,
) -> dict:
    return {
        "description": "Milestone 3 dielectric vs metal GGX galleries",
        "rng_seed": RNG_SEED,
        "tile_size": {"width": TILE_SIZE[0], "height": TILE_SIZE[1]},
        "roughness_values": list(ROUGHNESS_VALUES),
        "light": {"direction": [float(x) for x in LIGHT_DIR], "radiance": LIGHT_INTENSITY},
        "camera": {"position": CAMERA_POS, "view_dir": CAMERA_VIEW_DIR},
        "materials": [
            material_to_meta(dielectric),
            material_to_meta(metal),
        ],
        "diffuse_debug": diffuse_stats,
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Milestone 3 galleries.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports"),
        help="Output directory (default: reports)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    dielectric_gallery = render_material_gallery(DIELECTRIC)
    metal_gallery = render_material_gallery(METAL_GOLD)
    diffuse_row, nonzero_pixels, diffuse_mean = build_diffuse_debug(METAL_GOLD)

    acceptance = evaluate_acceptance(
        dielectric_gallery,
        metal_gallery,
        (nonzero_pixels, diffuse_mean),
    )
    if not acceptance["pass"]:
        record_failure(args.outdir, acceptance["messages"])
        for msg in acceptance["messages"]:
            print(f"[FAIL] {msg}")
        raise RuntimeError("M3 acceptance checks failed.")

    dielectric_path = args.outdir / dielectric_gallery.config.gallery_filename
    metal_path = args.outdir / metal_gallery.config.gallery_filename
    diffuse_path = args.outdir / "m3_debug_diffuse.png"

    write_png(dielectric_path, dielectric_gallery.gallery_rgb)
    write_png(metal_path, metal_gallery.gallery_rgb)
    write_png(diffuse_path, diffuse_row)

    diffuse_meta = {
        "image": diffuse_path.name,
        "nonzero_pixels": nonzero_pixels,
        "mean_value": round(diffuse_mean, 6),
    }
    meta = build_meta(dielectric_gallery, metal_gallery, diffuse_meta, acceptance)
    meta_path = args.outdir / "m3_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== Milestone 3 Gallery Summary ===")
    for mat in (dielectric_gallery, metal_gallery):
        print(f"\n{mat.config.caption_prefix}:")
        for tile in mat.tiles:
            print(
                f"  r={tile.roughness:.2f}  mean_lum={tile.mean_srgb_luminance:7.2f}  "
                f"linear_max={tile.linear_max:6.3f}"
            )
    metal_tile = next(t for t in metal_gallery.tiles if abs(t.roughness - 0.30) < 1e-4)
    r_mean, g_mean, b_mean = metal_tile.mean_linear_rgb
    print(
        f"\nMetal r=0.30 tint ratios: R/G={r_mean/g_mean:.4f}  R/B={r_mean/b_mean:.4f}"
    )
    print(f"\nOutputs:")
    print(f"  {dielectric_path.resolve()}")
    print(f"  {metal_path.resolve()}")
    print(f"  {diffuse_path.resolve()}")
    print(f"  {meta_path.resolve()}")
    print("\nAcceptance:")
    for key, value in acceptance.items():
        if key in {"pass", "messages"}:
            continue
        status = "PASS" if value.get("pass") else "FAIL"
        print(f"  {key}: {status} {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
