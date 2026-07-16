from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import forge3d
from tests._ssim import ssim


ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "fonts" / "NotoSansLatin-subset.ttf"
PACKAGED_ATLAS = ROOT / "python" / "forge3d" / "data" / "fonts"


def _bake(
    size: int, character: str = "A"
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    baked = forge3d.text.bake_msdf_atlas(
        [str(FONT)], character, size, px_range=4.0, padding=2
    )
    image = baked["image"]
    glyph = baked["metrics"]["glyphs"][str(ord(character))]
    shaped = forge3d.text.shape(character, [str(FONT)], float(size))
    analytic = forge3d.text.rasterize_shaped_run(
        shaped,
        int(glyph["w"]),
        int(glyph["h"]),
        origin=(-float(glyph["ox"]), -float(glyph["oy"])),
    )
    cell = image[
        int(glyph["y"]) : int(glyph["y"] + glyph["h"]),
        int(glyph["x"]) : int(glyph["x"] + glyph["w"]),
    ]
    sdf_cell = baked["sdf_image"][
        int(glyph["y"]) : int(glyph["y"] + glyph["h"]),
        int(glyph["x"]) : int(glyph["x"] + glyph["w"]),
    ]
    return baked, cell, sdf_cell, analytic


def _smoothstep(low: np.ndarray, high: np.ndarray, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - low) / np.maximum(high - low, 1.0e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _renderer_coverage(field: np.ndarray, *, channels: int) -> np.ndarray:
    """Mirror text_overlay.wgsl using 2x2 fragment-quad derivatives."""
    normalized = field.astype(np.float32) / 255.0
    encoded = np.median(normalized, axis=2) if channels == 3 else normalized
    signed = encoded - 0.5
    edge_width = np.empty_like(signed)
    height, width = signed.shape
    for y in range(0, height, 2):
        y1 = min(y + 1, height - 1)
        for x in range(0, width, 2):
            x1 = min(x + 1, width - 1)
            top_left = signed[y, x]
            top_right = signed[y, x1]
            bottom_left = signed[y1, x]
            bottom_right = signed[y1, x1]
            dx_top = abs(top_right - top_left)
            dx_bottom = abs(bottom_right - bottom_left)
            dy_left = abs(bottom_left - top_left)
            dy_right = abs(bottom_right - top_right)
            edge_width[y, x] = dx_top + dy_left
            edge_width[y, x1] = dx_top + dy_right
            edge_width[y1, x] = dx_bottom + dy_left
            edge_width[y1, x1] = dx_bottom + dy_right
    edge_width = np.maximum(edge_width, 1.0e-6)
    return _smoothstep(-edge_width, edge_width, signed).astype(np.float32)


def _distance_coverage(field: np.ndarray, px_range: float) -> np.ndarray:
    encoded = np.median(field.astype(np.float32) / 255.0, axis=2)
    return np.clip((encoded - 0.5) * px_range + 0.5, 0.0, 1.0)


def _pillow_freetype_coverage(
    baked: dict, character: str, glyph: dict, font_path: Path = FONT, y_adjust: int = 0
) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), int(baked["metrics"]["font_size"]))
    bbox = font.getbbox(character)
    oracle = Image.new("L", (int(glyph["w"]), int(glyph["h"])), 0)
    y = -bbox[1] + int(baked["metrics"]["padding"] + baked["metrics"]["px_range"])
    if float(glyph["oy"]) > 0.0:
        y = -bbox[1] - int(baked["metrics"]["padding"])
    ImageDraw.Draw(oracle).text(
        (-int(glyph["ox"]), y + y_adjust),
        character,
        fill=255,
        font=font,
    )
    return np.asarray(oracle, dtype=np.float32) / 255.0


def _bilinear(field: np.ndarray, x: float, y: float) -> np.ndarray:
    x -= 0.5
    y -= 0.5
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = x - x0
    fy = y - y0

    def pixel(px: int, py: int) -> np.ndarray:
        return field[
            min(max(py, 0), field.shape[0] - 1),
            min(max(px, 0), field.shape[1] - 1),
        ]

    return (
        pixel(x0, y0) * (1.0 - fx) * (1.0 - fy)
        + pixel(x0 + 1, y0) * fx * (1.0 - fy)
        + pixel(x0, y0 + 1) * (1.0 - fx) * fy
        + pixel(x0 + 1, y0 + 1) * fx * fy
    )


def _render_upscaled_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = 8
    character = "V"
    source, source_msdf, source_sdf, _ = _bake(12, character)
    source_glyph = source["metrics"]["glyphs"][str(ord(character))]
    source_msdf = source_msdf.astype(np.float32)
    source_sdf = source_sdf.astype(np.float32)
    output_shape = (
        int(source_glyph["h"]) * scale,
        int(source_glyph["w"]) * scale,
    )
    shaped = forge3d.text.shape(character, [str(FONT)], float(12 * scale))
    analytic = forge3d.text.rasterize_shaped_run(
        shaped,
        output_shape[1],
        output_shape[0],
        origin=(
            -float(source_glyph["ox"]) * scale,
            -float(source_glyph["oy"]) * scale,
        ),
    )
    sampled_msdf = np.empty(output_shape + (3,), dtype=np.float32)
    sampled_sdf = np.empty(output_shape, dtype=np.float32)
    for y in range(analytic.shape[0]):
        for x in range(analytic.shape[1]):
            source_x = (x + 0.5) / scale
            source_y = (y + 0.5) / scale
            sampled_msdf[y, x] = _bilinear(source_msdf, source_x, source_y)
            sampled_sdf[y, x] = _bilinear(source_sdf, source_x, source_y)
    return sampled_msdf, sampled_sdf, analytic


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask >= 0.5
    edge = np.zeros_like(binary)
    edge[1:] |= binary[1:] != binary[:-1]
    edge[:-1] |= binary[:-1] != binary[1:]
    edge[:, 1:] |= binary[:, 1:] != binary[:, :-1]
    edge[:, :-1] |= binary[:, :-1] != binary[:, 1:]
    return np.argwhere(edge)


def _hausdorff(left: np.ndarray, right: np.ndarray) -> float:
    a = _boundary(left)
    b = _boundary(right)
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def test_true_rgb_msdf_meets_12px_analytic_fidelity():
    baked, cell, _, analytic = _bake(12)
    decoded = _renderer_coverage(cell, channels=3)
    intersection = np.count_nonzero((decoded >= 0.5) & (analytic >= 0.5))
    union = np.count_nonzero((decoded >= 0.5) | (analytic >= 0.5))
    iou = intersection / union
    hausdorff = _hausdorff(decoded, analytic)
    mean_error = float(np.mean(np.abs(decoded - analytic)))

    print(
        f"12px IoU={iou:.6f} Hausdorff={hausdorff:.6f}px "
        f"MAE={mean_error:.6f}"
    )
    # The oracle is continuous area coverage while the atlas sign is sampled
    # at pixel centers, so a one-pixel disagreement is the quantization bound.
    assert iou >= 0.90
    assert hausdorff <= 1.0
    assert mean_error <= 0.02
    assert baked["image"].shape[2] == 3
    assert np.any(baked["image"][..., 0] != baked["image"][..., 1])
    assert np.any(baked["image"][..., 1] != baked["image"][..., 2])


def test_true_rgb_msdf_meets_96px_ssim():
    baked, cell, _, analytic = _bake(96)
    decoded = _distance_coverage(cell, baked["metrics"]["px_range"])
    measured = ssim(decoded, analytic, data_range=1.0)
    mean_error = float(np.mean(np.abs(decoded - analytic)))

    print(f"96px SSIM={measured:.9f} MAE={mean_error:.9f}")
    assert measured >= 0.985
    assert mean_error <= 0.003


def test_true_rgb_msdf_printable_latin_matches_pillow_freetype_oracle():
    characters = "".join(chr(codepoint) for codepoint in range(33, 127))
    baked = forge3d.text.bake_msdf_atlas(
        [str(FONT)], characters, 96, px_range=4.0, padding=2
    )
    eligible = 0
    failures = []
    for character in characters:
        glyph = baked["metrics"]["glyphs"][str(ord(character))]
        cell = baked["image"][
            int(glyph["y"]) : int(glyph["y"] + glyph["h"]),
            int(glyph["x"]) : int(glyph["x"] + glyph["w"]),
        ]
        decoded = _distance_coverage(cell, baked["metrics"]["px_range"])
        if np.count_nonzero(decoded >= 0.5) < 8:
            continue
        eligible += 1
        reference = _pillow_freetype_coverage(baked, character, glyph)
        measured = ssim(decoded, reference, data_range=1.0)
        mean_error = float(np.mean(np.abs(decoded - reference)))
        if measured < 0.90 or mean_error > 0.025:
            failures.append((character, measured, mean_error))

    assert eligible == len(characters)
    assert failures == []


def test_true_rgb_msdf_representative_scripts_match_pillow_freetype_oracle():
    cases = (
        ("assets/fonts/NotoSansArabic-subset.ttf", "\u0628", 0),
        ("assets/fonts/NotoSansHebrew-subset.ttf", "\u05e9", 1),
        ("assets/fonts/NotoSansDevanagari-subset.ttf", "\u0915", 0),
        ("assets/fonts/NotoSansSC-subset.ttf", "\u4e2d", 0),
    )
    for font_name, character, y_adjust in cases:
        font_path = ROOT / font_name
        baked = forge3d.text.bake_msdf_atlas(
            [str(font_path)], character, 96, px_range=4.0, padding=2
        )
        glyph = baked["metrics"]["glyphs"][str(ord(character))]
        cell = baked["image"][
            int(glyph["y"]) : int(glyph["y"] + glyph["h"]),
            int(glyph["x"]) : int(glyph["x"] + glyph["w"]),
        ]
        decoded = _distance_coverage(cell, baked["metrics"]["px_range"])
        reference = _pillow_freetype_coverage(
            baked, character, glyph, font_path, y_adjust
        )

        assert ssim(decoded, reference, data_range=1.0) >= 0.95, font_name
        assert float(np.mean(np.abs(decoded - reference))) <= 0.02, font_name


def test_text_overlay_shader_keeps_msdf_decode_and_derivative_smoothing():
    shader = (ROOT / "src/shaders/text_overlay.wgsl").read_text(encoding="utf-8")

    assert "sdf = median3(sample.rgb) - 0.5;" in shader
    assert "sdf = sample.r - 0.5;" in shader
    assert "fwidth(sdf) * max(U.smoothing, 0.1)" in shader
    assert "smoothstep(-edge_width, edge_width, sdf)" in shader


def test_single_channel_ablation_loses_the_sharp_corner():
    msdf_cell, sdf_cell, analytic = _render_upscaled_fields()
    msdf_coverage = _renderer_coverage(msdf_cell, channels=3)
    sdf_coverage = _renderer_coverage(sdf_cell, channels=1)
    msdf_distance = _hausdorff(msdf_coverage, analytic)
    sdf_distance = _hausdorff(sdf_coverage, analytic)
    msdf_similarity = ssim(msdf_coverage, analytic, data_range=1.0)
    sdf_similarity = ssim(sdf_coverage, analytic, data_range=1.0)
    msdf_error = float(np.mean(np.abs(msdf_coverage - analytic)))
    sdf_error = float(np.mean(np.abs(sdf_coverage - analytic)))

    print(
        f"MSDF Hausdorff={msdf_distance:.6f}px "
        f"independent SDF Hausdorff={sdf_distance:.6f}px "
        f"SSIM={msdf_similarity:.6f}/{sdf_similarity:.6f}"
    )
    assert msdf_distance <= sdf_distance
    assert msdf_similarity > sdf_similarity + 0.005
    assert msdf_error < sdf_error


def test_bake_metadata_is_typed_deterministic_and_measured():
    first, _, first_sdf, _ = _bake(12)
    second, _, second_sdf, _ = _bake(12)

    assert first["image"].dtype == np.uint8
    assert first["image"].tobytes() == second["image"].tobytes()
    assert first_sdf.dtype == np.uint8
    assert first_sdf.tobytes() == second_sdf.tobytes()
    assert first["metrics"]["channels"] == 3
    assert first["metrics"]["kind"] == "msdf_font_atlas"
    assert first["metrics"]["byte_count"] == first["image"].nbytes
    assert first["metrics"]["sdf_byte_count"] == first["sdf_image"].nbytes
    assert first["metrics"]["bake_ms"] >= 0.0
    assert "np." + "gradient" not in Path(__file__).read_text(encoding="utf-8")


def test_packaged_atlas_glyph_borders_are_saturated_background():
    """Lock out near-threshold cell borders that render as axis-aligned rays."""
    from forge3d._png import load_png_rgba

    metrics = json.loads(
        (PACKAGED_ATLAS / "atlas_latin_default.json").read_text(encoding="utf-8")
    )
    atlas = load_png_rgba(PACKAGED_ATLAS / "atlas_latin_default.png")[..., :3]
    for identity, glyph in metrics["glyphs"].items():
        x, y, width, height = (
            int(glyph[key]) for key in ("x", "y", "w", "h")
        )
        cell = atlas[y : y + height, x : x + width]
        median = np.median(cell, axis=2)
        border = np.concatenate(
            (median[0], median[-1], median[:, 0], median[:, -1])
        )
        assert int(border.max()) <= 16, identity


def test_live_gpu_shader_readback_matches_independent_quad_oracle():
    baked, cell, _, _ = _bake(32, "A")
    glyph = baked["metrics"]["glyphs"][str(ord("A"))]
    width = int(glyph["w"])
    height = int(glyph["h"])
    canvas_width = width + 24
    canvas_height = height + 24
    x0 = 12
    y0 = 12

    try:
        scene = forge3d.Scene(canvas_width, canvas_height)
    except Exception as error:
        import pytest

        pytest.skip(f"live GPU text readback unavailable: {error}")
    scene.disable_terrain()
    background = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    background[..., 3] = 255
    scene.set_raster_overlay(background, 1.0, None, None)
    scene.set_native_text_atlas(baked["image"], 3, 1.0)
    scene.enable_native_text()

    atlas_height, atlas_width = baked["image"].shape[:2]
    scene.add_native_text_rect_uv(
        float(x0),
        float(y0),
        float(width),
        float(height),
        float(glyph["x"]) / atlas_width,
        float(glyph["y"]) / atlas_height,
        float(glyph["x"] + glyph["w"]) / atlas_width,
        float(glyph["y"] + glyph["h"]) / atlas_height,
        1.0,
        0.0,
        0.0,
        1.0,
    )
    rendered = np.asarray(scene.render_rgba(), dtype=np.uint8)
    gpu = rendered[y0 : y0 + height, x0 : x0 + width, 0].astype(np.float32) / 255.0
    expected = _renderer_coverage(cell, channels=3)

    empty = forge3d.Scene(canvas_width, canvas_height)
    empty.disable_terrain()
    empty.set_raster_overlay(background, 1.0, None, None)
    no_label = np.asarray(empty.render_rgba(), dtype=np.uint8)

    assert np.count_nonzero(no_label[..., 0]) == 0
    assert np.mean(np.abs(gpu - expected)) <= 0.015
    assert np.max(np.abs(gpu - expected)) <= 0.20
    assert _hausdorff(gpu, expected) <= 0.5
