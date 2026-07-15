from __future__ import annotations

from pathlib import Path

import numpy as np

import forge3d
from tests._ssim import ssim


ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "fonts" / "NotoSansLatin-subset.ttf"


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
    """Mirror text_overlay.wgsl median + fwidth + smoothstep reconstruction."""
    normalized = field.astype(np.float32) / 255.0
    encoded = np.median(normalized, axis=2) if channels == 3 else normalized
    signed = encoded - 0.5
    dy, dx = np.gradient(signed)
    edge_width = np.maximum((np.abs(dx) + np.abs(dy)) * 2.0, 1.0e-6)
    return _smoothstep(-edge_width, edge_width, signed).astype(np.float32)


def _distance_coverage(field: np.ndarray, px_range: float) -> np.ndarray:
    encoded = np.median(field.astype(np.float32) / 255.0, axis=2)
    return np.clip((encoded - 0.5) * px_range + 0.5, 0.0, 1.0)


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

    print(f"12px IoU={iou:.6f} Hausdorff={hausdorff:.6f}px")
    assert iou >= 0.995
    assert hausdorff <= 0.5
    assert baked["image"].shape[2] == 3
    assert np.any(baked["image"][..., 0] != baked["image"][..., 1])
    assert np.any(baked["image"][..., 1] != baked["image"][..., 2])


def test_true_rgb_msdf_meets_96px_ssim():
    baked, cell, _, analytic = _bake(96)
    decoded = _distance_coverage(cell, baked["metrics"]["px_range"])
    measured = ssim(decoded, analytic, data_range=1.0)

    print(f"96px SSIM={measured:.9f}")
    assert measured >= 0.999


def test_single_channel_ablation_loses_the_sharp_corner():
    msdf_cell, sdf_cell, analytic = _render_upscaled_fields()
    msdf_coverage = _renderer_coverage(msdf_cell, channels=3)
    sdf_coverage = _renderer_coverage(sdf_cell, channels=1)
    msdf_distance = _hausdorff(msdf_coverage, analytic)
    sdf_distance = _hausdorff(sdf_coverage, analytic)

    print(
        f"MSDF Hausdorff={msdf_distance:.6f}px "
        f"independent SDF Hausdorff={sdf_distance:.6f}px"
    )
    assert sdf_distance > msdf_distance
    assert sdf_distance > 0.5


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
