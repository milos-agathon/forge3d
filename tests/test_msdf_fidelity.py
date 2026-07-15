from __future__ import annotations

from pathlib import Path

import numpy as np

import forge3d
from tests._ssim import ssim


ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "fonts" / "NotoSansLatin-subset.ttf"


def _bake(size: int) -> tuple[dict, np.ndarray, np.ndarray]:
    baked = forge3d.text.bake_msdf_atlas(
        [str(FONT)], "A", size, px_range=4.0, padding=2
    )
    image = baked["image"]
    glyph = baked["metrics"]["glyphs"][str(ord("A"))]
    shaped = forge3d.text.shape("A", [str(FONT)], float(size))
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
    return baked, cell, analytic


def _coverage(field: np.ndarray, px_range: float) -> np.ndarray:
    median = np.median(field.astype(np.float32) / 255.0, axis=2)
    return np.clip((median - 0.5) * px_range + 0.5, 0.0, 1.0)


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


def _render_12px_from_msdf(channel: int | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    target_size = 12
    scale = 8
    px_range = 4
    padding = 2
    layout, _, analytic = _bake(target_size)
    source = forge3d.text.bake_msdf_atlas(
        [str(FONT)],
        "A",
        target_size * scale,
        px_range=px_range * scale,
        padding=padding * scale,
    )
    target_glyph = layout["metrics"]["glyphs"][str(ord("A"))]
    source_glyph = source["metrics"]["glyphs"][str(ord("A"))]
    source_cell = source["image"][
        int(source_glyph["y"]) : int(source_glyph["y"] + source_glyph["h"]),
        int(source_glyph["x"]) : int(source_glyph["x"] + source_glyph["w"]),
    ].astype(np.float32) / 255.0
    coverage = np.zeros_like(analytic)
    for y in range(coverage.shape[0]):
        for x in range(coverage.shape[1]):
            covered = 0
            for sy in range(8):
                for sx in range(8):
                    source_x = (
                        x + (sx + 0.5) / 8.0 + float(target_glyph["ox"])
                    ) * scale - float(source_glyph["ox"])
                    source_y = (
                        y + (sy + 0.5) / 8.0 + float(target_glyph["oy"])
                    ) * scale - float(source_glyph["oy"])
                    sample = _bilinear(source_cell, source_x, source_y)
                    distance = sample[channel] if channel is not None else np.median(sample)
                    covered += distance >= 0.5
            coverage[y, x] = covered / 64.0
    return source, coverage, analytic


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
    baked, decoded, analytic = _render_12px_from_msdf()
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
    baked, cell, analytic = _bake(96)
    decoded = _coverage(cell, baked["metrics"]["px_range"])
    measured = ssim(decoded, analytic, data_range=1.0)

    print(f"96px SSIM={measured:.9f}")
    assert measured >= 0.999


def test_single_channel_ablation_loses_the_sharp_corner():
    _, red_coverage, analytic = _render_12px_from_msdf(channel=0)
    distance = _hausdorff(red_coverage, analytic)

    print(f"single-channel Hausdorff={distance:.6f}px")
    assert distance > 0.5


def test_bake_metadata_is_typed_deterministic_and_measured():
    first, _, _ = _bake(12)
    second, _, _ = _bake(12)

    assert first["image"].dtype == np.uint8
    assert first["image"].tobytes() == second["image"].tobytes()
    assert first["metrics"]["channels"] == 3
    assert first["metrics"]["kind"] == "msdf_font_atlas"
    assert first["metrics"]["byte_count"] == first["image"].nbytes
    assert first["metrics"]["bake_ms"] >= 0.0
