from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")
pytest.importorskip("rasterio")

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "swiss_terrain_landcover_viewer.py"


def load_module():
    spec = importlib.util.spec_from_file_location("swiss_terrain_landcover_viewer", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compose_snapshot_requires_distinct_raw_and_output_paths(tmp_path: Path) -> None:
    module = load_module()
    raw_path = tmp_path / "raw.png"
    Image.new("RGBA", (256, 256), (255, 255, 255, 255)).save(raw_path)

    with pytest.raises(ValueError, match="separate raw and output paths"):
        module.compose_snapshot(raw_path, raw_path)


def test_snap_overlay_to_legend_preserves_palette_and_masks_dark_unknowns() -> None:
    module = load_module()
    rgb = np.array([[[227.0, 226.0, 195.0], [60.0, 123.0, 76.0], [8.0, 8.0, 8.0]]], dtype=np.float32)

    snapped, valid = module.snap_overlay_to_legend(rgb)

    assert tuple(int(v) for v in snapped[0, 0]) == (227, 226, 195)
    assert tuple(int(v) for v in snapped[0, 1]) == (57, 125, 73)
    assert not bool(valid[0, 2])


def test_rgb_to_classes_snaps_near_palette_values() -> None:
    module = load_module()
    rgb = np.array([[[60.0, 123.0, 76.0], [8.0, 8.0, 8.0]]], dtype=np.float32)

    classes = module.rgb_to_classes(rgb)

    assert int(classes[0, 0]) == 1
    assert int(classes[0, 1]) == -1


def test_despeckle_landcover_classes_replaces_isolated_outlier() -> None:
    module = load_module()
    classes = np.array([[1, 1, 1], [1, 4, 1], [1, 1, 1]], dtype=np.int16)

    cleaned = module.despeckle_landcover_classes(classes, passes=1)

    assert int(cleaned[1, 1]) == 1


def test_classes_to_rgba_uses_display_palette_and_makes_invalid_pixels_transparent() -> None:
    module = load_module()
    classes = np.array([[1, -1]], dtype=np.int16)

    rgba = module.classes_to_rgba(classes)

    assert tuple(int(v) for v in module.LANDCOVER_SOURCE_PALETTE_RGB[1]) == (57, 125, 73)
    assert tuple(int(v) for v in rgba[0, 0]) == (43, 106, 61, 255)
    assert tuple(int(v) for v in rgba[0, 1]) == (0, 0, 0, 0)


def test_compose_snapshot_adds_title_legend_and_caption(tmp_path: Path) -> None:
    module = load_module()
    raw_path = tmp_path / "raw.png"
    final_path = tmp_path / "final.png"

    raw = Image.new("RGBA", (720, 720), (255, 255, 255, 255))
    draw = ImageDraw.Draw(raw)
    draw.polygon(
        [(120, 360), (300, 260), (560, 300), (640, 430), (520, 600), (230, 560)],
        fill=(88, 140, 102, 255),
    )
    draw.rectangle((260, 310, 430, 430), fill=(214, 153, 68, 255))
    raw.save(raw_path)

    module.compose_snapshot(raw_path, final_path)
    composed = np.asarray(Image.open(final_path).convert("RGB"), dtype=np.uint8)

    top_band = np.any(composed[:120] < 235, axis=2)
    left_band = np.any(composed[100:300, :180] < 235, axis=2)
    bottom_band = np.any(composed[int(composed.shape[0] * 0.88) :] < 235, axis=2)

    assert np.count_nonzero(top_band) > 500
    assert np.count_nonzero(left_band) > 500
    assert np.count_nonzero(bottom_band) > 200
