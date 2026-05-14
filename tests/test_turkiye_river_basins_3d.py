from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")
pytest.importorskip("geopandas")
pytest.importorskip("rasterio")

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "turkiye_river_basins_3d.py"


def load_module():
    spec = importlib.util.spec_from_file_location("turkiye_river_basins_3d", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    forge3d_stub = types.ModuleType("forge3d")
    examples_dir = str(EXAMPLE_PATH.parent)
    added_examples_dir = False
    previous_forge3d = sys.modules.get("forge3d")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
        added_examples_dir = True
    sys.modules["forge3d"] = forge3d_stub
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_forge3d is None:
            sys.modules.pop("forge3d", None)
        else:
            sys.modules["forge3d"] = previous_forge3d
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


def test_snapshot_dimensions_match_reference_plate_aspect() -> None:
    module = load_module()

    width, height = module._snapshot_dimensions(4200)

    assert width == 4200
    assert height == 2874
    assert width / height == pytest.approx(7706 / 5274, rel=1e-3)


def test_default_camera_uses_oblique_terrain_view() -> None:
    module = load_module()

    assert 18.0 <= module.TERRAIN["theta"] <= 34.0
    assert module.CAMERA["exaggeration"] >= 0.75


def test_relief_pass_is_stronger_than_color_pass() -> None:
    module = load_module()

    assert module.RELIEF_TERRAIN["shadow"] > module.TERRAIN["shadow"]
    assert module.RELIEF_TERRAIN["ambient"] < module.TERRAIN["ambient"]
    assert module.RELIEF_PBR["normal_strength"] > module.PBR["normal_strength"]
    assert module.RELIEF_PBR["sun_visibility"]["mode"] == "hard"


def test_combine_render_passes_uses_relief_luminance(tmp_path: Path) -> None:
    module = load_module()
    color = Image.new("RGBA", (8, 4), (255, 255, 255, 255))
    relief = Image.new("RGBA", (8, 4), (255, 255, 255, 255))
    color_arr = np.asarray(color).copy()
    relief_arr = np.asarray(relief).copy()
    color_arr[:, 1:7, :3] = (90, 180, 210)
    relief_arr[:, 1:4, :3] = (40, 40, 40)
    relief_arr[:, 4:7, :3] = (220, 220, 220)
    color_path = tmp_path / "color.png"
    relief_path = tmp_path / "relief.png"
    Image.fromarray(color_arr, mode="RGBA").save(color_path)
    Image.fromarray(relief_arr, mode="RGBA").save(relief_path)

    combined = module._combine_render_passes(color_path, relief_path)
    combined_arr = np.asarray(combined.convert("RGBA"))

    dark_value = int(combined_arr[2, 2, :3].max())
    light_value = int(combined_arr[2, 5, :3].max())
    assert light_value > dark_value + 20
    assert combined_arr[2, 2, 2] >= combined_arr[2, 2, 0]


def test_reframe_snapshot_outputs_white_landscape_plate() -> None:
    module = load_module()
    source = Image.new("RGBA", (300, 300), (255, 255, 255, 255))
    arr = np.asarray(source).copy()
    arr[80:230, 20:280, :3] = (40, 150, 220)
    arr[80:230, 20:280, 3] = 255
    source = Image.fromarray(arr, mode="RGBA")

    reframed = module._reframe_snapshot(source, (400, 274))
    mask = module._subject_mask_from_white(reframed)
    ys, xs = np.nonzero(mask)

    assert reframed.size == (400, 274)
    assert tuple(reframed.getpixel((0, 0))) == (255, 255, 255, 255)
    assert xs.min() >= 0
    assert xs.max() < 400
    assert ys.min() <= int(274 * 0.12)
    assert ys.max() >= int(274 * 0.88)
    assert (ys.max() - ys.min()) > int(274 * 0.76)
    assert (xs.max() - xs.min()) > int(400 * 0.90)


def test_reference_attribution_is_preserved() -> None:
    module = load_module()

    assert "©2023 Milos Popovic" in module.POSTER_CREDIT
    assert "©2026 Milos Popovic" not in module.POSTER_CREDIT


def test_palette_starts_with_reference_southern_blue() -> None:
    module = load_module()

    assert module.BASIN_PALETTE[0].lower() in {"#33a4db", "#2aaeea", "#2bb6cc"}


def test_higher_order_rivers_are_drawn_as_primary_channels() -> None:
    module = load_module()

    widths = [module._river_width_px(order, 2400) for order in (3, 5, 7, 9)]

    assert widths == sorted(widths)
    assert widths[0] < widths[-1] * 0.35
    assert module.RIVER_ALPHA_MAP[3] < module.RIVER_ALPHA_MAP[9]


def test_subject_brightening_preserves_deep_relief_shadows() -> None:
    module = load_module()
    source = Image.new("RGBA", (4, 1), (255, 255, 255, 255))
    arr = np.asarray(source).copy()
    arr[0, 1, :3] = (24, 60, 84)
    arr[0, 2, :3] = (80, 170, 210)
    source = Image.fromarray(arr, mode="RGBA")

    brightened = np.asarray(module._brighten_subject(source).convert("RGBA"))

    assert int(brightened[0, 1, :3].max()) < 105
    assert int(brightened[0, 2, 2]) > int(brightened[0, 2, 0]) + 80


def test_overlay_style_sidecar_invalidates_stale_cached_overlay(tmp_path: Path) -> None:
    module = load_module()
    overlay_path = tmp_path / "overlay.png"
    overlay_path.write_bytes(b"not a real png")

    assert not module._overlay_is_current(overlay_path)

    module._overlay_style_path(overlay_path).write_text("old-style\n", encoding="utf-8")
    assert not module._overlay_is_current(overlay_path)

    module._overlay_style_path(overlay_path).write_text(
        module.OVERLAY_STYLE_VERSION + "\n",
        encoding="utf-8",
    )
    assert module._overlay_is_current(overlay_path)
