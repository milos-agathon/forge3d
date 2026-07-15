from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import forge3d
from forge3d.export import VectorScene, generate_svg


ROOT = Path(__file__).resolve().parents[1]
LATIN_FONT = ROOT / "assets" / "fonts" / "NotoSans-subset.ttf"
HEBREW_FONT = ROOT / "assets" / "fonts" / "NotoSansHebrew-subset.ttf"


def test_native_rasterization_is_float32_deterministic_coverage():
    shaped = forge3d.text.shape("Map", [str(LATIN_FONT)], 18.0)
    first = forge3d.text.rasterize_shaped_run(
        shaped, 96, 48, origin=(4.0, 32.0)
    )
    second = forge3d.text.rasterize_shaped_run(
        shaped, 96, 48, origin=(4.0, 32.0)
    )

    assert first.dtype == np.float32
    assert first.shape == (48, 96)
    assert first.tobytes() == second.tobytes()
    assert 0.0 <= float(first.min()) < float(first.max()) <= 1.0
    assert np.any((first > 0.0) & (first < 1.0))


def test_positioned_outline_order_depends_on_supplied_line_partition():
    shaped = forge3d.text.shape(
        "Map שלום", [str(LATIN_FONT), str(HEBREW_FONT)], 18.0
    )
    whole = shaped.svg_path([(0, 8)])
    wrapped = shaped.svg_path([(0, 6), (6, 8)])

    assert whole != wrapped
    assert whole.startswith("M")
    assert wrapped.startswith("M")


def test_python_svg_labels_are_native_outlines_with_identical_halo_geometry():
    scene = VectorScene()
    scene.add_label("Map", (50, 50), font_size=18, halo_width=2.0)

    svg = generate_svg(scene, width=100, height=100)
    paths = re.findall(r'<path\b[^>]*\bd="([^"]+)"[^>]*/>', svg)

    assert "<text" not in svg
    assert len(paths) == 2
    assert paths[0] == paths[1]

    transforms = re.findall(r'transform="translate\(([-0-9.]+) ([-0-9.]+)\)"', svg)
    assert len(transforms) == 2
    font = ROOT / "python" / "forge3d" / "data" / "fonts" / "NotoSansLatin-subset.ttf"
    bounds = forge3d.text.shape("Map", [str(font)], 18.0).outline_bounds()
    assert bounds is not None
    tx, ty = map(float, transforms[0])
    assert abs(tx + (bounds[0] + bounds[2]) * 0.5 - 50.0) <= 0.01
    assert abs(ty + (bounds[1] + bounds[3]) * 0.5 - 50.0) <= 0.01


def test_cpu_text_source_has_no_bitmap_or_exception_fallback():
    source = (ROOT / "python" / "forge3d" / "_map_scene_render.py").read_text(
        encoding="utf-8"
    )

    assert "_draw_text_fallback" not in source
    assert "ImageFont" not in source
    assert "ImageDraw" not in source
    assert ".text(" not in source
    assert "ord(char)" not in source


def test_furniture_text_has_no_pillow_or_system_font_route():
    for name in ("scale_bar.py", "north_arrow.py"):
        source = (ROOT / "python" / "forge3d" / name).read_text(encoding="utf-8")
        assert "PIL" not in source
        assert "ImageFont" not in source
        assert "DejaVuSans" not in source
        assert "Arial" not in source
        assert "load_default" not in source


def test_scale_bar_and_north_arrow_call_shared_native_text(monkeypatch):
    from forge3d import _map_scene_render
    from forge3d.north_arrow import NorthArrow
    from forge3d.scale_bar import ScaleBar

    calls: list[str] = []

    def record(image, text, anchor, **kwargs):
        calls.append(str(text))
        image[max(0, anchor[1]), max(0, anchor[0])] = (1, 2, 3, 255)

    monkeypatch.setattr(_map_scene_render, "_draw_text", record)
    scale = ScaleBar(100.0).render()
    north = NorthArrow().render()

    assert calls == ["10 km", "N"]
    assert np.any(scale[..., 3] > 0)
    assert np.any(north[..., 3] > 0)
