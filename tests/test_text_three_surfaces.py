from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import forge3d
from forge3d.export import SvgTextStyleError, VectorScene, generate_svg


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


def test_python_svg_honors_explicit_font_path_and_rejects_noop_styles(monkeypatch):
    from forge3d import text

    cjk = ROOT / "assets" / "fonts" / "NotoSansSC-subset.ttf"
    original_shape = text.shape
    calls: list[list[str]] = []

    def record_shape(value, fonts, size, **kwargs):
        calls.append(list(fonts))
        return original_shape(value, fonts, size, **kwargs)

    monkeypatch.setattr(text, "shape", record_shape)
    explicit = VectorScene()
    explicit.add_label("地图", (50, 50), font_family=str(cjk), halo_width=0.0)
    generate_svg(explicit, width=100, height=100)

    assert calls and Path(calls[0][0]).resolve() == cjk.resolve()

    unsupported_family = VectorScene()
    unsupported_family.add_label("Map", (50, 50), font_family="Host Font")
    with pytest.raises(SvgTextStyleError) as family_error:
        generate_svg(unsupported_family)
    assert family_error.value.diagnostics[0]["reason"] == "unsupported_font_family"

    unsupported_weight = VectorScene()
    unsupported_weight.add_label("Map", (50, 50), font_weight="bold")
    with pytest.raises(SvgTextStyleError) as weight_error:
        generate_svg(unsupported_weight)
    assert weight_error.value.diagnostics[0]["reason"] == "unsupported_font_weight"


def test_cpu_text_source_has_no_bitmap_or_exception_fallback():
    source = (ROOT / "python" / "forge3d" / "_map_scene_render.py").read_text(
        encoding="utf-8"
    )

    assert "_draw_text_fallback" not in source
    assert "ImageFont" not in source
    assert "ImageDraw" not in source
    assert ".text(" not in source
    assert "ord(char)" not in source


def test_cpu_text_rasterizes_only_the_local_outline_roi(monkeypatch):
    from forge3d import _map_scene_render, text

    calls: list[tuple[int, int, tuple[float, float]]] = []

    class FakeShaped:
        @staticmethod
        def outline_bounds():
            return (0.0, -10.0, 20.0, 2.0)

    monkeypatch.setattr(text, "shape", lambda *_args, **_kwargs: FakeShaped())

    def rasterize(_shaped, width, height, origin=(0.0, 0.0), line_ranges=None):
        del line_ranges
        calls.append((width, height, origin))
        return np.zeros((height, width), dtype=np.float32)

    monkeypatch.setattr(text, "rasterize_shaped_run", rasterize)
    image = np.zeros((2048, 2048, 4), dtype=np.uint8)
    _map_scene_render._draw_text(
        image,
        "ROI",
        (100, 100),
        color=(255, 255, 255, 255),
        halo=(0, 0, 0, 0),
        font_size=12,
    )

    assert len(calls) == 1
    assert calls[0][0] < 32 and calls[0][1] < 32


def test_furniture_text_has_no_pillow_or_system_font_route():
    for name in ("scale_bar.py", "north_arrow.py", "legend.py", "map_plate.py"):
        source = (ROOT / "python" / "forge3d" / name).read_text(encoding="utf-8")
        assert "ImageFont" not in source
        assert "DejaVuSans" not in source
        assert "Arial" not in source
        assert "load_default" not in source
        assert ".text(" not in source
    for name in ("scale_bar.py", "north_arrow.py", "legend.py"):
        source = (ROOT / "python" / "forge3d" / name).read_text(encoding="utf-8")
        assert "PIL" not in source


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


def test_legend_and_map_plate_title_call_shared_native_text(monkeypatch):
    from forge3d import _map_scene_render
    from forge3d.legend import Legend, LegendConfig
    from forge3d.map_plate import MapPlate

    calls: list[str] = []

    def record(image, text, anchor, **kwargs):
        calls.append(str(text))
        image[max(0, anchor[1]), max(0, anchor[0])] = (1, 2, 3, 255)

    monkeypatch.setattr(_map_scene_render, "_draw_text", record)
    colors = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    legend = Legend(
        colors,
        (0.0, 10.0),
        LegendConfig(tick_count=3, title="Elevation", label_suffix=""),
    ).render()
    plate = MapPlate.__new__(MapPlate)
    plate._title = SimpleNamespace(text="Native title", font_size=18, color=(0, 0, 0, 255))
    title = plate._render_title()

    assert calls == ["0", "5", "10", "Elevation", "Native title"]
    assert np.any(legend[..., 3] > 0)
    assert title is not None and np.any(title[..., 3] > 0)
