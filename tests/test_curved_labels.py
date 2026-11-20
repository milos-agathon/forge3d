from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import forge3d
from forge3d import _forge3d as native

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("office", "NotoSans-subset.ttf"),
    ("سلام", "NotoSansArabic-subset.ttf"),
    ("नमस्ते", "NotoSansDevanagari-subset.ttf"),
    ("בְ", "NotoSansHebrew-marks-subset.ttf"),
    ("בָּ", "NotoSansHebrew-marks-subset.ttf"),
    ("שָׁלוֹם", "NotoSansHebrew-marks-subset.ttf"),
    ("בּ", "NotoSansHebrew-marks-subset.ttf"),
    ("שׁ", "NotoSansHebrew-marks-subset.ttf"),
]
PATH = [(20.0, 76.0, 0.0), (160.0, 44.0, 0.0), (300.0, 76.0, 0.0)]


def authority(text, font, path=PATH, size=24.0):
    shaped = forge3d.text.shape(text, [str(ROOT / "assets/fonts" / font)], 1.0)
    payload = native.layout_label_candidates(
        "curved", "river", text, path, shaped.to_dict()["positioned_glyphs"], size,
    )
    assert payload is not None
    return shaped, payload["candidates"][0]


def absolute(candidate, size):
    anchor = np.array(candidate["anchor"][:2])
    return np.array([anchor + np.array(g["origin"]) * size for g in candidate["positioned_glyphs"]])


@pytest.mark.parametrize("text,font", CASES)
def test_curved_shaped_glyphs_reverse_upright_and_preserve_marks(text, font):
    shaped, forward = authority(text, font)
    _, reverse = authority(text, font, list(reversed(PATH)))
    source = shaped.to_dict()["positioned_glyphs"]
    if text == "office":
        assert len(source) < len(text), "fixture must contain a genuine ligature"
    placed = forward["positioned_glyphs"]
    assert [(g["font_index"], g["glyph_id"], g["cluster"]) for g in placed] == [
        (g["font_index"], g["glyph_id"], g["cluster"]) for g in source
    ]
    np.testing.assert_allclose(absolute(forward, 24.0), absolute(reverse, 24.0), atol=0.25, rtol=0)
    for glyph in placed:
        assert abs(glyph["rotation"]) <= math.pi / 2
    base_rotation = None
    base_source = base_placed = None
    for original, glyph, point in zip(source, placed, absolute(forward, 24.0)):
        if original["advance"][0] > 0:
            base_rotation = glyph["rotation"]
            base_source, base_placed = np.array(original["origin"]), point
        else:
            assert glyph["rotation"] == base_rotation
            rotation = np.array([[math.cos(base_rotation), -math.sin(base_rotation)],
                                 [math.sin(base_rotation), math.cos(base_rotation)]])
            expected_offset = rotation @ (np.array(original["origin"])-base_source) * 24.0
            np.testing.assert_allclose(point-base_placed, expected_offset, atol=0.25, rtol=0)


@pytest.mark.parametrize("text,font", CASES)
def test_curved_centers_are_on_the_baseline_within_quarter_pixel(text, font):
    shaped, candidate = authority(text, font)
    points = absolute(candidate, 24.0)
    source = shaped.to_dict()["positioned_glyphs"]
    pen = 0.0
    deviations = []
    for original, glyph, origin in zip(source, candidate["positioned_glyphs"], points):
        advance = original["advance"][0]
        if advance > 0:
            angle = glyph["rotation"]
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            local_offset = np.array(original["origin"]) - (pen, 0.0)
            center = origin + rotation @ (np.array([advance * 0.5, 0.0]) - local_offset) * 24.0
            distances = []
            for first, last in zip(PATH, PATH[1:]):
                a, b = np.array(first[:2]), np.array(last[:2])
                t = np.clip(np.dot(center-a, b-a) / np.dot(b-a, b-a), 0, 1)
                distances.append(np.linalg.norm(center - (a+t*(b-a))))
            deviations.append(min(distances))
        pen += advance
    print(f"curved {text!r}: max baseline-normal deviation={max(deviations):.9f}px")
    assert max(deviations) <= 0.25


@pytest.mark.parametrize("text,font", CASES)
def test_public_labelplan_accepts_curved_shaped_runs_and_replays_canonically(text, font):
    plan = forge3d.LabelPlan.compile(labels=[{
        "id": "river", "text": text, "curved_text": True,
        "geometry": {"type": "LineString", "coordinates": PATH},
        "typography": {"font_size": 24.0, "font_chain": [str(ROOT / "assets/fonts" / font)]},
    }], camera={}, viewport=(320, 128), glyph_atlas={"font_chain": [str(ROOT / "assets/fonts" / font)]})
    assert not plan.rejected
    assert len(plan.accepted) == 1
    assert plan.accepted[0].candidate.details["geometry_authority"] == "layout_curved_text"
    restored = forge3d.LabelPlan.from_dict(plan.to_dict())
    assert restored.to_dict() == plan.to_dict()


def test_physical_mapscene_consumes_curved_arabic_plan_without_mutation(tmp_path):
    from forge3d.text_atlas import BakedAtlas, save_atlas

    font = str(ROOT / "assets/fonts/NotoSansArabic-subset.ttf")
    text = "سلام"
    shaped = forge3d.text.shape(text, [font], 24.0)
    atlas_range = 2.0 * (1.0 + np.sqrt(0.5)) * 96.0 / 24.0
    baked = forge3d.text.bake_msdf_atlas([font], shaped, 96.0, px_range=atlas_range, padding=2, channels=4)
    atlas_metrics = dict(baked["metrics"], font_source=font)
    image, metrics = save_atlas(BakedAtlas(baked["image"], atlas_metrics),
                                tmp_path / "arabic.png", tmp_path / "arabic.json")
    output = tmp_path / "curved-arabic.png"
    scene = forge3d.MapScene(
        terrain=forge3d.TerrainSource(data=np.zeros((8, 8), dtype=np.float32),
            crs="EPSG:32610", metadata={"width": 8, "height": 8, "source_id": "inline-dem"},
            elevation_sampling_available=True),
        camera=forge3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=120.0),
        lighting=forge3d.LightingPreset(name="daylight"),
        output=forge3d.OutputSpec(width=320, height=128, format="png", path=str(output)),
        layers=[forge3d.LabelLayer(layer_id="river", occlusion="none",
            labels=[{"id": "river", "text": text, "curved_text": True,
                "geometry": {"type": "LineString", "coordinates": PATH},
                "typography": {"font_size": 24.0, "halo_width": 1.0}}],
            glyph_atlas={"font_chain": [font], "image_path": str(image), "metrics_path": str(metrics)})],
    )
    plan = scene.compile_plan()
    before = plan.label_plans["river"].to_dict()
    assert len(before["accepted"]) == 1
    report = scene.render()
    assert report.status == "ok"
    assert scene.last_render_backend == "gpu_terrain"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert scene.compiled_label_plans["river"].to_dict() == before
    assert "<path" in scene.label_svg() and "<text" not in scene.label_svg()
    probe = native.device_probe(None)
    assert probe["status"] == "ok" and not probe["software_fallback"]
    assert probe["device_type"] in {"DiscreteGpu", "IntegratedGpu"}
    print(f"curved Arabic MapScene: {probe}, bake_ms={baked['metrics']['bake_ms']:.6f}, output={output}")
