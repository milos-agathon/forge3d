from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
from forge3d.text_atlas import (
    bake_atlas,
    default_latin_atlas_paths,
    load_atlas_metrics,
    save_atlas,
    validate_atlas_metrics,
)


def test_bake_atlas_produces_sdf_image_and_roundtrips_metrics(tmp_path):
    atlas = bake_atlas(charset="AB", font_size=20, px_range=4, padding=2)

    assert atlas.image.ndim == 3
    assert atlas.image.shape[2] == 4
    assert atlas.image.dtype == np.uint8
    assert atlas.image[..., 0].min() < 128
    assert atlas.image[..., 0].max() > 128
    assert set(atlas.metrics["glyphs"]) == {str(ord("A")), str(ord("B"))}

    png_path, json_path = save_atlas(atlas, tmp_path / "atlas.png", tmp_path / "atlas.json")
    loaded = load_atlas_metrics(json_path)

    assert png_path.exists()
    assert loaded["glyphs"][str(ord("A"))]["w"] > 0
    assert loaded["channels"] == 1


def test_validate_atlas_metrics_rejects_malformed_payload():
    with pytest.raises(ValueError, match="missing field"):
        validate_atlas_metrics({"font_size": 12, "glyphs": {}})

    with pytest.raises(ValueError, match="missing metric"):
        validate_atlas_metrics(
            {
                "font_size": 12,
                "line_height": 16,
                "baseline": 12,
                "glyphs": {"65": {"x": 0}},
            }
        )


def test_default_latin_atlas_is_packaged_and_used_by_fontatlas():
    png_path, json_path = default_latin_atlas_paths()
    atlas = f3d.FontAtlas.default_latin()

    assert png_path.exists()
    assert json_path.exists()
    assert atlas.covers("A")
    assert atlas.source_path == str(json_path)
    assert atlas.coverage["atlas_kind"] == "sdf_font_atlas"
    assert atlas.coverage["image_path"] == str(png_path)
    assert f3d.load_atlas_metrics(json_path)["glyphs"][str(ord("A"))]["adv"] > 0


def test_label_layer_accepts_fontatlas_binding():
    atlas = f3d.FontAtlas.default_latin()
    layer = f3d.LabelLayer(
        layer_id="labels",
        labels=[{"id": "a", "text": "A", "geometry": {"type": "Point", "coordinates": [8, 8]}}],
        atlas=atlas,
    )

    payload = layer.to_dict()

    assert layer.atlas is atlas
    assert layer.glyph_atlas["glyphs"] == sorted(atlas.glyphs)
    assert payload["atlas"]["source_path"] == atlas.source_path
    assert payload["glyph_atlas"]["source_path"] == atlas.source_path


def test_default_latin_atlas_is_in_package_data_and_sdist_manifest():
    package_data = Path("pyproject.toml").read_text(encoding="utf-8")
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    assert "data/fonts/*.png" in package_data
    assert "data/fonts/*.json" in package_data
    assert "python/forge3d/data/fonts" in manifest
    assert "*.png" in manifest
    assert "*.json" in manifest


def test_label_plan_normalizes_halo_typography_aliases():
    from forge3d.label_plan import LabelPlan

    plan = LabelPlan.compile(
        labels=[
            {
                "id": "a",
                "text": "A",
                "geometry": {"type": "Point", "coordinates": [10, 10]},
            }
        ],
        camera={},
        viewport=(64, 64),
        glyph_atlas={"glyphs": ["A"]},
        typography={"text_halo_width": 3.0, "text_halo_color": [0.0, 0.0, 0.0, 0.75]},
    )

    assert len(plan.accepted) == 1
    typography = plan.accepted[0].typography
    assert typography["halo_width_px"] == 3.0
    assert typography["halo_width"] == 3.0
    assert typography["halo_color"] == [0.0, 0.0, 0.0, 0.75]
