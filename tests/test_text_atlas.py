from __future__ import annotations

import hashlib
import json
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
from forge3d._png import load_png_rgba


def test_bake_atlas_delegates_to_native_rgb_msdf_and_roundtrips(tmp_path):
    atlas = bake_atlas(charset="AB", font_size=20, px_range=4, padding=2)

    assert atlas.image.ndim == 3
    assert atlas.image.shape[2] == 3
    assert atlas.image.dtype == np.uint8
    assert np.any(atlas.image[..., 0] != atlas.image[..., 1])
    assert atlas.metrics["kind"] == "msdf_font_atlas"
    assert atlas.metrics["channels"] == 3
    assert atlas.metrics["byte_count"] == atlas.image.nbytes
    assert atlas.metrics["bake_ms"] >= 0.0
    assert set(atlas.metrics["glyphs"]) == {str(ord("A")), str(ord("B"))}

    png_path, json_path = save_atlas(atlas, tmp_path / "atlas.png", tmp_path / "atlas.json")
    loaded = load_atlas_metrics(json_path)

    assert png_path.exists()
    assert np.array_equal(load_png_rgba(png_path)[..., :3], atlas.image)
    assert loaded["glyphs"][str(ord("A"))]["w"] > 0
    assert loaded["channels"] == 3


def test_validate_atlas_metrics_rejects_malformed_payload():
    with pytest.raises(ValueError, match="missing field"):
        validate_atlas_metrics({"font_size": 12, "glyphs": {}})

    with pytest.raises(ValueError, match="missing metric"):
        validate_atlas_metrics(
            {
                "kind": "msdf_font_atlas",
                "font_size": 12,
                "line_height": 16,
                "baseline": 12,
                "px_range": 4,
                "padding": 2,
                "channels": 3,
                "width": 8,
                "height": 8,
                "bake_ms": 0,
                "byte_count": 192,
                "font_source": "font.ttf",
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
    assert atlas.coverage["atlas_kind"] == "msdf_font_atlas"
    assert atlas.coverage["image_path"] == str(png_path)
    assert f3d.load_atlas_metrics(json_path)["glyphs"][str(ord("A"))]["adv"] > 0
    assert f3d.load_atlas_metrics(json_path)["channels"] == 3


def test_default_bake_is_deterministic_and_uses_committed_font():
    first = bake_atlas(charset="Map", font_size=18, px_range=4, padding=2)
    second = bake_atlas(charset="Map", font_size=18, px_range=4, padding=2)

    assert first.image.tobytes() == second.image.tobytes()
    assert first.metrics["font_source"] == "forge3d/data/fonts/NotoSansLatin-subset.ttf"
    assert (default_latin_atlas_paths()[0].parent / "NotoSansLatin-subset.ttf").exists()


def test_independent_atlas_saves_are_byte_identical(tmp_path):
    first = bake_atlas(charset="Map", font_size=18, px_range=4, padding=2)
    second = bake_atlas(charset="Map", font_size=18, px_range=4, padding=2)

    first_png, first_json = save_atlas(first, tmp_path / "first.png", tmp_path / "first.json")
    second_png, second_json = save_atlas(second, tmp_path / "second.png", tmp_path / "second.json")

    assert first_png.read_bytes() == second_png.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    assert json.loads(first_json.read_text(encoding="utf-8"))["bake_ms"] == 0.0


def test_shaped_atlas_rejects_a_different_supplied_font_chain():
    root = Path(__file__).resolve().parents[1]
    latin = root / "assets" / "fonts" / "NotoSansLatin-subset.ttf"
    arabic = root / "assets" / "fonts" / "NotoSansArabic-subset.ttf"
    shaped = f3d.text.shape("Map", [str(latin)], 18.0)

    with pytest.raises(f3d.text.TextShapingError) as caught:
        f3d.text.bake_msdf_atlas([str(arabic)], shaped, 18.0)

    assert caught.value.diagnostics[0]["reason"] == "atlas_font_mismatch"


def test_text_atlas_has_no_bitmap_or_optional_dependency_tokens():
    source = Path("python/forge3d/text_atlas.py").read_text(encoding="utf-8")
    for token in ("PIL", "ImageFont", "ImageDraw", "scipy", "_sdf", "bitmap-mask"):
        assert token not in source


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
    assert "data/fonts/*.ttf" in package_data
    assert "data/fonts/*.txt" in package_data
    assert "data/fonts/*.md" in package_data
    assert "python/forge3d/data/fonts" in manifest
    assert "*.png" in manifest
    assert "*.json" in manifest
    assert "*.ttf" in manifest
    assert "*.txt" in manifest
    assert "*.md" in manifest


def test_packaged_font_notices_map_every_binary_to_exact_hash_and_notice():
    font_dir = Path("python/forge3d/data/fonts")
    notices = json.loads((font_dir / "FONT-NOTICES.json").read_text(encoding="utf-8"))
    expected = {
        "NotoSansArabic-subset.ttf": "Copyright 2022 The Noto Project Authors (https://github.com/notofonts/arabic)",
        "NotoSansDevanagari-subset.ttf": "Copyright 2022 The Noto Project Authors (https://github.com/notofonts/devanagari)",
        "NotoSansHebrew-subset.ttf": "Copyright 2022 The Noto Project Authors (https://github.com/notofonts/hebrew)",
        "NotoSansLatin-subset.ttf": "Copyright 2022 The Noto Project Authors (https://github.com/notofonts/latin-greek-cyrillic)",
        "NotoSansSC-subset.ttf": "Copyright 2014-2021 Adobe (http://www.adobe.com/), with Reserved Font Name 'Source'",
    }

    assert set(notices["fonts"]) == set(expected)
    assert (font_dir / notices["license_file"]).read_text(encoding="utf-8").startswith(
        "This Font Software is licensed under the SIL Open Font License, Version 1.1."
    )
    for name, copyright_notice in expected.items():
        binary = font_dir / name
        entry = notices["fonts"][name]
        assert entry["copyright_notice"] == copyright_notice
        assert entry["upstream"].startswith("https://github.com/google/fonts/")
        assert entry["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    for name, expected_hash in notices["generated_assets"].items():
        data = (font_dir / name).read_bytes()
        if name.endswith(".json"):
            data = data.replace(b"\r\n", b"\n")
        assert expected_hash == hashlib.sha256(data).hexdigest()


def test_font_inventory_covers_every_distributed_ttf_and_atlas_asset():
    root = Path(".")
    inventory = json.loads((root / "python/forge3d/data/fonts/FONT-INVENTORY.json").read_text(encoding="utf-8"))
    ttf_paths = {
        path.as_posix()
        for directory in (root / "assets/fonts", root / "python/forge3d/data/fonts")
        for path in directory.glob("*.ttf")
    }
    inventory_ttf_paths = {
        path
        for record in inventory["fonts"]
        for path in record["archive_paths"]
        if path.startswith("assets/") or path.startswith("python/")
    }

    assert inventory["subset_tool"] == {"name": "fontTools pyftsubset", "version": "4.58.5"}
    assert inventory_ttf_paths == ttf_paths
    for record in inventory["fonts"]:
        repo_paths = [
            root / path
            for path in record["archive_paths"]
            if path.startswith("assets/") or path.startswith("python/")
        ]
        assert repo_paths
        assert record["upstream_url"].startswith("https://raw.githubusercontent.com/google/fonts/")
        assert record["copyright"]
        assert record["subset_inputs"]
        assert all((root / path).is_file() for path in record["license_paths"])
        for path in repo_paths:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    for asset in inventory["generated_assets"]:
        path = root / asset["path"]
        assert path.is_file()
        data = path.read_bytes()
        if path.suffix == ".json":
            data = data.replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == asset["sha256"]


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
