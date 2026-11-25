from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import forge3d
from forge3d import _forge3d as native
from tests._deltae import delta_e_2000, srgb_to_lab
from tests._ssim import ssim

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("NotoSansLatin-subset.ttf", "office AV"),
    ("NotoSansArabic-subset.ttf", "سلام"),
    ("NotoSansDevanagari-subset.ttf", "नमस्ते"),
    ("NotoSansHebrew-marks-subset.ttf", "שָׁלוֹם"),
]


@pytest.mark.parametrize("font,text", CASES)
@pytest.mark.parametrize("curved", [False, True])
def test_physical_gpu_cpu_svg_pairwise_letterforms(font, text, curved, tmp_path):
    from xml.etree import ElementTree as ET
    from forge3d.map_scene import _compiled_label_render_style, _mapscene_native_label_batches
    from forge3d.text_atlas import BakedAtlas, save_atlas

    font_path = ROOT / "assets/fonts" / font
    size, bake_size = 32.0, 96.0
    width, height, halo_width = 448, 128, 1.0
    shaped = forge3d.text.shape(text, [font_path], size)
    # The encoding represents +/- px_range/2 source pixels. It must contain
    # the halo and a pixel's half diagonal after scaling to the display size.
    atlas_range = 2.0 * (halo_width + np.sqrt(0.5)) * bake_size / size
    baked = forge3d.text.bake_msdf_atlas(
        [font_path], shaped, bake_size, px_range=atlas_range, padding=2, channels=4,
    )
    np.testing.assert_array_equal(baked["image"][..., 3], baked["sdf_image"])
    metrics = dict(baked["metrics"], font_source=str(font_path))
    atlas_png, atlas_json = save_atlas(
        BakedAtlas(baked["image"], metrics), tmp_path / "atlas.png", tmp_path / "atlas.json",
    )
    geometry = (
        {"type": "LineString", "coordinates": [(20, 84, 0), (220, 50, 0), (420, 84, 0)]}
        if curved else {"type": "Point", "coordinates": [220, 80]}
    )
    map_scene = forge3d.MapScene(
        terrain=forge3d.TerrainSource(data=np.zeros((8, 8), dtype=np.float32), crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "source_id": "inline-dem"}),
        camera=forge3d.OrbitCamera(target=(0, 0, 0), distance=120.0),
        lighting=forge3d.LightingPreset(name="daylight"),
        output=forge3d.OutputSpec(width=width, height=height, format="png"),
        layers=[forge3d.LabelLayer(layer_id="labels", occlusion="none",
            labels=[{"id": "label", "text": text, "curved_text": curved, "geometry": geometry,
                "typography": {"font_size": size, "halo_width": halo_width,
                               "color": "#ffffffff", "halo_color": "#404040ff"}}],
            glyph_atlas={"font_chain": [str(font_path)], "image_path": str(atlas_png),
                         "metrics_path": str(atlas_json)})],
    )
    scene_plan = map_scene.compile_plan()
    label_plan = scene_plan.label_plans["labels"]
    before = label_plan.to_dict()
    assert len(label_plan.accepted) == 1 and not label_plan.rejected
    label = label_plan.accepted[0]
    placements, origin, _, _, _, _, _ = _compiled_label_render_style(
        label, "labels", width, height, size,
    )
    svg = map_scene.label_svg()
    assert map_scene.compiled_plan is scene_plan
    assert scene_plan.label_plans["labels"].to_dict() == before
    (tmp_path / "label.svg").write_text(svg, encoding="utf-8")
    root = ET.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert not root.findall(".//svg:text", ns)
    halo_path, fill_path = root.findall(".//svg:path", ns)
    assert halo_path.attrib["d"] == fill_path.attrib["d"] and fill_path.attrib["d"]
    assert halo_path.attrib["stroke-linejoin"] == "round"
    assert halo_path.attrib["stroke-width"] == str(2 * halo_width)
    assert (fill_path.attrib["fill"], halo_path.attrib["fill"]) == ("#ffffff", "#404040")
    assert fill_path.attrib["opacity"] == halo_path.attrib["opacity"] == "1.0"
    # CPU and SVG decode different path inputs through the same analytic rasterizer.
    cpu_fill, cpu_halo = (
        forge3d.text.rasterize_shaped_run(shaped, width, height, origin=origin,
            positioned_glyphs=placements, halo_width=halo)
        for halo in (0.0, halo_width)
    )
    svg_fill = forge3d.text.rasterize_svg_path(fill_path.attrib["d"], width, height)
    svg_halo = forge3d.text.rasterize_svg_path(
        halo_path.attrib["d"], width, height, halo_width=float(halo_path.attrib["stroke-width"]) / 2,
    )

    def composite(fill, halo):
        return np.rint(np.clip(fill + (64 / 255) * halo * (1.0-fill), 0, 1) * 255).astype(np.uint8)

    cpu, exported = composite(cpu_fill, cpu_halo), composite(svg_fill, svg_halo)
    batch, = _mapscene_native_label_batches(map_scene.recipe, scene_plan.label_plans, width, height)
    atlas_height, atlas_width = batch["atlas"].shape[:2]
    scene = forge3d.Scene(width, height)
    scene.disable_terrain()
    backdrop = np.zeros((height, width, 4), dtype=np.uint8)
    backdrop[..., 3] = 255
    scene.set_raster_overlay(backdrop, 1.0, None, None)
    texture_bytes_before = native.global_memory_metrics()["texture_bytes"]
    scene.set_native_text_atlas(batch["atlas"], batch["channels"], 1.0, px_range=batch["px_range"])
    tracked_atlas_bytes = native.global_memory_metrics()["texture_bytes"] - texture_bytes_before
    assert tracked_atlas_bytes == atlas_width * atlas_height * 4
    scene.enable_native_text()
    for index, rect in enumerate(batch["rects"]):
        if index in batch["label_starts"]:
            scene.begin_native_text_label()
        scene.add_native_text_rect_uv_halo(*rect)
    gpu = np.asarray(scene.render_rgba())[..., 0]
    assert not [d for d in native.native_degradations()
                if d["kind"] in {"uncaptured_gpu_error", "validation_error"}]
    from forge3d._png import save_png
    for name, pixels in [("gpu", gpu), ("cpu", cpu), ("svg", exported)]:
        save_png(tmp_path / f"{name}.png", np.repeat(pixels[..., None], 3, axis=2))
    adapter = native.device_probe(None)
    assert adapter["status"] == "ok" and not adapter["software_fallback"]
    assert adapter["device_type"] in {"DiscreteGpu", "IntegratedGpu"}


    print(f"{text!r} curved={curved} adapter={adapter} bake_ms={metrics['bake_ms']:.6f} atlas_bytes={metrics['byte_count']} tracked_gpu_atlas_bytes={tracked_atlas_bytes}")
    glyph_pixels = cpu_fill > 0
    ys, xs = np.where(glyph_pixels)
    roi = (slice(ys.min(), ys.max()+1), slice(xs.min(), xs.max()+1))
    failures = []
    for pair, a, b in [("GPU/CPU",gpu,cpu),("GPU/SVG",gpu,exported),("CPU/SVG",cpu,exported)]:
        rgb_a, rgb_b = np.repeat(a[...,None],3,axis=2), np.repeat(b[...,None],3,axis=2)
        delta = delta_e_2000(srgb_to_lab(rgb_a),srgb_to_lab(rgb_b))
        fraction = float(np.mean(delta[glyph_pixels] < 2.0))
        similarity = ssim(a[roi],b[roi])
        print(f"{text!r} curved={curved} {pair}: DeltaE<2 fraction={fraction:.9f}, SSIM={similarity:.9f}")
        if fraction < 0.99 or similarity <= 0.99:
            failures.append((pair,fraction,similarity))
    assert not failures, failures


@pytest.mark.parametrize("opacity,local_alpha", [
    (0.5, 1.0), (1.0, 0.5), (0.5, 0.5),
])
def test_physical_rgba_union_applies_translucency_once(opacity, local_alpha):
    scene = forge3d.Scene(32, 32)
    scene.disable_terrain()
    background = np.zeros((32, 32, 4), dtype=np.uint8)
    background[..., 3] = 255
    scene.set_raster_overlay(background, 1.0, None, None)
    scene.set_native_text_atlas(np.full((2, 2, 4), 255, dtype=np.uint8), 4, 1.0, px_range=8.0)
    scene.enable_native_text()
    scene.set_native_text_alpha(opacity)
    scene.begin_native_text_label()
    for _ in range(2):
        scene.add_native_text_rect_uv(4, 4, 24, 24, 0, 0, 1, 1, 1, 1, 1, local_alpha)
    pixels = np.asarray(scene.render_rgba())
    legacy = forge3d.Scene(32, 32)
    legacy.disable_terrain()
    legacy.set_raster_overlay(background, 1.0, None, None)
    legacy.set_native_text_atlas(np.full((2, 2, 4), 255, dtype=np.uint8), 3, 1.0)
    legacy.enable_native_text()
    legacy.set_native_text_alpha(opacity)
    legacy.add_native_text_rect_uv(4, 4, 24, 24, 0, 0, 1, 1, 1, 1, 1, local_alpha)
    reference = np.asarray(legacy.render_rgba())
    # Exact compatibility comparison; do not prescribe tie rounding of
    # 0.5*255. The interior is fully covered in both reconstruction paths.
    np.testing.assert_array_equal(pixels[8:24, 8:24], reference[8:24, 8:24])
    assert 0 < int(reference[12, 12, 0]) < 255
    print(f"union translucency global={opacity} local={local_alpha}: "
          f"RGBA={pixels[12,12].tolist()} legacy={reference[12,12].tolist()}")
    assert not [d for d in native.native_degradations()
                if d["kind"] in {"uncaptured_gpu_error", "validation_error"}]
    probe = native.device_probe(None)
    assert probe["status"] == "ok" and not probe["software_fallback"]
    assert probe["device_type"] in {"DiscreteGpu", "IntegratedGpu"}


@pytest.mark.parametrize("channels,opacity,expected", [
    (3, 1.0, [255, 255, 255]),
    (3, 0.5, [127, 127, 127]),
    (1, 0.5, [127, 127, 127]),
    (1, 1.0, [255, 255, 255]),
])
def test_legacy_atlas_does_not_paint_a_halo_inside_the_glyph(channels, opacity, expected):
    """Preserve the single-draw shader opacity on fully covered legacy texels."""
    scene = forge3d.Scene(32, 32)
    scene.disable_terrain()
    backdrop = np.zeros((32, 32, 4), dtype=np.uint8)
    backdrop[..., 3] = 255
    scene.set_raster_overlay(backdrop, 1.0, None, None)
    scene.set_native_text_atlas(np.full((2, 2, 3), 255, dtype=np.uint8), channels, 1.0)
    scene.enable_native_text()
    scene.set_native_text_alpha(opacity)
    scene.add_native_text_rect_uv_halo(4,4,24,24,0,0,1,1,
        1,1,1,1,0,0,0,0,0)
    scene.add_native_text_rect_uv_halo(4,4,24,24,0,0,1,1,
        0,0,0,0,0,0,1,1,1)
    pixels = np.asarray(scene.render_rgba())
    assert pixels[12,12,:3].tolist() == expected
    assert not [d for d in native.native_degradations()
                if d["kind"] in {"uncaptured_gpu_error", "validation_error"}]
    adapter = native.device_probe(None)
    assert adapter["status"] == "ok" and not adapter["software_fallback"]
    assert adapter["device_type"] in {"DiscreteGpu", "IntegratedGpu"}
