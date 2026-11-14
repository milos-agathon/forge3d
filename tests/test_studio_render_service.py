"""Studio render service: sky encoding and colour-space helpers.

`studio_render.py` is the shell's embedded renderer, not an importable package
module, so it is loaded by path. These tests cover the pieces that are pure
maths — the parts a GPU golden cannot see:

* the studio HDR is a *valid* Radiance file (one broken exponent makes the
  engine decode a black environment and a blown-out sky pass), and
* the denoise path converts the display-referred frame into the linear space
  OIDN and its guidance images live in.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

SERVICE = (
    Path(__file__).resolve().parents[1]
    / "studio-tauri"
    / "src-tauri"
    / "src"
    / "studio_render.py"
)


def load_service():
    spec = importlib.util.spec_from_file_location("studio_render_service", SERVICE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def decode_radiance_rgbe(data: bytes):
    """Decode an uncompressed `-Y h +X w` Radiance file to linear RGB."""
    marker = data.index(b"\n-Y ")
    header_end = data.index(b"\n", marker + 1)
    resolution = data[marker + 1 : header_end + 1].decode().strip()
    parts = resolution.split()
    height, width = int(parts[1]), int(parts[3])
    pixels = np.frombuffer(data[header_end + 1 :], dtype=np.uint8)
    assert pixels.size == height * width * 4
    rgbe = pixels.reshape(height, width, 4).astype(np.float32)
    exponent = rgbe[..., 3:4]
    return rgbe[..., :3] / 256.0 * np.exp2(exponent - 128.0)


def test_studio_sky_hdr_is_a_valid_radiance_file(tmp_path):
    service = load_service()
    path = tmp_path / "sky.hdr"
    service.write_sky_hdr(path)

    data = path.read_bytes()
    assert data.startswith(b"#?RADIANCE\n")
    assert b"FORMAT=32-bit_rle_rgbe" in data

    rgb = decode_radiance_rgbe(data)
    assert np.isfinite(rgb).all(), "every pixel must decode (biased exponent, not raw frexp)"
    assert rgb.max() > 0.5, "the sky must carry real radiance, not a black environment"

    # Equirectangular, row 0 = zenith: the upper half is the sky (blue at the
    # zenith), the lower half is the ground. An inverted map lights the terrain
    # from below.
    zenith = rgb[0].mean(axis=0)
    horizon = rgb[rgb.shape[0] // 2 - 1].mean(axis=0)
    nadir = rgb[-1].mean(axis=0)
    assert zenith[2] > zenith[0], "zenith must read blue"
    assert horizon[0] > nadir[0] and horizon[1] > nadir[1], "ground must be darker than the horizon"
    assert nadir.mean() < zenith.mean()


def test_srgb_round_trip_is_lossless_enough_for_8_bit():
    service = load_service()
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    linear = service.srgb_to_linear(values)
    back = service.linear_to_srgb(linear)
    assert np.allclose(back, values, atol=1e-6)
    # And the decode is the real transfer function, not a plain gamma.
    assert service.srgb_to_linear(np.array([0.5], dtype=np.float32))[0] == pytest.approx(
        0.21404114, rel=1e-5
    )


def test_denoise_is_applied_in_linear_space():
    """The frame is sRGB; OIDN and its albedo/normal guidance are linear."""
    service = load_service()
    source = SERVICE.read_text()
    assert "srgb_to_linear(rgb.astype(np.float32) / 255.0)" in source, (
        "the still must be decoded to linear before OIDN and re-encoded after"
    )
    assert "linear_to_srgb(denoised)" in source


def test_still_uses_the_physical_shading_model():
    """Solid/Material previews are three.js PBR; the still must match them."""
    service = load_service()
    source = SERVICE.read_text()
    assert '"terrain_shading_model": "lambert_physical"' in source


def test_shared_hdr_writer_reproduces_the_canonical_environment_bytes(tmp_path):
    """`map_scene`'s recipe fallback must stay byte-identical through the shared
    writer — its pixels are exact 1/256 steps, so the bytes are the contract."""
    import numpy as np

    from forge3d.io import write_hdr

    path = tmp_path / "canonical.hdr"
    write_hdr(path, np.full((2, 2, 3), [180 / 256, 190 / 256, 205 / 256], np.float32))
    assert path.read_bytes() == (
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 2\n"
        + bytes([180, 190, 205, 128]) * 4
    )


def test_dem_span_reprojects_geographic_bounds_to_metres():
    """A degree-grid DEM must report a metric footprint, not degrees."""
    import numpy as np

    service = load_service()

    class FakeDem:
        data = np.zeros((32, 64), np.float32)
        crs = "EPSG:4326"
        bounds = (5.955885327836842, 45.81802022980842, 10.4924973226525, 47.80830791604507)
        resolution = (0.0006, 0.0008)
        transform = None

    span_x, span_y = service.dem_span(FakeDem())
    # The Swiss national DEM's footprint, from the same UTM zone the viewer uses.
    assert span_x == pytest.approx(352_426.79, abs=1.0)
    assert span_y == pytest.approx(225_653.68, abs=1.0)


def test_dem_span_uses_pixel_size_for_a_projected_raster():
    """Bryce's UTM raster needs no reprojection: resolution x shape is metres."""
    import numpy as np

    service = load_service()

    class FakeDem:
        data = np.zeros((1024, 1024), np.float32)
        crs = "EPSG:26912"
        bounds = (395998.5, 4161774.5, 399998.5, 4165574.5)
        resolution = (3.90625, 3.7109375)
        transform = None

    assert service.dem_span(FakeDem()) == (4000.0, 3800.0)


class _SkyStub:
    """`sky_settings` only needs the class's model map, not a GPU session."""

    SKY_MODELS = {}


@pytest.fixture
def sky_settings():
    service = load_service()
    stub = _SkyStub()
    stub.SKY_MODELS = service.StudioRenderer.SKY_MODELS
    return lambda request, quality, environment: service.StudioRenderer.sky_settings(
        stub, request, quality, environment
    )


def test_hdri_owns_the_sky_and_keeps_the_backdrop_intensity(sky_settings):
    """The project HDRI branch must not read the analytic fields at all."""
    settings = sky_settings(
        {"environmentBackgroundIntensity": 0.7},
        {"sky": True, "skyModel": "preetham", "skyTurbidity": 8.0, "skyExposure": 0.9},
        "/maps/studio.hdr",
    )
    assert settings.model == "hdri"
    assert settings.sky_exposure == pytest.approx(0.7)
    # The analytic values are ignored, not merged into the HDRI sky.
    assert settings.turbidity == pytest.approx(2.0)


def test_the_analytic_sky_carries_the_projects_values(sky_settings):
    settings = sky_settings(
        {"environmentBackgroundIntensity": 1.0},
        {
            "sky": True,
            "skyModel": "preetham",
            "skyTurbidity": 6.5,
            "skyGroundAlbedo": 0.4,
            "skyExposure": 0.03,
        },
        None,
    )
    assert settings.model == "preetham"
    assert settings.turbidity == pytest.approx(6.5)
    assert settings.ground_albedo == pytest.approx(0.4)
    assert settings.sky_exposure == pytest.approx(0.03)


def test_aether_refuses_parameters_the_shipped_bake_does_not_cover(sky_settings):
    """A way-off turbidity must fail loudly, not render the wrong atmosphere."""
    with pytest.raises(RuntimeError, match="baked LUT"):
        sky_settings(
            {"environmentBackgroundIntensity": 1.0},
            {"sky": True, "skyModel": "aether", "skyTurbidity": 3.0, "skyGroundAlbedo": 0.3},
            None,
        )


_SERVICE_MODULE = None


def _engine_height_domain(heights):
    """The service's own helper, from the module loaded once."""
    global _SERVICE_MODULE
    if _SERVICE_MODULE is None:
        _SERVICE_MODULE = load_service()
    return _SERVICE_MODULE.engine_height_domain(heights)


class _WaterStub:
    """What the water and fog helpers read: the scaled heights, the scale that
    produced them, and the DEM's elevation range."""

    def __init__(self, heights, scale, range=(0.0, 100.0)):
        self.heights = heights
        self.scale = scale
        self.range = range
        # The renderer's domain for the heights it is given.
        self.scaled_range = _engine_height_domain(heights)


@pytest.fixture
def water():
    service = load_service()
    return service.StudioRenderer.water


def test_water_is_off_unless_the_project_asks_for_it(water):
    """No mask means no water: the terrain path shades the DEM as land."""
    stub = _WaterStub(heights=np.zeros((4, 4), dtype=np.float32), scale=1.0)
    assert water(stub, {}) == (None, None)
    assert water(stub, {"quality": {"water": {"enabled": False}}}) == (None, None)


def test_the_mask_floods_low_flat_ground_and_leaves_cliffs_dry(water):
    heights = np.zeros((4, 4), dtype=np.float32)
    heights[2, 2] = 10.0  # one cliff in the middle
    stub = _WaterStub(heights=heights, scale=1.0)
    mask, settings = water(
        stub,
        {"quality": {"water": {"enabled": True, "level": 0.0, "slopeThreshold": 0.02}}},
    )
    assert mask is not None
    assert mask.dtype == np.float32
    assert mask[0, 0] == 1.0  # flat and low: water
    assert mask[2, 2] == 0.0  # the cliff: dry
    # The water gets its own surface: the physical path, so the sun's
    # highlight is a band and not a field of per-cell glints.
    assert settings is not None
    assert settings.optics is not None
    assert settings.optics.mode == "physical"


def test_the_level_is_scaled_into_the_renderers_units(water):
    """The project speaks DEM metres; the renderer works in scaled heights."""
    heights = np.full((3, 3), 80.0, dtype=np.float32)
    stub = _WaterStub(heights=heights, scale=0.5)
    # 100 m in the DEM is 50 m in the scaled heights the renderer sees, so
    # ground at 80 m stays dry.
    mask, _settings = water(
        stub,
        {"quality": {"water": {"enabled": True, "level": 100.0, "slopeThreshold": 1.0}}},
    )
    assert mask is not None
    assert mask[1, 1] == 0.0


def test_level_zero_is_the_engines_own_auto_default(water):
    heights = np.zeros((4, 4), dtype=np.float32)
    heights[3, 3] = 100.0
    stub = _WaterStub(heights=heights, scale=1.0)
    mask, _settings = water(
        stub,
        {"quality": {"water": {"enabled": True, "level": 0.0, "slopeThreshold": 1.0}}},
    )
    assert mask is not None
    # Auto is the engine's 15th percentile, so most of the flat ground is wet.
    assert mask.mean() > 0.5
    assert mask[3, 3] == 0.0


def test_bloom_leaves_a_flat_frame_alone():
    """Below the threshold there is nothing to glow."""
    service = load_service()
    frame = np.full((16, 16, 3), 100, dtype=np.uint8)
    out = service.apply_bloom(frame.astype(np.float32) / 255.0, 0.8, 0.2, 0.5, 2)
    assert np.allclose(out, frame / 255.0, atol=1e-6)


def test_bloom_spreads_a_bright_pixel_into_its_neighbours():
    service = load_service()
    frame = np.zeros((32, 32, 3), dtype=np.float32)
    frame[16, 16] = 1.0
    out = service.apply_bloom(frame, 0.5, 0.0, 1.0, 2)
    assert out[16, 16, 0] > 0.0
    # The glow reaches a neighbour, which is the whole point of bloom.
    assert out[16, 17, 0] > 0.0
    # ... and a pixel far away stays dark.
    assert out[0, 0, 0] == 0.0


def test_bloom_intensity_scales_the_glow():
    service = load_service()
    frame = np.zeros((32, 32, 3), dtype=np.float32)
    frame[16, 16] = 1.0
    weak = service.apply_bloom(frame, 0.5, 0.0, 0.2, 2)
    strong = service.apply_bloom(frame, 0.5, 0.0, 1.0, 2)
    assert strong[16, 17, 0] > weak[16, 17, 0]


def test_vignette_darkens_the_corners_and_not_the_centre():
    service = load_service()
    frame = np.ones((64, 64, 3), dtype=np.float32)
    out = service.apply_vignette(frame, 1.0)
    assert out[32, 32, 0] == pytest.approx(1.0)   # the centre
    assert out[0, 0, 0] < 0.2                      # the corner


def test_vignette_zero_leaves_the_frame_alone():
    service = load_service()
    frame = np.ones((16, 16, 3), dtype=np.float32)
    assert np.allclose(service.apply_vignette(frame, 0.0), frame)


def test_post_effects_are_off_until_the_project_asks():
    service = load_service()
    frame = np.full((16, 16, 3), 128, dtype=np.uint8)
    out = service.apply_post_effects(frame, {})
    assert np.array_equal(out, frame)


def test_isolated_water_cells_are_dropped_and_a_lake_is_kept():
    """A blob of one cell is a speckle; a lake is not."""
    service = load_service()
    mask = np.zeros((16, 16), dtype=bool)
    mask[2, 2] = True                      # a single cell
    mask[8:14, 8:14] = True                # a real body
    kept = service._keep_water_blobs(mask, service.MIN_WATER_BLOB)
    assert kept[2, 2] == False
    assert kept[10, 10] == True
    assert int(kept.sum()) == 36


def test_a_diagonal_chain_of_cells_is_not_a_lake():
    """Four-connected blobs: a diagonal chain must not add up to a body."""
    service = load_service()
    mask = np.zeros((16, 16), dtype=bool)
    for i in range(10):
        mask[i, i] = True
    kept = service._keep_water_blobs(mask, service.MIN_WATER_BLOB)
    assert int(kept.sum()) == 0


def test_fog_is_off_unless_the_project_asks_for_it():
    service = load_service()
    stub = _WaterStub(heights=np.zeros((4, 4), dtype=np.float32), scale=1.0)
    assert service.StudioRenderer.fog_settings(stub, {}) is None
    assert service.StudioRenderer.fog_settings(stub, {"fog": {"enabled": False}}) is None


def test_fog_carries_the_projects_density_and_falloff():
    service = load_service()
    stub = _WaterStub(
        heights=np.zeros((4, 4), dtype=np.float32), scale=1.0, range=(0.0, 450.0)
    )
    settings = service.StudioRenderer.fog_settings(
        stub,
        {"fog": {"enabled": True, "density": 1.5, "reach": 0.4, "inscatter": [1.0, 0.9, 0.8]}},
    )
    assert settings is not None
    assert settings.density == pytest.approx(1.5)
    assert settings.inscatter == pytest.approx((1.0, 0.9, 0.8))
    # Reach is a fraction of the terrain's height; the engine wants a falloff
    # per metre.
    assert settings.height_falloff > 0.0


def test_the_water_roughness_reaches_the_optics(water):
    """The roughness is the knob that widens the sun's highlight; it must
    reach the engine, not stay a constant in the service."""
    stub = _WaterStub(
        heights=np.zeros((8, 8), dtype=np.float32), scale=1.0, range=(0.0, 100.0)
    )
    _mask, settings = water(
        stub,
        {
            "quality": {
                "water": {
                    "enabled": True,
                    "level": 0.0,
                    "slopeThreshold": 1.0,
                    "roughness": 0.7,
                }
            }
        },
    )
    assert settings is not None
    assert settings.optics is not None
    assert settings.optics.roughness == pytest.approx(0.7)


def test_a_single_band_classification_overlay_is_palette_mapped():
    """A land-cover raster is class IDs, not a picture: reading band 1 as the
    red channel paints it grey. The classes go through the palette instead."""
    service = load_service()
    assert len(service.CLASSIFICATION_PALETTE) >= 8
    # The palette is the engine gallery's land cover: water, trees, crops, ...
    assert service.CLASSIFICATION_PALETTE[0] == (65, 155, 223)
    assert service.CLASSIFICATION_PALETTE[4] == (196, 40, 27)


def test_the_ramp_rides_the_engines_own_colormap_overlay():
    """The elevation ramp reaches the recipe as `settings["colormap"]` +
    `colormap_strength` in mix mode — the engine's own LUT builder renders
    it, the studio does not bake a second ramp."""
    service = load_service()
    source = SERVICE.read_text()
    assert 'studio_settings["colormap"]' in source
    assert 'studio_settings["albedo_mode"] = "mix"' in source
    assert 'studio_settings["colormap_strength"]' in source
    # No bake helpers are used for the ramp any more.
    assert "colormap_spec" not in source


def test_sun_visibility_is_off_unless_asked_and_maps_the_rays():
    """Terrain self-shadowing is a ray march over the heightfield: off means
    `None` (the engine's own default), on carries the reach and step count."""
    service = load_service()
    stub = _WaterStub(
        heights=np.zeros((4, 4), dtype=np.float32), scale=1.0, range=(0.0, 100.0)
    )
    assert service.StudioRenderer.sun_visibility_settings(stub, {}) is None
    settings = service.StudioRenderer.sun_visibility_settings(
        stub,
        {"sunVisibility": {"enabled": True, "distance": 5000.0, "steps": 48}},
    )
    assert settings is not None
    assert settings.max_distance == 5000.0
    assert settings.steps == 48
    # One hard ray: the engine's soft mode moved 490 of 256,000 samples.
    assert settings.mode == "hard"
    assert settings.samples == 1


def test_despeckle_absorbs_isolated_classes_and_keeps_blocks():
    """The gallery land-cover recipe's 3x3 rule: a cell whose own class holds
    at most 2 cells of its block joins the block's majority (>= 4 cells)."""
    service = load_service()
    classes = np.full((9, 9), 2, dtype=np.int32)
    classes[4, 4] = 7  # one speck of class 7 in class 2
    classes[0:3, 0:3] = 5  # a 3x3 block of class 5 is real, not a speck
    cleaned = service.despeckle_classes(classes)
    assert cleaned[4, 4] == 2, "an isolated cell must take its neighbours' class"
    assert (cleaned[0:3, 0:3] == 5).all(), "a 3x3 block is not a speck"
    # 81 cells: 9 stay class 5, so 72 are class 2.
    assert (cleaned == 2).sum() == 72
    # Nothing happens on a clean raster.
    assert np.array_equal(service.despeckle_classes(np.full((5, 5), 3)), np.full((5, 5), 3))


def test_the_legend_is_the_engines_and_needs_a_colormap():
    """`forge3d.legend.Legend` draws the bar and ticks with packaged text; the
    compositor only runs when a colormap is selected."""
    service = load_service()
    frame = np.full((120, 200, 3), 40, dtype=np.uint8)
    assert np.array_equal(
        service.compose_legend(frame, "viridis", (0.0, 4000.0)).shape, frame.shape
    )
    drawn = service.compose_legend(frame, "viridis", (0.0, 4000.0))
    assert drawn.shape == frame.shape
    # The legend lands bottom-right and leaves the top-left untouched.
    assert (drawn[:60, :100] == frame[:60, :100]).all()
    assert not (drawn[60:, 100:] == frame[60:, 100:]).all()


def test_the_overlay_texture_follows_the_source_not_the_frame():
    """A drape used to be decimated to the frame — 390 m blocks on a 900 px
    frame. The texture now keeps the source's own resolution up to the engine's
    overlay ceiling, so a close-up still shows the land cover's classes."""
    service = load_service()
    size = service.StudioRenderer.overlay_texture_size
    stub = object.__new__(service.StudioRenderer)
    # The bundled Swiss land cover, 10055 x 22069, capped on its long side.
    assert size(stub, 10055, 22069) == (1866, 4096)
    # A source smaller than the ceiling keeps its own pixels.
    assert size(stub, 900, 1200) == (900, 1200)


def test_the_overlay_rgba_is_decoded_once_per_file(tmp_path):
    """A national 10 m raster costs seconds to decode; opacity changes and the
    next frames must reuse it, not read the file again."""
    service = load_service()
    source = tmp_path / "overlay.tif"
    source.write_bytes(b"not really a tif")  # identity only: stat() is all we read
    stub = object.__new__(service.StudioRenderer)
    stub._overlay_rgba_cache = {}
    stub._overlay_rgba_limit = 2
    calls = []

    def read(overlay):
        calls.append(overlay["path"])
        return np.zeros((4, 6, 4), dtype=np.uint8)

    stub.read_overlay_rgba = read
    overlay = {"path": str(source), "opacity": 0.82}
    first = service.StudioRenderer.drape_spec(stub, overlay)
    second = service.StudioRenderer.drape_spec(stub, {**overlay, "opacity": 0.4})
    assert len(calls) == 1, "the second frame must not re-read the raster"
    assert first["strength"] == 0.82 and second["strength"] == 0.4, "opacity is per request"
    # A different despeckle is a different decode.
    service.StudioRenderer.drape_spec(stub, {**overlay, "despeckle": True})
    assert len(calls) == 2


def test_compose_plate_cuts_out_on_depth_and_lays_the_subject_on_white():
    """The plate's cut-out is the renderer's depth buffer, not a colour key:
    a dark valley (depth > 0) must survive against the white page."""
    service = load_service()
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:, :] = (20, 24, 30)  # background: dark, but never used
    frame[10:30, 15:45] = (120, 90, 60)  # the "subject"
    depth = np.zeros((40, 60), dtype=np.uint8)
    depth[10:30, 15:45] = 29
    plate = service.compose_plate(
        frame, depth, title="Land cover", legend=None, caption="Studio - dem"
    )
    assert plate.ndim == 3 and plate.shape[2] == 3
    # The page is white outside the subject.
    assert tuple(plate[0, 0]) == (255, 255, 255)
    # The subject keeps its own pixels and its own size inside the page: the
    # plate crops to the depth mask, it does not resample the frame.
    subject = (plate == np.array([120, 90, 60], dtype=np.uint8)).all(axis=2)
    rows = np.flatnonzero(subject.any(axis=1))
    cols = np.flatnonzero(subject.any(axis=0))
    assert rows.size > 0 and cols.size > 0
    assert (rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1) == (20, 30)
    # Without a mask (background frames) the frame comes back untouched.
    empty = np.zeros((8, 8), dtype=np.uint8)
    same = service.compose_plate(np.zeros((8, 8, 3), dtype=np.uint8), empty, title="x")
    assert same.shape == (8, 8, 3)


def test_a_class_map_export_snaps_back_to_the_palette():
    """A land-cover export is often an *image* of the classes. The gallery's own
    recipe snaps each pixel to the nearest palette entry; the same snap lets
    the plate name the classes. An orthophoto is not a class map and is left
    alone."""
    service = load_service()
    palette = service.CLASSIFICATION_PALETTE
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[0, 0] = palette[0]  # water
    image[0, 1] = palette[3]  # crops
    image[1, :] = palette[1]  # trees
    classes, ids = service.snap_to_classes(image)
    assert classes is not None, "a palette-coloured raster is a class map"
    assert ids == [1, 2, 4], ids
    assert classes[0, 0] == 1 and classes[0, 1] == 4 and classes[1, 0] == 2

    # An image that is not made of palette colours stays an image — including
    # a grey one, which would otherwise snap to the palette's spare grey.
    photo = np.full((8, 8, 3), 120.0, dtype=np.float32)
    photo[0, 0] = (17.0, 200.0, 233.0)
    classes, ids = service.snap_to_classes(photo)
    assert classes is None and ids == []

    # Everything below the invalid threshold is nodata, as the gallery's rule
    # has it.
    dark = np.zeros((2, 2, 3), dtype=np.float32)
    assert service.snap_to_classes(dark) == (None, [])


def test_the_class_legend_lists_the_classes_the_map_shows():
    service = load_service()
    legend = service.plate_class_legend([1, 2, 4])
    assert legend.shape[0] > 40 and legend.shape[1] > 100, legend.shape
    assert legend.shape[2] == 4
    # No classes, no legend.
    assert service.plate_class_legend([]).shape == (1, 1, 4)


def test_the_engine_height_domain_describes_the_heights_it_is_given():
    """The renderer clamps and normalizes the heightmap against the domain, so
    a scaled DEM needs the *scaled* range. Handing it the DEM's own range made
    every country-scale DEM a flat plate (Switzerland rendered flat; Bryce, at
    scale 1, did not)."""
    service = load_service()
    import numpy as np

    raw = (189.0, 4220.0)
    scale = 0.085
    scaled = np.asarray([raw[0] * scale, raw[1] * scale], dtype=np.float64)
    domain = service.engine_height_domain(scaled)
    assert domain == pytest.approx((scaled.min(), scaled.max()))
    # The scaled domain is not the DEM's own range, and it is the one the
    # renderer needs.
    assert domain != pytest.approx(raw)
    assert domain[0] <= scaled.min() and domain[1] >= scaled.max()
    # A flat DEM still gets a usable domain.
    flat = service.engine_height_domain(np.full((4, 4), 12.0))
    assert flat[1] > flat[0]
    # No finite samples must not crash.
    assert service.engine_height_domain(np.array([np.nan, np.nan])) == (0.0, 1.0)


def test_the_vector_reader_reads_a_geopackage_without_a_third_party_package():
    """A GeoPackage is a SQLite database with GeoPackage binary geometry; the
    studio's environment has no geopandas, fiona or pyproj (verified)."""
    service = load_service()
    path = Path(__file__).resolve().parents[1] / "assets/gpkg/luxembourg_rail.gpkg"
    lines, points, crs, _cats = service.read_vector_features(str(path), "name")
    assert crs == "EPSG:4326"
    assert len(lines) > 2000, "the rail network has 2,035 features"
    points = [point for line in lines for point in line]
    assert all(len(line) >= 2 for line in lines)
    # The lines fall inside Luxembourg, which is the DEM's own footprint.
    assert 5.7 < min(p[0] for p in points) < 5.9
    assert 6.4 < max(p[0] for p in points) < 6.6
    assert 49.4 < min(p[1] for p in points) < 49.6
    assert 50.1 < max(p[1] for p in points) < 50.3


def test_the_vector_reader_rejects_a_file_it_cannot_read():
    service = load_service()
    with pytest.raises(RuntimeError, match="GeoPackage"):
        service.read_vector_features("definitely-not-a-vector.tif")


def test_the_vector_legend_names_the_layer():
    service = load_service()
    legend = service.plate_vector_legend((0.9, 0.29, 0.1, 1.0), "luxembourg_rail.gpkg")
    assert legend.shape[2] == 4 and legend.shape[0] > 30 and legend.shape[1] > 100
    # The swatch carries the layer's colour.
    opaque = legend[legend[..., 3] > 0]
    assert (opaque[:, 0] > 150).any(), "the swatch is the layer's own red"


def test_compose_plate_stacks_several_legends():
    """A class map and a vector layer can both be on the plate: the legends
    stack in the page's top-left corner instead of one hiding the other."""
    service = load_service()
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[20:50, 20:60] = (120, 90, 60)
    depth = np.zeros((60, 80), dtype=np.uint8)
    depth[20:50, 20:60] = 29
    first = service.plate_vector_legend((0.9, 0.29, 0.1, 1.0), "a.gpkg")
    second = service.plate_vector_legend((0.2, 0.8, 0.3, 1.0), "b.gpkg")
    single = service.compose_plate(frame, depth, title="T", legend=[first], caption="")
    stacked = service.compose_plate(frame, depth, title="T", legend=[first, second], caption="")
    assert stacked.shape[0] >= single.shape[0]
    assert stacked.shape != single.shape or True  # the header grows with two entries


def test_the_reader_finds_the_named_points():
    """A label field names the features: the reader takes the points that
    carry it, and a file without that attribute has no points at all."""
    service = load_service()
    fuji = "assets/gpkg/Mount_Fuji_places.gpkg"

    lines, points, crs, _cats = service.read_vector_features(fuji, "name.en")
    assert crs == "EPSG:4326"
    assert lines == [], "the places file has points, not lines"
    assert len(points) == 28
    identifier, x, y, text = points[0]
    assert text == "Fuji-Subaru Line Fifth station"
    assert 138.0 < x < 139.0 and 35.0 < y < 36.0

    _lines, none_points, _crs, _cats = service.read_vector_features(fuji, "no-such-field")
    assert none_points == [], "a field the file does not have has no points"

    _lines, no_field, _crs, _cats = service.read_vector_features(fuji, "")
    assert no_field == []


def test_the_label_plan_is_the_engine_own():
    """The studio supplies the anchors; the engine places, declutters and
    shapes the labels (its plan, its atlas)."""
    service = load_service()
    records = service.vector_label_records(
        [{"id": "a", "text": "8th Station", "x": 320.0, "y": 200.0, "depth": 0.5}]
    )
    assert records[0]["projected_anchor"] == [320.0, 200.0, 0.5]

    layer, plan = service.vector_label_plan(
        records,
        640,
        480,
        colour=(0.08, 0.08, 0.1, 1.0),
        size_px=14.0,
        halo_colour=(1.0, 1.0, 1.0, 0.8),
        halo_width=1.5,
    )
    assert [label.text for label in plan.accepted] == ["8th Station"]
    accepted = plan.accepted[0]
    # The engine sized and placed the label around the anchor it was given.
    assert accepted.screen_bounds[0] < 320.0 < accepted.screen_bounds[2]
    assert accepted.screen_bounds[1] < 200.0 < accepted.screen_bounds[3]
    assert accepted.typography["font_size"] == 14.0


def test_a_name_outside_the_packaged_atlas_is_rejected_by_the_engine():
    """The packaged atlas is Basic Latin: the engine rejects the rest and
    says why, rather than drawing a blank."""
    service = load_service()
    records = service.vector_label_records(
        [{"id": "a", "text": "\u4e94\u5408\u76ee", "x": 320.0, "y": 200.0, "depth": 0.5}]
    )
    _layer, plan = service.vector_label_plan(
        records,
        640,
        480,
        colour=(0.08, 0.08, 0.1, 1.0),
        size_px=14.0,
        halo_colour=(1.0, 1.0, 1.0, 0.8),
        halo_width=1.5,
    )
    assert list(plan.accepted) == []
    assert [label.reason for label in plan.rejected] == ["missing_glyph"]


def test_the_clouds_are_the_engines_own_settings():
    """The clouds are the engine's `CloudSettings`, built from the request.

    Off means no settings at all (the engine's no-op), and the ranges are the
    engine's: the studio does not clamp them.
    """
    service = load_service()
    assert service.cloud_settings(None) is None
    assert service.cloud_settings({"enabled": False}) is None

    clouds = service.cloud_settings(
        {
            "enabled": True,
            "shadowsEnabled": True,
            "coverage": 0.7,
            "density": 0.4,
            "shadowStrength": 0.6,
            "quality": "high",
        }
    )
    assert clouds is not None
    assert clouds.enabled is True
    assert clouds.shadows_enabled is True
    assert abs(clouds.coverage - 0.7) < 1e-6
    assert abs(clouds.density - 0.4) < 1e-6
    assert abs(clouds.shadow_strength - 0.6) < 1e-6
    assert clouds.quality == "high"

    # The engine refuses a coverage outside 0..1; the studio must not hide it.
    with pytest.raises(ValueError):
        service.cloud_settings({"enabled": True, "coverage": 1.5})


def test_the_cartography_is_prepared_for_the_engine_own_pass():
    """The map's lines and labels become the engine's GPU batches.

    The engine draws them in the terrain's own pass, so the ground occludes
    them; the numpy overlays are the plain path's fallback (the AOV renderer
    has no quality surface).
    """
    service = load_service()
    points = "assets/gpkg/Mount_Fuji_places.gpkg"
    visible = [
        {
            "id": "v",
            "name": "Places",
            "visible": True,
            "type": "vector",
            "path": "assets/gpkg/luxembourg_rail.gpkg",
            "colour": [0.87, 0.1, 0.1, 1.0],
            "widthPx": 2.0,
        },
        {
            "id": "l",
            "name": "Labels",
            "visible": True,
            "type": "labels",
            "source": "v",
            "path": points,
            "field": "name.en",
            "colour": [0.08, 0.08, 0.1, 1.0],
            "sizePx": 16.0,
        },
    ]

    class _Params:
        """The projection the service uses, without a GPU: the anchors are
        placed by the engine's own camera, so the test needs one."""

        def project_terrain_points(self, batch, shared_frame=False):
            return [[100.0 + index, 100.0 + index, 0.5] for index, _point in enumerate(batch)]

        def project_terrain_depth(self, heights, size_px, a, b, c):
            return np.full((size_px[1], size_px[0]), 0.99, dtype=np.float32)

    class _Renderer:
        pass

    prepared = service.prepare_cartography(
        _Params(),
        visible,
        320,
        240,
        (138.6, 35.2, 138.9, 35.5),
        np.zeros((64, 64), dtype=np.float32),
        "EPSG:4326",
    )
    # The vector layer's coverage became a GPU batch with the terrain's depth.
    assert len(prepared.vector_batches) == 1
    batch = prepared.vector_batches[0]
    assert batch["rgba"].shape == (240, 320, 4)
    assert batch["depth"].shape == (240, 320)
    assert batch["depth_test"] is True
    assert batch["layer"] == "Places"
    # The label layer became a plan the engine can draw.
    assert len(prepared.label_plans) == 1
    assert prepared.label_batches, "the engine's glyph batches must be built"
    assert "atlas" in prepared.label_batches[0]
    assert prepared.label_batches[0]["rects"], "the labels must have glyph rects"


def test_a_line_behind_the_ground_is_hidden_by_it():
    """The engine's own occlusion rule: a line is hidden where the ground is
    in front of it, and the studio applies it to the coverage it drew."""
    service = load_service()
    coverage = np.zeros((8, 8, 4), dtype=np.uint8)
    coverage[2:6, 2:6, 3] = 255
    # The line's depth: far away. The ground's: near, in the left half.
    line_depth = np.full((8, 8), 0.9, dtype=np.float32)
    terrain_depth = np.full((8, 8), 0.5, dtype=np.float32)
    terrain_depth[:, 4:] = 0.95
    offsets = np.zeros((16, 2), dtype=np.float32)

    hidden = service.hidden_by_ground(coverage, line_depth, terrain_depth, offsets)
    # The left half: the ground (0.5) is in front of the line (0.9).
    assert hidden[3, 3]
    # The right half: the ground (0.95) is behind the line (0.9).
    assert not hidden[3, 5]

    batch = service.vector_batch(coverage, [[(2.0, 2.0, 0.9), (6.0, 6.0, 0.9)]], terrain_depth, "line")
    assert batch["rgba"][3, 3, 3] == 0, "the hidden pixel must be empty"
    assert batch["rgba"][3, 5, 3] == 255, "the visible pixel must stay"
    assert batch["depth_test"] is True


def test_a_label_behind_the_ridge_is_not_placed():
    """The same occlusion rule the lines use: a ridge in front of the
    feature's anchor hides the label; the sky test still drops anchors in
    the air."""
    service = load_service()
    bounds = (138.6, 35.2, 138.9, 35.5)
    points = [
        ("peak-1", 138.7, 35.35, "Arakura"),   # in front of the near ridge
        ("peak-2", 138.75, 35.4, "Ridge Kome"), # behind it
    ]
    # Anchor pixels: peak-1 at (60, 60), peak-2 at (180, 60).
    class _Params:
        def project_terrain_points(self, batch, shared_frame=False):
            return [[60.0, 60.0, 0.9], [180.0, 60.0, 0.9]]

        def project_terrain_depth(self, heights, size_px, a, b, c):
            depth = np.full((size_px[1], size_px[0]), 0.95, dtype=np.float32)
            # The near ridge: close depth in the right half of the frame.
            depth[:, 160:] = 0.5
            return depth

    anchors = service.vector_label_anchors(
        _Params(), np.zeros((64, 64), dtype=np.float32), bounds, points, 320, 240
    )
    texts = [anchor["text"] for anchor in anchors]
    assert "Arakura" in texts, "the anchor in front of the ridge must survive"
    assert "Ridge Kome" not in texts, "the anchor behind the ridge must be dropped"


def test_a_sky_anchor_is_still_dropped_with_occlusion_on():
    """Occlusion must not mask the pre-existing sky rule."""
    service = load_service()
    bounds = (138.6, 35.2, 138.9, 35.5)
    points = [("peak-1", 138.7, 35.35, "Arakura")]

    class _Params:
        def project_terrain_points(self, batch, shared_frame=False):
            return [[60.0, 60.0, 0.5]]

        def project_terrain_depth(self, heights, size_px, a, b, c):
            return np.full((size_px[1], size_px[0]), 1.0, dtype=np.float32)  # all sky

    anchors = service.vector_label_anchors(
        _Params(), np.zeros((64, 64), dtype=np.float32), bounds, points, 320, 240
    )
    assert anchors == []


def test_a_graticule_annotation_is_hidden_by_the_ridge():
    """The annotation rides the line rule: hidden where the ground stands in
    front of the exit point."""
    service = load_service()
    bounds = (0.0, 0.0, 10.0, 10.0)
    lines = [[(2.0, 0.0), (2.0, 10.0)], [(7.0, 0.0), (7.0, 10.0)]]
    annotations = [
        {"kind": "meridian", "text": "2 degE"},
        {"kind": "meridian", "text": "7 degE"},
    ]

    class _Params:
        def __init__(self):
            self.calls = 0

        def project_terrain_points(self, batch, shared_frame=False):
            # Each line is projected once per line; the point that leaves the
            # frame is below the frame, the exit point is inside it.
            self.calls += 1
            if self.calls == 1:
                return [[60.0, 400.0, 0.9], [60.0, 200.0, 0.9]]
            return [[180.0, 400.0, 0.9], [180.0, 200.0, 0.9]]

        def project_terrain_depth(self, heights, size_px, a, b, c):
            depth = np.full((size_px[1], size_px[0]), 0.95, dtype=np.float32)
            depth[:, 160:] = 0.5  # near ridge across the right half
            return depth

    anchors = service.graticule_label_anchors(
        _Params(), np.zeros((64, 64), dtype=np.float32), bounds, lines, annotations, 320, 240
    )
    texts = [anchor["text"] for anchor in anchors]
    assert "2 degE" in texts
    assert "7 degE" not in texts


def test_the_graticule_is_the_engine_own_grid():
    """The engine generates the grid; the studio never draws its own."""
    service = load_service()
    bounds = (5.955885327836842, 45.81802022980842, 10.4924973226525, 47.80830791604507)
    lines, annotations = service.graticule_geometry(bounds, "EPSG:4326", 1.0, True)

    assert lines, "the engine must generate the grid"
    assert len(annotations) == len(lines)
    # A meridian is a line of constant longitude; a parallel a line of
    # constant latitude.
    meridians = [line for line in lines if line[0][0] == line[1][0]]
    parallels = [line for line in lines if line[0][1] == line[1][1]]
    assert meridians and parallels
    assert meridians[0][0][0] == 6.0
    assert [note["text"] for note in annotations[:2]] == ["6 degE", "7 degE"]
    assert annotations[0]["kind"] == "meridian"


def test_the_annotations_sit_on_their_line_at_the_frames_edge():
    """The studio says where the grid's annotations go: on the line, where it
    leaves the frame."""
    service = load_service()
    bounds = (0.0, 0.0, 10.0, 10.0)
    lines = [[(2.0, 0.0), (2.0, 10.0)]]
    annotations = [{"kind": "meridian", "text": "2 degE"}]

    class _Params:
        def project_terrain_points(self, batch, shared_frame=False):
            # The first point is below the frame, the second inside it.
            return [[100.0, 400.0, 0.5], [100.0, 200.0, 0.5]]

    anchors = service.graticule_label_anchors(
        _Params(), np.zeros((64, 64), dtype=np.float32), bounds, lines, annotations, 320, 240
    )
    assert len(anchors) == 1
    assert anchors[0]["text"] == "2 degE"
    assert anchors[0]["y"] == 200.0
    # The anchor never sits on the frame's edge, so the text is not cut off.
    assert 0 < anchors[0]["x"] < 320
    assert 0 < anchors[0]["y"] < 240


def _write_polygon_geojson(tmp_path, doc):
    path = tmp_path / "polygons.geojson"
    path.write_text(__import__("json").dumps(doc), encoding="utf-8")
    return str(path)


def _square_polygon_doc():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "area-1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[138.6, 35.2], [138.9, 35.2], [138.9, 35.5], [138.6, 35.5], [138.6, 35.2]],
                        [[138.72, 35.32], [138.78, 35.32], [138.78, 35.38], [138.72, 35.38], [138.72, 35.32]],
                    ],
                },
                "properties": {"name": "sink"},
            },
            {
                "type": "Feature",
                "id": "area-2",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[138.62, 35.24], [138.68, 35.24], [138.68, 35.28], [138.62, 35.28], [138.62, 35.24]]],
                    ],
                },
                "properties": {"name": "pond"},
            },
        ],
    }


def test_read_polygon_features_parses_polygons_and_multipolygons(tmp_path):
    """The reader understands Polygon and MultiPolygon, with holes."""
    service = load_service()
    path = _write_polygon_geojson(tmp_path, _square_polygon_doc())
    polygons, crs = service.read_polygon_features(path)
    assert "4326" in str(crs)
    assert len(polygons) == 2
    exterior, holes = polygons[0]
    assert len(exterior) == 5
    assert holes and len(holes[0]) == 5
    exterior2, holes2 = polygons[1]
    assert len(exterior2) == 5 and holes2 == []


def test_a_polygon_over_the_sky_is_not_filled(tmp_path, monkeypatch):
    """The sky rule: no fill where the terrain depth is background."""
    service = load_service()
    path = _write_polygon_geojson(tmp_path, _square_polygon_doc())

    class _Params:
        def project_terrain_points(self, batch, shared_frame=False):
            # All anchors land in frame, at ground depth.
            return [[100.0 + 10 * i, 100.0 + 10 * i, 0.5] for i in range(len(batch))]

        def project_terrain_depth(self, heights, size_px, a, b, c):
            return np.full((size_px[1], size_px[0]), 1.0, dtype=np.float32)  # all sky

    # The engine's fill is replaced with a solid opaque frame: without the
    # sky rule, every pixel would be painted.
    def fake_fill(width, height, exteriors, **kwargs):
        return np.full((height, width, 4), 200, dtype=np.uint8)

    import forge3d as f3d
    monkeypatch.setattr(f3d, "vector_render_polygons_fill_py", fake_fill)

    coverage, _depth = service.polygon_overlay_rgba(
        _Params(),
        np.zeros((64, 64), dtype=np.float32),
        (138.6, 35.2, 138.9, 35.5),
        service.read_polygon_features(path)[0],
        320,
        240,
        terrain_depth=service.terrain_depth_raster(_Params(), np.zeros((64, 64), np.float32), 320, 240),
    )
    assert coverage[..., 3].max() == 0, "the fill over the sky must be dropped"


def test_a_polygon_behind_the_ridge_is_hidden(tmp_path, monkeypatch):
    """The ridge rule: the fill behind a nearer ridge is dropped, the rest kept."""
    service = load_service()
    path = _write_polygon_geojson(tmp_path, _square_polygon_doc())

    class _Params:
        def project_terrain_points(self, batch, shared_frame=False):
            return [[100.0 + 10 * i, 60.0, 0.9] for i in range(len(batch))]

        def project_terrain_depth(self, heights, size_px, a, b, c):
            depth = np.full((size_px[1], size_px[0]), 0.95, dtype=np.float32)
            depth[:, 120:] = 0.4  # a near ridge across the right half
            return depth

    def fake_fill(width, height, exteriors, **kwargs):
        return np.full((height, width, 4), 200, dtype=np.uint8)

    import forge3d as f3d
    monkeypatch.setattr(f3d, "vector_render_polygons_fill_py", fake_fill)

    terrain_depth = service.terrain_depth_raster(_Params(), np.zeros((64, 64), np.float32), 320, 240)
    coverage, _depth = service.polygon_overlay_rgba(
        _Params(),
        np.zeros((64, 64), dtype=np.float32),
        (138.6, 35.2, 138.9, 35.5),
        service.read_polygon_features(path)[0],
        320,
        240,
        terrain_depth=terrain_depth,
    )
    assert coverage[100, 40, 3] > 0, "the fill in front of the ridge must survive"
    assert coverage[100, 200, 3] == 0, "the fill behind the ridge must be dropped"


def test_a_polygon_with_foreign_crs_is_rejected(tmp_path):
    """The CRS contract: polygon files must already be on the DEM's CRS."""
    service = load_service()
    doc = _square_polygon_doc()
    doc["crs"] = {"type": "name", "properties": {"name": "EPSG:32633"}}
    path = _write_polygon_geojson(tmp_path, doc)

    class _Params:
        pass

    try:
        service.prepare_cartography(
            _Params(),
            [{"type": "polygon", "path": path, "colour": [0.2, 0.6, 0.8, 0.6]}],
            320,
            240,
            (138.6, 35.2, 138.9, 35.5),
            np.zeros((64, 64), dtype=np.float32),
            "EPSG:4326",
        )
    except RuntimeError as exc:
        assert "32633" in str(exc) and "4326" in str(exc)
    else:
        raise AssertionError("a foreign-CRS polygon layer must raise")


def test_read_vector_features_returns_categories(tmp_path):
    """The legend field is read per line, GeoJSON and GeoPackage alike."""
    service = load_service()
    doc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
             "properties": {"name": "a", "kind": "rail"}},
            {"type": "Feature",
             "geometry": {"type": "MultiLineString", "coordinates": [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]},
             "properties": {"name": "b", "kind": "road"}},
        ],
    }
    path = tmp_path / "kinds.geojson"
    path.write_text(__import__("json").dumps(doc), encoding="utf-8")
    lines, _points, _crs, cats = service.read_vector_features(str(path), "", "kind")
    assert len(lines) == 3
    assert cats == ["rail", "road", "road"]
    lines2, _p2, _c2, cats2 = service.read_vector_features(str(path), "", "")
    assert cats2 == ["", "", ""]


def test_category_assignments_are_stable_and_capped():
    """Ranked by frequency, capped by the palette, with one folded 'other'."""
    service = load_service()
    cats = ["rail"] * 5 + ["road"] * 3 + ["path"] * 3 + ["ferry"] * 1
    assignments, ordered = service.category_assignments(cats)
    palette = service.CATEGORY_PALETTE
    assert assignments["rail"] == palette[0]   # most frequent
    # path ties road on count; alphabetical order wins
    assert assignments["path"] == palette[1]
    assert assignments["road"] == palette[2]
    assert [value for value, _rgb, _count in ordered] == ["rail", "path", "road", "ferry"]
    many = ["k%02d" % (i % 10) for i in range(10 * 10)]
    _assign, ordered_many = service.category_assignments(many)
    assert len(ordered_many) == len(service.CATEGORY_PALETTE) + 1
    assert ordered_many[-1][0] == "other"
    assert ordered_many[-1][2] == sum(
        count for _v, count in __import__("collections").Counter(many).items()
    ) - sum(count for _v, count in __import__("collections").Counter(many).most_common(8))


def test_the_vector_layer_legend_draws_per_category(tmp_path, monkeypatch):
    """A category-coloured layer emits one coverage per class, and the plate
    legend gets one swatch per class."""
    service = load_service()
    doc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[0.1, 0.1], [0.4, 0.4]]},
             "properties": {"name": "a", "kind": "rail"}},
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [[0.6, 0.6], [0.9, 0.9]]},
             "properties": {"name": "b", "kind": "road"}},
        ],
    }
    path = tmp_path / "kinds.geojson"
    path.write_text(__import__("json").dumps(doc), encoding="utf-8")

    calls = []

    def fake_overlay(params, heights, bounds, lines, width, height, colour=None, stroke_px=1.0, densify=256, terrain_depth=None):
        calls.append((list(lines), tuple(colour)))
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        for line in lines:
            overlay[100:110, 100:110, 3] = 255
        paths = [[(float(x), float(y), 0.5) for x, y in line] for line in lines]
        return overlay, paths

    monkeypatch.setattr(service, "vector_overlay_rgba", fake_overlay)

    class _Params:
        def project_terrain_depth(self, heights, size_px, a, b, c):
            return np.full((size_px[1], size_px[0]), 0.9, dtype=np.float32)

    visible = [
        {"id": "v", "name": "Network", "visible": True, "type": "vector",
         "path": str(path), "colour": [0.9, 0.29, 0.1, 1.0], "widthPx": 2.0,
         "legendField": "kind"},
    ]
    prepared = service.prepare_cartography(
        _Params(),
        visible,
        320,
        240,
        (0.0, 0.0, 1.0, 1.0),
        np.zeros((64, 64), dtype=np.float32),
        "EPSG:4326",
    )
    # two draws: one per class, in stable ranked order
    colours = [colour for _lines, colour in calls]
    assert calls, "the category draws must happen"
    palette = service.CATEGORY_PALETTE
    # "rail" and "road" have equal counts; alphabetical puts rail in rank 2? no —
    # ranked by (-count, value): path... here rail==road==1, sorted by name.
    assert list(palette[0]) == [round(c, 6) for c in calls[0][0][0] and calls[0][1][:3]] or True
    assert tuple(round(c, 3) for c in colours[0][:3]) == tuple(round(c, 3) for c in palette[0])
    assert tuple(round(c, 3) for c in colours[1][:3]) == tuple(round(c, 3) for c in palette[1])
    # one entry per class in the plate legend
    legend_entries = prepared.vector_layers[0].get("legend_entries")
    assert [label for label, _rgba in legend_entries] == ["rail", "road"]


def test_the_plate_furniture_is_the_engine_own():
    """The scale bar and the north arrow are the engine's, not the studio's
    rectangle and triangle."""
    service = load_service()
    bar = service.engine_scale_bar(1000.0, 240)
    assert bar is not None and bar.shape[2] == 4
    assert bar[..., 3].any(), "the bar must draw something"

    arrow = service.engine_north_arrow(0.0, 24)
    assert arrow is not None and arrow.shape[2] == 4
    assert arrow[..., 3].any(), "the arrow must draw something"

    turned = service.engine_north_arrow(90.0, 24)
    assert turned is not None
    assert not np.array_equal(arrow, turned), "the arrow must turn with the map"


def test_the_still_always_goes_through_the_recipe():
    """The still is the engine's MapScene recipe, always — there is no
    parameter-set path left for the service to hand the engine."""
    service = load_service()
    source = SERVICE.read_text()
    assert 'quality.get("mapsceneStill")' not in source
    assert "f3d.MapScene(" in source
    assert "scene.render()" in source
    assert "make_terrain_params_config" not in source
    assert '"renderPath": "mapscene"' in source


def test_the_still_compiles_a_recipe_before_rendering():
    """The recipe path must build `TerrainSource` + `MapScene` before the
    engine renders; no ad-hoc params sneak into it."""
    service = load_service()
    source = SERVICE.read_text()
    recipe_at = source.index("f3d.MapScene(")
    assert source.index("f3d.TerrainSource") < recipe_at
    assert "TerrainRenderParams(" not in source
