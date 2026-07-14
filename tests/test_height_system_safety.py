# tests/test_height_system_safety.py
# MENSURA win 6 (part): the height-system boundary is explicit and typed.
# DEM ingestion carries a height-system tag; the orthometric/ellipsoidal
# bridge goes through the geoid; wgs84_to_ecef only accepts a typed
# ellipsoidal height at the Rust level (source-asserted).
# RELEVANT FILES: src/gis/terrarium.rs, src/gis/domain.rs, src/geo/geoid.rs,
#                 src/tiles3d/bounds.rs

import struct
from pathlib import Path

import numpy as np
import pytest

import forge3d
from forge3d import gis


def _height_system(info):
    """Read height_system from a RasterInfo object or its dict form."""
    return info["height_system"] if isinstance(info, dict) else info.height_system

REPO_SRC = Path(__file__).parent.parent / "src"


def test_terrarium_decode_carries_orthometric_height_system_tag():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 128  # 128*256 - 32768 = 0 m
    result = gis.decode_terrarium_dem(rgb)
    assert result["height_system"] == "orthometric_egm96"


def test_prepare_dem_carries_height_system_tag():
    dem = np.linspace(0.0, 100.0, 64, dtype=np.float32).reshape(8, 8)
    result = gis.prepare_dem(dem)
    assert "height_system" in result
    assert result["height_system"] == "unspecified"


def test_prepare_dem_converts_declared_egm96_pixels_to_ellipsoidal():
    raw = np.full((1, 2, 2), 100.0, dtype=np.float32)
    result = gis.prepare_dem(
        {
            "array": raw,
            "height_system": "orthometric_egm96",
            "info": {
                "width": 2,
                "height": 2,
                "band_count": 1,
                "crs_authority": {"name": "EPSG", "code": "4326"},
                "bounds": (10.0, 50.0, 12.0, 52.0),
            },
        }
    )
    assert result["height_system"] == "ellipsoidal"
    value = float(np.asarray(result["array"])[0, 0, 0])
    expected = 100.0 + forge3d.geoid_undulation(51.5, 10.5)
    assert abs(value - expected) < 1e-6


def test_orthometric_and_ellipsoidal_differ_by_exactly_n():
    lat, lon = -14.6212170, 305.0211140
    n = forge3d.geoid_undulation(lat, lon)
    for h in (-431.0, 0.0, 8848.86):
        ell = forge3d.orthometric_to_ellipsoidal(h, lat, lon)
        assert abs((ell - h) - n) < 1e-12
        assert abs(forge3d.ellipsoidal_to_orthometric(ell, lat, lon) - h) < 1e-12


def test_wgs84_to_ecef_signature_is_typed_ellipsoidal_only():
    # Source-level lock: the Rust conversion consumed by 3D Tiles bounding
    # volumes must accept ONLY a typed ellipsoidal height — no bare f64/f32
    # height parameter (that was the pre-MENSURA silent orthometric mix-in).
    bounds_rs = (REPO_SRC / "tiles3d" / "bounds.rs").read_text(encoding="utf-8")
    assert "height: Height<Ellipsoidal>" in bounds_rs, (
        "tiles3d wgs84_to_ecef must take Height<Ellipsoidal>"
    )
    assert "height: f64" not in bounds_rs and "height: f32" not in bounds_rs, (
        "tiles3d wgs84_to_ecef must not accept an untyped height"
    )
    # And the f64 conversion is the geo engine's, not a local reimplementation.
    assert "wgs84_geodetic_to_ecef" in bounds_rs


def test_compile_fail_doctests_are_present_in_units_rs():
    # cargo test --doc proves these actually fail to compile; this test locks
    # that at least the five required proofs stay present in the source.
    units_rs = (REPO_SRC / "geo" / "units.rs").read_text(encoding="utf-8")
    count = units_rs.count("```compile_fail")
    assert count >= 5, f"expected >= 5 compile_fail doctests, found {count}"
    for marker in (
        "let _ = l + a;",           # metre + degree
        "let _ = e - o;",           # ellipsoidal - orthometric
        "let _ = a - b;",           # epoch mismatch and CRS mismatch
        "as f32",                    # raw world f64 -> f32 cast
    ):
        assert marker in units_rs, f"missing compile_fail proof body: {marker}"


def test_dem_conversion_is_per_pixel_exact():
    dem = np.full((3, 3), 100.0)
    bounds = (102.0, 46.5, 103.0, 47.25)
    out = forge3d.dem_orthometric_to_ellipsoidal(dem, bounds)
    # Different pixels get different N; every pixel differs from the raw DEM
    # by its own local undulation.
    lat0 = bounds[3] - 0.5 * (bounds[3] - bounds[1]) / 3
    lon0 = bounds[0] + 0.5 * (bounds[2] - bounds[0]) / 3
    n00 = forge3d.geoid_undulation(lat0, lon0)
    assert abs(out[0, 0] - (100.0 + n00)) < 1e-9


def test_raster_info_exposes_height_system_field_default_unspecified(tmp_path):
    # M-03: RasterInfo carries a first-class height_system field (not just a
    # sidecar dict key), defaulting to the honest "unspecified" on a plain read.
    path = tmp_path / "plain.tif"
    data = np.ones((1, 4, 4), dtype=np.float32)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=(0.1, 0, 10, 0, -0.1, 52))
    info = gis.read_raster_info(str(path))
    assert _height_system(info) == "unspecified"


def test_height_system_survives_reprojection(tmp_path):
    # M-03: horizontal reprojection preserves the vertical-datum tag through
    # the shared operation_info builder.
    path = tmp_path / "dem.tif"
    data = np.ones((1, 8, 8), dtype=np.float32)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=(0.01, 0, 13, 0, -0.01, 52))
    result = gis.reproject_raster(str(path), "EPSG:3857", resampling="nearest")
    assert result["info"]["height_system"] == "unspecified"


def test_prepare_dem_returned_info_carries_converted_height_system():
    # M-03: the converted vertical datum lands on the RasterInfo field, not
    # only the sidecar key, so it survives being fed back into other ops.
    raw = np.full((1, 2, 2), 100.0, dtype=np.float32)
    result = gis.prepare_dem(
        {
            "array": raw,
            "height_system": "orthometric_egm96",
            "info": {
                "width": 2,
                "height": 2,
                "band_count": 1,
                "crs_authority": {"name": "EPSG", "code": "4326"},
                "bounds": (10.0, 50.0, 12.0, 52.0),
            },
        }
    )
    assert result["height_system"] == "ellipsoidal"
    assert result["info"]["height_system"] == "ellipsoidal"


def test_invalid_height_system_declaration_is_rejected():
    # M-03 policy: an unrecognized vertical-datum tag is rejected, never coerced.
    raw = np.full((1, 2, 2), 100.0, dtype=np.float32)
    with pytest.raises(Exception):
        gis.prepare_dem(
            {
                "array": raw,
                "height_system": "wgs84_banana",
                "info": {
                    "width": 2,
                    "height": 2,
                    "band_count": 1,
                    "crs_authority": {"name": "EPSG", "code": "4326"},
                    "bounds": (10.0, 50.0, 12.0, 52.0),
                },
            }
        )


def test_raster_info_as_dict_includes_height_system(tmp_path):
    # M-03: RasterInfo.as_dict() must match the op-result dict form and carry
    # height_system (it previously omitted it, an asymmetry with
    # raster_info_to_py_dict).
    path = tmp_path / "plain.tif"
    data = np.ones((1, 4, 4), dtype=np.float32)
    gis.write_raster(str(path), data, crs="EPSG:4326", transform=(0.1, 0, 10, 0, -0.1, 52))
    info = gis.read_raster_info(str(path))
    d = info.as_dict()
    assert d["height_system"] == "unspecified"


def test_height_system_persists_across_geotiff_write_read(tmp_path):
    # M-03: an explicitly declared vertical datum survives a GeoTIFF write->read
    # round trip via the private forge3d ASCII tag (65001).
    for declared in ("orthometric_egm96", "ellipsoidal", "chart_datum"):
        path = tmp_path / f"dem_{declared}.tif"
        data = np.ones((1, 4, 4), dtype=np.float32)
        gis.write_raster(
            str(path),
            data,
            crs="EPSG:4326",
            transform=(0.1, 0, 10, 0, -0.1, 52),
            height_system=declared,
        )
        info = gis.read_raster_info(str(path))
        assert info.height_system == declared, declared
    # An undeclared write stays "unspecified" (no tag written).
    plain = tmp_path / "plain.tif"
    gis.write_raster(str(plain), np.ones((1, 4, 4), dtype=np.float32),
                     crs="EPSG:4326", transform=(0.1, 0, 10, 0, -0.1, 52))
    assert gis.read_raster_info(str(plain)).height_system == "unspecified"


def test_prepare_dem_accepts_chart_datum_without_promoting_it():
    # M-03: chart_datum is a valid declaration; carried, not silently promoted
    # to ellipsoidal (no tidal model shipped), and not rejected.
    raw = np.full((1, 2, 2), 5.0, dtype=np.float32)
    result = gis.prepare_dem(
        {
            "array": raw,
            "height_system": "chart_datum",
            "info": {
                "width": 2,
                "height": 2,
                "band_count": 1,
                "crs_authority": {"name": "EPSG", "code": "4326"},
                "bounds": (10.0, 50.0, 12.0, 52.0),
            },
        }
    )
    assert result["height_system"] == "chart_datum"
    assert float(np.asarray(result["array"])[0, 0, 0]) == 5.0  # unchanged


def test_height_system_survives_full_and_windowed_read(tmp_path):
    # M-03: the vertical datum tag is read back on both full and windowed
    # read_raster (the windowed path clones the base RasterInfo).
    path = tmp_path / "dem.tif"
    data = np.ones((1, 8, 8), dtype=np.float32)
    gis.write_raster(str(path), data, crs="EPSG:4326",
                     transform=(0.1, 0, 10, 0, -0.1, 52), height_system="orthometric_egm96")
    full = gis.read_raster(str(path))
    assert full["info"]["height_system"] == "orthometric_egm96"
    windowed = gis.read_raster(str(path), window=(1, 0, 4, 4))
    assert windowed["info"]["height_system"] == "orthometric_egm96"


def _write_dem(path, *, height_system=None, crs="EPSG:4326",
               transform=(0.05, 0, 10, 0, -0.05, 52), shape=(1, 16, 16)):
    kwargs = {} if height_system is None else {"height_system": height_system}
    gis.write_raster(str(path), np.ones(shape, dtype=np.float32),
                     crs=crs, transform=transform, **kwargs)
    return str(path)


def _write_tiled_geotiff(path, *, height_system="orthometric_egm96"):
    """Hand-build a minimal TILED float32 GeoTIFF (EPSG:4326, 32x32, 16x16
    tiles) carrying the forge3d height-system ASCII tag 65001.

    forge3d's own write_raster deliberately rejects tiled output in G-002a1,
    so a genuinely tiled fixture must be assembled at the byte level. Each of
    the four tiles is filled with its tile index, which lets the reader prove
    real tile reassembly rather than a striped fallback.
    """
    size, tile = 32, 16
    tiles_across = size // tile
    tile_bytes = tile * tile * 4
    tile_data = b"".join(
        struct.pack(f"<{tile * tile}f", *([float(i)] * tile * tile))
        for i in range(tiles_across * tiles_across)
    )
    tile_offsets = [8 + i * tile_bytes for i in range(tiles_across * tiles_across)]

    ext = b""  # external tag values, placed right after the tile data
    ext_base = 8 + len(tile_data)

    def external(fmt, values):
        nonlocal ext
        offset = ext_base + len(ext)
        ext += struct.pack(f"<{len(values)}{fmt}", *values)
        return offset

    offsets_off = external("I", tile_offsets)
    counts_off = external("I", [tile_bytes] * len(tile_offsets))
    scale_off = external("d", [0.05, 0.05, 0.0])
    tiepoint_off = external("d", [0.0, 0.0, 0.0, 10.0, 52.0, 0.0])
    # GeoKeyDirectory: header + GTModelType=geographic, GTRasterType=area,
    # GeographicType=EPSG:4326.
    geokeys_off = external(
        "H",
        [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326],
    )
    hs_bytes = height_system.encode("ascii") + b"\0"
    hs_off = ext_base + len(ext)
    ext += hs_bytes

    ifd_offset = ext_base + len(ext)
    entries = [
        (256, 3, 1, size),          # ImageWidth
        (257, 3, 1, size),          # ImageLength
        (258, 3, 1, 32),            # BitsPerSample
        (259, 3, 1, 1),             # Compression = NONE
        (262, 3, 1, 1),             # Photometric = BlackIsZero
        (277, 3, 1, 1),             # SamplesPerPixel
        (322, 3, 1, tile),          # TileWidth
        (323, 3, 1, tile),          # TileLength
        (324, 4, len(tile_offsets), offsets_off),   # TileOffsets
        (325, 4, len(tile_offsets), counts_off),    # TileByteCounts
        (339, 3, 1, 3),             # SampleFormat = IEEE float
        (33550, 12, 3, scale_off),  # ModelPixelScale
        (33922, 12, 6, tiepoint_off),  # ModelTiepoint
        (34735, 3, 16, geokeys_off),   # GeoKeyDirectory
        (65001, 2, len(hs_bytes), hs_off),  # forge3d height system
    ]
    ifd = struct.pack("<H", len(entries))
    for tag, dtype, count, value in entries:
        ifd += struct.pack("<HHII", tag, dtype, count, value)
    ifd += struct.pack("<I", 0)  # no next IFD

    header = struct.pack("<2sHI", b"II", 42, ifd_offset)
    path.write_bytes(header + tile_data + ext + ifd)
    return str(path)


def test_height_system_survives_cog_tiled_read(tmp_path):
    # M-03: the tiled / COG-compatible read path (read_cog, full and windowed)
    # preserves the vertical-datum tag on a GENUINELY TILED GeoTIFF —
    # write_raster cannot produce one, so the fixture is byte-assembled.
    path = _write_tiled_geotiff(tmp_path / "cog.tif")
    full = gis.read_cog(path)
    assert full["is_cog_like"] is True
    assert full["tile_info"]["tiling"] == "tiled"
    assert full["info"]["height_system"] == "orthometric_egm96"
    # The four 16x16 tiles hold their tile index; correct reassembly proves the
    # tag rode along a real tiled decode, not a striped fallback.
    arr = np.asarray(full["array"]).reshape(32, 32)
    assert (arr[0, 0], arr[0, 31], arr[31, 0], arr[31, 31]) == (0.0, 1.0, 2.0, 3.0)
    windowed = gis.read_cog(path, window=(8, 8, 16, 16))
    assert windowed["info"]["height_system"] == "orthometric_egm96"
    win = np.asarray(windowed["array"]).reshape(16, 16)
    assert (win[0, 0], win[0, 15], win[15, 0], win[15, 15]) == (0.0, 1.0, 2.0, 3.0)


def test_unknown_persisted_height_system_tag_is_rejected(tmp_path):
    # M-03: a PRESENT tag-65001 value that names an unknown vertical datum is
    # rejected on read — never silently coerced to "unspecified", which would
    # erase the (unintelligible) declaration. The bogus value is byte-patched
    # into a valid file because the write path already rejects it.
    path = tmp_path / "bogus.tif"
    _write_dem(path, height_system="chart_datum")
    raw = path.read_bytes()
    assert raw.count(b"chart_datum") == 1
    path.write_bytes(raw.replace(b"chart_datum", b"chart_bogus"))
    with pytest.raises(Exception, match="invalid_height_system"):
        gis.read_raster_info(str(path))
    # A byte-identical file with the untouched tag still reads fine.
    good = tmp_path / "good.tif"
    _write_dem(good, height_system="chart_datum")
    assert gis.read_raster_info(str(good)).height_system == "chart_datum"


def test_height_system_survives_resample(tmp_path):
    # M-03: resampling changes horizontal sampling only; the vertical datum tag
    # is preserved through operation_info.
    path = _write_dem(tmp_path / "resample.tif", height_system="orthometric_egm96")
    result = gis.resample_raster(path, (8, 8), method="nearest")
    assert result["info"]["height_system"] == "orthometric_egm96"


def test_height_system_survives_align(tmp_path):
    # M-03: aligning a raster to a target grid preserves the vertical datum tag.
    path = _write_dem(tmp_path / "align.tif", height_system="orthometric_egm96")
    target = _write_dem(tmp_path / "target.tif",
                        transform=(0.1, 0, 10, 0, -0.1, 52), shape=(1, 8, 8))
    result = gis.align_raster_grid(path, target, resampling="nearest")
    assert result["info"]["height_system"] == "orthometric_egm96"


@pytest.mark.parametrize("declared", ["chart_datum", "unspecified"])
def test_chart_datum_and_unspecified_survive_reprojection_without_promotion(tmp_path, declared):
    # M-03: value-preserving horizontal reprojection carries chart_datum and
    # unspecified UNCHANGED — never silently promoted to ellipsoidal.
    hs = None if declared == "unspecified" else declared
    path = _write_dem(tmp_path / f"{declared}.tif", height_system=hs)
    result = gis.reproject_raster(path, "EPSG:3857", resampling="nearest")
    assert result["info"]["height_system"] == declared


def test_horizontal_crs_does_not_imply_height_system(tmp_path):
    # M-03: declaring a horizontal CRS (even a projected one) never sets a
    # vertical datum; an undeclared height stays "unspecified".
    path = _write_dem(tmp_path / "proj.tif", crs="EPSG:3857",
                      transform=(1000.0, 0, 0, 0, -1000.0, 6_000_000.0))
    assert gis.read_raster_info(path).height_system == "unspecified"


def test_height_system_survives_domain_helper_outputs(tmp_path):
    # M-03: domain helpers (here prepare_dem -> prepare_terrain_derivatives)
    # carry the vertical datum on their emitted RasterInfo. prepare_dem also
    # performs the orthometric->ellipsoidal tag change; the derivative helper
    # must preserve that ellipsoidal tag by value.
    prepared = gis.prepare_dem(
        {
            "array": np.ones((1, 8, 8), dtype=np.float32),
            "height_system": "orthometric_egm96",
            "info": {
                "width": 8,
                "height": 8,
                "band_count": 1,
                "crs_authority": {"name": "EPSG", "code": "4326"},
                "bounds": (10.0, 50.0, 12.0, 52.0),
                "transform": (0.25, 0, 10.0, 0, -0.25, 52.0),
            },
        }
    )
    assert prepared["info"]["height_system"] == "ellipsoidal"
    derivatives = gis.prepare_terrain_derivatives(prepared)
    assert derivatives["info"]["height_system"] == "ellipsoidal"


def test_terrarium_mosaic_uses_the_orthometric_contract(tmp_path):
    # M-03: the ACTUAL mosaic path (build_terrarium_dem) tags EGM96 orthometric.
    # It is offline-testable: with pre-populated cached PNG tiles it never
    # touches the network. Bounds (10, 20, 20, 30) at zoom 1 resolve to the
    # single slippy tile z=1/x=1/y=0.
    PIL_Image = pytest.importorskip("PIL.Image")
    # Terrarium encoding of 100 m: r*256 + g + b/256 - 32768.
    tile = PIL_Image.new("RGB", (256, 256), (128, 100, 0))
    tile_path = tmp_path / "1" / "1" / "0.png"
    tile_path.parent.mkdir(parents=True)
    tile.save(tile_path)

    result = gis.build_terrarium_dem((10.0, 20.0, 20.0, 30.0), 1, cache=str(tmp_path))
    assert result["height_system"] == "orthometric_egm96"
    assert result["info"]["height_system"] == "orthometric_egm96"
    assert result["tile_count"] == 1
    assert result["manifest"][0]["status"] == "hit"
    assert float(np.asarray(result["array"])[0, 0, 0]) == 100.0
    # The single-tile decode path shares the same contract.
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 128
    assert gis.decode_terrarium_dem(rgb)["height_system"] == "orthometric_egm96"
