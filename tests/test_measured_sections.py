import csv
from dataclasses import replace
import io
import json
import math
import os

import numpy as np
import pytest

from forge3d.measured_sections import Transect, render_section, sample_transect

# WGS 84 semi-major axis (metres). The equator is a circle of this radius.
EQUATOR_M_PER_DEGREE = 6378137 * math.pi / 180
DISTANCE_TOLERANCE_M = 1e-8  # MENSURA committed GeodTest inverse contract
VERTICAL = "EPSG:3855"


def write_grid(path, *, crs="EPSG:4326", transform=None, nodata=-99999.0, unit="metre"):
    from rasterio import open as open_raster
    from rasterio.transform import from_origin

    values = np.array([[-2, 0, -99999, 3], [-3, -4, np.nan, 2]], dtype=np.float32)
    with open_raster(path, "w", driver="GTiff", count=1, width=4, height=2, dtype="float32",
                     crs=crs, transform=transform or from_origin(0, 1 / 512, 1 / 256, 1 / 256),
                     nodata=nodata) as dst:
        dst.write(values, 1)
        dst.set_band_unit(1, unit)
    return path


def measure(path, transect, vertical_crs=VERTICAL):
    return sample_transect(path, transect, source_id="Analytic test raster", vertical_crs=vertical_crs)


@pytest.fixture
def measured(tmp_path):
    path = write_grid(tmp_path / "equator.tif")
    # A 5-pixel equator transect puts six samples at cell centers. The
    # margin below six intervals is deliberate, independent of roundoff.
    transect = Transect((-1 / 512, 0), (9 / 512, 0), 1.01 * EQUATOR_M_PER_DEGREE / 256)
    return path, measure(path, transect)


def test_analytic_distance_heights_validity_and_reverse(measured):
    path, result = measured
    rows = result.samples
    expected_m = [i * EQUATOR_M_PER_DEGREE / 256 for i in range(6)]
    assert [row.distance_m for row in rows] == pytest.approx(expected_m, rel=0, abs=DISTANCE_TOLERANCE_M)
    expected_lon = [(2 * i - 1) / 512 for i in range(6)]
    # The direct-geodesic contract is 1e-9 degrees, far below the 1/256-degree cells.
    assert [row.longitude for row in rows] == pytest.approx(expected_lon, rel=0, abs=1e-9)
    assert [row.latitude for row in rows] == pytest.approx([0] * len(rows), rel=0, abs=1e-9)
    assert [row.elevation_m for row in rows] == [None, -2, 0, None, 3, None]
    assert [row.status for row in rows] == ["out_of_bounds", "valid", "valid", "nodata", "valid", "out_of_bounds"]
    assert [(row.row, row.column) for row in rows] == [(None, None), (0, 0), (0, 1), (0, 2), (0, 3), (None, None)]
    t = result.transect
    reverse = measure(path, Transect(t.end, t.start, t.spacing_m))
    for a, b, distance in zip(rows, reversed(reverse.samples), reversed(expected_m)):
        assert replace(a, index=b.index, distance_m=b.distance_m) == b
        assert b.distance_m == pytest.approx(distance, rel=0, abs=DISTANCE_TOLERANCE_M)


def test_affine_edge_ownership_and_nonfinite(tmp_path):
    path = write_grid(tmp_path / "edges.tif")
    t = Transect((0, 1 / 512), (4 / 256, -3 / 512), 3000)
    a, b = measure(path, t).samples
    assert (a.longitude, a.latitude, a.row, a.column, a.status) == (*t.start, 0, 0, "valid")
    assert (b.longitude, b.latitude, b.status) == (*t.end, "out_of_bounds")
    for x, y, status in [(4 / 256, 0, "out_of_bounds"), (0, -3 / 512, "out_of_bounds"),
                         (2 / 256, -1 / 512, "nodata"), (1 / 256, 0, "valid")]:
        a = measure(path, replace(t, start=(x, y), end=(0.01, 0))).samples[0]
        assert a.status == status
    # A coordinate one binary64 step west of the outer edge remains outside.
    a = measure(path, replace(t, start=(math.nextafter(0, -1), 0))).samples[0]
    assert a.status == "out_of_bounds"


@pytest.mark.parametrize("crs,vertical,unit,error", [
    (None, VERTICAL, "m", "declare a horizontal CRS"),
    ("EPSG:3857", VERTICAL, "m", "horizontal CRS must be EPSG:4326"),
    ("EPSG:4269", VERTICAL, "m", "horizontal CRS must be EPSG:4326"),
    ("EPSG:9518", "EPSG:5773", "m", "differs"),
    ("EPSG:4326", "EPSG:4979", "m", "vertical height CRS"),
    ("EPSG:4326", "guess", "m", "vertical CRS explicitly"),
    ("EPSG:4326", VERTICAL, "feet", "metres"),
])
def test_reject_unsupported_reference(tmp_path, crs, vertical, unit, error):
    path = write_grid(tmp_path / "reference.tif", crs=crs, unit=unit)
    with pytest.raises(ValueError, match=error):
        measure(path, Transect((0, 0), (0.01, 0), 100), vertical)


def test_csv_and_json_keep_signed_values_and_nulls(measured):
    _, section = measured
    decoded = json.loads(section.to_json())
    csv_rows = list(csv.DictReader(io.StringIO(section.to_csv())))
    assert [row["elevation_m"] for row in csv_rows] == ["", "-2.0", "0.0", "", "3.0", ""]
    assert decoded["samples"][3]["elevation_m"] is None
    assert decoded["source"]["vertical_crs"] == VERTICAL
    assert decoded["source"]["horizontal_crs"] == "EPSG:4326"
    assert len(decoded["source"]["sha256"]) == 64
    assert decoded["interpolation"] == "nearest (containing cell)"
    assert decoded["interval_m"] < section.transect.spacing_m
    assert section.to_json() == json.dumps(decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_gaps_are_drawn_without_a_connecting_profile_line(measured):
    from forge3d._measured_section_plate import ACCENT, GAP, PAPER, PLATE_SIZE, _axes, _draw_profile

    _, section = measured
    axes = _axes(section)
    width, height = PLATE_SIZE
    canvas = np.full((height, width, 4), PAPER, dtype=np.uint8)
    pixels = _draw_profile(canvas, section, axes, [])
    x0, y0, x1, y1 = axes["rect"]
    d = section.samples[3].distance_m
    gap_x = round(x0 + (x1 - x0) * d / axes["distance_m"][1])
    gap_column = pixels[y0:y1, gap_x]
    assert np.any(np.all(gap_column == GAP, axis=1))
    assert not np.any(np.all(gap_column == ACCENT, axis=1))
    assert np.any(np.all(pixels[y0:y1, x0:x1] == ACCENT, axis=2))


def test_reject_changed_source_before_render(tmp_path, measured):
    _, section = measured
    new = write_grid(tmp_path / "different.tif", crs="EPSG:9518")
    with pytest.raises(ValueError, match="differs from the sampled source"):
        render_section(section, new, tmp_path / "different", title="No output")
    assert not (tmp_path / "different").exists()


@pytest.mark.skipif(os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1",
                    reason="uses the same native GPU lane as linked publications")
def test_rendered_section_preserves_gaps_and_reuses_only_the_base_map(measured, tmp_path, pro_license):
    from forge3d._png import load_png_rgba

    source, section = measured
    cache = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    render_section(section, source, first, title="Analytic terrain", cache=cache)
    updated = measure(source, replace(section.transect, end=(7 / 512, 0)))
    render_section(updated, source, second, title="Analytic terrain", cache=cache)

    before = json.loads((first / "layout.json").read_text())
    after = json.loads((second / "layout.json").read_text())
    assert before["map_render"]["cache_hit"] is False
    assert after["map_render"]["cache_hit"] is True
    assert before["sha256"]["map.png"] == after["sha256"]["map.png"]
    for name in ("section.png", "samples.json", "samples.csv"):
        assert before["sha256"][name] != after["sha256"][name]
    assert before["annotations"][-1]["map_pixel"] is None
    assert after["annotations"][-1]["map_pixel"] is not None
    assert before["annotations"][-1]["profile_pixel"][1] is None
    assert after["annotations"][-1]["profile_pixel"][1] is not None
    assert load_png_rgba(first / "section.png").shape == (1130, 1500, 4)
