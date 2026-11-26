"""MUTATIO identities, coverage boundaries, and two-sided normal-model experiments."""
import csv
from statistics import NormalDist

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import Affine, from_origin

from forge3d import elevation_change


def raster(path, values, *, transform=from_origin(600000, 200000, 2, 2),
           crs="EPSG:2056", registration="Area", reference="LN02", mask=None, unit=None):
    values = np.asarray(values, dtype="float64")
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype=values.dtype, transform=transform, crs=crs, nodata=-9999) as ds:
        ds.write(values, 1)
        ds.update_tags(AREA_OR_POINT=registration)
        if reference:
            ds.update_tags(vertical_reference=reference)
        if mask is not None:
            ds.write_mask(np.array(mask, dtype="uint8") * 255)
        if unit:
            ds.set_band_unit(1, unit)
    return path


def table(product):
    with (product / "volumes.csv").open() as stream:
        return {(row["scope"], row["quantity"]): row for row in csv.DictReader(stream)}


def test_public_workflow_masks_units_registration_and_thresholds(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    # Inspect text artists on the very figure saved by the public workflow.
    drawn = []
    save = Figure.savefig

    def capture(figure, *args, **kwargs):
        drawn.extend(text.get_text() for axis in figure.axes for text in axis.texts)
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    # US survey feet, shear and rotation exercise both factors in the determinant.
    t = Affine(4, 1, 1000000, .5, -3, 200000)
    params = dict(transform=t, crs="EPSG:2263", registration="Point")
    def write(name, values, **kw):
        return raster(tmp_path / name, values, **params, **kw)
    z1 = write("a.tif", [[0, 0, 0, -9999], [0, 0, 0, 0]],
               mask=[[1, 0, 1, 1], [1, 1, 1, 1]])
    z2 = write("b.tif", [[9, 1, 0, 0], [-8, 1, 1, np.nan]])
    se = write("error.tif", [[4, 0, 0, 4], [0, 0, -9999, 0]])
    dest = elevation_change(z1, z2, tmp_path / "out", sigma1=3., sigma2=se, confidence=.9)
    q = -NormalDist().inv_cdf((1 - .9) / 2)
    with rasterio.open(dest / "change.tif") as ds:
        assert ds.transform == t and ds.tags()["AREA_OR_POINT"] == "Point"
        assert ds.crs == rasterio.crs.CRS.from_epsg(2263)
        d, s, threshold, classes = ds.read()
        expected = np.array([[1, np.nan, 0, np.nan], [-1, 0, np.nan, np.nan]])
        np.testing.assert_equal(classes, expected)
        np.testing.assert_equal(s, [[5, np.nan, 3, np.nan], [3, 3, np.nan, np.nan]])
        np.testing.assert_equal(threshold, q * s)
        np.testing.assert_equal(np.isnan(d), np.isnan(expected))
    area = abs(t.determinant) * (1200 / 3937) ** 2
    rows = table(dest)
    for (scope, quantity), row in rows.items():
        included = np.isfinite(d) if scope == "all" else np.abs(classes) == 1
        if quantity == "gain":
            included &= d > 0
        elif quantity == "loss":
            included &= d < 0
        assert int(row["cells"]) == included.sum()
        assert float(row["volume_m3"]) == pytest.approx(area * d[included].sum())
        assert float(row["independent_half_width_m3"]) == pytest.approx(area * q * np.linalg.norm(s[included]))
        assert float(row["correlated_half_width_m3"]) == pytest.approx(area * q * s[included].sum())
    with Image.open(dest / "map.png") as image:
        legend = image.info["Description"]
        assert "90% confidence" in legend
        assert f"{3*q:.6g} to {5*q:.6g} m across cells" in legend
        assert any("90% confidence" in text and f"{3*q:.6g} to {5*q:.6g} m" in text
                   for text in drawn)


@pytest.mark.parametrize(("change", "message"), [
    ({"crs": "EPSG:3857"}, "CRS"), ({"crs": None}, "CRS"),
    ({"registration": "Point"}, "registration"), ({"reference": "LHN95"}, "vertical"),
    ({"reference": None}, "vertical"), ({"unit": "ft"}, "units"),
    ({"transform": from_origin(600001, 200000, 2, 2)}, "grid"),
    ({"values": [[0, 0, 0]]}, "grid"),
])
def test_reject_before_output(tmp_path, change, message):
    a = raster(tmp_path / "a.tif", [[0, 0]])
    b = raster(tmp_path / "b.tif", **({"values": [[0, 0]]} | change))
    dest = tmp_path / "rejected"
    with pytest.raises(ValueError, match=message):
        elevation_change(a, b, dest, confidence=.95, sigma1=1., sigma2=1.)
    assert not dest.exists()


def test_stable_bias_opt_in_and_raster_errors(tmp_path):
    z1 = raster(tmp_path / "first.tif", [[0, 0, 0, 0]])
    z2 = raster(tmp_path / "second.tif", [[2, 3, 4, 9]])
    stable = raster(tmp_path / "stable.tif", [[1, 1, 1, 0]])
    angles = raster(tmp_path / "slope.tif", [[0, 30, 60, 90]])
    se = raster(tmp_path / "s1.tif", [[1, 2, 3, 4]])
    for apply in (False, True):
        dest = elevation_change(z1, z2, tmp_path / str(apply), stable_ground=stable,
                                correct_bias=apply, slope_degrees=angles, sigma1=se,
                                sigma2=lambda slope: slope / 30, confidence=.95)
        with rasterio.open(dest / "change.tif") as ds:
            assert float(ds.tags()["stable_bias_m"]) == 3
            assert float(ds.tags()["stable_sample_sd_m"]) == 1
            assert float(ds.tags()["applied_bias_m"]) == (3 if apply else 0)
            np.testing.assert_equal(ds.read(1), [[-1, 0, 1, 6]] if apply else [[2, 3, 4, 9]])
            np.testing.assert_equal(ds.read(2), [np.hypot([1, 2, 3, 4], [0, 1, 2, 3])])


def test_excludes_boundary_and_rejects_invalid_ancillaries(tmp_path):
    q = -NormalDist().inv_cdf((1 - .95) / 2)
    a = raster(tmp_path / "a.tif", [[0, 0]])
    b = raster(tmp_path / "b.tif", [[q, -q]])
    kwargs = dict(sigma1=1., sigma2=0., confidence=.95)
    with rasterio.open(elevation_change(a, b, tmp_path / "ok", **kwargs) / "change.tif") as ds:
        np.testing.assert_equal(ds.read(4), [[0, 0]])
    invalid = raster(tmp_path / "invalid.tif", [[0, -1]])
    missing = raster(tmp_path / "missing.tif", [[-9999, -9999]])
    shifted = raster(tmp_path / "shift.tif", [[0, 0]], transform=from_origin(600001, 200000, 2, 2))
    for changes in (dict(confidence=1), dict(sigma1=invalid), dict(sigma2=missing),
                    dict(sigma1=shifted), dict(stable_ground=shifted),
                    dict(vertical_reference1="LHN95"), dict(correct_bias=True)):
        dest = tmp_path / "rejected"
        with pytest.raises(ValueError):
            elevation_change(a, b, dest, **(kwargs | changes))
        assert not dest.exists()


def test_recover_known_volume(tmp_path):
    rng = np.random.default_rng(86)
    shape = (64, 64)
    first = rng.normal(0, .12, shape)
    second = rng.normal(0, .16, shape)
    block = np.s_[16:48, 16:48]
    second[block] += 3
    truth = 32 * 32 * 4 * 3
    dest = elevation_change(raster(tmp_path / "a.tif", first), raster(tmp_path / "b.tif", second),
                            tmp_path / "recovery", sigma1=.12, sigma2=.16, confidence=.95)
    with rasterio.open(dest / "change.tif") as ds:
        np.testing.assert_equal(ds.read(2), np.full(shape, np.hypot(.12, .16)))
        assert np.all(ds.read(4)[block] == 1)
    for scope in ("all", "significant"):
        row = table(dest)[scope, "net"]
        print("Recovery", truth, row)
        assert abs(float(row["volume_m3"]) - truth) <= float(row["independent_half_width_m3"])


def test_no_change_binomial(tmp_path):
    from scipy.stats import binomtest

    # Each row is an independently seeded pair of unchanged surveys.
    pairs = [np.random.default_rng(seed).normal(size=(2, 1024)) for seed in range(64)]
    a, b = np.stack(pairs, axis=1)
    dest = elevation_change(raster(tmp_path / "a.tif", a), raster(tmp_path / "b.tif", b),
                            tmp_path / "noise", sigma1=1., sigma2=1., confidence=.95)
    with rasterio.open(dest / "change.tif") as ds:
        flags = np.count_nonzero(ds.read(4))
    p = binomtest(flags, a.size, .05, alternative="two-sided").pvalue
    print(f"Unchanged: {flags}/{a.size}, two-sided exact binomial p={p:.12g}, alpha=.05")
    assert p >= .05



@pytest.mark.parametrize("bad", ["no_affine", "band_datum", "contradictory_keys", "ancillary_affine"])
def test_structured_preflight(tmp_path, bad):
    from forge3d import PreparationError

    params = dict(transform=None) if bad == "no_affine" else {}
    a = raster(tmp_path / "a.tif", [[0, 0]], **params)
    b = raster(tmp_path / "b.tif", [[1, 2]], **params)
    errors = dict(sigma1=0., sigma2=1.)
    if bad in {"band_datum", "contradictory_keys"}:
        with rasterio.open(a, "r+") as ds:
            ds.update_tags(1, vertical_reference="LN02")
        with rasterio.open(b, "r+") as ds:
            ds.update_tags(1, **({"vertical_reference": "LHN95"} if bad == "band_datum" else
                                 {"height_system": "LHN95", "vertical_reference": "LN02"}))
    if bad == "ancillary_affine":
        errors["sigma2"] = raster(tmp_path / "uncertain.tif", [[1, 1]], transform=None)
    with pytest.raises(PreparationError) as err:
        elevation_change(a, b, tmp_path / "rejected", confidence=.95, **errors)
    expected = "missing_geotransform" if "affine" in bad else "conflicting_vertical_reference"
    assert err.value.diagnostic.code == expected
    assert err.value.diagnostic.remediation
    assert not (tmp_path / "rejected").exists()


def test_explicit_identity_and_band_only_height_system(tmp_path):
    paths = [raster(tmp_path / name, [[0]], transform=Affine.identity(), reference=None)
             for name in ("a.tif", "b.tif")]
    for path in paths:
        with rasterio.open(path, "r+") as ds:
            ds.update_tags(1, height_system="LN02")
    result = elevation_change(*paths, tmp_path / "ok", sigma1=1., sigma2=1., confidence=.95)
    with rasterio.open(result / "change.tif") as ds:
        assert ds.transform == Affine.identity() and ds.tags()["vertical_reference"] == "ln02"


def test_control_mean_covariance(tmp_path):
    # 100 four-square-metre cells, 2 shared controls and 1 m raw difference SE.
    z = np.zeros((10, 10))
    d = np.full_like(z, 2.1)
    d[0, :2] = [-1.5, 1.5]
    control = np.zeros_like(z)
    control[0, :2] = 1
    result = elevation_change(raster(tmp_path / "a.tif", z), raster(tmp_path / "b.tif", d),
                              tmp_path / "corrected", confidence=.95, sigma1=.6, sigma2=.8,
                              stable_ground=raster(tmp_path / "control.tif", control), correct_bias=True)
    q = -NormalDist().inv_cdf((1 - .95) / 2)
    with rasterio.open(result / "change.tif") as ds:
        np.testing.assert_allclose(ds.read(3), q * np.sqrt(1.5 - control))
        np.testing.assert_equal(ds.read(4), np.where(control, np.sign(d), 0.))
    for (scope, quantity), row in table(result).items():
        chosen = np.ones_like(d, dtype=bool) if scope == "all" else control == 1
        if quantity == "gain":
            chosen &= d > 0
        elif quantity == "loss":
            chosen &= d < 0
        weights = 4 * chosen - 4 * chosen.sum() * control / 2
        assert float(row["independent_half_width_m3"]) == pytest.approx(q * np.linalg.norm(weights))
        assert float(row["correlated_half_width_m3"]) == 0
    row = table(result)["all", "net"]
    print("Corrected independent / correlated 95% half-widths:",
          row["independent_half_width_m3"], row["correlated_half_width_m3"])


def test_empirical_controls_and_cross_tile_covariance(tmp_path):
    # Controls straddle TIFF tiles; the selected gain includes one control.
    change = np.tile([5., -5.], 300)[None, :]
    control = np.zeros_like(change)
    control[0, [0, -1]] = 1
    change[0, [0, -1]] = [1., 3.]
    errors = np.linspace(.5, 2., change.size)[None, :]
    a = raster(tmp_path / "a.tif", np.zeros_like(change))
    b = raster(tmp_path / "b.tif", change)
    c = raster(tmp_path / "control.tif", control)
    se = raster(tmp_path / "error.tif", errors)
    q = -NormalDist().inv_cdf((1 - .95) / 2)
    for empirical in (False, True):
        model = {} if empirical else dict(sigma1=0, sigma2=se)
        result = elevation_change(a, b, tmp_path / str(empirical), confidence=.95,
                                  stable_ground=c, correct_bias=True, **model)
        raw = np.full_like(change, np.sqrt(2)) if empirical else errors
        with rasterio.open(result / "change.tif") as ds:
            np.testing.assert_allclose(ds.read(2), raw)
            expected = np.sqrt(raw ** 2 * (1 - control) + np.sum((raw * control) ** 2) / 4)
            np.testing.assert_allclose(ds.read(3), q * expected)
            assert ds.tags()["stable_bias_m"] == "2.0"
        weights = 4 * (change > 2) - 4 * (change > 2).sum() * control / 2
        gain = table(result)["all", "gain"]
        if empirical:
            # With the same SD everywhere the common-mode term is exactly zero.
            assert float(gain["correlated_half_width_m3"]) == 0
        else:
            assert float(gain["correlated_half_width_m3"]) == pytest.approx(q * abs(np.sum(weights * raw)))
        assert float(gain["independent_half_width_m3"]) == pytest.approx(q * np.linalg.norm(weights * raw))
    with pytest.raises(ValueError, match="Incomplete survey error"):
        elevation_change(a, b, tmp_path / "partial", confidence=.95, sigma1=1, stable_ground=c)


@pytest.mark.parametrize("confidence", [1e-17, np.nextafter(1., 0.)])
def test_central_confidence_extremes(tmp_path, confidence):
    from scipy.special import erfinv

    a = raster(tmp_path / "a.tif", [[0.]])
    b = raster(tmp_path / "b.tif", [[1e-18]])
    result = elevation_change(a, b, tmp_path / "out", sigma1=1., sigma2=0., confidence=confidence)
    with rasterio.open(result / "change.tif") as ds:
        np.testing.assert_allclose(ds.read(3), np.sqrt(2.) * erfinv(confidence), atol=0, rtol=1e-14)
        assert ds.read(4).item() == 0
    with Image.open(result / "map.png") as im:
        assert "100% confidence" not in im.info["Description"]


def _wgs84_strip_area(width, south, north):
    # Closed-form ellipsoid zone integral evaluated at the two bounding parallels.
    import math

    a = 6378137.
    f = 1 / 298.257223563
    e = math.sqrt(f * (2 - f))

    def primitive(latitude):
        u = math.sin(math.radians(latitude))
        return u / (1 - (e * u) ** 2) + math.atanh(e * u) / e

    return math.radians(width) * a ** 2 * (1 - e ** 2) / 2 * (primitive(north) - primitive(south))


def test_geographic_area_and_d02_legend(tmp_path):
    import json
    from forge3d.raster_style import CategoricalRasterStyle, raster_style_from_dict

    t = from_origin(179.5, 62, 1, 1)
    z = np.zeros((2, 2))
    changes = [[1, -2], [-1, 3]]
    kw = dict(transform=t, crs="EPSG:4326")
    result = elevation_change(raster(tmp_path / "a.tif", z, **kw),
                              raster(tmp_path / "b.tif", changes, **kw), tmp_path / "out",
                              sigma1=0., sigma2=.1, confidence=.95)
    areas = np.array([_wgs84_strip_area(1, y - 1, y) for y in (62, 61)])
    assert areas[0] < areas[1]
    row = table(result)["all", "net"]
    assert float(row["volume_m3"]) == pytest.approx(np.sum(areas[:, None] * changes))
    with rasterio.open(result / "change.tif") as ds:
        assert ds.crs.to_epsg() == 4326 and ds.transform == t
        style = raster_style_from_dict(json.loads(ds.tags(4)["forge3d_style"]))
    assert isinstance(style, CategoricalRasterStyle)
    legend = style.legend()
    assert [entry["class_id"] for entry in legend["entries"]] == [-1, 0, 1]
    assert "95% confidence" in legend["caption"] and "0.195996 m" in legend["caption"]
    with Image.open(result / "map.png") as im:
        assert legend["caption"] in im.info["Description"]

def test_geographic_volume_and_uncertainty_under_column_subdivision(tmp_path):
    area = _wgs84_strip_area(10, 60, 61)
    confidence = .95
    error = .1
    q = -NormalDist().inv_cdf((1 - confidence) / 2)
    nets = []
    for columns in (1, 2):
        z = np.zeros((1, columns))
        grid = dict(crs="EPSG:4326", transform=from_origin(10, 61, 10 / columns, 1))
        result = elevation_change(raster(tmp_path / "first.tif", z, **grid),
                                  raster(tmp_path / "second.tif", z + 1, **grid),
                                  tmp_path / str(columns), confidence=confidence, sigma1=0, sigma2=error)
        rows = table(result)
        for scope in ("all", "significant"):
            for quantity in ("gain", "net"):
                row = rows[scope, quantity]
                # The published reference is rounded to whole cubic metres.
                assert round(float(row["volume_m3"])) == round(area) == 61231408787
                assert round(float(row["correlated_half_width_m3"])) == round(q * error * area)
                assert round(float(row["independent_half_width_m3"])) == round(q * error * area / np.sqrt(columns))
        nets.append(rows["all", "net"])
        print(f'{columns} column(s):', nets[-1])
    for field in ("volume_m3", "correlated_half_width_m3"):
        assert float(nets[0][field]) == float(nets[1][field])


def test_rotated_geographic_volume_under_subdivision(tmp_path):
    # On a sphere the mean cosine of an affine latitude is its midpoint cosine
    # times a sinc for each of the two independent latitude increments.
    radius = 6371000.
    transform = Affine(8, -2, 170, 3, 5, 52)
    rad = np.pi / 180
    area = (radius ** 2 * abs(transform.determinant) * rad ** 2
            * np.cos(56 * rad) * np.sinc(3 / 360) * np.sinc(5 / 360))
    for n in (1, 2):
        z = np.zeros((n, n))
        grid = dict(transform=transform * Affine.scale(1 / n),
                    crs=f"+proj=longlat +R={radius} +type=crs")
        result = elevation_change(raster(tmp_path / "a.tif", z, **grid),
                                  raster(tmp_path / "b.tif", z - 1, **grid), tmp_path / str(n),
                                  sigma1=.1, sigma2=0, confidence=.95)
        rows = table(result)
        assert round(float(rows["all", "loss"]["volume_m3"])) == round(-area)
        assert round(float(rows["significant", "net"]["volume_m3"])) == round(-area)


@pytest.mark.parametrize("south", [-90, -91])
def test_geographic_south_pole_roundoff(tmp_path, south):
    from forge3d import PreparationError
    from rasterio.transform import from_bounds

    width, height = 4, 169
    grid = dict(crs="EPSG:4326", transform=from_bounds(-180, south, 180, 90, width, height))
    z = np.zeros((height, width))
    first = raster(tmp_path / "first.tif", z, **grid)
    second = raster(tmp_path / "second.tif", z + 1, **grid)
    output = tmp_path / "global"
    with rasterio.open(first) as src:
        assert src.bounds.bottom < -90
    if south == -91:
        with pytest.raises(PreparationError) as error:
            elevation_change(first, second, output, sigma1=0, sigma2=.1, confidence=.95)
        assert error.value.diagnostic.code == "invalid_cell_area"
        assert not output.exists()
    else:
        rows = table(elevation_change(first, second, output, sigma1=0, sigma2=.1, confidence=.95))
        # WGS84 surface area for +1 m: 2πa² [1 + (1−e²) atanh(e)/e],
        # evaluated in decimal arithmetic and rounded once to binary64.
        expected = 510_065_621_724_088.5
        for scope in ("all", "significant"):
            net = rows[scope, "net"]
            assert int(net["cells"]) == width * height
            assert float(net["volume_m3"]) == pytest.approx(expected, rel=np.finfo(float).eps)
