import builtins
from collections import Counter
from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest
import rasterio

import forge3d
from forge3d import dem_audit as da
from forge3d.io import load_dem
from forge3d.preparation import (
    PreparationGrid,
    PreparationRecipe,
    PreparationSource,
    prepare_scene,
)
from _extrema_fixture import surfaces, TRANSFORM


def test_mixed_defects_locations_measures_and_inspection_sheet(tmp_path):
    smooth, fractal, damaged = surfaces()
    assert "dem_audit" in forge3d.__all__
    for clean in (smooth, fractal):
        assert da.audit(clean, TRANSFORM).findings == []
    before = damaged.tobytes()
    report = da.audit(damaged, TRANSFORM)
    assert before == damaged.tobytes()
    assert Counter(f["class"] for f in report.findings) == dict(
        spike=1, pit=1, void=1, striping=2, seam=1, flat_fill=1, quantization=1
    )
    f = {item["class"]: item for item in report.findings}
    for kind, (x, y), height in [("spike", (39, 46), 72), ("pit", (26, 54), -54)]:
        assert f[kind]["cell"] == [x, y]
        assert f[kind]["xy"] == list(TRANSFORM * (x + 0.5, y + 0.5))
        assert f[kind]["height"] == pytest.approx(height, abs=0.25)
    assert f["void"]["rings"][0] == [[9, 12], [9, 16], [14, 16], [14, 12], [9, 12]]
    assert f["void"]["enclosed"] and f["void"]["cells"] == 20
    assert f["void"]["area"] == 20 * abs(TRANSFORM.determinant)
    assert f["flat_fill"]["rings"][0] == [
        [30, 70],
        [30, 78],
        [38, 78],
        [38, 70],
        [30, 70],
    ]
    assert f["flat_fill"]["cells"] == 64 and f["flat_fill"]["elevation"] == 123.125
    assert f["seam"]["bounds"] == [64, 0, 65, 96] and f["seam"][
        "step"
    ] == pytest.approx(9, abs=0.25)
    assert sorted(
        (p["axis"], p["period_cells"])
        for p in report.findings
        if p["class"] == "striping"
    ) == [("column", 2.0), ("row", 4.0)]
    assert f["quantization"]["step"] == 0.25
    assert f["quantization"]["analysis_bounds"] == [0, 0, 96, 96]
    report.save(tmp_path / "inspection")
    saved = json.loads((tmp_path / "inspection.json").read_text())
    assert (
        saved["findings"] == report.findings
        and saved["status"] == "candidates for inspection"
    )
    rgb = np.asarray(Image.open(tmp_path / "inspection.png"))
    for color in (
        (255, 84, 41),
        (56, 137, 255),
        (0, 214, 230),
        (184, 82, 255),
        (255, 113, 165),
    ):
        assert np.all(rgb == color, axis=2).any()


def test_original_file_mask_scale_nodata_and_loader_fill(tmp_path, monkeypatch):
    path = tmp_path / "source.tif"
    a = np.arange(81, dtype="float32").reshape(9, 9)
    a[1, 2] = -9999.0
    a[5:, :4] = 31.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=9,
        width=9,
        count=1,
        dtype=a.dtype,
        crs="EPSG:32632",
        transform=TRANSFORM,
        nodata=-9999,
    ) as ds:
        ds.write(a, 1)
        mask = np.full(a.shape, 255, dtype="uint8")
        mask[4, 5] = 0
        ds.write_mask(mask)
        ds.scales, ds.offsets = (2.0,), (9.0,)
    before = path.read_bytes()
    r = da.audit(path)
    assert r.metadata["valid_cells"] == 79 and r.metadata["crs"] == "EPSG:32632"
    assert sum(f["cells"] for f in r.findings if f["class"] == "void") == 2
    assert next(f for f in r.findings if f["class"] == "flat_fill")["elevation"] == 71.0
    grid = PreparationGrid("EPSG:32632", 9, 9, TRANSFORM.to_gdal())
    steps = [
        {"operation": "fill_missing", "region": {"rings": f["rings"]}}
        for f in r.findings
        if f["class"] == "void"
    ]
    source = PreparationSource(
        "terrain",
        path,
        "elevation",
        grid=grid,
        units="m",
        vertical_reference="NAP",
        repairs=steps,
    )
    layer = prepare_scene(PreparationRecipe(grid, [source], "NAP")).layers["terrain"]
    assert layer.valid_mask.all()
    for row, col in ((1, 2), (4, 5)):
        neighbors = a[[row - 1, row + 1, row, row], [col, col, col - 1, col + 1]]
        assert layer.data[row, col] in (neighbors * 2 + 9)
    assert [step["newly_valid"] for step in layer.metadata["operations"][:2]] == [1, 1]
    default = load_dem(path)
    assert default.fill_record == dict(
        requested=True, requested_method="nearest", method="nearest", cells_filled=1
    )
    assert default.data[1, 2] in (a[0, 2], a[2, 2], a[1, 1], a[1, 3])
    raw = load_dem(path, fill_nodata_values=False)
    assert raw.data.tobytes() == a.tobytes() and raw.fill_record["cells_filled"] == 0
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("exercise existing loader fallback")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    fallback = load_dem(path)
    assert (
        fallback.fill_record["method"] == "mean"
        and fallback.data[1, 2] == a[a != -9999].mean()
    )
    assert path.read_bytes() == before


def test_named_repairs_replay_locality_and_statistics(tmp_path):
    _, _, z = surfaces()
    path = tmp_path / "input.npy"
    np.save(path, z)
    steps = [
        dict(
            operation="fill_missing",
            region={"bounds": [9, 12, 14, 16]},
            method="nearest",
        ),
        dict(operation="remove_spikes", region={"cells": [[39, 46], [26, 54]]}),
        dict(
            operation="notch_filter",
            region={"bounds": [0, 0, 24, 32]},
            axis="row",
            frequencies=[0.25],
        ),
        dict(operation="dequantize", region={"bounds": [48, 48, 64, 64]}, step=0.25),
    ]
    array = np.ma.masked_invalid(z.copy())
    records = []
    for step in steps:
        args = step.copy()
        operation = args.pop("operation")
        out, record = getattr(da, operation)(array, **args)
        measured = np.zeros(z.shape, bool)
        for row, first, end in record["changed_runs"]:
            measured[row, first:end] = True
        assert measured.sum() == record["cells_changed"] > 0
        mask = np.zeros(z.shape, bool)
        if "cells" in args["region"]:
            for c, r in args["region"]["cells"]:
                mask[r, c] = True
        else:
            c, r, d, s = args["region"]["bounds"]
            mask[r:s, c:d] = True
        assert not measured[~mask].any()
        assert array.data[~measured].tobytes() == out.data[~measured].tobytes()
        assert array.mask[~measured].tobytes() == out.mask[~measured].tobytes()
        diff = abs(out.data - array.data)[measured & ~array.mask]
        assert record["mean_change"] == (diff.mean() if diff.size else None)
        assert record["maximum_change"] == (diff.max() if diff.size else None)
        assert record["parameters"] == args
        if operation == "notch_filter":
            assert (
                np.std(
                    np.mean(out.data[:32, :24], axis=1)
                    - np.mean(array.data[:32, :24], axis=1)
                )
                > 1
            )
        if operation == "dequantize":
            assert record["maximum_change"] <= args["step"] / 2
        records.append(record)
        array = out

    grid = PreparationGrid("EPSG:32632", 96, 96, TRANSFORM.to_gdal())
    source = PreparationSource(
        "terrain",
        path,
        "elevation",
        grid=grid,
        units="m",
        vertical_reference="NAP",
        repairs=steps,
    )
    recipe = PreparationRecipe(grid, [source], "NAP")
    prepared = prepare_scene(recipe)
    assert prepared.layers["terrain"].data.tobytes() == array.data.tobytes()
    ref = prepared.layers["terrain"].metadata
    assert ref["operations"][:4] == records
    # Both nested mutation and replacing a declaration must leave the executed
    # history, its hash, the snapshot recipe and its replay mutually consistent.
    steps[0]["region"]["bounds"][:] = [0, 0, 1, 1]
    steps[2]["frequencies"][0] = 0.5
    assert records[0]["parameters"]["region"]["bounds"] == [9, 12, 14, 16]
    assert ref["operations"][:4] == records
    prepared.save(tmp_path / "snapshot")
    replayed = prepare_scene(
        PreparationRecipe.load(tmp_path / "snapshot" / "recipe.json")
    )
    layer = replayed.layers["terrain"]
    assert ref["operations"] == layer.metadata["operations"]
    assert ref["repairs_sha256"] == layer.metadata["repairs_sha256"]
    assert layer.data.tobytes() == array.data.tobytes()
    assert layer.valid_mask.tobytes() == (~array.mask).tobytes()
    assert np.load(path).tobytes() == z.tobytes()
    original = replace(source, repairs=[])
    assert "repairs" not in original.to_dict()
    altered = prepare_scene(
        replace(recipe, sources=[replace(source, repairs=steps[:1])])
    )
    assert altered.layers["terrain"].metadata["repairs_sha256"] != ref["repairs_sha256"]


def test_all_missing_and_declared_mask_protection(tmp_path):
    z = np.ma.masked_all((4, 4))
    r = da.audit(z)
    assert len(r.findings) == 1 and not r.findings[0]["enclosed"]
    with pytest.raises(ValueError, match="donor"):
        da.fill_missing(z, region={"bounds": [0, 0, 4, 4]})
    path = tmp_path / "dem.npy"
    np.save(path, np.arange(16.0).reshape(4, 4))
    grid = PreparationGrid("EPSG:32632", 4, 4, TRANSFORM.to_gdal())
    allowed = np.ones((4, 4), bool)
    allowed[1, 1] = False
    np.save(tmp_path / "mask.npy", allowed)
    source = PreparationSource(
        "dem",
        path,
        "elevation",
        grid=grid,
        units="m",
        vertical_reference="NAP",
        mask_path=tmp_path / "mask.npy",
        repairs=[
            dict(
                operation="fill_missing",
                method="nearest",
                region={"bounds": [0, 0, 4, 4]},
            )
        ],
    )
    with pytest.raises(ValueError, match="mask"):
        prepare_scene(PreparationRecipe(grid, [source], "NAP"))


def test_decimal_lattice_in_source_precision():
    _, terrain, _ = surfaces()
    z = (np.rint(terrain / 0.1) * 0.1).astype("float32")
    q = next(f for f in da.audit(z).findings if f["class"] == "quantization")
    assert q["aligned_fraction"] == 1.0
    assert q["step"] == pytest.approx(0.1, abs=np.spacing(z.max()))


def test_direct_and_recipe_use_same_integer_repair(tmp_path):
    z = np.zeros((5, 5), dtype=np.int16)
    z[2, 2] = 99
    z[3, 1:3] = 1
    region = {"cells": [[2, 2]]}
    out, record = da.remove_spikes(z, region=region)
    path = tmp_path / "integer.npy"
    np.save(path, z)
    grid = PreparationGrid("EPSG:32632", 5, 5, TRANSFORM.to_gdal())
    source = PreparationSource(
        "dem",
        path,
        "elevation",
        grid=grid,
        units="m",
        vertical_reference="NAP",
        repairs=[dict(operation="remove_spikes", region=region)],
    )
    layer = prepare_scene(PreparationRecipe(grid, [source], "NAP")).layers["dem"]
    assert out.dtype == z.dtype and out[2, 2] == 0
    np.testing.assert_array_equal(layer.data, out.data)
    assert record["maximum_change"] == 99
    assert layer.metadata["operations"][0] == record


def test_unselected_nonfinite_mask_bits_survive_repair():
    z = np.ma.array(np.arange(25.0).reshape(5, 5), mask=np.zeros((5, 5), bool))
    z.data[0, 0] = np.nan
    z.mask[1, 1] = True
    z.mask[2, 2] = True
    out, record = da.fill_missing(z, region={"cells": [[2, 2]]})
    untouched = np.ones(z.shape, bool)
    untouched[2, 2] = False
    assert record["changed_runs"] == [[2, 2, 3]]
    assert out.data[untouched].tobytes() == z.data[untouched].tobytes()
    assert out.mask[untouched].tobytes() == z.mask[untouched].tobytes()


@pytest.mark.parametrize("kind", ["striping", "quantization"])
def test_population_findings_require_a_separate_spatial_selection(kind):
    _, z, _ = surfaces()
    if kind == "striping":
        end = 64
        z[:, :end] += np.resize([0.0, 6.0, 0.0, -6.0], 96)[:, None]
        repair = da.notch_filter
        kwargs = {"frequencies": [0.25], "axis": "row"}
    else:
        end = 88
        z[:, :end] = np.round(z[:, :end] * 4) / 4
        repair = da.dequantize
        kwargs = {"step": 0.25}
    finding = next(f for f in da.audit(z).findings if f["class"] == kind)
    assert finding["analysis_bounds"] == [0, 0, 96, 96]
    assert "bounds" not in finding and finding["repair_region"] is None
    with pytest.raises(ValueError, match="region"):
        repair(z, region=finding, **kwargs)
    with pytest.raises(ValueError, match="region"):
        repair(z, region=finding["repair_region"], **kwargs)
    out, record = repair(z, region={"bounds": [0, 0, end, 96]}, **kwargs)
    assert record["cells_changed"] > 0
    assert z[:, end:].tobytes() == out.data[:, end:].tobytes()


@pytest.mark.parametrize("step", [-9.0, 9.0])
@pytest.mark.parametrize("axis", ["row", "column"])
def test_signed_seam_on_rising_terrain(axis, step):
    z = np.tile(np.arange(96.0) * 12.0, (96, 1))
    z[:, 64:] += step
    if axis == "row":
        z = z.T
    found = [f for f in da.audit(z).findings if f["class"] == "seam"]
    expected = [64, 0, 65, 96] if axis == "column" else [0, 64, 96, 65]
    assert [(f["axis"], f["bounds"], f["step"]) for f in found] == [
        (axis, expected, step)
    ]


def test_corner_spikes_have_one_sided_support():
    _, z, _ = surfaces()
    corners = {(0, 0), (0, 95), (95, 0), (95, 95)}
    for c, r in corners:
        z[r, c] += 72
    findings = da.audit(z).findings
    assert {tuple(f["cell"]) for f in findings if f["class"] == "spike"} == corners


def test_numpy_scalar_settings_serialize_and_invalid_settings_are_structured(tmp_path):
    z = np.ones((4, 4))
    report = da.audit(z, flat_cells=np.int64(16), stripe_fraction=np.float32(0.84))
    report.save(tmp_path / "params")
    assert (
        json.loads((tmp_path / "params.json").read_text())["parameters"]
        == report.metadata["parameters"]
    )
    for key, value in (
        ("flat_cells", 0.5),
        ("stripe_fraction", np.nan),
        ("spike_multiple", "5"),
        ("seam_length", True),
        ("quantized_fraction", np.inf),
    ):
        with pytest.raises(da.AuditParameterError) as error:
            da.audit(z, **{key: value})
        diagnostic = error.value.diagnostic
        assert diagnostic.code == "invalid_audit_parameter"
        assert diagnostic.details == {"parameter": key} and diagnostic.remediation
