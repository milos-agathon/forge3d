import gzip
import json

import numpy as np
import pytest

from tools.europe_smoke import config, derive, population

# ------------------------------------------------------------------ a fake
# A miniature domain that exercises every invariant without touching the
# 384 MB GHSL raster: three countries, a margin ring, and a cell whose
# population exceeds uint16 so the dtype choice is tested, not assumed.


def _fake_grid():
    nlat, nlon = config.GRID_SHAPE
    rng = np.random.default_rng(7)
    grid = rng.integers(0, 900, size=(nlat, nlon)).astype("float64")
    grid += rng.random((nlat, nlon)) * 0.5      # fractional part to round away
    grid[120, 173] = 22_693_914.4               # Cairo-sized cell
    grid[0, :] = 0.0                            # an empty margin row
    return grid


def _fake_table(grid):
    """Split each cell between up to three countries, largest-remainder once."""
    nlat, nlon = grid.shape
    names = ["AAA", "BBB", "CCC"]
    vals, keys = [], []
    for r in range(nlat):
        for c in range(nlon):
            v = grid[r, c]
            if v <= 0:
                continue
            shares = (0.5, 0.3, 0.2) if (r + c) % 2 == 0 else (1.0, 0.0, 0.0)
            for i, s in enumerate(shares):
                if s > 0:
                    vals.append(v * s)
                    keys.append((names[i], r, c))
    arr = np.array(vals)
    rounded = population.largest_remainder(arr, total=int(round(arr.sum())))
    rows = [(k[0], k[1], k[2], int(n)) for k, n in zip(keys, rounded) if n > 0]
    stats = {
        "countries": names,
        "n_countries": 3,
        "total_pixels": nlat * nlon * 48 * 48,
        "unassigned_pixels_before_fill": 17,
        "unassigned_people_before_fill": 12.0,
        "unassigned_pixels_after_fill": 0,
        "table_rows": len(rows),
    }
    return rows, stats


@pytest.fixture
def fake_derive(monkeypatch, tmp_path):
    grid = _fake_grid()
    rows, stats = _fake_table(grid)
    calls = {"aggregate": 0, "table": 0}

    def fake_aggregate(lon, lat, return_raw_sum=False):
        calls["aggregate"] += 1
        return (grid.copy(), float(grid.sum())) if return_raw_sum else grid.copy()

    def fake_table(cache_dir, return_stats=False):
        calls["table"] += 1
        return (list(rows), dict(stats)) if return_stats else list(rows)

    monkeypatch.setattr(derive.population, "aggregate", fake_aggregate)
    monkeypatch.setattr(derive.population, "country_cell_table", fake_table)
    return tmp_path, grid, calls


# ------------------------------------------------------------------- shape


def test_run_writes_every_array_the_sidecar_names(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    assert m["schema"] == derive.SCHEMA
    for name, entry in m["arrays"].items():
        p = derive.derived_dir(tmp) / entry["file"]
        assert p.exists(), name
        assert p.stat().st_size == entry["bytes"]


def test_arrays_are_raw_little_endian_with_no_container_header(fake_derive):
    """The whole point of the format: file length == elements x itemsize."""
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    for name, e in m["arrays"].items():
        n = int(np.prod(e["shape"]))
        itemsize = np.dtype(e["dtype"]).itemsize
        assert e["bytes"] == n * itemsize, f"{name} carries {e['bytes'] - n * itemsize} B of header"
        assert e["byte_order"] == "little"
    assert m["byte_order"] == "little"


def test_dtypes_are_the_ones_the_measured_extremes_require(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    a = m["arrays"]
    # Cairo is 22.69 M people: uint16 would truncate it 346x (§1.3)
    assert a["population_domain"]["dtype"] == "uint32"
    assert a["population_display"]["dtype"] == "uint32"
    assert a["country_cell_people"]["dtype"] == "uint32"
    # 136*241-1 = 32775 flat cells fit uint16 with room to spare
    assert a["country_cell_cell"]["dtype"] == "uint16"
    assert a["country_cell_country"]["dtype"] == "uint16"


def test_base64_length_is_exact_not_four_thirds(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    for e in m["arrays"].values():
        assert e["base64_bytes"] == 4 * ((e["bytes"] + 2) // 3)


def test_measured_gzip_sizes_are_real(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    for name, e in m["arrays"].items():
        blob = (derive.derived_dir(tmp) / e["file"]).read_bytes()
        assert e["gzip_bytes"] == len(gzip.compress(blob, 9)), name


# --------------------------------------------------------------- integrity


def test_domain_grid_is_exactly_the_marginal_of_the_table(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    grid = derive.read_array("population_domain", tmp, m)
    cell = derive.read_array("country_cell_cell", tmp, m)
    people = derive.read_array("country_cell_people", tmp, m)
    marginal = np.zeros(grid.size, dtype="uint64")
    np.add.at(marginal, cell.astype("int64"), people.astype("uint64"))
    assert np.array_equal(marginal.reshape(grid.shape), grid.astype("uint64"))


def test_display_block_is_bit_identical_to_the_window_it_claims(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    grid = derive.read_array("population_domain", tmp, m)
    block = derive.read_array("population_display", tmp, m)
    off = m["arrays"]["population_display"]["axes"]["domain_offset"]
    r0, c0 = off["row0"], off["col0"]
    nr, nc = config.DISPLAY_BLOCK_SHAPE
    assert block.shape == (nr, nc)
    assert np.array_equal(block, grid[r0:r0 + nr, c0:c0 + nc])
    assert int(block.sum(dtype="uint64")) == m["totals"]["display_block_people"]


def test_display_offset_matches_the_spec_cell_centres(fake_derive):
    """§1.3: the block is lon -24.8..44.8, lat 30.0..72.0."""
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    lon, lat = config.grid_axes()
    off = m["arrays"]["population_display"]["axes"]["domain_offset"]
    assert lat[off["row0"]] == pytest.approx(72.0)
    assert lon[off["col0"]] == pytest.approx(-24.8)


def test_country_row_offsets_slice_each_country_purely(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    country = derive.read_array("country_cell_country", tmp, m)
    people = derive.read_array("country_cell_people", tmp, m)
    offs = derive.read_array("country_row_offsets", tmp, m)
    assert offs.size == len(m["countries"]) + 1
    assert int(offs[0]) == 0 and int(offs[-1]) == country.size
    for i, name in enumerate(m["countries"]):
        lo, hi = int(offs[i]), int(offs[i + 1])
        assert (country[lo:hi] == i).all()
        if hi > lo:
            assert int(people[lo:hi].sum(dtype="uint64")) == m["per_country_people"][name]


def test_totals_close_exactly_domain_equals_display_plus_ring(fake_derive):
    tmp, _, _ = fake_derive
    t = derive.run(build_dir=tmp)["totals"]
    assert t["domain_people"] == t["display_block_people"] + t["margin_ring_people"]
    assert t["domain_people"] == t["table_people"]
    assert t["domain_people"] == round(t["domain_people_float"])


def test_printable_names_the_display_block_and_the_ring_it_excludes(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    assert m["printable"]["array"] == "population_display"
    assert m["printable"]["people"] == m["totals"]["display_block_people"]
    assert str(m["totals"]["margin_ring_people"]) in m["printable"]["rule"]


def test_read_array_rejects_a_corrupted_file(fake_derive):
    tmp, _, _ = fake_derive
    m = derive.run(build_dir=tmp)
    p = derive.derived_dir(tmp) / m["arrays"]["population_domain"]["file"]
    blob = bytearray(p.read_bytes())
    blob[0] ^= 0xFF
    p.write_bytes(bytes(blob))
    with pytest.raises(RuntimeError, match="sha256"):
        derive.read_array("population_domain", tmp, m)


# ------------------------------------------------------------------- gates


def test_gate7_reports_the_measured_conservation_error(fake_derive):
    tmp, _, _ = fake_derive
    g = derive.run(build_dir=tmp)["gates"]["7"]
    assert g["verdict"] == "PASS"
    assert g["rel_err"] <= 1e-12


def test_gate8_asserts_zero_unassigned_after_the_fill(fake_derive):
    tmp, _, _ = fake_derive
    g = derive.run(build_dir=tmp)["gates"]["8"]
    assert g["verdict"] == "PASS"
    assert g["unassigned_pixels_after_fill"] == 0
    assert g["per_country_closes"] is True
    # the before-fill figures are [print], and must still be recorded
    assert g["unassigned_pixels_before_fill"] == 17


def test_a_table_that_does_not_close_is_rejected(monkeypatch, tmp_path):
    grid = _fake_grid()
    rows, stats = _fake_table(grid)
    rows[0] = (rows[0][0], rows[0][1], rows[0][2], rows[0][3] + 1000)
    monkeypatch.setattr(derive.population, "aggregate",
                        lambda lon, lat, return_raw_sum=False:
                        (grid, float(grid.sum())) if return_raw_sum else grid)
    monkeypatch.setattr(derive.population, "country_cell_table",
                        lambda cache_dir, return_stats=False:
                        (rows, stats) if return_stats else rows)
    with pytest.raises(RuntimeError, match="closes on"):
        derive.run(build_dir=tmp_path)


# ------------------------------------------------------------- resumability


def test_second_run_reuses_the_previous_outputs(fake_derive):
    tmp, _, calls = fake_derive
    derive.run(build_dir=tmp)
    assert calls["aggregate"] == 1
    again = derive.run(build_dir=tmp)
    assert again["skipped"] is True
    assert calls["aggregate"] == 1, "resume must not re-aggregate 75.5 M pixels"


def test_force_reruns_even_with_valid_outputs(fake_derive):
    tmp, _, calls = fake_derive
    derive.run(build_dir=tmp)
    derive.run(build_dir=tmp, force=True)
    assert calls["aggregate"] == 2


def test_a_deleted_array_forces_a_rerun(fake_derive):
    tmp, _, calls = fake_derive
    m = derive.run(build_dir=tmp)
    (derive.derived_dir(tmp) / m["arrays"]["country_cell_cell"]["file"]).unlink()
    derive.run(build_dir=tmp)
    assert calls["aggregate"] == 2


def test_a_truncated_array_forces_a_rerun(fake_derive):
    tmp, _, calls = fake_derive
    m = derive.run(build_dir=tmp)
    p = derive.derived_dir(tmp) / m["arrays"]["population_domain"]["file"]
    p.write_bytes(p.read_bytes()[:-4])
    derive.run(build_dir=tmp)
    assert calls["aggregate"] == 2


def test_a_same_size_bit_flip_forces_a_rerun(fake_derive):
    """Size alone cannot detect corruption in bytes Plan 2 base64-inlines.

    build.py republishes the manifest's recorded sha256 into the report and
    never calls read_array, so a size-only resume ships a manifest that lies
    about its own payload. All six arrays are 331,016 B; hashing them is free.
    """
    tmp, _, calls = fake_derive
    m = derive.run(build_dir=tmp)
    p = derive.derived_dir(tmp) / m["arrays"]["country_cell_people"]["file"]
    blob = bytearray(p.read_bytes())
    blob[7] ^= 0x01
    p.write_bytes(bytes(blob))
    assert p.stat().st_size == m["arrays"]["country_cell_people"]["bytes"]
    again = derive.run(build_dir=tmp)
    assert again.get("skipped") is False
    assert calls["aggregate"] == 2
    derive.read_array("country_cell_people", tmp, again)   # consistent again


def test_a_changed_geometry_forces_a_rerun(fake_derive, monkeypatch):
    tmp, _, calls = fake_derive
    derive.run(build_dir=tmp)
    monkeypatch.setattr(config, "DISPLAY_WINDOW", (-25.0, 30.0, 45.0, 71.6))
    assert derive.geometry_fingerprint() != json.loads(
        derive.manifest_path(tmp).read_text())["geometry_fingerprint"]


def test_format_version_is_part_of_the_fingerprint(fake_derive, monkeypatch):
    tmp, _, _ = fake_derive
    before = derive.geometry_fingerprint()
    monkeypatch.setattr(derive, "FORMAT_VERSION", derive.FORMAT_VERSION + 1)
    assert derive.geometry_fingerprint() != before


# ------------------------------------------------------- the real raster


@pytest.mark.slow
def test_real_ghsl_reproduces_the_spec_totals(tmp_path, ghsl_artifact):
    m = derive.run(build_dir=tmp_path)
    t = m["totals"]
    assert t["domain_people"] / 1e6 == pytest.approx(1257.39, rel=2e-4)
    assert t["display_block_people"] / 1e6 == pytest.approx(1037.66, rel=2e-4)
    assert t["margin_ring_people"] / 1e6 == pytest.approx(219.74, rel=1e-3)
    assert m["gates"]["8"]["largest_cell_people"] == pytest.approx(22_693_914, rel=1e-4)
    assert m["gates"]["7"]["verdict"] == "PASS"
    assert m["gates"]["8"]["verdict"] == "PASS"
