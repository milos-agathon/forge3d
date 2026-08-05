"""Serialise the population products in the shape Plan 2 has to inline (§8).

Plan 1's fetch stage stops at NetCDF and CSV. Neither is loadable by a WebGL
page, and neither is what §8 budgets. This module turns the GHSL aggregation
into the exact bytes Plan 2 base64-inlines, plus one sidecar that records what
those bytes mean and what they cost.

WHY RAW LITTLE-ENDIAN + A JSON SIDECAR, and not npz or parquet
--------------------------------------------------------------
Plan 2 must land every array inside one self-contained ``.html`` and decode it
with ``Uint8Array.fromBase64()`` -> a typed-array view (§8: 1702 MB/s, against
55 MB/s for ``fetch('data:')``). That decode path wants a contiguous block of
little-endian fixed-width integers and nothing else.

  * **npz** is a ZIP of ``.npy`` members, and ``.npy`` carries a Python-literal
    header. A browser cannot read it, so the bundler would have to transcode --
    a second on-disk representation of the same numbers, and a second place for
    the lat-descending / C-order convention to be silently lost. Rejected.
  * **parquet** needs ``pyarrow``, which is **not installed** in this
    interpreter, and the plan forbids new production dependencies. It is also
    unreadable in the browser and columnar-with-metadata for a payload that has
    no schema evolution. Rejected.
  * **raw uint32/uint16 + JSON sidecar** is zero-transcode: the ``.bin`` on
    disk is byte-identical to the buffer behind the ``Uint32Array``. The
    sidecar carries dtype, shape, order, axis direction, byte order, sha256 and
    the *measured* gzip and base64 sizes, so §8's payload table stops being an
    estimate. Chosen.

Byte order is asserted, not assumed: every target of §3 is little-endian, and a
big-endian build must fail loudly rather than ship byte-swapped population.

The (country, cell) table is a **struct of arrays**, not an array of structs.
Measured on the real 15,671-row table: three separately gzipped columns cost
70,462 B against 78,035 B for interleaved 8-byte records -- **9.7% cheaper**,
because each column gets to compress against its own entropy instead of being
striped through two neighbours. Each column is also one typed-array view with
no stride arithmetic in the page. (``np.savez_compressed`` of the same three
columns is 71,252 B, i.e. no cheaper, and unreadable in JS.)

Rows are sorted **country-major, then by cell**. That is *not* a compression
win -- cell-major measures 70,333 B, 0.18% smaller, a wash. It is a structural
one: it makes each country's rows contiguous, so the sidecar can ship
``country_row_offsets`` and Plan 2 slices country ``c``'s rows with two
integers instead of building an index over 15,671 rows at load time.

The uint32 domain grid is the **exact marginal of the rounded table**, not an
independently rounded copy of the float64 aggregation. One rounding, one source
of truth: ``sum_country table[country, cell] == grid[cell]`` holds cell by cell,
so Plan 2 can use either without reconciling them.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import config, population

# Bump when the on-disk layout changes; it is part of the fingerprint, so a
# bump invalidates every cached derived product.
FORMAT_VERSION = 1
SCHEMA = "europe-smoke-derived/1"

MANIFEST_NAME = "derived_manifest.json"

_ARRAY_FILES = {
    "population_domain": "population_domain_u32.bin",
    "population_display": "population_display_u32.bin",
    "country_cell_country": "country_cell_country_u16.bin",
    "country_cell_cell": "country_cell_cell_u16.bin",
    "country_cell_people": "country_cell_people_u32.bin",
    "country_row_offsets": "country_row_offsets_u32.bin",
}


# ------------------------------------------------------------------ digests


def digest(obj) -> str:
    """Stable 16-hex-char digest of any JSON-able object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def geometry() -> dict:
    """Everything a derived population product depends on. No time axis.

    Population does not move with the CAMS clock, so keying it on the geometry
    alone is what lets a second day's fetch reuse yesterday's 75.5 M-pixel
    aggregation.
    """
    ghsl = config.GHSL_TIF
    return {
        "format_version": FORMAT_VERSION,
        "area": list(config.AREA),
        "lattice": config.LATTICE,
        "grid_shape": list(config.GRID_SHAPE),
        "display_window": list(config.DISPLAY_WINDOW),
        "display_block_shape": list(config.DISPLAY_BLOCK_SHAPE),
        "block": population.BLOCK,
        "ghsl": {
            "path": str(ghsl),
            "bytes": ghsl.stat().st_size if ghsl.exists() else None,
        },
    }


def geometry_fingerprint() -> str:
    return digest(geometry())


# ------------------------------------------------------------------- output


def derived_dir(build_dir: Path | None = None) -> Path:
    return Path(build_dir or config.BUILD_DIR) / "derived"


def manifest_path(build_dir: Path | None = None) -> Path:
    return derived_dir(build_dir) / MANIFEST_NAME


def _sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def _b64_len(n: int) -> int:
    """Exact base64 length of ``n`` bytes -- 4*ceil(n/3), not 4n/3 (§8)."""
    return 4 * ((n + 2) // 3)


def _write_array(out_dir: Path, key: str, arr: np.ndarray, axes: dict | None = None) -> dict:
    """Write one raw little-endian array and describe it for the sidecar."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "derived arrays are little-endian by contract (§3/§8 decode path); "
            f"this interpreter is {sys.byteorder}-endian"
        )
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    buf = arr.tobytes(order="C")
    path = out_dir / _ARRAY_FILES[key]
    path.write_bytes(buf)
    entry = {
        "file": path.name,
        "dtype": str(arr.dtype),
        "shape": [int(n) for n in arr.shape],
        "order": "C",
        "byte_order": "little",
        "bytes": len(buf),
        "sha256": _sha256_bytes(buf),
        "gzip_bytes": len(gzip.compress(buf, 9)),
        "base64_bytes": _b64_len(len(buf)),
    }
    if axes:
        entry["axes"] = axes
    return entry


def _outputs_current(manifest: dict, out_dir: Path, fingerprint: str) -> bool:
    """True if a previous run's outputs are still on disk and still valid."""
    if manifest.get("geometry_fingerprint") != fingerprint:
        return False
    if manifest.get("format_version") != FORMAT_VERSION:
        return False
    for key, entry in manifest.get("arrays", {}).items():
        path = out_dir / entry["file"]
        if not path.exists() or path.stat().st_size != entry["bytes"]:
            return False
        # [VERIFIER FIX] Hash, not just size. These bytes are base64-inlined
        # into Plan 2 and build.py republishes the recorded sha256 into the
        # report without ever calling read_array, so a same-size bit flip
        # would resume silently and ship a manifest that lies about its own
        # payload. All six arrays together are 331,016 B.
        if _sha256_bytes(path.read_bytes()) != entry.get("sha256"):
            return False
    return True


# --------------------------------------------------------------------- run


def run(build_dir: Path | None = None, cache_dir: Path | None = None,
        force: bool = False) -> dict:
    """Aggregate, attribute, serialise. Returns (and writes) the sidecar.

    Resumable: if the sidecar on disk was written for the current geometry and
    every array it names is still present, the right size and the right hash,
    nothing is recomputed and ``skipped`` is True.
    """
    build_dir = Path(build_dir or config.BUILD_DIR)
    cache_dir = Path(cache_dir or config.CACHE_DIR)
    out_dir = derived_dir(build_dir)
    fingerprint = geometry_fingerprint()

    mpath = manifest_path(build_dir)
    if not force and mpath.exists():
        try:
            previous = json.loads(mpath.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            previous = {}
        if _outputs_current(previous, out_dir, fingerprint):
            previous["skipped"] = True
            return previous

    out_dir.mkdir(parents=True, exist_ok=True)
    lon, lat = config.grid_axes()
    nlat, nlon = lat.size, lon.size

    # ---- gate 7: the block-48 reduce must conserve mass exactly ----------
    grid_f, raw_sum = population.aggregate(lon, lat, return_raw_sum=True)
    if grid_f.shape != tuple(config.GRID_SHAPE):
        raise RuntimeError(f"aggregate gave {grid_f.shape}, expected {config.GRID_SHAPE}")
    domain_float = float(grid_f.sum())
    rel_err = abs(domain_float - raw_sum) / max(raw_sum, 1.0)
    gate7 = {
        "name": "mass conservation of the GHSL block-48 reduce",
        "rtol": 1e-12,
        "rel_err": rel_err,
        "raw_window_sum": raw_sum,
        "grid_sum": domain_float,
        "verdict": "PASS" if rel_err <= 1e-12 else "FAIL",
    }

    # ---- (country, cell) attribution ------------------------------------
    rows, table_stats = population.country_cell_table(cache_dir, return_stats=True)
    if not rows:
        raise RuntimeError("country_cell_table produced no rows")

    names = table_stats["countries"]
    index_of = {n: i for i, n in enumerate(names)}
    n_rows = len(rows)

    country_idx = np.empty(n_rows, dtype="uint16")
    cell_idx = np.empty(n_rows, dtype="uint16")
    people = np.empty(n_rows, dtype="uint32")

    max_cell = nlat * nlon - 1
    if max_cell > np.iinfo(np.uint16).max:
        raise RuntimeError(
            f"flat cell index reaches {max_cell}; uint16 cell column would wrap. "
            "Widen country_cell_cell to uint32 and bump FORMAT_VERSION."
        )
    if len(names) > np.iinfo(np.uint16).max:
        raise RuntimeError("more countries than a uint16 index can carry")

    # Sort country-major, then by cell, so every country's rows are contiguous
    # and sliceable from country_row_offsets. (Cell-major gzips 0.18% smaller
    # -- a wash; the contiguity is the reason.)
    order = sorted(range(n_rows), key=lambda i: (index_of[rows[i][0]],
                                                 rows[i][1] * nlon + rows[i][2]))
    for out_i, i in enumerate(order):
        name, r, c, n = rows[i]
        if not 0 <= r < nlat or not 0 <= c < nlon:
            raise RuntimeError(f"table row {i} cell ({r},{c}) leaves the grid")
        if n <= 0 or n > np.iinfo(np.uint32).max:
            raise RuntimeError(f"table row {i} people={n} does not fit uint32")
        country_idx[out_i] = index_of[name]
        cell_idx[out_i] = r * nlon + c
        people[out_i] = n

    if n_rows and not (np.diff(country_idx.astype("int64")) >= 0).all():
        raise RuntimeError("country column is not non-decreasing; offsets would lie")
    row_offsets = np.searchsorted(
        country_idx, np.arange(len(names) + 1, dtype="uint16"), side="left"
    ).astype("uint32")
    row_offsets[-1] = n_rows

    table_total = int(people.sum(dtype="uint64"))

    # ---- the uint32 grid IS the marginal of the table -------------------
    grid_u32 = np.zeros(nlat * nlon, dtype="uint64")
    np.add.at(grid_u32, cell_idx.astype("int64"), people.astype("uint64"))
    if grid_u32.max() > np.iinfo(np.uint32).max:
        raise RuntimeError("a CAMS cell exceeds uint32 population")
    grid_u32 = grid_u32.astype("uint32").reshape(nlat, nlon)

    domain_int = int(grid_u32.sum(dtype="uint64"))
    expected_int = int(round(domain_float))
    if domain_int != table_total:
        raise RuntimeError(f"grid marginal {domain_int} != table total {table_total}")
    if domain_int != expected_int:
        raise RuntimeError(
            f"table closes on {domain_int} but the float grid rounds to {expected_int}"
        )

    cell_diff = np.abs(grid_u32.astype("float64") - grid_f)
    max_cell_diff = float(cell_diff.max())
    if max_cell_diff > len(names):
        raise RuntimeError(
            f"per-cell |int - float| reached {max_cell_diff}; largest-remainder "
            f"rounding cannot move a cell by more than the {len(names)} countries "
            "that touch it"
        )

    # ---- the display block is a slice of that same array ----------------
    lon_min, lat_min, lon_max, lat_max = config.DISPLAY_WINDOW
    ci = np.flatnonzero((lon >= lon_min - 1e-9) & (lon <= lon_max + 1e-9))
    ri = np.flatnonzero((lat >= lat_min - 1e-9) & (lat <= lat_max + 1e-9))
    if (ri.size, ci.size) != tuple(config.DISPLAY_BLOCK_SHAPE):
        raise RuntimeError(
            f"display block {(ri.size, ci.size)} != {config.DISPLAY_BLOCK_SHAPE}"
        )
    if not (np.diff(ri) == 1).all() or not (np.diff(ci) == 1).all():
        raise RuntimeError("display block is not a contiguous window")
    block_u32 = np.ascontiguousarray(grid_u32[ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1])
    # checked redundancy: the block ships as its own array so Plan 2 can inline
    # it alone, but it must be bit-identical to the slice it claims to be
    if not np.array_equal(block_u32, grid_u32[np.ix_(ri, ci)]):
        raise RuntimeError("display block is not the window it claims to be")

    display_int = int(block_u32.sum(dtype="uint64"))
    ring_int = domain_int - display_int

    # Compute the per-country totals THROUGH the offsets, so a wrong offset
    # table fails the closure check rather than shipping silently.
    per_country = {}
    for i, name in enumerate(names):
        lo, hi = int(row_offsets[i]), int(row_offsets[i + 1])
        if hi < lo:
            raise RuntimeError(f"country_row_offsets is not monotone at {name}")
        if hi > lo:
            if not (country_idx[lo:hi] == i).all():
                raise RuntimeError(f"country_row_offsets slice for {name} is impure")
            per_country[name] = int(people[lo:hi].sum(dtype="uint64"))
    if sum(per_country.values()) != domain_int:
        raise RuntimeError("per-country totals do not close on the domain total")

    gate8 = {
        "name": "population plausibility and exact closure",
        "domain_people": domain_int,
        "display_block_people": display_int,
        "margin_ring_people": ring_int,
        "table_rows": n_rows,
        "table_people": table_total,
        "per_country_closes": True,
        "unassigned_pixels_after_fill": table_stats["unassigned_pixels_after_fill"],
        "unassigned_pixels_before_fill": table_stats["unassigned_pixels_before_fill"],
        "unassigned_people_before_fill": table_stats["unassigned_people_before_fill"],
        "unassigned_people_share_before_fill": (
            table_stats["unassigned_people_before_fill"] / max(domain_float, 1.0)),
        "max_abs_cell_diff_int_vs_float": max_cell_diff,
        "largest_cell_people": int(grid_u32.max()),
        "note": (
            "the before-fill figures are [print], not [assert]: they move by "
            "~251 k people between GDAL/shapely builds (§5.4)"
        ),
        "verdict": (
            "PASS" if table_stats["unassigned_pixels_after_fill"] == 0 else "FAIL"),
    }

    # ---- serialise ------------------------------------------------------
    arrays = {
        "population_domain": _write_array(
            out_dir, "population_domain", grid_u32,
            axes={
                "lat": f"descending cell centres {lat[0]} .. {lat[-1]} step {-config.LATTICE}",
                "lon": f"ascending cell centres {lon[0]} .. {lon[-1]} step {config.LATTICE}",
                "index": "flat = row * %d + col" % nlon,
            }),
        "population_display": _write_array(
            out_dir, "population_display", block_u32,
            axes={
                "lat": f"descending cell centres {lat[ri[0]]} .. {lat[ri[-1]]}",
                "lon": f"ascending cell centres {lon[ci[0]]} .. {lon[ci[-1]]}",
                "domain_offset": {"row0": int(ri[0]), "col0": int(ci[0])},
            }),
        "country_cell_country": _write_array(
            out_dir, "country_cell_country", country_idx,
            axes={"meaning": "index into manifest.countries"}),
        "country_cell_cell": _write_array(
            out_dir, "country_cell_cell", cell_idx,
            axes={"meaning": "flat domain cell index, row * %d + col" % nlon}),
        "country_cell_people": _write_array(
            out_dir, "country_cell_people", people,
            axes={"meaning": "people in (country, cell), integer, closes exactly"}),
        "country_row_offsets": _write_array(
            out_dir, "country_row_offsets", row_offsets,
            axes={"meaning": (
                "len(countries)+1 offsets; country i owns rows "
                "[offsets[i], offsets[i+1]) of the three parallel columns")}),
    }

    payload_bytes = sum(a["bytes"] for a in arrays.values())
    payload_gzip = sum(a["gzip_bytes"] for a in arrays.values())

    manifest = {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "geometry_fingerprint": fingerprint,
        "geometry": geometry(),
        "byte_order": "little",
        "arrays": arrays,
        "countries": names,
        "totals": {
            "domain_people": domain_int,
            "domain_people_float": domain_float,
            "display_block_people": display_int,
            "margin_ring_people": ring_int,
            "table_people": table_total,
        },
        "printable": {
            "array": "population_display",
            "people": display_int,
            "rule": (
                "§7.7 / §1.3: every printed human figure uses the 175 x 106 "
                "display block. The domain total includes a margin ring of "
                f"{ring_int} people that must never surface in the UI."
            ),
        },
        "per_country_people": per_country,
        "payload": {
            "binary_bytes": payload_bytes,
            "gzip_bytes": payload_gzip,
            "base64_bytes": sum(a["base64_bytes"] for a in arrays.values()),
            "gzip_then_base64_bytes": _b64_len(payload_gzip),
        },
        "gates": {"7": gate7, "8": gate8},
        "skipped": False,
    }
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    # Return what a resumed run would return, so a fresh build and a resumed
    # build contribute byte-identical content to the build report.
    return json.loads(mpath.read_text(encoding="utf-8"))


def load(build_dir: Path | None = None) -> dict:
    return json.loads(manifest_path(build_dir).read_text(encoding="utf-8"))


def read_array(name: str, build_dir: Path | None = None,
               manifest: dict | None = None) -> np.ndarray:
    """Read one serialised array back, exactly as Plan 2's JS will see it."""
    manifest = manifest or load(build_dir)
    entry = manifest["arrays"][name]
    buf = (derived_dir(build_dir) / entry["file"]).read_bytes()
    if len(buf) != entry["bytes"]:
        raise RuntimeError(f"{entry['file']}: {len(buf)} B, manifest says {entry['bytes']}")
    if _sha256_bytes(buf) != entry["sha256"]:
        raise RuntimeError(f"{entry['file']}: sha256 mismatch")
    return np.frombuffer(buf, dtype=entry["dtype"]).reshape(entry["shape"])


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="derive the serialised population products")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    m = run(force=args.force)
    t = m["totals"]
    print(("reused" if m.get("skipped") else "wrote ") + f" {manifest_path()}")
    print(f"  domain        {t['domain_people'] / 1e6:10.2f} M")
    print(f"  display block {t['display_block_people'] / 1e6:10.2f} M  <- the only printable figure")
    print(f"  margin ring   {t['margin_ring_people'] / 1e6:10.2f} M")
    print(f"  table rows    {m['gates']['8']['table_rows']}")
    print(f"  payload       {m['payload']['binary_bytes']} B raw, "
          f"{m['payload']['gzip_bytes']} B gzip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
