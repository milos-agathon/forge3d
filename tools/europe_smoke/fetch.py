"""End-to-end acquisition: probe -> CAMS -> FIRMS -> boundary, with a report.

MERGE NOTE. ``run()`` below is where four independent fixes meet, so the order
of its stages is load-bearing:

* the provenance gate is first and refuses only on level ``fail``; a ``pending``
  acquired artifact (Natural Earth) is printed as a warning, because fetch is
  part of the chain that creates it and refusing there deadlocks the build;
* every CAMS payload is cached under a digest of the REQUEST, never an ordinal,
  and both arms' delivered time axes are asserted against times derived from
  THIS run's probe axis -- the forecast assertion is the one the plan never had;
* each arm is canonicalised against the variables that arm actually asked for
  (analysis is segmented by variable, §0.4 R1a-d), and the full §1.1 required
  set is re-asserted on the assembled arms;
* the report write is optional so exactly one process owns ``build_report.json``
  when ``build.py`` orchestrates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

from . import boundary, cams, config, firms, probe, provenance, seam

# §0.4 forbids caching an ADS *constraint* across days; the payload of a
# specific request is still cacheable, but ONLY under a key that is the request
# itself. A filename like ``analysis_0.zip`` is an ordinal, not a key: the axis
# moves every day while the ordinal does not, so yesterday's bytes get served
# for today's request. The digest below closes that.
DIGEST_LEN = 16


def _canonical(obj):
    """Recursively normalise a request for stable serialisation.

    Dict key order is irrelevant to ADS, so it must be irrelevant to the
    digest. List order is NOT normalised -- it is part of the request we send.
    """
    if isinstance(obj, dict):
        return {str(k): _canonical(v)
                for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite value in request: {obj!r}")
        return 0.0 if f == 0.0 else f          # canonicalise -0.0
    raise TypeError(f"request carries an unserialisable value: {obj!r} ({type(obj)})")


def canonical_request_json(request: dict) -> str:
    """The exact bytes the digest is taken over. Stable across runs."""
    return json.dumps(_canonical(request), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def request_digest(request: dict) -> str:
    return hashlib.sha256(
        canonical_request_json(request).encode("utf-8")).hexdigest()[:DIGEST_LEN]


def cache_target(request: dict, cache_dir: Path, label: str) -> Path:
    """``<cache_dir>/<label>_<digest>.zip`` -- the request IS the cache key."""
    return Path(cache_dir) / f"{label}_{request_digest(request)}.zip"


def _sidecar_path(target: Path) -> Path:
    return target.with_name(target.name + ".request.json")


def read_sidecar(target: Path) -> dict | None:
    p = _sidecar_path(target)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_sidecar(target: Path, request: dict, digest: str) -> None:
    _sidecar_path(target).write_text(json.dumps({
        "dataset": config.ADS_DATASET,
        "digest": digest,
        "request": _canonical(request),
        "retrieved_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "bytes": target.stat().st_size,
        "sha256": provenance.sha256_file(target),
    }, indent=2, sort_keys=True), encoding="utf-8")


def retrieve(request: dict, cache_dir: Path, label: str, *, client=None) -> Path:
    """Fetch ``request`` from ADS into a cache entry keyed by the request.

    A cache hit requires all of: the digest in the filename (implicit), a
    sidecar recording a byte-identical request, the same dataset, and a payload
    whose size and sha256 still match what was recorded. Anything else is a
    miss and is re-fetched. The download lands on a ``.part`` file and is
    renamed only on success, so an interrupted run never becomes a cache entry.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = request_digest(request)
    target = cache_target(request, cache_dir, label)

    if target.exists():
        side = read_sidecar(target)
        if side is None:
            print(f"cams: {target.name} has no sidecar; re-fetching")
        elif side.get("request") != _canonical(request):
            raise RuntimeError(
                f"{target}: cached request does not match the requested one under the "
                f"same digest {digest}. Delete the file and its sidecar to re-fetch.")
        elif side.get("dataset") != config.ADS_DATASET:
            raise RuntimeError(
                f"{target}: cached for dataset {side.get('dataset')!r}, "
                f"not {config.ADS_DATASET!r}")
        elif (target.stat().st_size != side.get("bytes")
              or provenance.sha256_file(target) != side.get("sha256")):
            print(f"cams: {target.name} drifted from its sidecar; re-fetching")
        else:
            print(f"cams: cached {target.name} (retrieved {side['retrieved_utc']})")
            return target

    if client is None:
        import cdsapi
        client = cdsapi.Client()
    part = target.with_suffix(".zip.part")
    print(f"cams: retrieving {target.name} (may queue)")
    try:
        client.retrieve(config.ADS_DATASET, request, str(part))
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, target)
    _write_sidecar(target, request, digest)
    return target


def record_delivery(target: Path, ds: xr.Dataset) -> dict:
    """Record what the cache entry actually delivered, next to the request."""
    side = read_sidecar(target) or {}
    times = np.asarray(ds["time"].values)
    delivered = {
        "n_time": int(times.size),
        "first": str(times[0]) if times.size else None,
        "last": str(times[-1]) if times.size else None,
        "grid": [int(ds.sizes.get("latitude", 0)), int(ds.sizes.get("longitude", 0))],
        "variables": sorted(str(v) for v in ds.data_vars),
    }
    side["delivered"] = delivered
    _sidecar_path(target).write_text(
        json.dumps(side, indent=2, sort_keys=True), encoding="utf-8")
    return delivered


def _open(target: Path) -> xr.Dataset:
    """Open a delivered payload, extracting a zip through the safe helper.

    The NetCDF list comes from the members we just extracted, never from a glob
    of the output directory -- a leftover ``.nc`` from an earlier, larger
    delivery must not be merged into this one.
    """
    target = Path(target)
    if zipfile.is_zipfile(target):
        outdir = target.with_suffix("")
        written = provenance.safe_extract(target, outdir)
        paths = sorted(p for p in written if p.suffix.lower() == ".nc")
    else:
        paths = [target]
    if not paths:
        raise RuntimeError(f"{target} contained no NetCDF")
    if len(paths) == 1:
        return xr.open_dataset(paths[0])
    return xr.merge([xr.open_dataset(p) for p in paths], compat="override")


def run(days: int = 10, dry_run: bool = False, today: dt.date | None = None,
        axis=None, write_report: bool = True,
        report_path: Path | None = None) -> dict:
    # Gate. Blocking = a required artifact missing, any size/hash mismatch, a
    # missing credential. Warnings (unpinned, pending) are printed and carried
    # into the report, but must NOT refuse -- fetch is part of the chain that
    # creates the acquired artifacts, so refusing on their absence deadlocks
    # the build.
    ok, findings = provenance.check_all()
    if not ok:
        bad = [f.name for f in findings if f.blocking]
        raise RuntimeError(
            f"provenance check failed for {bad}; refusing to fetch "
            f"({'; '.join(f.detail for f in findings if f.blocking)})"
        )
    for f in findings:
        if f.level == provenance.WARN:
            print(f"provenance warning: {f.name}: {f.status}: {f.detail}")

    today = today or dt.datetime.now(dt.UTC).date()
    axis = axis or probe.resolve_axis(today - dt.timedelta(days=days), today)

    requests = cams.analysis_requests(axis) + [cams.forecast_request(axis)]
    horizon = axis.t_now + dt.timedelta(hours=72)
    n_forecast = sum(
        1 for h in range(3, 121, 3)
        if axis.t_init + dt.timedelta(hours=h) > axis.t_now
        and axis.t_init + dt.timedelta(hours=h) <= horizon)

    labels = [f"analysis_{i}" for i in range(len(requests) - 1)] + ["forecast"]
    cams_cache_dir = config.CACHE_DIR / "cams"
    plan = {
        "dry_run": dry_run,
        "requests": requests,
        "cache_keys": [
            {"label": lab, "digest": request_digest(r),
             "file": cache_target(r, cams_cache_dir, lab).name}
            for lab, r in zip(labels, requests)
        ],
        "t_now": axis.t_now.isoformat(),
        "t_init": axis.t_init.isoformat(),
        "n_analysis_expected": len(axis.requested_analysis_times()),
        "n_forecast_expected": n_forecast,
    }
    if dry_run:
        for i, r in enumerate(requests):
            print(f"--- request {i}: {r['type'][0]} {r['variable']} {r['date']} "
                  f"{r['time']}  -> {plan['cache_keys'][i]['file']}")
        return plan

    report = provenance.BuildReport()
    report.update(**plan, provenance=provenance.summarise(findings))

    # ---- CAMS -----------------------------------------------------------
    cams_resolved: dict[str, str] = {}
    cams_shortfall: dict[str, list[dict]] = {}
    cache_records: list[dict] = []

    def _canonical_arm(name: str, request: dict, label: str):
        """Retrieve, open, flatten and canonicalise ONE arm.

        The arm is held only to the variables it asked for: analysis is
        segmented by variable (§0.4 R1a-d), so the wind arm carries no
        omaod550 and would hard-fail against the default REQUIRED set.
        """
        target = retrieve(request, cams_cache_dir, label)
        required, optional = cams.expected_vars(request)
        out = cams.canonicalise(
            cams.flatten_time(_open(target)),
            required=required, optional=optional)
        cams_resolved.update({c: r.delivered for c, r in out.resolved.items()})
        if out.shortfall:
            cams_shortfall[name] = [
                {"variable": s.canonical, "kind": s.kind, "detail": s.detail}
                for s in out.shortfall]
            for s in out.shortfall:
                print(f"cams: SHORTFALL {name} {s.canonical} [{s.kind}] {s.detail}")
        side = read_sidecar(target) or {}
        cache_records.append({"label": label, "file": Path(target).name,
                              "delivered": record_delivery(target, out.ds),
                              "retrieved_utc": side.get("retrieved_utc")})
        return out.ds

    arms = []
    for i, request in enumerate(requests[:-1]):
        arms.append(_canonical_arm(f"analysis_{i}", request, labels[i]))
    analysis = xr.merge(arms, compat="override", join="outer")
    forecast = _canonical_arm("forecast", requests[-1], labels[-1])

    report.set("cams_cache", cache_records)
    report.set("cams_resolved", cams_resolved)
    report.set("cams_shortfall", cams_shortfall)

    # Gate 16: the full §1.1 REQUIRED set must hold on the ASSEMBLED arms, not
    # just per-request. xr.merge preserves the units attrs [measured].
    for name, ds in (("analysis", analysis), ("forecast", forecast)):
        cams.assert_variables(ds)
        cams.assert_grid(ds)
    # Both assertions compare against times derived from THIS run's probe axis,
    # never against whatever the file happens to contain -- that is what makes a
    # stale-but-same-name payload impossible to swallow. The forecast line is
    # the one the plan never had; test_run_stops_on_a_forecast_arm_from_an_
    # older_run fails the moment it is removed.
    cams.assert_axis(analysis, np.array(axis.requested_analysis_times(),
                                        dtype="datetime64[ns]"))
    cams.assert_axis(forecast, np.array(
        cams.requested_forecast_times(axis, requests[-1]), dtype="datetime64[ns]"))

    merged, merge_report = cams.merge_arms(
        analysis, forecast, np.datetime64(axis.t_now), np.datetime64(axis.t_init))
    merged = merged.sel(time=merged["time"] <= np.datetime64(horizon))
    report.set("merge", merge_report)
    report.set("seam", seam.evaluate(merged, np.datetime64(axis.t_now)))

    out_nc = config.BUILD_DIR / "cams_merged.nc"
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(out_nc)
    report.set("cams_nc", str(out_nc))
    report.set("n_steps", int(merged.sizes["time"]))

    # ---- FIRMS ----------------------------------------------------------
    key = config.FIRMS_KEY.read_text(encoding="utf-8-sig").strip()
    csv_text = firms.fetch(axis.start, axis.d_an, key)
    csv_path = config.BUILD_DIR / "firms.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    records = firms.parse_csv(csv_text)
    report.update(firms_total=len(records),
                  firms_map_records=len(firms.thin(records)),
                  firms_window=[str(axis.start), str(axis.d_an)],
                  firms_sources=list(config.FIRMS_SOURCES),
                  firms_csv=str(csv_path))

    # ---- boundary -------------------------------------------------------
    boundary.build_land_union(config.CACHE_DIR)
    report.set("boundary", str(boundary.osm_boundary_path(config.CACHE_DIR)))

    # Re-verify the artifacts this run just acquired, so the report states
    # their post-acquisition status and an unpinned NE zip is visible with the
    # hash the operator must review.
    report.set("provenance_post", provenance.summarise(provenance.check_all()[1]))

    if write_report:
        path = report.write(report_path)
        print(f"build report -> {path}")
    return report.data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    run(days=args.days, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
