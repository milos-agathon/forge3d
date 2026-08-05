"""One command that produces Plan 1's whole deliverable, and one report.

    python -m tools.europe_smoke.build --days 10

Runs probe -> fetch -> derive -> basemap and writes a single
``build_report.json`` carrying every gate verdict (5, 6, 7, 8, 9, 10, 11, 16),
the fitted advection gain, the delivered axis, the FIRMS window and counts, the
population totals and the measured basemap byte size.

ORDER. §10.1 lists "probe -> fetch -> basemap -> derive". This runs derive
before basemap deliberately: derive is offline, cheap and has no failure mode
that a retry fixes, while the basemap render is ~700 DEM tiles through a
monkey-patched sourceless engine. Putting the fragile stage last means a failed
render leaves everything else banked.

RESUMABILITY. Each stage records a fingerprint of the inputs it actually
depends on and the outputs it wrote. A stage is skipped when the fingerprint
matches and every recorded output is still on disk at the recorded size.

  * ``fetch``   keys on the **axis** fingerprint (geometry + the probed time
    axis + the FIRMS window). A new analysis hour invalidates it.
  * ``derive``  keys on the **geometry** fingerprint only. Population does not
    move with the CAMS clock, so tomorrow's fetch reuses today's 75.5 M-pixel
    aggregation.
  * ``basemap`` keys on geometry + render parameters. Also clock-independent.
  * ``probe``   is never skipped. ``constraints.json`` is an ADS build-time
    snapshot and must never be reused across days (§0.4).

So a failed basemap render costs a basemap render, not a re-fetch; and a
re-fetch tomorrow costs a fetch, not a re-aggregation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from . import (advection, basemap, cams, config, derive, hillshade, probe,
               provenance)

SCHEMA = "europe-smoke-build-report/1"
STAGE_NAMES = ("fetch", "derive", "basemap")

EARTH_R = 6378137.0
REFERENCE_CANVAS_W = 1920          # §6.6 states the rail invariant at 1920 px
# §6.6 writes the invariant as "rail_width < 450*(W/1920)". 450 is not an
# independent bound -- it is a rounding of the 449.9 px of out-of-data basemap
# the spec measured on its own widened span, so testing the measurement against
# it is circular (and fails by 0.1 px on the spec's own numbers). We MEASURE the
# out-of-data width and publish it as the ceiling Plan 2's layout must respect.
SPEC_RAIL_REFERENCE_PX = 450.0
WEBP_BUDGET_BYTES = 1_500_000      # §8: basemap WebP <= 1.5 MB binary

ESCALATION_AREA = (79.2, -41.6, 22.8, 61.6)
ESCALATION_GRID_SHAPE = (142, 259)

# Manifest entries the build CREATES rather than requires up front. Natural
# Earth is downloaded by boundary.load_countries during derive, so on a clean
# machine it is legitimately absent at stage 0 and a hard failure there would
# make the build unable to bootstrap itself. Gate 16 re-runs the full check
# after the stages, by which point it must exist.
DERIVABLE_ARTIFACTS = frozenset({"natural_earth"})


# ----------------------------------------------------------------- helpers


def _merc_x(lon_deg: float) -> float:
    return math.radians(lon_deg) * EARTH_R


def _merc_y(lat_deg: float) -> float:
    return math.log(math.tan(math.pi / 4.0 + math.radians(lat_deg) / 2.0)) * EARTH_R


def stage_dir(build_dir: Path | None = None) -> Path:
    return Path(build_dir or config.BUILD_DIR) / "stages"


def stage_path(name: str, build_dir: Path | None = None) -> Path:
    return stage_dir(build_dir) / f"{name}.json"


def axis_fingerprint(axis, days: int) -> str:
    """Everything the fetched cube depends on."""
    return derive.digest({
        "geometry": derive.geometry(),
        "days": days,
        "start": str(axis.start),
        "d_an": str(axis.d_an),
        "h_win": list(axis.h_win),
        "h_new": list(axis.h_new),
        "t_now": axis.t_now.isoformat(),
        "t_init": axis.t_init.isoformat(),
        "run": f"{axis.run_date} {axis.run_hour}",
        "firms_sources": list(config.FIRMS_SOURCES),
    })


def basemap_fingerprint(width: int, height: int, dem_zoom: int,
                        max_dem_size: int, webp_qualities) -> str:
    return derive.digest({
        "geometry": derive.geometry(),
        "width": width, "height": height,
        "dem_zoom": dem_zoom, "max_dem_size": max_dem_size,
        "webp_qualities": list(webp_qualities),
        "sentinel_relation_id": config.SENTINEL_RELATION_ID,
    })


def _outputs_ok(outputs) -> tuple[bool, str]:
    for o in outputs:
        p = Path(o["path"])
        if not p.exists():
            return False, f"missing output {p}"
        if o.get("bytes") is not None and p.stat().st_size != o["bytes"]:
            return False, f"{p} is {p.stat().st_size} B, stage recorded {o['bytes']} B"
    return True, "outputs present"


def run_stage(name: str, fingerprint: str, fn, force: bool = False,
              build_dir: Path | None = None) -> dict:
    """Run ``fn`` unless a matching, complete previous run is on disk.

    ``fn`` returns ``(data, [Path, ...])``.
    """
    path = stage_path(name, build_dir)
    if not force and path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            previous, reason = None, f"unreadable stage record: {exc}"
        else:
            reason = None
        if previous is not None:
            if previous.get("fingerprint") != fingerprint:
                reason = "fingerprint changed since the last run"
            else:
                ok, detail = _outputs_ok(previous.get("outputs", []))
                reason = None if ok else detail
            if reason is None:
                previous["skipped"] = True
                print(f"[{name}] reusing previous run ({fingerprint})")
                return previous
        print(f"[{name}] rerunning: {reason}")

    print(f"[{name}] running ({fingerprint})")
    started = dt.datetime.now(dt.UTC)
    data, outputs = fn()
    # Round-trip through JSON now, so a stage's contribution to the report is
    # identical whether it just ran or was reloaded from disk. Without this the
    # advection fit's ``scores`` dict has float keys when fresh and string keys
    # when resumed, and two builds of the same inputs disagree.
    data = json.loads(json.dumps(data, default=str))
    record = {
        "stage": name,
        "fingerprint": fingerprint,
        "started_utc": started.isoformat(timespec="seconds"),
        "completed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seconds": (dt.datetime.now(dt.UTC) - started).total_seconds(),
        "outputs": [{"path": str(p), "bytes": Path(p).stat().st_size}
                    for p in outputs],
        "data": data,
        "skipped": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def package_versions() -> dict:
    """§10.2: the resolved versions are part of the result, not incidental."""
    import importlib

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "xarray", "rasterio", "geopandas", "shapely",
                 "pyproj", "scipy", "PIL", "netCDF4", "cdsapi"):
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001 - a missing optional is data
            out[name] = f"unavailable ({type(exc).__name__})"
    try:
        import rasterio

        out["gdal"] = str(rasterio.__gdal_version__)
    except Exception:  # noqa: BLE001
        pass
    return out


# ------------------------------------------------------------------- gates


def gate5(merged, axis) -> dict:
    """CAMS axis + lattice + grid, re-measured from the merged cube."""
    lat = np.asarray(merged["latitude"].values, dtype="float64")
    lon = np.asarray(merged["longitude"].values, dtype="float64")
    resid = {}
    for name, ax in (("latitude", lat), ("longitude", lon)):
        resid[name] = float(
            np.abs(ax / config.LATTICE - np.round(ax / config.LATTICE)).max())

    delivered_all = np.asarray(merged["time"].values).astype("datetime64[m]")
    t_now = np.datetime64(axis.t_now, "m")
    delivered_analysis = set(delivered_all[delivered_all <= t_now].tolist())
    requested = set(np.asarray(axis.requested_analysis_times(),
                               dtype="datetime64[m]").tolist())
    missing = sorted(requested - delivered_analysis)
    extra = sorted(delivered_analysis - requested)

    want_vars = set(config.VAR_RESOLUTION)
    have_vars = set(map(str, merged.data_vars))
    missing_vars = sorted(want_vars - have_vars)
    # MERGE NOTE. cams.canonicalise treats bcaod550/aod550 as OPTIONAL at fetch
    # time -- their absence is a recorded Shortfall, not a crash, so the arm
    # still loads. The BUILD is stricter: a report that cannot carry the §7.2 /
    # §7.3 composition readouts is not a complete build, so `ok` stays "every
    # name in VAR_RESOLUTION resolved". The split below makes the two
    # severities visible instead of collapsing them.
    missing_required = sorted(set(config.REQUIRED_VARS) - have_vars)
    missing_optional = sorted(set(config.OPTIONAL_VARS) - have_vars)

    checks = {
        "grid_shape": {"value": [int(lat.size), int(lon.size)],
                       "expected": list(config.GRID_SHAPE),
                       "ok": (lat.size, lon.size) == tuple(config.GRID_SHAPE)},
        "latitude_descends": {"ok": bool((np.diff(lat) < 0).all())},
        "lattice_residual": {"value": resid, "tolerance": config.LATTICE_TOL,
                             "ok": max(resid.values()) <= config.LATTICE_TOL},
        "analysis_axis": {"n_requested": len(requested),
                          "n_delivered": len(delivered_analysis),
                          "missing": [str(t) for t in missing[:5]],
                          "extra": [str(t) for t in extra[:5]],
                          "ok": not missing and not extra},
        "variables_resolved": {"delivered": sorted(have_vars),
                               "missing": missing_vars,
                               "missing_required": missing_required,
                               "missing_optional": missing_optional,
                               "ok": not missing_vars},
    }
    return {
        "name": "CAMS axis + lattice + grid",
        "checks": checks,
        "n_steps": int(merged.sizes["time"]),
        "verdict": "PASS" if all(c["ok"] for c in checks.values()) else "FAIL",
    }


def gate10(projector, width: int, height: int) -> dict:
    """frame_rect subset [0,1], plus the §6.6 rail headroom."""
    x0f, y0f, x1f, y1f = (float(v) for v in projector.frame_rect)
    contained = 0.0 <= x0f < x1f <= 1.0 and 0.0 <= y0f < y1f <= 1.0

    x0, y0, x1, y1 = (float(v) for v in projector.map_bounds_mercator)
    lon_min, lat_min, lon_max, lat_max = config.DISPLAY_WINDOW
    span = x1 - x0
    left_frac = (_merc_x(lon_min) - x0) / span
    right_frac = (x1 - _merc_x(lon_max)) / span
    # [VERIFIER FIX] Out-of-data width measured on the CANVAS, not on the map.
    # The map occupies only frame_rect of the canvas, so the rail-free region
    # is the canvas gutter plus the map's own out-of-data margin. With
    # frame_rect (0,0,1,1) this reduces exactly to left_frac * W and still
    # reproduces §6.6's 449.9353310866 px; with frame_rect (0.1,0,0.9,1) the
    # old form understated the true 551.948 px by 102 px.
    frame_w = x1f - x0f
    left_px = (x0f + left_frac * frame_w) * REFERENCE_CANVAS_W
    right_px = ((1.0 - x1f) + right_frac * frame_w) * REFERENCE_CANVAS_W
    headroom_px = min(left_px, right_px)

    aspect = width / float(height)
    widened = aspect >= 16.0 / 9.0 - 0.05

    out = {
        "name": "frame_rect containment and the §6.6 rail invariant",
        "frame_rect": [x0f, y0f, x1f, y1f],
        "frame_rect_contained": contained,
        "render_aspect": aspect,
        "out_of_data_px_at_1920": {"left": left_px, "right": right_px},
        "rail_ceiling_px_at_1920": headroom_px,
        "spec_rail_reference_px": SPEC_RAIL_REFERENCE_PX,
        "covers_display_window": bool(
            x0 <= _merc_x(lon_min) and x1 >= _merc_x(lon_max)
            and y0 <= _merc_y(lat_min) and y1 >= _merc_y(lat_max)),
        "rule": (
            "Plan 2 must satisfy rail_width < rail_ceiling_px_at_1920 * (W/1920). "
            "The ceiling is measured here, not taken from §6.6's rounded 450."
        ),
    }
    if not contained or not out["covers_display_window"]:
        out["verdict"] = "FAIL"
        out["reason"] = ("frame_rect leaves [0,1]" if not contained
                         else "basemap does not cover the display window")
    elif not widened:
        out["verdict"] = "N/A"
        out["reason"] = (
            f"rendered at the data aspect {aspect:.4f}, not the 16:9 viewport "
            "aspect of §6.6, so there is no rail region to bound. frame_rect "
            "and display-window coverage PASS; the rail half of gate 10 must "
            "be re-run on the widened basemap Plan 2 lays out."
        )
    elif headroom_px <= 0.0:
        out["verdict"] = "FAIL"
        out["reason"] = (
            "the basemap is flush with the display window on at least one side, "
            "so any rail would cover live data; widen the basemap or letterbox"
        )
    else:
        out["verdict"] = "PASS"
        if headroom_px < SPEC_RAIL_REFERENCE_PX:
            out["note"] = (
                f"ceiling {headroom_px:.1f} px is below §6.6's 450 px reference; "
                "Plan 2 must size rails from the measured ceiling, not from 450"
            )
    return out


def _engine_patch_installed():
    """Did the cos-latitude hillshade actually SHADE something on this engine?

    ``None`` when the engine module was never loaded (a stubbed render).
    ``False`` when it was loaded but no shading pass ran through the patch.
    ``True`` only when the wrapper ran at least once and the replacement it
    installed was the cos-latitude one.

    MERGE NOTE. The pipeline fix probed ``m._dem_hillshade.__qualname__``,
    which was right while ``install_hillshade_patch`` had no caller. The
    hillshade fix makes ``basemap.configure()`` install a wrapper around the
    SHADING PASS and rebind ``_dem_hillshade`` only for the duration of that
    call, restoring the engine's own function in a ``finally``. Probing the
    module global therefore reads False even on a correct render; the patch
    state is the thing to ask. The distinction the pipeline fix was defending
    -- "the correction is right for this geometry" vs "the delivered pixels
    were shaded with it" -- is preserved, and now it can answer True.
    """
    m = getattr(basemap, "_ENGINE", None)
    if m is None:
        return None
    try:
        state = basemap.patch_state(m)
    except RuntimeError:
        return False
    shade = state.get("hillshade")
    return bool(state.get("shading_calls", 0) > 0
                and shade is not None and getattr(shade, "is_coslat", False)
                and getattr(shade, "calls", 0) > 0)


def _gate9(bounds, shape) -> dict:
    out = hillshade.latitude_uniformity(bounds, shape)
    installed = _engine_patch_installed()
    out["engine_patch_installed"] = installed
    if installed:
        out["applies_to"] = (
            "the cos-lat replacement evaluated at the delivered raster's "
            "mercator bounds and pixel shape, AND the engine's shading pass ran "
            "through that same replacement on this render "
            "(basemap.assert_hillshade_patch_ran proved it), so the delivered "
            "basemap pixels were produced with the correction this gate scores."
        )
    else:
        out["applies_to"] = (
            "the cos-lat replacement evaluated at the delivered raster's mercator "
            "bounds and pixel shape. It does NOT certify the pixels in basemap.png: "
            "no shading pass ran through the patch on this build, so "
            "engine_patch_installed is "
            + ("False" if installed is False else "unknown (engine not loaded)")
        )
    return out


def gate16(findings, fit: dict, gate5_result: dict, axis, firms_data: dict,
           versions: dict) -> dict:
    prov_ok = all(
        f.status in {provenance.VERIFIED, provenance.PRESENT, provenance.REMOTE}
        for f in findings
    )
    k_ok = fit["k"] <= config.MAX_ADVECTION_GAIN
    vars_ok = gate5_result["checks"]["variables_resolved"]["ok"]
    return {
        "name": "provenance and build-report completeness",
        "artifacts_and_credentials": {
            "ok": prov_ok,
            "findings": [f._asdict() for f in findings],
        },
        "advection_gain": {"k": fit["k"], "cap": config.MAX_ADVECTION_GAIN,
                           "clamped": fit["clamped"],
                           "k_unclamped": fit["k_unclamped"], "ok": k_ok},
        "variables_resolved": {"ok": vars_ok,
                               "missing": gate5_result["checks"]
                               ["variables_resolved"]["missing"]},
        "delivered_axis_recorded": True,
        "firms_window_recorded": bool(firms_data.get("window")),
        "package_versions": versions,
        "verdict": "PASS" if (prov_ok and k_ok and vars_ok) else "FAIL",
    }


# ------------------------------------------------------------------- build


def build(days: int = 10, today: dt.date | None = None, axis=None,
          width: int = 4000, height: int = 4241, dem_zoom: int = 7,
          max_dem_size: int = 4000, webp_qualities=(86, 90),
          force: tuple[str, ...] = (), skip_basemap: bool = False,
          build_dir: Path | None = None, report_path: Path | None = None) -> dict:
    """probe -> fetch -> derive -> basemap -> one report."""
    from . import fetch as fetch_mod

    build_dir = Path(build_dir or config.BUILD_DIR)
    build_dir.mkdir(parents=True, exist_ok=True)
    force = tuple(force)
    if "all" in force:
        force = STAGE_NAMES

    report = provenance.BuildReport()
    started = dt.datetime.now(dt.UTC)
    versions = package_versions()
    report.update(
        schema=SCHEMA,
        started_utc=started.isoformat(timespec="seconds"),
        build_dir=str(build_dir),
        package_versions=versions,
        geometry=derive.geometry(),
        geometry_fingerprint=derive.geometry_fingerprint(),
        forced_stages=list(force),
    )

    # ---- stage 0: provenance. Fail fast, before any ADS queue time is spent.
    # Non-strict only for bootstrap: a derivable artifact may legitimately be
    # absent or unpinned before the stage that creates it, but a non-derivable
    # warning and a wrong-on-disk derivable (size/mismatch) block immediately.
    _, findings = provenance.check_all()
    for line in provenance.format_findings(findings):
        print(line)
    healthy = {provenance.VERIFIED, provenance.PRESENT, provenance.REMOTE}
    derivable_before_build = {
        provenance.PENDING, provenance.MISSING, provenance.UNPINNED,
    }
    deferred = [
        f for f in findings
        if f.name in DERIVABLE_ARTIFACTS and f.status in derivable_before_build
    ]
    blocking = [
        f for f in findings
        if f.status not in healthy and f not in deferred
    ]
    for f in deferred:
        print(f"[note] {f.name} is absent but the build creates it; "
              "gate 16 re-checks at the end")
    if blocking:
        report.update(
            verdict="FAIL",
            failures=["gate 16: " + ", ".join(f.name for f in blocking)],
            gates={"16": {"name": "provenance", "verdict": "FAIL",
                          "blocking": [f._asdict() for f in blocking],
                          "findings": [f._asdict() for f in findings]}},
            completed_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        )
        # [VERIFIER FIX] the report belongs beside the artifacts it describes;
        # config.REPORT_PATH == config.BUILD_DIR / "build_report.json", so the
        # default is unchanged and a redirected build no longer splits them.
        path = report.write(report_path or build_dir / "build_report.json")
        print(f"provenance FAILED; build report -> {path}")
        return report.data
    report.set("provenance_deferred", [f.name for f in deferred])

    # ---- stage 1: probe. Never cached across days (§0.4).
    today = today or dt.datetime.now(dt.UTC).date()
    if axis is None:
        axis = probe.resolve_axis(today - dt.timedelta(days=days), today)
    lag_h = (dt.datetime.now(dt.UTC).replace(tzinfo=None) - axis.t_now).total_seconds() / 3600.0
    axis_data = {
        "start": str(axis.start), "d_an": str(axis.d_an), "d_prev": str(axis.d_prev),
        "h_win": list(axis.h_win), "h_new": list(axis.h_new),
        "t_now": axis.t_now.isoformat(), "t_init": axis.t_init.isoformat(),
        "run": f"{axis.run_date} {axis.run_hour}",
        "n_analysis_requested": len(axis.requested_analysis_times()),
        "t_now_lag_hours": lag_h,
        "window_days_requested": days,
        "window_days_delivered": (axis.d_an - axis.start).days + 1,
        "analysis_hours_on_newest_day": len(axis.h_new),
    }
    shortfalls = []
    if axis_data["window_days_delivered"] < days + 1:
        shortfalls.append(
            f"analysis window is {axis_data['window_days_delivered']} days, "
            f"nominal {days + 1}")
    if len(axis.h_new) < 4:
        shortfalls.append(
            f"newest day carries {len(axis.h_new)} analysis hours, not 4")
    axis_data["recorded_shortfalls"] = shortfalls
    report.update(axis=axis_data, axis_fingerprint=axis_fingerprint(axis, days))

    # ---- stage 2: fetch --------------------------------------------------
    fetch_fp = axis_fingerprint(axis, days)

    def _do_fetch():
        data = fetch_mod.run(days=days, today=today, axis=axis, write_report=False)
        outputs = [Path(data["cams_nc"]), Path(data["firms_csv"]),
                   Path(data["boundary"])]
        return data, outputs

    fetch_rec = run_stage("fetch", fetch_fp, _do_fetch,
                          force="fetch" in force, build_dir=build_dir)
    fetch_data = fetch_rec["data"]
    merged_nc = Path(fetch_data["cams_nc"])

    # ---- stage 3: derive -------------------------------------------------
    derived = derive.run(build_dir=build_dir, force="derive" in force)
    print(f"[derive] {'reused' if derived.get('skipped') else 'wrote'} "
          f"{derived['payload']['binary_bytes']} B of arrays")

    # ---- advection fit, keyed on the same axis as the cube ---------------
    def _do_advection():
        import xarray as xr

        with xr.open_dataset(merged_nc) as ds:
            fit = advection.fit_gain(ds)
            emp = advection.containment(ds, k=fit["k"])
        return {"fit": fit, "empirical": emp}, []

    adv_rec = run_stage("advection", fetch_fp, _do_advection,
                        force="fetch" in force or "advection" in force,
                        build_dir=build_dir)
    fit = adv_rec["data"]["fit"]
    empirical = adv_rec["data"]["empirical"]

    # ---- stage 4: basemap ------------------------------------------------
    basemap_rec = None
    if not skip_basemap:
        bm_fp = basemap_fingerprint(width, height, dem_zoom, max_dem_size,
                                    webp_qualities)

        def _do_basemap():
            img, projector = basemap.render(width, height, config.CACHE_DIR,
                                            dem_zoom=dem_zoom,
                                            max_dem_size=max_dem_size)
            rgb = img.convert("RGB")
            out_dir = build_dir / "basemap"
            out_dir.mkdir(parents=True, exist_ok=True)
            encodings, files = {}, []
            for q in webp_qualities:
                p = out_dir / f"basemap_q{q}.webp"
                rgb.save(p, "WEBP", quality=int(q), method=6)
                n = p.stat().st_size
                encodings[str(q)] = {
                    "path": str(p), "bytes": n,
                    "base64_bytes": 4 * ((n + 2) // 3),
                    "within_budget": n <= WEBP_BUDGET_BYTES,
                }
                files.append(p)
            png = out_dir / "basemap.png"
            rgb.save(png)
            files.append(png)

            bounds = [float(v) for v in projector.map_bounds_mercator]
            data = {
                "width": width, "height": height, "dem_zoom": dem_zoom,
                "max_dem_size": max_dem_size,
                "image_size": list(rgb.size),
                "frame_rect": [float(v) for v in projector.frame_rect],
                "map_bounds_mercator": bounds,
                "webp": encodings,
                "png_bytes": png.stat().st_size,
                "gate10": gate10(projector, width, height),
                "gate9": _gate9(bounds, rgb.size[::-1]),
            }
            return data, files

        basemap_rec = run_stage("basemap", bm_fp, _do_basemap,
                                force="basemap" in force, build_dir=build_dir)

    # ---- gates -----------------------------------------------------------
    import xarray as xr

    with xr.open_dataset(merged_nc) as merged:
        g5 = gate5(merged, axis)

    g6 = dict(fetch_data.get("seam") or {})
    g6.setdefault("name", "analysis/forecast seam continuity")
    g6.setdefault("verdict", "FAIL")

    g7 = derived["gates"]["7"]
    g8 = derived["gates"]["8"]

    g11a = advection.static_containment()
    g11 = {
        "name": "sample containment",
        "a_static": g11a,
        "b_empirical": empirical,
        "verdict": "PASS" if (g11a["verdict"] == "PASS" and empirical["pass"])
                   else "FAIL",
    }

    firms_block = {
        "window": fetch_data.get("firms_window"),
        "sources": fetch_data.get("firms_sources"),
        "detections_total": fetch_data.get("firms_total"),
        "map_records_after_thinning": fetch_data.get("firms_map_records"),
        "map_cap": config.FIRMS_MAP_CAP,
        "csv": fetch_data.get("firms_csv"),
        "note": ("counters come from the full record set, never from the "
                 f"{config.FIRMS_MAP_CAP}-record map cap (§4.3)"),
    }

    # Gate 16 re-runs the check now that the derivable artifacts exist. It is
    # STRICT: only verified/present/remote findings pass, so an unpinned hash
    # cannot ride along to release.
    _, final_findings = provenance.check_all(strict=True)
    g16 = gate16(final_findings, fit, g5, axis, firms_block, versions)

    gates = {"5": g5, "6": g6, "7": g7, "8": g8, "11": g11, "16": g16}
    if basemap_rec is not None:
        gates["9"] = basemap_rec["data"]["gate9"]
        gates["10"] = basemap_rec["data"]["gate10"]
    else:
        for n in ("9", "10"):
            gates[n] = {"name": f"gate {n}", "verdict": "NOT RUN",
                        "reason": "basemap stage skipped (--skip-basemap)"}

    failures = [f"gate {n}: {g.get('reason', g.get('name', ''))}"
                for n, g in sorted(gates.items(), key=lambda kv: int(kv[0]))
                if g.get("verdict") == "FAIL"]
    not_run = [n for n, g in gates.items() if g.get("verdict") == "NOT RUN"]
    not_applicable = {n: g.get("reason") for n, g in gates.items()
                      if g.get("verdict") == "N/A"}

    report.update(
        stages={
            "fetch": {"skipped": fetch_rec["skipped"],
                      "seconds": fetch_rec.get("seconds"),
                      "fingerprint": fetch_rec["fingerprint"]},
            "derive": {"skipped": bool(derived.get("skipped")),
                       "fingerprint": derived["geometry_fingerprint"]},
            "advection": {"skipped": adv_rec["skipped"],
                          "seconds": adv_rec.get("seconds")},
            "basemap": ({"skipped": basemap_rec["skipped"],
                         "seconds": basemap_rec.get("seconds"),
                         "fingerprint": basemap_rec["fingerprint"]}
                        if basemap_rec else {"skipped": True,
                                             "reason": "--skip-basemap"}),
        },
        cams={
            "nc": str(merged_nc),
            "nc_bytes": merged_nc.stat().st_size,
            "n_steps": fetch_data.get("n_steps"),
            "n_analysis_expected": fetch_data.get("n_analysis_expected"),
            "n_forecast_expected": fetch_data.get("n_forecast_expected"),
            "merge": fetch_data.get("merge"),
            "requests": fetch_data.get("requests"),
        },
        firms=firms_block,
        population={
            "domain_people": derived["totals"]["domain_people"],
            "display_block_people": derived["totals"]["display_block_people"],
            "margin_ring_people": derived["totals"]["margin_ring_people"],
            "printable": derived["printable"],
            "table_rows": derived["gates"]["8"]["table_rows"],
            "largest_cell_people": derived["gates"]["8"]["largest_cell_people"],
            "per_country_top20": dict(sorted(
                derived["per_country_people"].items(),
                key=lambda kv: -kv[1])[:20]),
            "manifest": str(derive.manifest_path(build_dir)),
            "payload": derived["payload"],
            "arrays": {k: {kk: v[kk] for kk in
                           ("file", "dtype", "shape", "bytes", "gzip_bytes",
                            "base64_bytes", "sha256")}
                       for k, v in derived["arrays"].items()},
        },
        advection={"fit": fit, "containment": empirical,
                   "kernel_lags_h": list(advection.KERNEL_LAGS),
                   "kernel_weights": list(advection.KERNEL_WEIGHTS)},
        basemap=(basemap_rec["data"] if basemap_rec else
                 {"skipped": True, "reason": "--skip-basemap"}),
        gates=gates,
        failures=failures,
        gates_not_run=not_run,
        gates_not_applicable=not_applicable,
        verdict=("PASS" if not failures and not not_run
                 else "FAIL" if failures else "INCOMPLETE"),
        completed_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        seconds=(dt.datetime.now(dt.UTC) - started).total_seconds(),
    )

    path = report.write(report_path or build_dir / "build_report.json")
    _print_summary(report.data, path)
    if not empirical.get("pass", True):
        _print_escalation(empirical)
    return report.data


def _print_summary(data: dict, path: Path) -> None:
    print("\n" + "=" * 72)
    print(f"BUILD {data['verdict']}   ->  {path}")
    print("=" * 72)
    ax = data.get("axis", {})
    print(f"axis      {ax.get('start')} .. {ax.get('d_an')}  "
          f"NOW {ax.get('t_now')} (lag {ax.get('t_now_lag_hours', float('nan')):.1f} h)  "
          f"run {ax.get('run')}")
    for s in ax.get("recorded_shortfalls", []):
        print(f"          shortfall: {s}")
    cm = data.get("cams", {})
    print(f"cams      {cm.get('n_steps')} steps, {cm.get('nc_bytes')} B")
    fm = data.get("firms", {})
    print(f"firms     {fm.get('detections_total')} detections "
          f"{fm.get('window')} -> {fm.get('map_records_after_thinning')} map records")
    pop = data.get("population", {})
    print(f"pop       domain {pop.get('domain_people', 0) / 1e6:.2f} M | "
          f"PRINTABLE display block {pop.get('display_block_people', 0) / 1e6:.2f} M | "
          f"ring {pop.get('margin_ring_people', 0) / 1e6:.2f} M")
    adv = data.get("advection", {}).get("fit", {})
    print(f"advection k = {adv.get('k')} (unclamped {adv.get('k_unclamped')}, "
          f"clamped={adv.get('clamped')}, baseline k0 {adv.get('baseline_k0')})")
    bm = data.get("basemap", {})
    if bm.get("webp"):
        for q, enc in sorted(bm["webp"].items()):
            print(f"basemap   WebP q{q}: {enc['bytes']} B "
                  f"({enc['base64_bytes']} B base64) "
                  f"{'within' if enc['within_budget'] else 'OVER'} the 1.5 MB budget")
    print("gates")
    for n, g in sorted(data.get("gates", {}).items(), key=lambda kv: int(kv[0])):
        print(f"  {n:>2}  {g.get('verdict', '?'):<8} {g.get('name', '')}")
        if g.get("reason"):
            print(f"      {g['reason']}")


def _print_escalation(empirical: dict) -> None:
    print("\n" + "!" * 72)
    print("GATE 11(b) FAILED - containment escalation required (§0.1)")
    print(f"  frac_over_D    {empirical['frac_over_D']:.5f} "
          f"(limit {empirical['thresholds']['over_D']})")
    print(f"  frac_over_0.8D {empirical['frac_over_0.8D']:.5f} "
          f"(limit {empirical['thresholds']['over_0.8D']})")
    print("  WIDEN THE AREA. Never shrink D.")
    print(f"    config.AREA       = {ESCALATION_AREA}")
    print(f"    config.GRID_SHAPE = {ESCALATION_GRID_SHAPE}")
    print("  Then re-run this command. The axis and geometry fingerprints both")
    print("  change, so fetch and derive rerun automatically; the 106x175")
    print("  display block is identical on the escalation grid (§1.3), so the")
    print("  printed population figure does not move.")
    print("!" * 72)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--force", default="",
                    help="comma-separated stages to rerun: fetch,derive,advection,basemap,all")
    ap.add_argument("--skip-basemap", action="store_true",
                    help="run everything except the ~700-tile DEM render")
    ap.add_argument("--width", type=int, default=4000)
    ap.add_argument("--height", type=int, default=4241)
    ap.add_argument("--dem-zoom", type=int, default=7)
    ap.add_argument("--max-dem-size", type=int, default=4000)
    ap.add_argument("--webp-quality", default="86,90")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    data = build(
        days=args.days,
        force=tuple(s for s in args.force.split(",") if s),
        skip_basemap=args.skip_basemap,
        width=args.width, height=args.height,
        dem_zoom=args.dem_zoom, max_dem_size=args.max_dem_size,
        webp_qualities=tuple(int(q) for q in args.webp_quality.split(",")),
        report_path=Path(args.report) if args.report else None,
    )
    return 0 if data.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
