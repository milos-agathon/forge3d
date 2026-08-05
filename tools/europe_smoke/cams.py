"""CAMS request construction, variable resolution, and delivery assertions.

Nothing here hardcodes a NetCDF variable name. ECMWF short names are a
documentation artifact, not a contract, so every variable is resolved against
what was actually delivered and a failure prints the delivered list (§1.1).

Resolution is *tiered and exhaustive*, which is the whole of §1.1's assertion:

* tier 1 is the exact short name, compared case-insensitively for equality --
  never as a substring, because ``aod550`` is a substring of both ``omaod550``
  and ``bcaod550`` and a substring test there would make the total-AOD name
  unresolvable the moment the organic one is delivered alongside it;
* tier 2 is a substring of the variable name or of its ``long_name``;
* whichever tier fires first must yield **exactly one** candidate. Zero and
  two-or-more are both hard failures that print every candidate considered and
  every variable actually delivered.

Resolution alone is not enough to trust a name. ``aermssom`` (vertically
integrated mass of organic matter aerosol, ``kg m**-2``) matches the ``organic``
substring and would resolve as the smoke field; only the unit-class check in
:func:`assert_variables` rejects it. So resolution and assertion always run
together -- :func:`canonicalise` is the only entry point callers need.

Required vs optional (§1.1, §7.2, §7.3, §8):

* ``omaod550`` is the smoke field itself and ``u10``/``v10`` drive advection,
  the trajectories, and the §5.5 containment fit that validates the domain.
  Without any of the three there is no map and no gate 11. **Required.**
* ``bcaod550`` and ``aod550`` are the two ``?``-marked names whose ECMWF short
  names are documented but unverified. They feed only the §7.2/§7.3 composition
  readouts, which §7 already requires to display "--" when they cannot be
  computed, and §8 already contemplates dropping ``total_aod550`` from the atlas
  entirely. **Optional**: their absence is a recorded shortfall carrying the
  delivered ``data_vars`` list, never a silent skip. Pass
  ``strict_optional=True`` for the spec-literal reading in which a ``?``-marked
  name failing to resolve stops the build.

A shortfall is never "nothing matched" when something did. If every candidate
for an optional name was already claimed by another canonical name, the record
says so and names the claimer -- that ambiguity is precisely what §1.1 exists to
surface, and reporting it as a plain absence would hide the same
wrong-variable-served-silently failure this module was written to close.

MERGE NOTE. A shortfall is a *fetch-time* verdict: the arm still loads. The
build-report side is stricter -- ``build.gate5`` re-measures the merged cube
against the whole of ``config.VAR_RESOLUTION`` and reports a missing optional
as a gate-5/gate-16 failure of the BUILD, while separating
``missing_required`` from ``missing_optional`` so the two severities are
visible. Fetching does not crash; the build is not called complete.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import xarray as xr

from . import config
from .probe import Axis

_WS = re.compile(r"\s+")


class CamsContractError(RuntimeError):
    """The §1.1 variable contract was violated by the delivered dataset."""


class Resolution(NamedTuple):
    """How one canonical name was satisfied by the delivery."""

    canonical: str
    delivered: str
    tier: str            # "exact" | "substring"
    units: str
    dims: tuple[str, ...]


class Shortfall(NamedTuple):
    """An OPTIONAL variable that the delivery did not provide."""

    canonical: str
    kind: str            # "unresolved" | "ambiguous" | "claimed" | "rejected"
    detail: str          # always names the delivered data_vars


class CanonicalResult(NamedTuple):
    ds: xr.Dataset
    resolved: dict[str, Resolution]
    shortfall: tuple[Shortfall, ...]


# ------------------------------------------------------------------ resolution


def _dedupe(names: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication of a canonical-name sequence.

    An ADS request may legitimately list the same variable twice. Without this
    the injectivity check in :func:`assert_variables` fires against the name
    itself and prints the self-contradictory "'u10' and 'u10' both resolved to
    the single delivered variable 'u10'".
    """
    seen: dict[str, None] = {}
    for n in names:
        seen.setdefault(n, None)
    return tuple(seen)


def _delivered(ds: xr.Dataset) -> list[tuple[str, str]]:
    """``(lowercased, original)`` for every data variable.

    A list, not a dict: two delivered names differing only in case are a real
    ambiguity and a dict would silently keep just one of them.
    """
    return [(str(n).lower(), str(n)) for n in ds.data_vars]


def _haystack(ds: xr.Dataset, name: str) -> str:
    attrs = ds[name].attrs
    return " ".join((
        name.lower(),
        str(attrs.get("long_name", "")).lower(),
        str(attrs.get("standard_name", "")).lower(),
    ))


def candidates(ds: xr.Dataset, canonical: str) -> tuple[str, list[str]]:
    """Every delivered name that could be ``canonical``, and the tier that found them."""
    try:
        exact, substrings = config.VAR_RESOLUTION[canonical]
    except KeyError:
        raise CamsContractError(
            f"{canonical!r} is not in config.VAR_RESOLUTION "
            f"(known: {sorted(config.VAR_RESOLUTION)})"
        ) from None
    wanted = {w.lower() for w in exact}
    hits = [orig for low, orig in _delivered(ds) if low in wanted]
    if hits:
        return "exact", hits
    frags = tuple(f.lower() for f in substrings)
    hits = [orig for _, orig in _delivered(ds)
            if any(f in _haystack(ds, orig) for f in frags)]
    return "substring", hits


def _resolution_error(ds: xr.Dataset, canonical: str, tier: str,
                      hits: Sequence[str]) -> str:
    exact, substrings = config.VAR_RESOLUTION[canonical]
    what = "no delivered variable matches" if not hits else (
        f"{len(hits)} delivered variables match ambiguously: {sorted(hits)}"
    )
    return (
        f"CAMS variable {canonical!r} did not resolve to exactly one delivered "
        f"variable: {what} (tier={tier}, exact={list(exact)}, "
        f"substrings={list(substrings)}); delivered data_vars were "
        f"{sorted(str(n) for n in ds.data_vars)}"
    )


def resolve_var(ds: xr.Dataset, canonical: str) -> str:
    """Map a canonical name onto the one delivered variable name that is it.

    Raises :class:`CamsContractError` (a ``RuntimeError``) when zero or more
    than one delivered variable answers to ``canonical``. The message always
    lists the delivered ``data_vars``.
    """
    tier, hits = candidates(ds, canonical)
    if len(hits) != 1:
        raise CamsContractError(_resolution_error(ds, canonical, tier, hits))
    return hits[0]


# ------------------------------------------------------------------ assertions


def _norm_units(raw: object) -> str:
    return _WS.sub(" ", str(raw)).strip().lower()


def _unit_ok(canonical: str, units: object) -> bool:
    cls = config.UNIT_CLASS[canonical]
    members = {_norm_units(u) for u in config.UNIT_CLASS_MEMBERS[cls]}
    return _norm_units(units) in members


def _check_one(ds: xr.Dataset, canonical: str, delivered: str, tier: str) -> Resolution:
    """Dims and unit class for one already-resolved variable. Raises on violation."""
    var = ds[delivered]
    dims = tuple(str(d) for d in var.dims)
    if set(dims) not in config.ACCEPTED_DIM_SETS:
        raise CamsContractError(
            f"CAMS variable {canonical!r} (delivered as {delivered!r}) has dims "
            f"{dims}; expected one of "
            f"{[sorted(s) for s in config.ACCEPTED_DIM_SETS]}; delivered "
            f"data_vars were {sorted(str(n) for n in ds.data_vars)}"
        )
    if "units" not in var.attrs:
        raise CamsContractError(
            f"CAMS variable {canonical!r} (delivered as {delivered!r}) carries no "
            f"units attribute; §1.1 expects a "
            f"{config.UNIT_CLASS[canonical]} unit; delivered data_vars were "
            f"{sorted(str(n) for n in ds.data_vars)}"
        )
    units = var.attrs["units"]
    if not _unit_ok(canonical, units):
        cls = config.UNIT_CLASS[canonical]
        raise CamsContractError(
            f"CAMS variable {canonical!r} (delivered as {delivered!r}) has units "
            f"{str(units)!r}, which is not {cls} "
            f"({sorted(config.UNIT_CLASS_MEMBERS[cls])}); delivered data_vars "
            f"were {sorted(str(n) for n in ds.data_vars)}"
        )
    return Resolution(canonical, delivered, tier, str(units), dims)


def assert_variables(
    ds: xr.Dataset,
    required: Iterable[str] = config.REQUIRED_VARS,
) -> dict[str, Resolution]:
    """§1.1's assertion, executable.

    Every name in ``required`` must resolve to exactly one delivered variable;
    the resolved set must carry the expected dims and a dimensionless (``~``,
    ``1``) or ``m s**-1`` unit; and no two canonical names may claim the same
    delivered variable. Any violation raises :class:`CamsContractError` with the
    delivered ``data_vars`` list in the message.
    """
    resolved: dict[str, Resolution] = {}
    claimed: dict[str, str] = {}
    for canonical in _dedupe(required):
        tier, hits = candidates(ds, canonical)
        if len(hits) != 1:
            raise CamsContractError(_resolution_error(ds, canonical, tier, hits))
        delivered = hits[0]
        if delivered in claimed:
            raise CamsContractError(
                f"CAMS variables {claimed[delivered]!r} and {canonical!r} both "
                f"resolved to the single delivered variable {delivered!r}; "
                f"delivered data_vars were "
                f"{sorted(str(n) for n in ds.data_vars)}"
            )
        claimed[delivered] = canonical
        resolved[canonical] = _check_one(ds, canonical, delivered, tier)
    return resolved


# ------------------------------------------------------------------ canonicalise


def _canonical_coords(ds: xr.Dataset) -> xr.Dataset:
    for short, long in (("lat", "latitude"), ("lon", "longitude")):
        if short in ds.coords and long not in ds.coords:
            ds = ds.rename({short: long})
    return ds


def expected_vars(request: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(required, optional)`` canonical names for one ADS request.

    Analysis is segmented by variable (§0.4 R1a-d): the wind arm carries no
    aerosol and the aerosol arm carries no wind, so an arm may only be held to
    the variables it actually asked for.
    """
    canon: list[str] = []
    for ads_name in request["variable"]:
        mapped = config.ADS_TO_CANONICAL.get(ads_name)
        if mapped is None:
            raise CamsContractError(
                f"ADS variable {ads_name!r} has no canonical mapping "
                f"(known: {sorted(config.ADS_TO_CANONICAL)})"
            )
        canon.append(mapped)
    # a request may list the same variable twice; the contract must not turn
    # that into an injectivity failure.
    required = _dedupe(c for c in canon if c in config.REQUIRED_VARS)
    optional = _dedupe(c for c in canon if c in config.OPTIONAL_VARS)
    return required, optional


def canonicalise(
    ds: xr.Dataset,
    required: Iterable[str] = config.REQUIRED_VARS,
    optional: Iterable[str] = config.OPTIONAL_VARS,
    strict_optional: bool = False,
) -> CanonicalResult:
    """Resolve, verify, and rename to the canonical set.

    Required names must resolve, must have the expected dims, and must carry a
    unit of the expected class -- otherwise :class:`CamsContractError`. Optional
    names that fail any of those become :class:`Shortfall` records the caller
    writes into the build report; nothing is ever skipped silently. With
    ``strict_optional=True`` a shortfall is raised instead.
    """
    ds = _canonical_coords(ds)
    required = _dedupe(required)
    # a name cannot be both; required wins and there is nothing to report.
    optional = tuple(c for c in _dedupe(optional) if c not in required)

    resolved = assert_variables(ds, required)
    claimed = {r.delivered: c for c, r in resolved.items()}

    shortfall: list[Shortfall] = []
    for canonical in optional:
        tier, hits = candidates(ds, canonical)
        # keep `hits` intact -- rebinding it to the filtered list and then
        # reporting "no delivered variable matches" would be false for a
        # candidate that had in fact matched and been claimed.
        free = [h for h in hits if h not in claimed]
        if len(free) != 1:
            if hits and not free:
                # Every candidate was taken by another canonical name. Saying
                # "no delivered variable matches" here would be false, and it
                # is exactly the ambiguity §1.1 exists to surface.
                taken = ", ".join(f"{h!r} (claimed by {claimed[h]!r})"
                                  for h in sorted(hits))
                shortfall.append(Shortfall(
                    canonical, "claimed",
                    f"CAMS variable {canonical!r} matched only delivered "
                    f"variables already claimed by another canonical name: "
                    f"{taken} (tier={tier}); delivered data_vars were "
                    f"{sorted(str(n) for n in ds.data_vars)}"))
            else:
                kind = "unresolved" if not free else "ambiguous"
                shortfall.append(Shortfall(canonical, kind,
                                           _resolution_error(ds, canonical, tier, free)))
            continue
        try:
            resolved[canonical] = _check_one(ds, canonical, free[0], tier)
        except CamsContractError as exc:
            shortfall.append(Shortfall(canonical, "rejected", str(exc)))
            continue
        claimed[free[0]] = canonical

    if shortfall and strict_optional:
        raise CamsContractError(
            "optional CAMS variables did not resolve and strict_optional is set: "
            + " | ".join(s.detail for s in shortfall)
        )

    renames = {r.delivered: c for c, r in resolved.items() if r.delivered != c}
    for delivered, canonical in renames.items():
        if canonical in ds.data_vars:
            raise CamsContractError(
                f"cannot rename {delivered!r} to {canonical!r}: the delivery "
                f"already carries a different variable of that name; delivered "
                f"data_vars were {sorted(str(n) for n in ds.data_vars)}"
            )
    if renames:
        ds = ds.rename(renames)
    # ``Resolution.delivered`` deliberately keeps the pre-rename name: it is the
    # provenance the build report needs to show what ECMWF actually shipped.
    return CanonicalResult(ds, resolved, tuple(shortfall))


# ------------------------------------------------------------- time flattening


def _as_timedelta(values, attrs) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.timedelta64):
        return arr.astype("timedelta64[ns]")
    units = str(attrs.get("units", "hours")).strip().lower()
    per_unit = {"h": 3600.0, "hour": 3600.0, "hours": 3600.0,
                "s": 1.0, "second": 1.0, "seconds": 1.0,
                "m": 60.0, "minute": 60.0, "minutes": 60.0}.get(units)
    if per_unit is None:
        raise CamsContractError(f"unsupported forecast_period units {units!r}")
    return (arr.astype("float64") * per_unit * 1e9).astype("timedelta64[ns]")


def flatten_time(ds: xr.Dataset) -> xr.Dataset:
    """Collapse (reference time, leadtime) onto one ascending ``time`` dim.

    Ported from examples/greece_smoke_data_fetch.py:236, which is correct.
    Stacking is by NAME, so the delivered period-major dim order is immaterial.
    """
    ref_name = next((n for n in ("forecast_reference_time", "reference_time", "time")
                     if n in ds.dims), None)
    per_name = next((n for n in ("forecast_period", "leadtime_hour", "step")
                     if n in ds.dims), None)
    if ref_name is not None and per_name is not None:
        period = _as_timedelta(ds[per_name].values, dict(ds[per_name].attrs))
        ref = ds[ref_name].values.astype("datetime64[ns]")
        valid = ref[:, None] + period[None, :]
        ds = ds.stack(vt=(ref_name, per_name)).reset_index("vt", drop=True)
        ds = ds.assign_coords(vt=valid.reshape(-1)).rename(vt="time").sortby("time")
    elif "time" not in ds.dims and "valid_time" in ds.dims:
        ds = ds.rename(valid_time="time").sortby("time")
    else:
        ds = ds.sortby("time")
    ds = ds.drop_vars("valid_time", errors="ignore")
    keep = {"time", "latitude", "lat", "longitude", "lon"}
    extra = [d for d, n in ds.sizes.items() if n == 1 and d not in keep]
    if extra:
        ds = ds.squeeze(extra, drop=True)
    _, keep_idx = np.unique(ds["time"].values, return_index=True)
    return ds.isel(time=np.sort(keep_idx))


# ------------------------------------------------------------- delivery gates
#
# MERGE NOTE (systemic issue (a)): every gate below raises AssertionError
# EXPLICITLY rather than using the `assert` statement. `python -O` strips the
# statement form outright, so a build run optimised would sail past a wrong
# grid, a wrong axis or a non-identical forecast lead 0. The exception TYPE and
# the message text are unchanged, so `pytest.raises(AssertionError, match=...)`
# still matches. The repo has hit this before -- 246353b8 "fix(examples):
# France sea-clamp gate survives python -O".


def assert_grid(ds: xr.Dataset) -> None:
    lat = np.asarray(ds["latitude"].values, dtype="float64")
    lon = np.asarray(ds["longitude"].values, dtype="float64")
    if (lat.size, lon.size) != config.GRID_SHAPE:
        raise AssertionError(
            f"grid {(lat.size, lon.size)} != expected {config.GRID_SHAPE}"
        )
    if not (np.diff(lat) < 0).all():
        raise AssertionError("latitude must descend, as CAMS delivers it")
    for name, axis in (("latitude", lat), ("longitude", lon)):
        resid = np.abs(axis / config.LATTICE - np.round(axis / config.LATTICE)).max()
        if resid > config.LATTICE_TOL:
            raise AssertionError(
                f"{name} off the 0.4 lattice by {resid:.3e} "
                f"(tolerance {config.LATTICE_TOL})"
            )


def assert_axis(ds: xr.Dataset, expected_times) -> None:
    delivered = set(np.asarray(ds["time"].values).astype("datetime64[m]").tolist())
    expected = set(np.asarray(expected_times, dtype="datetime64[m]").tolist())
    missing = sorted(expected - delivered)
    extra = sorted(delivered - expected)
    if missing or extra:
        raise AssertionError(
            f"delivered axis != requested axis; missing={missing[:5]} extra={extra[:5]}"
        )


# --------------------------------------------------------------- request build


def _base(variables, dates, times):
    return {
        "variable": list(variables),
        "date": [dates],
        "time": list(times),
        "data_format": "netcdf_zip",
        "area": list(config.AREA),
    }


def analysis_requests(axis: Axis) -> list[dict]:
    """R1a-d (§0.4). Segmented by variable AND by date."""
    window = f"{axis.start}/{axis.d_prev}"
    newest = f"{axis.d_an}/{axis.d_an}"
    reqs = []
    for variables in (config.AEROSOL_VARS, config.WIND_VARS):
        reqs.append({**_base(variables, window, axis.h_win),
                     "type": ["analysis"], "leadtime_hour": ["0"]})
    if axis.h_new != axis.h_win or axis.d_an != axis.d_prev:
        for variables in (config.AEROSOL_VARS, config.WIND_VARS):
            reqs.append({**_base(variables, newest, axis.h_new),
                         "type": ["analysis"], "leadtime_hour": ["0"]})
    return reqs


def forecast_request(axis: Axis) -> dict:
    """R2 (§0.4): the newest run, 3-hourly to +120 h."""
    return {
        **_base(list(config.AEROSOL_VARS) + list(config.WIND_VARS),
                f"{axis.run_date}/{axis.run_date}", [axis.run_hour]),
        "type": ["forecast"],
        "leadtime_hour": [str(h) for h in range(0, 121, 3)],
    }


def requested_forecast_times(axis: Axis, request: dict) -> list[dt.datetime]:
    """The valid times the forecast request asks for: ``t_init + leadtime``.

    Derived from the probe's ``t_init`` and the request we are about to send,
    never from a delivered file -- so ``assert_axis(forecast, ...)`` catches a
    forecast payload from an older run, which nothing checked before.
    """
    return sorted(
        axis.t_init + dt.timedelta(hours=int(h)) for h in request["leadtime_hour"]
    )


# --------------------------------------------------------------- the arm merge
#
# Forecast lead 0 IS the analysis initial state -- the same field, routed
# through a second ADS request. Any difference is an encoding artifact, not
# meteorology, so the gate is an absolute-tolerance identity test on the
# WORST cell, never a central statistic.
#
# Why not the spec's `median(d) <= 0.05 * p99(analysis)` (design §0.4): smoke
# is sparse, so a lead-0 field that is wrong over every plume and right over
# the clean-air majority has median|d| == 0 exactly. Measured per step on both
# cached cubes, perturbing only the top 5% of cells by x3 + 0.5 gives
# median|d| = 0.000000 at max|d| = 0.6675..1.9163 (Greece, 28 of 546 cells)
# and 0.8541..4.3310 (Iberia, 41 of 816 cells) -- a median gate PASSES every
# one of them, against thresholds of 0.008838 and 0.023380. It is also ill-
# conditioned at the low end: `max(p99, 1e-12)` lets the threshold fall to
# 5e-14, below float32 noise, so a clean-air domain false-alarms.
# NOTE for the integrator: this supersedes the snippet in design §0.4, which
# must be amended to match rather than left contradicting this module.
#
# Sizing of the replacement, measured:
#   * both cached deliveries carry omaod550 as unpacked float32
#     (encoding dtype=float32, scale_factor=None), so an identical field
#     differs by exactly 0.0;
#   * the worst plausible artifact is CF int16 packing on one arm only:
#     range/65534 = 3.1e-5 for the observed 0..2 AOD span;
#   * LEAD0_ATOL = 1e-3 is ~32x that quantum, and floors the threshold so a
#     clean-air domain (p99.9 -> 0) cannot false-alarm;
#   * LEAD0_RTOL * p99.9 adds headroom on unusually bright fields without
#     ever letting the threshold collapse toward zero.
LEAD0_ATOL = 1e-3
LEAD0_RTOL = 1e-3

# The identity premise above holds only when ADS serves both arms from the
# same physical cycle. [measured on the 2026-08-05 delivery] the analysis at
# t_init carries ASSIMILATION INCREMENTS the same-hour forecast init does not:
# max|d| = 0.0066 on a field of p99.9 = 0.156 (4.3% of scale), 93% of cells
# differing at the 1e-6 level, u10 up to 0.12 m/s -- smooth, field-wide, and
# unmistakably the same weather. So the gate has two bands: exact identity
# (packing noise) passes as before, a bounded smooth increment is recorded as
# `assimilation_increment`, and anything beyond it still fails hard. A WRONG
# run does not depend on this gate alone: fetch.run pins the forecast arm's
# delivered axis to this run's request before merge_arms ever sees it.
LEAD0_INCREMENT_MAX_FRAC = 0.10    # max|d| ceiling, as a fraction of p99.9|a|
LEAD0_INCREMENT_P999_FRAC = 0.05   # p99.9|d| ceiling, same scale


def _time_keys(values) -> np.ndarray:
    """Integer-nanosecond keys for a time axis.

    ``np.asarray(x, dtype='datetime64[ns]').tolist()`` yields **ints**, not
    ``datetime`` objects, because ns precision overflows ``datetime``. Building
    a set from that and then testing ``np.datetime64(t) in ...`` silently never
    matches, and ``np.datetime64(some_int)`` raises outright. Keep one integer
    representation throughout and format only at the edges.

    (``assert_axis`` above uses the same ``.tolist()`` pattern, where it is SAFE
    only because the values are cast to ``datetime64[m]`` first. Do not
    'simplify' that cast away.)
    """
    return np.asarray(values, dtype="datetime64[ns]").astype("int64")


def _fmt(key) -> str:
    return str(np.datetime64(int(key), "ns").astype("datetime64[m]"))


def lead0_identity(a: np.ndarray, f: np.ndarray) -> dict:
    """Statistics for the analysis-vs-forecast-lead-0 identity gate.

    ``max`` is what the gate tests; the rest is what makes a failure
    diagnosable from the build report alone. ``n_cells`` is load-bearing, not
    decoration: a comparison with no finite cells produces a guarded ``0.0``
    everywhere and must be rejected by the caller, never read as agreement.
    """
    a = np.asarray(a, dtype="float64")
    f = np.asarray(f, dtype="float64")
    if a.shape != f.shape:
        raise AssertionError(f"lead 0 shape {f.shape} != analysis shape {a.shape}")
    d = np.abs(f - a)
    finite = np.isfinite(d)
    n_cells = int(finite.sum())
    n_total = int(d.size)
    scale = float(np.nanpercentile(np.abs(a), 99.9)) if n_cells else 0.0
    threshold = LEAD0_ATOL + LEAD0_RTOL * scale
    return {
        "max": float(np.nanmax(d)) if n_cells else 0.0,
        "median": float(np.nanmedian(d)) if n_cells else 0.0,
        "p99_9": float(np.nanpercentile(d, 99.9)) if n_cells else 0.0,
        "scale_p99_9": scale,
        "threshold": threshold,
        "n_cells": n_cells,
        "n_total": n_total,
        "n_differing": int((d[finite] > threshold).sum()) if n_cells else 0,
        "n_nonfinite": n_total - n_cells,
    }


def merge_arms(analysis: xr.Dataset, forecast: xr.Dataset, t_now, t_init,
               var: str = "omaod550"):
    """Join the two arms with an explicit analysis-wins priority (§0.4).

    Priority is enforced twice, deliberately. Clipping the forecast to
    ``valid_time > t_now`` removes every collision by construction; the arm
    tag plus lexsort is a second line of defence that still holds if a future
    caller widens the clip. ``np.unique(return_index=True)`` keeps the FIRST
    occurrence, so tagging analysis 0 and forecast 1 makes the analysis record
    provably the survivor rather than an accident of concatenation order.

    Every gate below raises AssertionError explicitly; see the note above
    ``assert_grid``.
    """
    t_now = np.datetime64(t_now, "ns")
    t_init = np.datetime64(t_init, "ns")
    t_now_k = int(t_now.astype("int64"))
    t_init_k = int(t_init.astype("int64"))

    an_set = set(_time_keys(analysis["time"].values).tolist())
    fc_set = set(_time_keys(forecast["time"].values).tolist())
    overlap = sorted(an_set & fc_set)

    report: dict = {
        "overlap": [_fmt(k) for k in overlap],
        "n_analysis": int(analysis.sizes["time"]),
        "n_forecast": int(forecast.sizes["time"]),
    }
    for k in overlap:
        if not (t_init_k <= k <= t_now_k):
            raise AssertionError(
                f"overlap at {_fmt(k)} lies outside [{_fmt(t_init_k)}, {_fmt(t_now_k)}]"
            )

    if t_init_k in an_set and t_init_k in fc_set:
        stats = lead0_identity(analysis.sel(time=t_init)[var].values,
                               forecast.sel(time=t_init)[var].values)
        report["lead0_identity"] = stats
        # An all-non-finite comparison has nothing to disagree about, so a
        # max-based gate would wave it through with max = 0.0. The plan's
        # median gate rejected it only by accident (median of NaN is NaN, and
        # NaN <= x is False); make that rejection deliberate instead of
        # losing it when the statistic changes.
        if stats["n_cells"] <= 0:
            raise AssertionError(
                f"lead 0 has no finite cells to compare (all {stats['n_total']} are "
                "non-finite) - the delivery is empty, not identical"
            )
        if stats["max"] <= stats["threshold"]:
            stats["verdict"] = "identical"
        else:
            inc_max = LEAD0_ATOL + LEAD0_INCREMENT_MAX_FRAC * stats["scale_p99_9"]
            inc_p999 = LEAD0_ATOL + LEAD0_INCREMENT_P999_FRAC * stats["scale_p99_9"]
            stats["increment_max_allowed"] = inc_max
            stats["increment_p99_9_allowed"] = inc_p999
            if stats["max"] <= inc_max and stats["p99_9"] <= inc_p999:
                stats["verdict"] = "assimilation_increment"
            else:
                raise AssertionError(
                    f"forecast lead 0 is not the analysis: max|d|={stats['max']:.6g} "
                    f"(allowed {inc_max:.6g}) or p99.9|d|={stats['p99_9']:.6g} "
                    f"(allowed {inc_p999:.6g}); {stats['n_differing']} of "
                    f"{stats['n_cells']} cells differ beyond even an assimilation "
                    "increment - revisit the arm definitions"
                )

    forecast = forecast.sel(time=forecast["time"] > t_now)
    if set(_time_keys(forecast["time"].values).tolist()) & an_set:
        raise AssertionError("clip failed: a forecast step at or before t_now survived")

    analysis = analysis.assign_coords(
        arm=("time", np.full(analysis.sizes["time"], "an", dtype="<U2")))
    forecast = forecast.assign_coords(
        arm=("time", np.full(forecast.sizes["time"], "fc", dtype="<U2")))

    merged = xr.concat([analysis, forecast], dim="time", coords="minimal",
                       compat="override", join="exact")
    order = np.lexsort((np.where(merged["arm"].values == "an", 0, 1),
                        _time_keys(merged["time"].values)))
    merged = merged.isel(time=order)
    _, keep = np.unique(_time_keys(merged["time"].values), return_index=True)
    merged = merged.isel(time=np.sort(keep))
    report["n_merged"] = int(merged.sizes["time"])
    return merged, report
