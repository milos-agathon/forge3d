"""Advection gain, offset clamping and the containment gate (§0.2, §5.5, §9.11).

The clamp acts on the OFFSET, never on the texture coordinate: clamping the
coordinate would paint a false plume edge inside the map whenever wind carries
smoke in from outside, and discarding would break the kernel's weight
normalisation and silently dim the plume.

Gate 11 has three parts and all three live here:

* ``static_bound()``   -- 11(a), the exact arithmetic bound, four inequalities.
* ``containment()``    -- 11(b), the empirical weighted exceedance fractions.
* ``debug_pass()``     -- 11(c), the per-pixel count of taps outside the data.
"""
from __future__ import annotations

import numpy as np

from . import config

# §0.2 causal kernel: exponential at the reference's measured 5.0697 h release
# half-life, truncated at 12 h (keeps 91.5% of the mass on the 6 h ladder).
KERNEL_LAGS = (0.0, 6.0, 12.0)
KERNEL_WEIGHTS = (0.6119, 0.2694, 0.1186)

D_LON = 11.6
D_LAT = 4.6
CURL_CLAMP_DEG = 0.4
# §0.1: the 5-tap blur reaches +/-2 texels; a texel is one 0.4 deg CAMS cell.
BLUR_TAP_DEG = 2 * config.LATTICE

GATE_FRAC_OVER_D = 0.005
GATE_FRAC_OVER_08D = 0.020

_KNEE = 0.8
_WIDTH = 0.2

# The lags that can actually engage the knee. Lag 0 carries an offset that is
# identically zero, so it can never exceed D; keeping it in the denominator
# would divide every reported fraction by a constant 1/(1-0.6119) = 2.5773 and
# report the dilution rather than the engagement rate. Gate 11(b) is defined
# over the engageable lags and renormalised across them. This is a definitional
# choice, and it matches the as-planned code, which already skipped lag 0 in
# both the sum and the norm -- the correction below is to the SPATIAL weight,
# not to this one.
_ENGAGEABLE = tuple((lag, w) for lag, w in zip(KERNEL_LAGS, KERNEL_WEIGHTS) if lag > 0.0)
_ENGAGEABLE_WEIGHT = sum(w for _, w in _ENGAGEABLE)


def soft_knee(r):
    """Exact below 0.8, C1 at the join, asymptotic to 1.0.

    Saturates to exactly 1.0 in float32 once r is around 4.5, so containment
    must be argued from the static bound in gate 11(a), not from r < 1.
    """
    r = np.asarray(r, dtype="float64")
    out = np.where(r <= _KNEE, r,
                   _KNEE + _WIDTH * (1.0 - np.exp(-(r - _KNEE) / _WIDTH)))
    return float(out) if out.ndim == 0 else out


def degree_offsets(u, v, lat_deg, hours: float):
    """Wind (m/s) -> (dlon, dlat) in degrees over ``hours``."""
    seconds = hours * 3600.0
    lat = np.asarray(lat_deg, dtype="float64")
    coslat = np.cos(np.radians(lat))
    if np.ndim(u) == 2 and lat.ndim == 1:
        coslat = coslat[:, None]
    dlat = np.asarray(v, dtype="float64") * seconds / 111_320.0
    dlon = np.asarray(u, dtype="float64") * seconds / (111_320.0 * coslat)
    return dlon, dlat


def clamp_offsets(dlon, dlat, d_lon: float = D_LON, d_lat: float = D_LAT):
    """Apply the per-axis soft knee. Returns clamped degree offsets."""
    return (np.sign(dlon) * soft_knee(np.abs(dlon) / d_lon) * d_lon,
            np.sign(dlat) * soft_knee(np.abs(dlat) / d_lat) * d_lat)


def fit_gain_from_scores(scores: dict[float, float], kmax: float) -> dict:
    """Pick the best-scoring gain, clamped to the validated cap."""
    best_k = max(scores, key=lambda k: scores[k])
    clamped = best_k > kmax
    k = min(best_k, kmax)
    return {
        "k": float(k),
        "k_unclamped": float(best_k),
        "clamped": bool(clamped),
        "score": float(scores[best_k]),
        "baseline_k0": float(scores.get(0.0, float("nan"))),
        "scores": {float(a): float(b) for a, b in scores.items()},
    }


def fit_gain(ds, kmax: float = config.MAX_ADVECTION_GAIN,
             candidates=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)) -> dict:
    """Fit k by advecting step n forward and correlating against step n+1."""
    from scipy.ndimage import map_coordinates

    aod = ds["omaod550"].values
    u = ds["u10"].values
    v = ds["v10"].values
    lat = ds["latitude"].values
    times = ds["time"].values

    scores: dict[float, float] = {}
    for k in candidates:
        corrs = []
        for n in range(aod.shape[0] - 1):
            hours = ((times[n + 1] - times[n]).astype("timedelta64[m]")
                     .astype("float64") / 60.0)
            dlon, dlat = degree_offsets(u[n] * k, v[n] * k, lat, hours)
            rows = np.arange(aod.shape[1])[:, None] - dlat / config.LATTICE
            cols = np.arange(aod.shape[2])[None, :] + dlon / config.LATTICE
            warped = map_coordinates(
                aod[n], [np.broadcast_to(rows, aod[n].shape),
                         np.broadcast_to(cols, aod[n].shape)],
                order=1, mode="nearest")
            a, b = warped.ravel(), aod[n + 1].ravel()
            if a.std() > 0 and b.std() > 0:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
        scores[float(k)] = float(np.mean(corrs)) if corrs else float("nan")
    return fit_gain_from_scores(scores, kmax)


# --------------------------------------------------------------------------
# gate 11(a): the static, exact bound
# --------------------------------------------------------------------------

def static_bound(area=None, display_window=None, d_lon: float = D_LON,
                 d_lat: float = D_LAT, curl: float = CURL_CLAMP_DEG,
                 blur_tap: float = BLUR_TAP_DEG) -> dict:
    """Gate 11(a): assert the four edge inequalities numerically.

    The clamped offset can reach exactly ``D`` (the soft knee saturates to 1.0
    in float32 around r = 4.5), so the bound is argued at saturation, not from
    a strict inequality. Worst reach from a display edge is

        D + C + B   =  11.6 + 0.4 + 0.8  =  12.8 deg lon
                       4.6 + 0.4 + 0.8   =   5.8 deg lat

    against a delivered bleed of 13.0 / 6.0. Every entry in ``edges`` carries
    the extreme tap coordinate and the slack; ``pass`` is the AND of all four.
    """
    area = config.AREA if area is None else area
    display_window = config.DISPLAY_WINDOW if display_window is None else display_window
    north, west, south, east = (float(x) for x in area)
    lon_min, lat_min, lon_max, lat_max = (float(x) for x in display_window)

    reach_lon = d_lon + curl + blur_tap
    reach_lat = d_lat + curl + blur_tap

    edges = {
        # name: (extreme tap coordinate, data edge)
        "west": (lon_min - reach_lon, west),
        "east": (lon_max + reach_lon, east),
        "south": (lat_min - reach_lat, south),
        "north": (lat_max + reach_lat, north),
    }
    out = {}
    for name, (tap, limit) in edges.items():
        slack = (tap - limit) if name in ("west", "south") else (limit - tap)
        out[name] = {
            "extreme_tap_deg": tap,
            "data_edge_deg": limit,
            "slack_deg": slack,
            "ok": bool(slack >= 0.0),
        }
    return {
        "reach_lon_deg": reach_lon,
        "reach_lat_deg": reach_lat,
        "bleed_lon_deg": min(lon_min - west, east - lon_max),
        "bleed_lat_deg": min(lat_min - south, north - lat_max),
        "terms": {"D_lon": d_lon, "D_lat": d_lat, "curl": curl, "blur_tap": blur_tap},
        "edges": out,
        "min_slack_deg": min(e["slack_deg"] for e in out.values()),
        "pass": bool(all(e["ok"] for e in out.values())),
    }


def static_containment() -> dict:
    """Gate 11(a) in the shape the build report consumes.

    MERGE NOTE. This is the same arithmetic as :func:`static_bound`, reported
    per-edge as (delivered margin, required margin, slack) rather than as
    (extreme tap coordinate, data edge, slack). Two independent fixes each
    added gate 11(a) with its own report shape, and both are kept: ``build.py``
    reads this one, ``gate11()`` reads ``static_bound``. They are cross-checked
    against each other by ``test_the_two_gate11a_shapes_agree``.

    Worst reachable sample is the display edge plus the saturated advection
    clamp D, plus the curl clamp C, plus the outermost blur tap. Assert the
    four inequalities numerically rather than arguing them from ``r < 1`` --
    the soft knee saturates to exactly 1.0*D in float32 once r is around 4.5,
    so strict inequality is not available (§0.2).
    """
    north, west, south, east = config.AREA
    lon_min, lat_min, lon_max, lat_max = config.DISPLAY_WINDOW
    need_lon = D_LON + CURL_CLAMP_DEG + BLUR_TAP_DEG
    need_lat = D_LAT + CURL_CLAMP_DEG + BLUR_TAP_DEG
    edges = {
        "west": (lon_min - west, need_lon),
        "east": (east - lon_max, need_lon),
        "south": (lat_min - south, need_lat),
        "north": (north - lat_max, need_lat),
    }
    checks = {
        name: {"delivered_margin_deg": float(have), "required_deg": float(need),
               "slack_deg": float(have - need), "ok": bool(have >= need)}
        for name, (have, need) in edges.items()
    }
    return {
        "name": "static sample containment",
        "terms": {"D_lon": D_LON, "D_lat": D_LAT, "curl_clamp": CURL_CLAMP_DEG,
                  "blur_tap": BLUR_TAP_DEG},
        "edges": checks,
        "min_slack_deg": min(c["slack_deg"] for c in checks.values()),
        "verdict": "PASS" if all(c["ok"] for c in checks.values()) else "FAIL",
    }


def assert_static_bound(**kwargs) -> dict:
    """``static_bound`` as a hard build-time check (§9.11 gate 11(a)).

    ``raise AssertionError``, not the ``assert`` statement: a bare ``assert``
    is compiled out under ``python -O`` and this gate would then silently
    return a failing verdict instead of stopping the build. Measured: with the
    statement form, ``python -O`` accepted rev 1's 0.2 deg bleed
    (min_slack -12.6 deg) without raising. The repo already carries this fix
    elsewhere -- "not assert: must survive python -O",
    ``examples/population_ghsl/france_population_pt_3d.py``.
    """
    res = static_bound(**kwargs)
    failed = {n: e for n, e in res["edges"].items() if not e["ok"]}
    if failed:
        raise AssertionError(
            "gate 11(a) FAILED: the clamped sample reaches outside the fetched "
            f"domain at {sorted(failed)}; details={failed}"
        )
    return res


# --------------------------------------------------------------------------
# shared helpers for 11(b) and 11(c)
# --------------------------------------------------------------------------

def _display_index(lat, lon, display_window):
    """Cell-centre indices of the display block.

    The comparison carries ``config.LATTICE_TOL``: 0.4 is not a binary
    fraction and the delivered CAMS coordinates land up to ~1e-13 off the
    lattice (§0.1), so a strict ``>=`` can drop an edge row of the block.
    """
    lon_min, lat_min, lon_max, lat_max = display_window
    tol = config.LATTICE_TOL
    ri = np.flatnonzero((lat >= lat_min - tol) & (lat <= lat_max + tol))
    ci = np.flatnonzero((lon >= lon_min - tol) & (lon <= lon_max + tol))
    if ri.size == 0 or ci.size == 0:
        raise ValueError(
            f"display window {display_window} selects no cells from the "
            f"delivered grid (lat {lat.min()}..{lat.max()}, "
            f"lon {lon.min()}..{lon.max()})"
        )
    return ri, ci


def mercator_screen_weights(lat_sel, n_lon: int) -> np.ndarray:
    """Per-display-pixel screen-area share on a linear Mercator, summing to 1.

    Screen y is ``ln(tan(pi/4 + phi/2))``, so ``dy/dphi = sec(phi)``: a row of
    the 0.4 deg lattice occupies screen height proportional to ``sec(phi)``.
    Screen x is linear in longitude, so all ``n_lon`` columns of a row are
    equal. Hence

        W[r, c] = sec(phi_r) / (n_lon * sum_r sec(phi_r))

    and ``W.sum() == 1`` over the display block. This is the whole point of the
    weighting: over the lon -25..45 / lat 30..72 window the 66-72N band holds
    23.998% of the screen against a 15.094% share of the latitude rows, so an
    unweighted count would under-report exactly the high-latitude pixels whose
    1/cos(phi) amplification produces the extreme degree offsets. (The exact
    Mercator integral over each 0.4 deg row gives 23.998% too -- the midpoint
    rule is accurate to 4e-6 here, so the approximation is not load-bearing.)
    """
    n_lon = int(n_lon)
    if n_lon <= 0:
        raise ValueError(f"n_lon must be positive, got {n_lon}")
    sec = np.asarray(lat_sel, dtype="float64")
    if sec.size == 0:
        raise ValueError("lat_sel is empty; the display block selects no rows")
    sec = 1.0 / np.cos(np.radians(sec))
    w = np.broadcast_to(sec[:, None], (sec.size, n_lon)).astype("float64")
    return w / w.sum()


def _wind_fields(ds):
    u = np.asarray(ds["u10"].values, dtype="float64")
    v = np.asarray(ds["v10"].values, dtype="float64")
    lat = np.asarray(ds["latitude"].values, dtype="float64")
    lon = np.asarray(ds["longitude"].values, dtype="float64")
    return u, v, lat, lon


# --------------------------------------------------------------------------
# gate 11(b): the empirical weighted exceedance
# --------------------------------------------------------------------------

def containment(ds, k: float = 1.0, display_window=None,
                d_lon: float = D_LON, d_lat: float = D_LAT) -> dict:
    """Gate 11(b): how often does a raw offset exceed D over display pixels?

    The reported numbers are TRUE fractions in [0, 1]. The sample space is
    ``(display pixel p, engageable lag l, time step n)`` and the weight is the
    product of three factors, each of which sums to 1 over its own axis:

    * ``W_screen(p)``  -- Mercator screen-area share (``mercator_screen_weights``),
      summing to 1 over the display block. The as-planned form broadcast a
      latitude-normalised vector across longitude, so it summed to ``n_lon``
      and inflated every fraction by exactly the display width: 175x on the
      Europe grid, measured 26x on the cached Greece cube and 34x on Iberia.
      "frac_over_D" was then a fraction in name only and could exceed 1 --
      168.26% on Greece at k = 1.5 [measured].
    * ``w_l / sum_l' w_l'`` over the ENGAGEABLE lags {6 h, 12 h}, unchanged
      from the as-planned code. Lag 0 has an identically zero offset.
    * ``1 / n_steps`` -- a uniform average over the delivered time axis.

    Consequence, and it is the test that pins the definition: a field in which
    every sample exceeds D returns exactly 1.0.

    ``k`` is NOT clamped here -- the §0.1 escalation study needs to probe above
    the cap -- but ``k_over_validated_cap`` records when it was. The build path
    must pass the clamped ``fit_gain()['k']``.
    """
    u, v, lat, lon = _wind_fields(ds)
    display_window = config.DISPLAY_WINDOW if display_window is None else display_window
    ri, ci = _display_index(lat, lon, display_window)

    w = mercator_screen_weights(lat[ri], ci.size)
    n_steps = u.shape[0]

    over_d = 0.0
    over_08 = 0.0
    per_lag = {}
    for lag, kw in _ENGAGEABLE:
        lag_d = 0.0
        lag_08 = 0.0
        for n in range(n_steps):
            dlon, dlat = degree_offsets(u[n] * k, v[n] * k, lat, lag)
            sub_lon = np.abs(dlon[np.ix_(ri, ci)])
            sub_lat = np.abs(dlat[np.ix_(ri, ci)])
            exceed_d = (sub_lon > d_lon) | (sub_lat > d_lat)
            exceed_08 = (sub_lon > 0.8 * d_lon) | (sub_lat > 0.8 * d_lat)
            lag_d += float((exceed_d * w).sum()) / n_steps
            lag_08 += float((exceed_08 * w).sum()) / n_steps
        per_lag[f"{lag:g}h"] = {"frac_over_D": lag_d, "frac_over_0.8D": lag_08}
        over_d += kw * lag_d
        over_08 += kw * lag_08

    frac_d = over_d / _ENGAGEABLE_WEIGHT
    frac_08 = over_08 / _ENGAGEABLE_WEIGHT
    return {
        "frac_over_D": frac_d,
        "frac_over_0.8D": frac_08,
        "pass": bool(frac_d <= GATE_FRAC_OVER_D and frac_08 <= GATE_FRAC_OVER_08D),
        "thresholds": {"over_D": GATE_FRAC_OVER_D, "over_0.8D": GATE_FRAC_OVER_08D},
        "k": float(k),
        "k_over_validated_cap": bool(k > config.MAX_ADVECTION_GAIN),
        "D": {"lon": d_lon, "lat": d_lat},
        "per_lag": per_lag,
        "display_block": (int(ri.size), int(ci.size)),
        # MERGE NOTE: published as a JSON-friendly list beside the tuple above
        # so a mis-selected block is visible rather than silently rescaling the
        # fraction -- the display width is a divisor of the weight.
        "display_block_shape": [int(ri.size), int(ci.size)],
        "display_block_expected": list(config.DISPLAY_BLOCK_SHAPE),
        "n_steps": int(n_steps),
        "weighting": ("mercator screen-area share over display pixels (sums to 1) "
                      "x kernel weight renormalised over the engageable lags "
                      "{6h, 12h} x uniform over time steps"),
    }


# --------------------------------------------------------------------------
# gate 11(c): the debug pass
# --------------------------------------------------------------------------

def sample_extent(coord, offset, extra: float):
    """Envelope of the outermost blur taps for a backward trace.

    ``x_s = coord - offset``; the curl clamp and the outer blur tap can each
    push a further ``extra`` degrees in EITHER direction, so the envelope is
    ``x_s -/+ extra``. Signing ``extra`` by the offset would silently exempt
    the ``offset == 0`` pixels, which are precisely the ones with no advective
    margin of their own.
    """
    x_s = coord - offset
    return x_s - extra, x_s + extra


def debug_pass(ds, k: float = 1.0, display_window=None, area=None,
               d_lon: float = D_LON, d_lat: float = D_LAT,
               curl: float = CURL_CLAMP_DEG,
               blur_tap: float = BLUR_TAP_DEG, halo: int = 1) -> dict:
    """Gate 11(c): count display pixels whose outermost tap leaves the data.

    Two counts, over ``(evaluated cell, engageable lag, time step)``:

    * ``preclamp_outside`` -- using the RAW offset. It counts the same
      population gate 11(b) measures, so it is 0 only when 11(b) is 0, and a
      non-zero value is the direct evidence that the soft knee is load-bearing
      rather than decorative. Report it; do not expect 0. (Measured: 54 of
      97920 Iberia samples at k = 1.5 leave the rectangle pre-clamp.)
    * ``postclamp_outside`` -- using the soft-knee-clamped offset. This is the
      hard invariant, guaranteed by gate 11(a), and it is what ``pass``
      reports; any non-zero value means 11(a)'s arithmetic and the delivered
      grid disagree and CLAMP_TO_EDGE would be reached in the shader.

    Coverage. The offset field is defined only at CAMS cell centres, but the
    shader samples every screen pixel in the continuous display window. Each
    cell is therefore tested over its whole half-cell footprint clipped to the
    window, and bilinear interpolation makes the offset at an intermediate
    point a convex combination of the surrounding cells -- so a max over the
    cells bounds the points between them.

    That argument needs a HALO. The display window's longitude edges are half
    a cell off the 0.4 lattice (-25.0 sits between the -25.2 and -24.8
    columns), so the outermost in-block cell centre is 0.2 deg inside the
    window and the screen pixel at lon -25.0 interpolates the in-block column
    at -24.8 against the column at -25.2, which is NOT in the display block.
    Evaluating only the display block misses that partner outright: a field
    whose extreme wind sits on the -25.2 column reports 0 taps outside with
    ``halo=0`` and 18 with ``halo=1`` [measured]. One ring is enough, because
    a cell more than one step outside the window cannot be a bilinear partner
    of any point inside it; its clipped footprint is empty and it is masked
    out. Pass ``halo=0`` only to reproduce the block-only count.
    """
    u, v, lat, lon = _wind_fields(ds)
    display_window = config.DISPLAY_WINDOW if display_window is None else display_window
    ri_d, ci_d = _display_index(lat, lon, display_window)
    lon_min, lat_min, lon_max, lat_max = (float(x) for x in display_window)

    if area is None:
        area = config.AREA
    north, west, south, east = (float(x) for x in area)

    halo = int(halo)
    ri = np.arange(max(int(ri_d[0]) - halo, 0), min(int(ri_d[-1]) + halo + 1, lat.size))
    ci = np.arange(max(int(ci_d[0]) - halo, 0), min(int(ci_d[-1]) + halo + 1, lon.size))

    half = 0.5 * config.LATTICE
    tol = config.LATTICE_TOL
    lon_g = lon[ci][None, :]
    lat_g = lat[ri][:, None]
    lon_a = np.maximum(lon_g - half, lon_min)
    lon_b = np.minimum(lon_g + half, lon_max)
    lat_a = np.maximum(lat_g - half, lat_min)
    lat_b = np.minimum(lat_g + half, lat_max)
    # A halo cell clips to a degenerate point on the window edge; 0.4 arithmetic
    # can order the two endpoints backwards by ~4e-15, so admit within tol and
    # then re-sort, rather than dropping the very cells the halo exists for.
    live = np.broadcast_to((lon_a <= lon_b + tol) & (lat_a <= lat_b + tol),
                           (ri.size, ci.size))
    lon_lo = np.minimum(lon_a, lon_b)
    lon_hi = np.maximum(lon_a, lon_b)
    lat_lo = np.minimum(lat_a, lat_b)
    lat_hi = np.maximum(lat_a, lat_b)
    extra = curl + blur_tap

    n_live = int(live.sum())
    if n_live == 0:
        raise ValueError(
            f"display window {display_window} yields no live footprint cells")

    counts = {"pre": 0, "post": 0}
    # signed overshoot: negative is spare margin, positive is degrees outside
    worst = {"pre": {"lon": -np.inf, "lat": -np.inf},
             "post": {"lon": -np.inf, "lat": -np.inf}}
    total = 0
    for lag, _kw in _ENGAGEABLE:
        for n in range(u.shape[0]):
            dlon, dlat = degree_offsets(u[n] * k, v[n] * k, lat, lag)
            raw_lon = dlon[np.ix_(ri, ci)]
            raw_lat = dlat[np.ix_(ri, ci)]
            cl_lon, cl_lat = clamp_offsets(raw_lon, raw_lat, d_lon, d_lat)

            for tag, o_lon, o_lat in (("pre", raw_lon, raw_lat),
                                      ("post", cl_lon, cl_lat)):
                sx_lo, _ = sample_extent(lon_lo, o_lon, extra)
                _, sx_hi = sample_extent(lon_hi, o_lon, extra)
                sy_lo, _ = sample_extent(lat_lo, o_lat, extra)
                _, sy_hi = sample_extent(lat_hi, o_lat, extra)
                over_lon = np.maximum(west - sx_lo, sx_hi - east)
                over_lat = np.maximum(south - sy_lo, sy_hi - north)
                bad = ((over_lon > 0.0) | (over_lat > 0.0)) & live
                counts[tag] += int(bad.sum())
                worst[tag]["lon"] = max(
                    worst[tag]["lon"],
                    float(np.where(live, over_lon, -np.inf).max()))
                worst[tag]["lat"] = max(
                    worst[tag]["lat"],
                    float(np.where(live, over_lat, -np.inf).max()))
            total += n_live
    pre, post = counts["pre"], counts["post"]

    return {
        "preclamp_outside": pre,
        "postclamp_outside": post,
        "samples": total,
        "preclamp_frac": pre / total if total else 0.0,
        # signed: negative is spare margin, positive is how far outside
        "worst_overshoot_deg": {"preclamp": worst["pre"], "postclamp": worst["post"]},
        "preclamp_clean": bool(pre == 0),
        "pass": bool(post == 0),
        "k": float(k),
        "area": (north, west, south, east),
        "display_block": (int(ri_d.size), int(ci_d.size)),
        "halo": halo,
        "evaluated_cells": n_live,
    }


def gate11(ds, k: float = 1.0, display_window=None, area=None,
           d_lon: float = D_LON, d_lat: float = D_LAT,
           curl: float = CURL_CLAMP_DEG,
           blur_tap: float = BLUR_TAP_DEG) -> dict:
    """All three parts of gate 11 in one call, for the build report.

    ``d_lon``/``d_lat`` are threaded through so the §0.1 escalation area
    (``[79.2, -41.6, 22.8, 61.6]`` with ``D`` = 15.2/5.8) can be gated as a
    whole rather than only through ``static_bound``.
    """
    a = static_bound(area=area, display_window=display_window, d_lon=d_lon,
                     d_lat=d_lat, curl=curl, blur_tap=blur_tap)
    b = containment(ds, k=k, display_window=display_window,
                    d_lon=d_lon, d_lat=d_lat)
    c = debug_pass(ds, k=k, display_window=display_window, area=area,
                   d_lon=d_lon, d_lat=d_lat, curl=curl, blur_tap=blur_tap)
    return {"static": a, "empirical": b, "debug": c,
            "pass": bool(a["pass"] and b["pass"] and c["pass"])}
