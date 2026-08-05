"""A cos-latitude-correct replacement for the engine's ``_dem_hillshade`` (S2).

The engine's version, read out of the bytecode, is

    alt   = math.radians(altitude_deg)
    az    = math.radians(360.0 - azimuth_deg + 90.0)
    gy,gx = np.gradient(h.astype(np.float32))        # INDEX units, no spacing
    dzdx, dzdy = gx * exaggeration, gy * exaggeration
    slope  = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    return clip(sin(alt)*cos(slope) + cos(alt)*sin(slope)*cos(az-aspect), 0, 1)

``np.gradient`` in index units answers "metres of rise per PIXEL", and on a
web-mercator raster a pixel is ``merc_per_px * cos(lat)`` metres of ground.  So
identical ground terrain produces a gradient proportional to ``cos(lat)``:
measured on a lat 30..72N raster, the mean |grad| of one identical ridge patch
is ~2.79x larger in the south than in the north, against the 2.7999x ratio of
the spacings themselves.  Dividing by the per-row ground spacing removes exactly
that factor and nothing else -- the same measurement after the division is
0.997.

Units warning, and it is load-bearing.  After the division ``exaggeration`` is a
true dimensionless z-factor (rise over run), not the engine's per-pixel number.
The engine's 5.5 in index units is equivalent to a z-factor of
``5.5 * ground_metres_per_pixel`` -- 1320 on the Greece z9 DEM, ~7000 on a
Europe z7 one -- which is why the engine's shading is saturated and degenerates
to pure aspect shading: on the non-flat pixels of the real z9 DEM its slope runs
p10 85.8 / p50 89.6 / p90 89.9 deg, while the corrected form at z-factor 5.5
runs p10 3.3 / p50 32.8 / p90 62.7 deg.  That saturation is also what *masks*
the bug in the rendered image -- equalising the gradients only shows up once the
slope term is off the plateau -- so a look sweep and this correction are not
independent.  S2 already requires the exaggeration sweep to be re-run once this
correction lands; ``config.HILLSHADE_Z_FACTOR`` is the single knob that sweep
turns, and gate 9 holds across the whole measured range (z 1..100 ->
0.0816..0.0848%), so recovering contrast cannot re-break it.

The engine signature is ``(h, exaggeration, azimuth_deg, altitude_deg)`` -- no
latitude, no bounds, no transform -- so the replacement closes over the DEM's
mercator bounds and shape, and ``basemap.install_hillshade_patch`` captures
those from the engine itself at the one moment they exist.
"""
from __future__ import annotations

import math

import numpy as np

# EPSG:3857 sphere radius. _build_terrarium_dem writes crs="EPSG:3857"
# [verified in the bytecode and against the cached DEM], so this is the
# raster's own datum, not a choice.
EARTH_R = 6378137.0


def _lat_from_merc_y(y_m) -> np.ndarray:
    """Inverse web-mercator: northing in metres -> latitude in degrees."""
    return np.degrees(2.0 * np.arctan(np.exp(np.asarray(y_m, dtype="float64") / EARTH_R))
                      - math.pi / 2.0)


def row_latitudes(bounds_mercator, shape) -> np.ndarray:
    """Latitude of each row centre of a north-up mercator raster."""
    _, y0, _, y1 = (float(v) for v in bounds_mercator)
    nrows = int(shape[0])
    merc_per_row = (y1 - y0) / float(nrows)
    # row 0 is the NORTH edge; sample row centres
    return _lat_from_merc_y(y1 - merc_per_row * (np.arange(nrows) + 0.5))


def row_ground_metres(bounds_mercator, shape) -> np.ndarray:
    """Ground metres per pixel along the y (row) axis, one value per row."""
    _, y0, _, y1 = (float(v) for v in bounds_mercator)
    merc_per_row = (y1 - y0) / float(int(shape[0]))
    return merc_per_row * np.cos(np.radians(row_latitudes(bounds_mercator, shape)))


def col_ground_metres(bounds_mercator, shape) -> np.ndarray:
    """Ground metres per pixel along the x (column) axis, one value per row.

    Separate from :func:`row_ground_metres` because ``_build_terrarium_dem``
    rounds the output height and width independently (two calls of
    ``max(1, round(dem.shape[i] / scale))``) before ``from_bounds``, so mercator
    pixels are not exactly square.

    The effect is SMALL, and this function is a correctness guard rather than a
    lever: on the delivered Europe z7 geometry -- a 9728x8960 tile mosaic
    rescaled to (4000, 3684) -- the two spacings are 2974.317645 and
    2974.487615 mercator metres, a difference of **0.0057%** [measured].  Keep
    the split (it is free, and nothing guarantees a future zoom or
    ``max_dem_size`` stays this close), but do not expect it to move a pixel.
    Mercator is conformal, so both axes carry the identical ``cos(lat)`` factor
    -- only the mercator-metres-per-pixel term differs.
    """
    x0, _, x1, _ = (float(v) for v in bounds_mercator)
    merc_per_col = (x1 - x0) / float(int(shape[1]))
    return merc_per_col * np.cos(np.radians(row_latitudes(bounds_mercator, shape)))


LOWLAND_MAX_M = 300.0
LOWLAND_RELIEF = 0.15
TARGET_HIGHLAND_SHARE = 0.21
HIGHLAND_SHARE_RANGE = (0.18, 0.24)


def _lon_from_merc_x(x_m) -> np.ndarray:
    return np.degrees(np.asarray(x_m, dtype="float64") / EARTH_R)


def _smootherstep01(x) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype="float32"), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def ground_tiers(dem, valid, bounds_mercator, display_window):
    """Measure the three-tier relief field from the delivered DEM.

    Lowlands (below ``LOWLAND_MAX_M``) keep ``LOWLAND_RELIEF`` of the full
    relief; the highland threshold ``h1`` is the elevation quantile that puts
    ``TARGET_HIGHLAND_SHARE`` of the DISPLAY-window pixels at full relief, and
    the ramp between them is a smootherstep. Returns
    ``(multiplier, h1_m, highland_share)``.
    """
    dem = np.asarray(dem, dtype="float32")
    valid = np.asarray(valid, dtype=bool)
    if dem.shape != valid.shape or dem.ndim != 2:
        raise ValueError(f"DEM/valid shape mismatch: {dem.shape} vs {valid.shape}")

    rows = row_latitudes(bounds_mercator, dem.shape)
    x0, _, x1, _ = (float(v) for v in bounds_mercator)
    xs = x0 + (np.arange(dem.shape[1]) + 0.5) * (x1 - x0) / dem.shape[1]
    cols = _lon_from_merc_x(xs)
    west, south, east, north = display_window
    display = ((rows[:, None] >= south) & (rows[:, None] <= north)
               & (cols[None, :] >= west) & (cols[None, :] <= east))
    display_pixels = int(np.count_nonzero(display))
    land = display & valid & np.isfinite(dem)
    land_values = dem[land]
    target_pixels = int(round(TARGET_HIGHLAND_SHARE * display_pixels))
    if display_pixels == 0 or target_pixels <= 0 or target_pixels >= land_values.size:
        raise ValueError("display/land coverage cannot support the 21% highland target")

    q = 1.0 - target_pixels / float(land_values.size)
    h1 = float(np.quantile(land_values, q, method="linear"))
    if h1 <= LOWLAND_MAX_M:
        raise ValueError(f"measured h1 {h1:.1f} m is not above 300 m")

    ramp = _smootherstep01((dem - LOWLAND_MAX_M) / (h1 - LOWLAND_MAX_M))
    multiplier = LOWLAND_RELIEF + (1.0 - LOWLAND_RELIEF) * ramp
    multiplier = np.where(valid, multiplier, 0.0).astype("float32")
    share = float(np.count_nonzero(display & valid & (dem >= h1)) / display_pixels)
    if not HIGHLAND_SHARE_RANGE[0] <= share <= HIGHLAND_SHARE_RANGE[1]:
        raise AssertionError(f"measured highland share {share:.4f} is outside 0.18..0.24")
    return multiplier, h1, share


class CosLatHillshade:
    """Drop-in replacement for the engine's ``_dem_hillshade``.

    Instances are callable with the engine's exact positional signature.  The
    public attributes exist so the production render can *prove* the
    replacement was invoked rather than merely installed:

        is_coslat  -- marker; the engine's own function does not have it
        calls      -- number of times the engine actually invoked this object
        bounds     -- the mercator bounds it was built for
        shape      -- the DEM shape it was built for
    """

    is_coslat = True

    def __init__(self, bounds_mercator, shape, relief_multiplier=None,
                 h1_m=None, highland_share=None):
        self.bounds = tuple(float(v) for v in bounds_mercator)
        self.shape = (int(shape[0]), int(shape[1]))
        self.row_metres = row_ground_metres(self.bounds, self.shape)
        self.col_metres = col_ground_metres(self.bounds, self.shape)
        self.latitudes = row_latitudes(self.bounds, self.shape)
        self.calls = 0
        self.relief_multiplier = (
            None if relief_multiplier is None
            else np.asarray(relief_multiplier, dtype="float32")
        )
        if self.relief_multiplier is not None and self.relief_multiplier.shape != self.shape:
            raise ValueError(
                f"relief multiplier shape {self.relief_multiplier.shape} != DEM {self.shape}"
            )
        self.h1_m = None if h1_m is None else float(h1_m)
        self.highland_share = None if highland_share is None else float(highland_share)
        self._validate()

    @classmethod
    def _from_spacings(cls, row_metres, col_metres, ncols: int | None = None
                       ) -> "CosLatHillshade":
        """Build directly from ground spacings, bypassing the mercator maths.

        Both arrays are indexed by ROW: ``col_metres[i]`` is the ground metres
        per COLUMN at row ``i``'s latitude.  Used by
        :func:`ruggedness_uniformity` to shade a synthetic band at one uniform
        spacing.
        """
        obj = cls.__new__(cls)
        obj.row_metres = np.asarray(row_metres, dtype="float64")
        obj.col_metres = np.asarray(col_metres, dtype="float64")
        if obj.col_metres.size != obj.row_metres.size:
            raise ValueError("row_metres and col_metres are both per-row and must "
                             f"match: {obj.row_metres.size} vs {obj.col_metres.size}")
        obj.shape = (obj.row_metres.size,
                     int(ncols) if ncols is not None else obj.row_metres.size)
        obj.bounds = None
        obj.latitudes = None
        obj.calls = 0
        obj.relief_multiplier = None
        obj.h1_m = None
        obj.highland_share = None
        obj._validate()
        return obj

    def _validate(self) -> None:
        for name, arr in (("row", self.row_metres), ("col", self.col_metres)):
            if not np.all(np.isfinite(arr)) or float(np.min(arr)) <= 0.0:
                raise ValueError(
                    f"degenerate {name} ground spacing for bounds {self.bounds} "
                    f"shape {self.shape}: min={float(np.min(arr))!r}")

    @property
    def latitude_span_ratio(self) -> float:
        """south/north ground-metres ratio: the bias this replacement removes."""
        return float(self.row_metres[-1] / self.row_metres[0])

    def __call__(self, h, exaggeration, azimuth_deg, altitude_deg) -> np.ndarray:
        arr = np.asarray(h, dtype="float32")
        if arr.shape[0] != self.row_metres.size:
            raise ValueError(
                f"cos-latitude hillshade was built for a {self.shape} DEM but was "
                f"handed {arr.shape}; the captured mercator bounds no longer describe "
                "this array, so the correction would be wrong. Re-install the patch."
            )
        self.calls += 1

        gy, gx = np.gradient(arr)
        # index-unit gradient -> true rise/run, per row
        dzdx = np.float32(exaggeration) * gx / self.col_metres.astype("float32")[:, None]
        dzdy = np.float32(exaggeration) * gy / self.row_metres.astype("float32")[:, None]

        # from here down this is the engine's own code, verbatim: math.radians
        # for the angles, np.sin/np.cos for the scalars (which promotes the
        # combination to float64 before the final cast), the 360-az+90 spelling,
        # the -dzdx in arctan2, and the clip.
        slope = np.arctan(np.hypot(dzdx, dzdy))
        aspect = np.arctan2(dzdy, -dzdx)
        alt = math.radians(float(altitude_deg))
        az = math.radians(360.0 - float(azimuth_deg) + 90.0)
        shaded = (np.sin(alt) * np.cos(slope)
                  + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
        # measured three-tier relief: scale the shading contrast around the
        # flat-plane illumination sin(alt), so lowlands flatten toward the base
        # tone instead of darkening
        flat = np.float32(np.sin(alt))
        shaded = np.clip(shaded, 0.0, 1.0).astype("float32")
        if self.relief_multiplier is not None:
            shaded = flat + self.relief_multiplier * (shaded - flat)
        return np.clip(shaded, 0.0, 1.0).astype("float32")


def make_coslat_hillshade(bounds_mercator, shape, *, dem=None, valid=None,
                          display_window=None) -> CosLatHillshade:
    """Build a drop-in replacement for the engine's ``_dem_hillshade``.

    With ``dem``/``valid`` (always together) the replacement carries the
    measured three-tier relief field; without them it is the plain cos-latitude
    correction the synthetic gates use.
    """
    if (dem is None) != (valid is None):
        raise ValueError("dem and valid must be provided together")
    if dem is None:
        return CosLatHillshade(bounds_mercator, shape)
    if display_window is None:
        raise ValueError("display_window is required when measuring ground tiers")
    relief, h1, share = ground_tiers(dem, valid, bounds_mercator, display_window)
    return CosLatHillshade(
        bounds_mercator, shape,
        relief_multiplier=relief, h1_m=h1, highland_share=share,
    )


def ridge_patch(row_spacing_m: float, col_spacing_m: float, *, side_m: float,
                wavelength_m: float, amplitude_m: float) -> np.ndarray:
    """A square ground patch of ridges, sampled at one uniform spacing.

    ``side_m`` is a GROUND extent, so the same patch sampled at two spacings
    covers the same terrain and the same number of periods -- only the sampling
    rate differs.  That is exactly the comparison gate 9 needs.
    """
    n_rows = max(8, int(round(side_m / float(row_spacing_m))))
    n_cols = max(8, int(round(side_m / float(col_spacing_m))))
    y_m = (np.arange(n_rows) * float(row_spacing_m))[:, None]
    x_m = (np.arange(n_cols) * float(col_spacing_m))[None, :]
    return (amplitude_m * np.sin(2 * np.pi * x_m / wavelength_m)
            + amplitude_m * np.sin(2 * np.pi * y_m / wavelength_m)).astype("float32")


def ruggedness_uniformity(bounds_mercator, shape, exaggeration: float,
                          azimuth_deg: float, altitude_deg: float, *,
                          shade=None, periods: int = 4,
                          samples_per_period: int = 96, guard_periods: int = 1,
                          amplitude_m: float = 200.0) -> dict:
    """Gate 9: does identical GROUND terrain read equally rugged at both ends?

    Renders one synthetic ridge patch twice -- once at the raster's northernmost
    row spacing, once at its southernmost -- and reports the relative difference
    in rendered ruggedness.  ``shade`` defaults to the cos-latitude replacement
    built for each sampling; pass the engine's own ``_dem_hillshade`` to prove
    the metric has teeth.

    Three construction choices, each closing a way the naive version lies:

    * The patch is defined by a GROUND extent, not a pixel count, so both
      samplings cover the same terrain and the same whole number of periods.
      A fixed pixel patch covers different ground at the two ends and lands on
      different phases of the ridge -- on the cached Greece DEM (227 vs 253
      m/px) a 192 px patch spans 0.145 of a 300 km wavelength, and the phase
      mismatch alone reads as a 17% "ruggedness" difference.
    * The wavelength is set from the COARSER spacing, so the field is resolved
      at >= ``samples_per_period`` px per wavelength at both ends.  A fixed
      short wavelength aliases in the south: 20 km ridges on a lat 30..72N /
      800-row raster are 2.24 px per wavelength there against 6.3 in the north,
      and the two bands are then compared through a 2.8x different
      central-difference transfer.
    * Each sampling is uniform, so no spurious y-gradient is introduced.  A
      single global field cannot carry a latitude-independent ground gradient
      on a mercator raster without one.

    ``guard_periods`` of margin are built and then cropped away, and the std is
    taken over exactly ``periods`` whole periods of the interior.  Without that
    crop, a fixed pixel border trims a different fraction of each patch and
    lands on a different phase: at 48 samples per period the Europe span reads
    0.94% with a fixed 2 px trim against 0.2815% with the interior crop, i.e.
    the trim was most of the signal.

    Measured on the lat 30..72N display window at the defaults (669.6 km
    wavelength, 285 px/wavelength north against 102 south): corrected 0.0816%,
    the engine's index-unit form 54.5201% worst-case over amplitude.  On the
    delivered z7 Europe geometry (4000x3684, 4.4832x spacing bias): corrected
    0.0864%, engine 70.5252%.  Cost at that size, 4-amplitude sweep: 1.18 s.
    """
    row_m = row_ground_metres(bounds_mercator, shape)
    col_m = col_ground_metres(bounds_mercator, shape)
    coarse = max(float(row_m[0]), float(row_m[-1]), float(col_m[0]), float(col_m[-1]))
    wavelength_m = coarse * float(samples_per_period)
    side_m = wavelength_m * float(periods + 2 * guard_periods)

    out = {}
    for name, i in (("north", 0), ("south", -1)):
        spy, spx = float(row_m[i]), float(col_m[i])
        field = ridge_patch(spy, spx, side_m=side_m, wavelength_m=wavelength_m,
                            amplitude_m=amplitude_m)
        fn = shade
        if fn is None:
            n = field.shape[0]
            fn = CosLatHillshade._from_spacings(np.full(n, spy), np.full(n, spx),
                                                field.shape[1])
        img = np.asarray(fn(field, exaggeration, azimuth_deg, altitude_deg))
        r0 = int(round(guard_periods * wavelength_m / spy))
        r1 = int(round((guard_periods + periods) * wavelength_m / spy))
        c0 = int(round(guard_periods * wavelength_m / spx))
        c1 = int(round((guard_periods + periods) * wavelength_m / spx))
        out[name] = {
            "std": float(img[r0:r1, c0:c1].std()),
            "px": list(field.shape),
            "scored_px": [r1 - r0, c1 - c0],
            "ground_metres_per_px": [spy, spx],
            "px_per_wavelength": wavelength_m / spx,
        }
    north, south = out["north"]["std"], out["south"]["std"]
    return {
        "north_std": north,
        "south_std": south,
        "rel_diff": abs(north - south) / max(north, south, 1e-12),
        "ground_metres_ratio": float(row_m[-1] / row_m[0]),
        "px_per_wavelength_north": out["north"]["px_per_wavelength"],
        "px_per_wavelength_south": out["south"]["px_per_wavelength"],
        "scored_px_north": out["north"]["scored_px"],
        "scored_px_south": out["south"]["scored_px"],
        "side_m": float(side_m),
        "wavelength_m": float(wavelength_m),
        "amplitude_m": float(amplitude_m),
    }


# amplitudes span the engine's whole response, from unsaturated (2 m over a
# ~335 km wavelength) to deep in the arctan plateau
GATE9_AMPLITUDES = (2.0, 15.0, 150.0, 1500.0)


def worst_ruggedness_uniformity(bounds_mercator, shape, exaggeration: float,
                                azimuth_deg: float, altitude_deg: float, *,
                                shade=None, amplitudes=GATE9_AMPLITUDES,
                                **kwargs) -> dict:
    """:func:`ruggedness_uniformity` over an amplitude sweep, worst case kept.

    One amplitude is not enough for a gate: the correction's residual is flat in
    amplitude (0.0794%..0.0816% on the Europe display span) but the engine's
    error is not.  Its sweep there runs 54.5201% / 3.3463% / 2.5108% / 0.3153%
    at 2 / 15 / 150 / 1500 m -- it collapses as the slope term saturates, so a
    gate pinned to a single large amplitude would pass the buggy form outright.
    """
    worst, sweep = None, {}
    for amp in amplitudes:
        metric = ruggedness_uniformity(bounds_mercator, shape, exaggeration,
                                       azimuth_deg, altitude_deg, shade=shade,
                                       amplitude_m=amp, **kwargs)
        sweep[amp] = metric["rel_diff"]
        if worst is None or metric["rel_diff"] > worst["rel_diff"]:
            worst = metric
    worst = dict(worst)
    worst["sweep_rel_diff"] = sweep
    return worst


def latitude_uniformity(bounds_mercator, shape, samples_per_period: float = 20.0,
                        slope_ratio: float = 0.015, exaggeration: float = 5.5,
                        azimuth_deg: float = 315.0, altitude_deg: float = 34.0,
                        tolerance: float = 0.20) -> dict:
    """Gate 9, instantiated on the geometry the basemap actually rendered.

    Builds a ridge field whose wavelength is constant in GROUND METRES, shades
    it with and without the cos-latitude correction, and compares the hillshade
    standard deviation of the northern and southern eighths.

    Two things about the construction are load-bearing, and both were found by
    running it rather than reasoning about it:

    * **The ridges run east-west, varying with row only.** The obvious
      construction -- ``cumsum`` a row-varying pixel spacing along the columns
      to get a north-south ridge of constant ground wavelength -- injects a
      *spurious meridional* gradient, because the accumulated phase at column
      ``c`` is proportional to ``c * m_per_px[row]`` and therefore drifts with
      row at a rate growing linearly with ``c``. Measured on the real
      4000x4241 geometry that construction reports a 31.5% north/south
      difference *after* the correction, against 38.5% before it -- i.e. it
      barely separates the corrected case from the broken one. Mercator is
      conformal, so the meridional and zonal scale factors are identical and
      an east-west ridge tests exactly the same correction with ``gx == 0``
      and no cross-term.
    * **The ridge is sized relative to the raster.** Ground metres per pixel
      span 4.5x across this domain, so a fixed 20 km wavelength aliases at the
      equatorward edge and the gate measures its own sampling error.
      ``samples_per_period`` is enforced against the COARSEST row; 20 keeps
      the central-difference ``sin(d)/d`` attenuation under 2% across the
      whole raster. The amplitude is set so peak ground slope is
      ``slope_ratio`` whatever the wavelength, which makes the statistic
      scale-free and keeps the shading off the 0/1 clip.

    This is the unit test's construction evaluated at the delivered DEM's real
    mercator bounds and pixel shape. It is *not* the Atlas-vs-Scandes terrain
    comparison of gate 9's prose: it proves the correction is right for this
    raster's geometry, not that two real mountain ranges happen to match. The
    returned dict says so, and ``build._gate9`` adds that it does not certify
    the delivered pixels either.
    """
    nrows, ncols = int(shape[0]), int(shape[1])
    m_per_px = row_ground_metres(bounds_mercator, (nrows, ncols))
    wavelength_m = float(samples_per_period * m_per_px.max())
    amplitude_m = float(slope_ratio * wavelength_m / (2.0 * math.pi))
    # 1-D cumulative ground distance down the meridian: depends on row only,
    # so it cannot manufacture an east-west gradient.
    y_m = np.cumsum(m_per_px)
    dem = np.tile(
        (amplitude_m * np.sin(2.0 * np.pi * y_m / wavelength_m))[:, None],
        (1, ncols))

    # The ridge varies with row only, so the band statistic is the std of a 1-D
    # profile. A band shorter than one period measures a fragment of a wave and
    # is not comparable end to end -- on a 320x180 probe that alone reports a
    # 50% difference with the correction fully applied. Decline instead of
    # reporting a false FAIL.
    band = max(nrows // 8, 1)
    period_rows_max = float(wavelength_m / m_per_px.min())
    resolved = band >= period_rows_max

    shade = make_coslat_hillshade(bounds_mercator, (nrows, ncols))(
        dem, exaggeration, azimuth_deg, altitude_deg)
    north = float(shade[:band].std())
    south = float(shade[-band:].std())
    rel = abs(north - south) / max(north, south, 1e-12)

    gy, gx = np.gradient(dem.astype("float64"))
    slope = np.arctan(exaggeration * np.hypot(gx, gy))
    un_north = float(slope[:band].std())
    un_south = float(slope[-band:].std())
    un_rel = abs(un_north - un_south) / max(un_north, un_south, 1e-12)

    lat_n = float(_lat_from_merc_y(np.array([bounds_mercator[3]]))[0])
    lat_s = float(_lat_from_merc_y(np.array([bounds_mercator[1]]))[0])
    return {
        "name": "hillshade latitude uniformity after the cos-lat correction",
        "corrected": {"north_std": north, "south_std": south, "rel_diff": rel},
        "uncorrected": {"north_std": un_north, "south_std": un_south,
                        "rel_diff": un_rel},
        "tolerance": tolerance,
        "lat_north": lat_n,
        "lat_south": lat_s,
        "shape": [nrows, ncols],
        "probe": {
            "wavelength_m": wavelength_m,
            "amplitude_m": amplitude_m,
            "samples_per_period_at_coarsest_row": samples_per_period,
            "ground_m_per_px": {"min": float(m_per_px.min()),
                                "max": float(m_per_px.max())},
            "band_rows": int(band),
            "period_rows_max": period_rows_max,
            "resolved": bool(resolved),
        },
        "method": (
            "synthetic constant-ground-wavelength ridge field on the delivered "
            "DEM geometry; not the Atlas-vs-Scandes terrain comparison of gate "
            "9's prose"
        ),
        "verdict": ("N/A" if not resolved
                    else "PASS" if rel < tolerance else "FAIL"),
        "reason": (None if resolved else
                   f"raster is {nrows} rows; the probe needs at least "
                   f"{int(math.ceil(period_rows_max * 8))} so an eighth-band "
                   "covers a full ridge period"),
        "control_has_teeth": bool(un_rel > tolerance),
    }
