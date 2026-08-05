# tools/europe_smoke/config.py
"""Every constant the Europe smoke build depends on, in one place.

No logic beyond trivial derivations. If a number here disagrees with the spec,
the spec wins and this file is the bug.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- geometry
# §0.1. (N, W, S, E) as ADS orders them. Every edge is an exact multiple of
# 0.4 so the documented inward snap (N/E floor, S/W ceil) is a no-op.
AREA = (78.0, -38.0, 24.0, 58.0)
LATTICE = 0.4
LATTICE_TOL = 1e-9

# (lat, lon), matching the NetCDF axis order after flattening.
GRID_SHAPE = (136, 241)

# lon_min, lat_min, lon_max, lat_max. Everything the user ever sees.
DISPLAY_WINDOW = (-25.0, 30.0, 45.0, 72.0)
DISPLAY_BLOCK_SHAPE = (106, 175)

# §5.5 / gate 11(b): the area was validated at this wind gain.
MAX_ADVECTION_GAIN = 1.5

# ---------------------------------------------------------------- basemap §2
# install_hillshade_patch() divides the engine's index-unit gradient by the
# per-row GROUND metres, so `exaggeration` stops being "rise per pixel" and
# becomes a dimensionless z-factor. The engine's literal 5.5 is kept as the
# starting value so this task's diff is the geometry correction and nothing
# else -- but its MEANING has changed, so it must be named here rather than
# inherited from the engine, and §2's exaggeration sweep must be re-run on the
# real z7 DEM. Measured on the cached Greece DEM decimated to ~1200 m/px (the z7
# sampling proxy), against the engine's saturated std of 0.1884:
#     z  1.0 -> std 0.0315 (0.167x)   slope p90  3.9 deg
#     z  5.5 -> std 0.1122 (0.596x)   slope p90 20.4 deg   <-- this default
#     z 12.0 -> std 0.1463 (0.776x)   slope p90 39.1 deg
#     z 20.0 -> std 0.1612 (0.855x)   slope p90 53.6 deg
# Gate 9 holds across z 1..100 (0.0816%..0.0848% [measured]), so raising the
# z-factor to recover contrast cannot re-break the correction.
HILLSHADE_Z_FACTOR = 5.5
# gate 9 threshold on the rendered north/south ruggedness difference, worst case
# over the amplitude sweep. Two measurements, both on the shipped code:
#   * the lat 30..72N display window, 1024 rows (spacing bias 2.7999x):
#     engine 54.5201%, corrected 0.0816%.
#   * THE DELIVERED z7 EUROPE DEM, derived from the engine's own tile snap of
#     the AREA bbox -- tiles x 50..84 / y 18..55, a 9728x8960 mosaic rescaled to
#     shape (4000, 3684), spanning 78.059N..21.955N, spacing bias 4.4832x:
#     engine 70.5252%, corrected 0.0864%. 816x separation, 5.8x headroom.
# The corrected residual is the test field's own discretisation transfer (430
# px/wavelength north against 96 south), not the correction: it is flat in
# amplitude and falls to 0.0176% at samples_per_period=192.
HILLSHADE_UNIFORMITY_MAX = 0.005

# basemap.contain_fit solves for the scale that puts the binding edge of the
# engine's frame_rect exactly ON the unit square. Recomputing that edge inside
# the engine goes through a different sequence of multiplies and divides, so the
# result can land a few ULPs past 1.0 and fail gate 10 by ~2e-16. This margin
# backs the fit off by one part in a billion -- 4e-6 px on the 4000 px
# production canvas, seven orders of magnitude below a pixel and seven above the
# rounding it guards against.
FRAME_FIT_MARGIN = 1e-9

# ---------------------------------------------------------------- paths
BUILD_DIR = Path("D:/europe_smoke/build")
CACHE_DIR = BUILD_DIR / "cache"
REPORT_PATH = BUILD_DIR / "build_report.json"

ENGINE_PYC = REPO_ROOT / "examples" / "wildfire_smoke_engine.cpython-313.pyc"
GHSL_TIF = Path("D:/ghsl-population/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif")
OSM_LAND_ZIP = (
    REPO_ROOT / "examples" / ".cache" / "iberia_france_smoke" / "osm"
    / "simplified-land-polygons-complete-3857.zip"
)

CDSAPIRC = Path.home() / ".cdsapirc"
FIRMS_KEY = Path.home() / ".firms_key"

# §1.4: the engine's boundary filename is hardcoded to a California pattern;
# we hand it our own relation id so it loads our polygon instead of fetching
# relation 165475.
SENTINEL_RELATION_ID = 900000003

# ---------------------------------------------------------------- CAMS
ADS_DATASET = "cams-global-atmospheric-composition-forecasts"
ADS_CONSTRAINTS_URL = (
    "https://ads.atmosphere.copernicus.eu/api/retrieve/v1/processes/"
    f"{ADS_DATASET}/constraints"
)

AEROSOL_VARS = (
    "organic_matter_aerosol_optical_depth_550nm",
    "black_carbon_aerosol_optical_depth_550nm",
    "total_aerosol_optical_depth_550nm",
)
WIND_VARS = ("10m_u_component_of_wind", "10m_v_component_of_wind")

# §1.1 resolution rule: exact short name first, then a substring of the
# variable name or long_name. Canonical name -> (exact candidates, substrings).
VAR_RESOLUTION = {
    "omaod550": (("omaod550",), ("omaod", "organic")),
    "bcaod550": (("bcaod550",), ("bcaod", "black_carbon", "black carbon")),
    "aod550": (("aod550",), ("total_aerosol", "total aerosol")),
    "u10": (("u10", "10u", "u10m"), ("u_component",)),
    "v10": (("v10", "10v", "v10m"), ("v_component",)),
}

# §1.1: the ADS `variable` string -> the canonical NetCDF name. Lets an arm be
# held to exactly the variables it asked for; analysis is segmented by variable
# (§0.4 R1a-d), so the wind arm carries no aerosol and vice versa.
ADS_TO_CANONICAL = {
    "organic_matter_aerosol_optical_depth_550nm": "omaod550",
    "black_carbon_aerosol_optical_depth_550nm": "bcaod550",
    "total_aerosol_optical_depth_550nm": "aod550",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}

# omaod550 IS the map; u10/v10 drive advection, the trajectories, and the §5.5
# containment fit that validates the domain. Nothing renders without all three.
REQUIRED_VARS = ("omaod550", "u10", "v10")
# The two '?'-marked names of §1.1: documented ECMWF short names, unverified
# against a real delivery. They feed only the §7.2/§7.3 composition readouts,
# which §7 already requires to display "--" when they cannot be computed, and §8
# already contemplates dropping total_aod550 from the atlas. Their absence is a
# recorded shortfall, not a build failure.
OPTIONAL_VARS = ("bcaod550", "aod550")

UNIT_CLASS = {
    "omaod550": "dimensionless",
    "bcaod550": "dimensionless",
    "aod550": "dimensionless",
    "u10": "velocity",
    "v10": "velocity",
}
# '~' is ECMWF's dimensionless marker (§1.1); '' and '1' are the CF spellings.
DIMENSIONLESS_UNITS = frozenset({"~", "1", "", "dimensionless", "(0 - 1)"})
VELOCITY_UNITS = frozenset({"m s**-1", "m s^-1", "m s-1", "m/s", "ms**-1", "m.s-1"})
UNIT_CLASS_MEMBERS = {
    "dimensionless": DIMENSIONLESS_UNITS,
    "velocity": VELOCITY_UNITS,
}

# The two dim layouts a variable may legally have. Checked as a SET: flatten_time
# measurably reorders (forecast_period, forecast_reference_time, latitude,
# longitude) to (latitude, longitude, time) [measured on the cached delivery].
RAW_DIMS = frozenset({"forecast_period", "forecast_reference_time",
                      "latitude", "longitude"})
FLAT_DIMS = frozenset({"time", "latitude", "longitude"})
ACCEPTED_DIM_SETS = (RAW_DIMS, FLAT_DIMS)

# ---------------------------------------------------------------- FIRMS
FIRMS_SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
FIRMS_MAX_DAY_RANGE = 5  # §1.2, documented limit is 1..5, NOT 10
FIRMS_MAP_CAP = 80_000

USER_AGENT = "forge3d-europe-smoke/1.0"


def on_lattice(value: float, lattice: float = LATTICE, tol: float = LATTICE_TOL) -> bool:
    """True if ``value`` is an exact multiple of ``lattice`` within ``tol``.

    The tolerance is load-bearing: 0.4 is not a binary fraction, so even the
    literals in this file are off by ~1e-14 (§0.1).
    """
    q = value / lattice
    return abs(q - round(q)) <= tol


def grid_axes() -> tuple[np.ndarray, np.ndarray]:
    """The (lon ascending, lat descending) axes CAMS will deliver."""
    north, west, south, east = AREA
    lon = west + LATTICE * np.arange(round((east - west) / LATTICE) + 1)
    lat = north - LATTICE * np.arange(round((north - south) / LATTICE) + 1)
    return lon, lat
