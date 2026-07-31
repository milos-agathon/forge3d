# Iberia & SW France wildfire smoke — CAMS data-mode animated map

**Date:** 2026-07-31 · **Status:** approved design, pre-implementation

## Goal

A 15 s animated map video of wildfire smoke over Iberia (mainland Spain +
Portugal) and France up to Bordeaux for the last week of July 2026, driven by
the same data sources as the Copernicus Fire Emissions Watch app (per the
user-supplied screenshot), rendered with the forge3d wildfire smoke engine —
and visually better than the app: full-relief hillshade terrain, fire-anchored
glows, physically advected smoke, and wind streamline particles in the
established editorial plate style.

## Mandated data sources (all CAMS, via the Atmosphere Data Store)

| Layer | Dataset | Variable(s) | Notes |
| --- | --- | --- | --- |
| Smoke field | `cams-global-atmospheric-composition-forecasts` | `organic_matter_aerosol_optical_depth_550nm` (`omaod550`) | The app's AOD tracer for dates ≥ May 2026. 0.4°, 00/12 UTC runs, 3-hourly leadtimes assembled into a continuous series for 2026-07-24..31, bbox-subset. |
| Fires | NASA FIRMS VIIRS NRT (SNPP/NOAA-20/NOAA-21) | per-detection lat/lon/FRP | **Amended 2026-07-31:** ADS froze the GFAS dataset on 2025-12-03 (moved to the ECMWF Data Portal), so July 2026 is unavailable there. User-approved substitute: the FRP observations GFAS itself assimilates. Drives cluster positions, intensities, per-day activity, counters. |
| Wind | `cams-global-atmospheric-composition-forecasts` | `10m_u_component_of_wind`, `10m_v_component_of_wind` | Same dataset/runs as the AOD so every moving element is CAMS. Drives BOTH the streamline trails and any engine wind sampling. |

Credit line: `Copernicus Atmosphere Monitoring Service (CAMS) · ECMWF ·
NASA FIRMS (VIIRS) · forge3d` under the Copernicus licence.

### Credentials protocol (hard requirement)

- ADS API key is provided by the user **only via a temp file path**; it is
  never pasted into chat, never echoed to the terminal, never committed.
- Setup writes `~/.cdsapirc` (`url: https://ads.atmosphere.copernicus.eu/api`)
  by copying from the temp file with a command that prints nothing.
- The user deletes the temp file after setup. Fetched NetCDFs live under
  `examples/.cache/iberia_france_smoke/` (git-ignored).

## Region & framing

- Bbox lon −9.9 → 3.6, lat 35.7 → 45.5 (web-mercator, north-up). Bordeaux
  (44.84° N) inside the frame as the northern anchor.
- Land boundary = OSM union: mainland Spain (relation R1311341, cached) +
  mainland Portugal + France clipped to the bbox, all coastline-clipped via
  the osmdata land polygons (same method as the Spain fetcher).
- Full Terrarium DEM z9 hillshade relief for ALL visible land (Portugal and
  SW France are first-class smoke territory — no flat neighbor fills).
  Background beyond land: flat dark blue per the engine's LOOK layer.
- Portrait canvas 1080×1290, map top-anchored under the title block.

## Architecture

Two new files cloned from the proven Spain pattern; the shipped Spain
deliverable is untouched; the compiled engine
(`examples/wildfire_smoke_engine.cpython-313.pyc`) is used as-is.

1. `examples/iberia_france_data_fetch.py` — sub-commands `boundary`, `cams`
   (AOD + wind, via `cdsapi`), `gfas`, `all`. Converts CAMS NetCDF to the
   engine's `WindField` JSON schema for winds; AOD NetCDF is loaded directly
   by `AodField.from_file(path, aod_var="omaod550")`.
2. `examples/iberia_france_smoke_video.py` — wrapper: configures the engine
   (bbox, boundary, dates, layout), builds fire clusters from GFAS FRP grid
   cells (positions, intensity ∝ FRP, per-day activity), runs the engine
   data-mode smoke (`aod_field` + calibrated `aod_scale`, `AOD_CLOCK`
   "season"), composites layers, renders frames, encodes MP4.

New dev dependency: `cdsapi` (venv-only, not a crate/wheel dependency).

## Layers (bottom → top)

terrain hillshade → graticule/sea labels/scale bar → wind streamline trails
(`WindTrails` class reused from the Spain wrapper, alpha trimmed under smoke)
→ CAMS smoke layer → fire glow cores over smoke → city labels → title/
counters/footer.

## Counters (grounded only)

- Date/time UTC ticking through 2026-07-24..30.
- Cumulative VIIRS fire detections (FIRMS, real per-day counts).
- Current-day VIIRS detections.
- No burned-area estimate (not in the mandated data).

## Calibration & validation

- `aod_scale` calibrated by measurement, not guessed: render probe frames,
  target the reference band (smoke core luma ≈ 100–140, p99 ≤ ~130,
  final/peak ≤ 0.35, coverage sane) per the map-smoke skill gates.
- LOOK iterated with preview frames at ~5 checkpoints (eyeball), density
  chain re-measured after any knob change (knobs interact).
- Engine transfer constants are NOT edited (shared with data-mode goldens).

## Deliverable

450 frames = 15 s @ 30 fps, 1080×1290 MP4 → `D:/iberia_france_smoke/`.

## Out of scope

- Procedural FIRMS/Open-Meteo path (superseded by the CAMS mandate).
- MERRA-2 (latency excludes the last-few-days window).
- Any engine (.pyc) modification; any change to the Spain deliverable.

## Risks

- ADS request queue latency (minutes to hours at peak) — fetch early, cache.
- Dataset licences must be accepted in the ADS web UI before API access.
- `omaod550` variable naming in the delivered NetCDF may differ (e.g. long
  names / paramId); the fetch script verifies and passes `aod_var` explicitly.
- GFAS 0.1° cells are coarser than FIRMS points; cluster building bins by
  cell centroid — accepted (it is the app's own granularity).
