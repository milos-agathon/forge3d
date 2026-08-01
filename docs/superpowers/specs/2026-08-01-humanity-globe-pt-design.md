# Humanity Globe — Path-Traced Rotating Population Globe

**Date:** 2026-08-01
**Status:** Approved design, pre-implementation
**Template lineage:** map-maker Archetype 3 (population spikes), legacy pre-OBLIQUA
light-field × overlay pipeline (`examples/population_ghsl/southeast_europe_population_pt_3d.py`,
`examples/population_ghsl/france_population_pt_3d.py`)

## Problem

`examples/population_global_gpw/humanity_globe_video.py` renders a rotating
orthographic globe of GPW-v4 2020 population density, but:

- "Spikes" are a 2D illusion: `_paint_exaggerated_population_layer` scales
  pixels radially outward in image space. No geometry, no occlusion, no shadows.
- Lighting is a CPU Lambert + rim-power hack, not physically based.
- The legend ("People per 30 km²") mislabels the data: GPW density is
  persons/km², and a 15-min cell is ~770 km² at the equator, shrinking with
  cos(lat).
- The lat-lon grid overweights high latitudes in spike count per screen area.
- Ocean and land share one grey base; there are no coastlines, no graticule,
  and the limb is a hard aliased edge with a fake rim light.

## Goal

A new script that renders the same 360° rotating-globe video with PROMETHEUS
path tracing — real 3D spikes casting soft shadows onto the sphere — and
publication-grade cartography, targeting 12 s @ 30 fps, 1080×1080 delivery,
seamless loop.

## Decisions (user-approved 2026-08-01)

| Decision | Choice |
| --- | --- |
| Rendering approach | Per-frame heightfield-dome PT (orthographic view of a sphere IS a heightfield) |
| Light mood / palette | Hard poster light (sun el 32, intensity 2.6, env 0.80) + scico Roma classed colours |
| Budget | 360 frames (12 s @ 30 fps), master ~2016 px downscaled to 1080; overnight run |
| Data honesty | Full fix: 5-min aggregate from 30-arcsec, cos(lat) equal-area thinning, persons/km² legend, continuous log heights |

## Key constraint: this checkout is pre-OBLIQUA

`python/forge3d/path_tracing.py` has no `albedo_map`, no `"orthographic"`
camera model, no `render_terrain_poster`. The implementation MUST use the
legacy pipeline: uniform-albedo PT light field, reduced to shade + tint,
multiplied into a light-free palette overlay in NumPy. `mesh_vertices` exists
but is out of scope (single albedo, unproven at spike scale).

## Architecture

New script `examples/population_global_gpw/humanity_globe_pt_video.py`,
importing reusable pieces (GPW download/aggregation, ffmpeg encode, frame
paths, fonts) from the existing `humanity_globe_video.py`. The original stays
untouched and runnable.

Units, each independently testable:

1. **Data prep** (`prepare_data`) — cached to npz
   - 30-arcsec GPW (existing download path) → mean-aggregate to a 5-min grid
     (2160×4320) with the existing windowed row-block aggregator, factor 10.
   - Land mask: Natural Earth 110m land GeoJSON (downloaded, cached),
     rasterized to the 5-min grid with `rasterio.features.rasterize` on
     plain-JSON geometry (no geopandas). Antarctica included. GPW nodata →
     density 0 over land.
   - Equal-area spike grid: per latitude row, max-pool density along
     longitude in blocks of `ceil(1/cos φ)` so candidate spike sites are
     ~uniform per unit area. Store both the full 5-min density (for colour
     classing) and the thinned spike-site list (lat, lon, density).

2. **Per-frame geometry** (`build_frame_heightfield`)
   - Frame f of N=360: center longitude = INITIAL + 360·f/N. Frame 360 ≡
     frame 0 → seamless loop.
   - PT grid ~1536²: dome `z = R_world·√(1−r²)` inside the disc, 0 outside
     (flat apron; disc margin small).
   - For each spike site visible on the near hemisphere: project (lat, lon)
     to view (x, y), stamp the nearest grid texel (1–2 texel footprint) with
     added height `S(density)·n_z`, where `n_z` is the view-axis component of
     the surface normal (radial spike foreshortened onto z; fades toward the
     limb — an accepted, documented limitation of the heightfield
     representation).
   - Spike height law: continuous, `S = MIN_DISPLAY + K·(log-scaled density)^HEIGHT_GAMMA`
     with HEIGHT_GAMMA ≈ 3 and MIN_DISPLAY a visible floor (WorldPop recipe);
     calibrate K so the tallest spike ≈ 3–5 % of globe radius in world units.
     Exact constants tuned during the probe stage.

3. **PT trace** (`trace_frame`) — Archetype-3 recipe
   - `hybrid_render_terrain_reference`: nadir camera, up=(0,0,−1),
     CAMERA_FOV_Y = 8.0, SUN_AZIMUTH = 225 (NW light → SE shadows; verified
     once with the quadrant probe before the full run), SUN_ELEVATION = 32,
     SUN_INTENSITY = 2.6, ENV_INTENSITY = 0.80, neutral sky env, albedo
     (0.62,)×3, spp = 1.
   - **Fixed accumulation count** (min_frames = max_frames, variance
     threshold effectively disabled): the variance-gate early exit is a
     proven flicker source in animation. Fixed per-frame seed.
   - Master field: 2×2 feather-blended overlapping camera cells @ ~1008 px
     + 48 px margin ≈ 2016 px (budget: 366 B/frame-px + 19 B/grid-texel ≈
     490 MB < 512 MiB gate). Fallback if the timing probe exceeds ~2×
     overnight budget: single-tile 1080.
   - Per-frame subprocess isolation (anti-TDR, France recipe), per-cell npz
     cache keyed by all trace parameters + frame index, retry/resume loop,
     output logged to a file (never piped).

4. **Overlay + modulation** (`compose_frame_rgb`)
   - Light-free overlay at master resolution: deep dark ocean, light neutral
     land (NE mask), spike footprints coloured by Roma class of their
     density. Alpha = disc mask.
   - Modulation: flat-ground normalization SHADE_LOW/HIGH_PCT = 0.5/99,
     TERRAIN_FLOOR/GAIN = 0.52/0.48, gamma 0.90; chromatic tint strength 1.0
     clamped (0.78, 1.25).
   - **Spike-footprint exemption mask** (`_spike_mask` recipe: surface > 3 %
     of tallest spike, nearest-resized, dilated 2 px; scale = 1, tint = 1
     inside) — needles stay clean and self-lit.
   - Ocean: own glassy curve from a heavily blurred shade (water-class
     recipe), no tint, no cavity.

5. **Cartographic finish** (compose stage, CPU)
   - Faint 15° graticule drawn on ocean pixels only.
   - Atmospheric rim: thin scattering-gradient ring replacing the rim-light
     hack; soft glow into a near-black background.
   - Downscale master → 1080; then title, legend, caption text (existing
     composer plumbing/fonts). Legend title corrected to persons per km²
     ("Population density (people per km²)"), same class breaks
     (1/5/10/50/100/500/1000), Roma swatches.
   - ffmpeg encode via the existing helper (30 fps).

## Error handling

- Tracer refusal ("no valid reservoirs") should be impossible (disc centered,
  every cell touches geometry) — treat as fatal with the cell id in the log.
- Device-lost / `Queue::submit` panics: retried by the per-frame subprocess
  wrapper; cached cells make retries cheap.
- Budget check computed up front from grid + frame sizes; abort before
  tracing if over the gate.
- Missing rasterio / NE download failure: fail with actionable message
  (mirrors existing GPW download error style).

## Testing

Unit (pure NumPy, no GPU, in `tests/`):
- Orthographic projection round-trip: lat/lon → view (x,y) → lat/lon within
  tolerance on the near hemisphere.
- cos(lat) thinning: spike-site areal density approximately uniform across
  latitude bands; total population-weighted mass preserved by max-pool
  within documented bounds.
- Dome heightfield: monotone radial profile, correct R_world, zero outside
  disc; spike stamps add exactly `S·n_z` at expected texels.
- Legend text contains "km²" and not "30 km²".

Render verification (GPU, probe-stage):
- Quadrant probe confirms shadow direction SE before the full run.
- Spike-shadow check: mean luminance in the shadow-side quadrant of the 10
  tallest spikes < lit-side quadrant.
- Flicker metric on a 10-frame probe: frame-to-frame mean-luminance delta
  within a documented bound on static regions (ocean).
- 3-frame timing probe decides master size (2016 vs 1080 fallback) before
  committing to the 360-frame run.

## Execution plan constraints (CLAUDE.md)

- Implementation in a fresh worktree/branch off `main` (current branch is
  dirty with unrelated work).
- Tasks delegated to Opus subagents; review after each task; final review;
  ask before opening a PR.
- Long renders: background, output redirected to a log file, `&&` chaining
  only.

## Out of scope

- Mesh-sphere PT (`mesh_vertices`) and limb-silhouette spikes.
- Night side / city lights / terminator.
- OBLIQUA features (`albedo_map`, true orthographic camera) — revisit if the
  checkout gains them.
