# Cigar Smoke Fire-Only Improvement Plan

**Date:** 2026-06-12
**Scope:** Fire rendering only in `examples/california_cigar_smoke_demo.py`
**Goal:** Improve only the fire in `examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4` so it looks closer to the fire treatment in `rapidsave.com_oc_heat_and_smoke_the_recordbreaking_2023-pezptq0xr7ub1.mp4`.

## Hard Boundary

Do not touch any other effect.

Allowed:

- `HybridSmokeSource` fire lifecycle metadata in `examples/california_cigar_smoke_demo.py`
- `make_hybrid_smoke_sources()` source timing so flame lifetime differs from smolder/smoke lifetime
- `hybrid_fire_sources_rgba()` in `examples/california_cigar_smoke_demo.py`
- A fire-only burn-scar/ash RGBA layer composited into the existing fire render path
- The existing fire-only call tuning around the glow-only fire bloom pass and final fire core pass
- Fire-only regression tests in `tests/test_california_cigar_smoke_hybrid.py`

Not allowed:

- Smoke simulation algorithm, smoke color, alpha, haze, residual haze, or compositing logic
- Physical smoke domain, step, render settings, or volumetric shader logic
- Terrain texture generation, camera, lighting, labels, timing, or video encoding
- `src/smoke/*`

Limited lifecycle interaction:

- Existing source injection gates may honor source `end_frame` as the smolder-tail end, so a source stops creating new smoke after its smolder tail. This is source timing, not a smoke algorithm change.

## Lifecycle Requirement

The reference video differs most strongly in fire lifecycle: individual hot points do not burn forever. They burn for a finite interval, fade through a short smolder stage, and then leave a dark ash/burn-scar footprint that persists under smoke.

Implementation requirements:

- Each source needs a finite `flame_end_frame` separate from its final smoke/smolder `end_frame`.
- Source generation must stagger `flame_end_frame` values across the visible 8-second window, accounting for warmup frames so not all flames expire before the first rendered frame.
- `hybrid_fire_sources_rgba()` must render bright cores, yellow flares, orange bloom, and chain/front features only for sources that are still in flame life.
- After `flame_end_frame`, a source may emit reduced smolder smoke/ember glow briefly, but white/yellow cores must disappear.
- A persistent burn-scar/ash layer must render inactive-after-flame source footprints as irregular charcoal/gray-brown patches.
- The burn-scar layer must be composited before smoke and final active fire, so smoke veils the burnt ground while live flames remain visible.
- Tests must prove that late frames retain ash/burn scar while hot fire cores have dropped substantially or disappeared for burned-out sources.

Old exclusions retained:

- Smoke color, alpha, haze, residual haze, or compositing logic
- Physical smoke domain, step, emitter, or render settings beyond source timing inputs
- Terrain texture, camera, lighting, labels, timing, or video encoding

## Observed Reference Differences

The reference fire reads as hot light, not a symbolic overlay:

- Brighter white-yellow cores with orange edges
- Stronger localized bloom around active hotspots
- Fewer uniform dots and more intensity variation
- Irregular clusters and short front-like chains
- Embers vary from tiny red points to bright flares
- Fire stays visible through smoke without creating a broad red mask
- Individual fire points burn out and become ash/burnt ground instead of remaining hot forever

The current generated fire reads as:

- Many similarly sized orange dots
- Weak white-hot core presence
- Broad reddish/orange haze around the burn area
- Less contrast against smoke and terrain
- More evenly scattered points inside the fire complex
- Hot points persist for the full render because source `end_frame` is beyond the video duration

## Lifecycle Implementation Plan

1. Add explicit lifecycle helpers:
   - active flame weight
   - smolder weight
   - burn-scar weight
2. Extend `HybridSmokeSource` with optional `flame_end_frame` metadata while preserving existing constructor compatibility.
3. Change generated source timing so `end_frame` becomes a smolder/smoke end, not a flame end.
4. Render only flame-live sources in cluster/front/fire-core logic.
5. Add `hybrid_burn_scar_rgba()` for irregular persistent ash and burnt-area marks.
6. Composite the scar map onto the terrain copy before under-smoke glow and smoke layers.
7. Add tests for:
   - generated source lifetimes are finite and staggered
   - late fire cores disappear after flame life
   - ash persists after flame life
   - smolder smoke can outlive visible flame briefly

## Current Implementation Status

The fire morphology pass and the finite-flame lifecycle/ash extension are implemented and audited.

Completed:

- `HybridSmokeSource` now carries optional `flame_end_frame` metadata while retaining constructor compatibility for existing tests and callsites.
- `make_hybrid_smoke_sources()` now staggers finite flame lifetimes across the visible render window and accounts for warmup frames.
- Source `end_frame` now represents the end of the reduced smolder/smoke tail rather than the end of bright flame.
- Shared lifecycle helpers drive active flame, smolder smoke, and burn-scar weights from the same source timing.
- `hybrid_fire_sources_rgba()` now groups nearby active sources into local clusters before drawing hard cores.
- Cluster-level patches, principal-axis front strokes, and short deterministic source-to-source chains reduce the point-emitter/bead-field look.
- Individual source marks are reduced inside clusters; only selected cluster anchors keep stronger hard cores, while weaker clustered sources become ember support.
- Bright cores, yellow flares, orange bloom, and cluster/front hot strokes are rendered only while source flame weight is active.
- Expired flame sources can contribute only a small red smolder ember and reduced smoke; they no longer produce white/yellow hot cores.
- `hybrid_burn_scar_rgba()` renders persistent irregular charcoal/gray-brown ash footprints for burned sources.
- The burn-scar layer is composited onto terrain before under-smoke fire glow, smoke, and final active fire.
- Fire-specific pass tuning keeps the under-smoke `glow_only=True` bloom narrower and makes the over-smoke core pass hotter without changing smoke, terrain, camera, labels, timing, or encoding.
- Fire-only regression tests now cover RGBA shape/dtype, finite/staggered lifetimes, warmup-aware source timing, white/yellow cores, localized bloom, source intensity variation, `glow_only=True`, connected warm components, front-dominant warm area, connected hot strokes, bloom locality, hot-core burnout, ash persistence, and smolder smoke after flame end.
- `examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4` and its preview have been re-rendered from the updated lifecycle implementation.
- All 240 rendered frames were decoded for full-frame hot/warm pixel lifecycle metrics, and representative frames at 1s, 4s, and 7s were visually inspected.

Frame audit evidence from `examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4`:

- SHA-256: `6f2fe9703366afb82c19c0f85cdb79be7c97466c17e3169739cdbcd08d62ee00`
- Video dimensions: 960x540, 30 FPS, 8.0 seconds, 240 frames.
- Hot-core pixels are present for 144 of 240 frames, then burn out; the last hot-core frame is index 177.
- Late-frame hot-core total from frames 180-239 is 0, proving the fire no longer burns forever.
- Warm fire pixels are present for 202 of 240 frames and continue briefly after hot cores disappear as smolder/embers.
- Full-frame hot-core pixel count range: 0 to 51, median 2.
- Full-frame warm pixel count range: 0 to 801, median 305.
- Representative frame metrics:
  - 1s: hot=23, warm=776
  - 4s: hot=10, warm=298
  - 7s: hot=0, warm=0
- Visual review of 1s, 4s, and 7s frames showed active early/mid flame clusters over a darkened burn footprint, and a late frame where smoke veils the burnt area after hot cores have disappeared.

## Remaining Requirements

None for this fire-only lifecycle pass. The required morphology, bloom locality, finite flame lifetime, smolder tail, ash/burn-scar rendering, tests, re-render, and 240-frame artifact metric audit are complete.

Closed requirements:

### A. Reduce the Point-Emitter Look

- Implemented by clustering active sources, drawing shared warm patches, reducing non-anchor clustered source marks, and keeping weak sources as embers.
- The final rendered video no longer visually reads as a uniform field of separate dots in most frames.

### B. Add Short Front-Like Chains

- Implemented deterministic chain marks between selected nearby clustered sources.
- Chains are short, broken, irregular, and source-local, using orange bloom/edges with yellow-white interiors on stronger fronts.
- Representative early and mid frames show active fronts as well as retained embers; late frames transition to smoke over ash/burn-scar footprints.

### C. Improve Spatial Variation Without Changing Smoke

- Implemented deterministic cluster/front variation using source seeds and frame index.
- Added asymmetric cluster polygons, rotated per-source marks, and short line strokes using PIL drawing operations.
- Smoke fields, smoke compositing, terrain, camera, labels, timing, and encoding were not changed.

### D. Keep Bloom Local

- Implemented with narrower blur radii, reduced clustered-source bloom, and constrained cluster/chain alpha.
- `glow_only=True` preserves grouped bloom while suppressing hard cores.
- Full-frame visual audit found no broad red/orange mask.

### E. Add Morphology-Focused Tests

- Added fire-only tests that fail for a pure point-field implementation.
- The tests operate on `hybrid_fire_sources_rgba()` only and avoid smoke assertions.
- Coverage includes connected warm components, connected hot strokes, front-dominant warm area, localized bloom, and `glow_only=True` behavior.
- Added lifecycle tests for warmup-aware generated source timing, hot-core burnout, ash persistence, and smolder smoke after flame end.

### F. Re-render and Re-audit Every Frame

- Re-rendered `examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4`.
- Decoded all 240 frames for full-frame lifecycle pixel metrics.
- Confirmed hot cores burn out before the late frames (`last_hot_frame_index=177`, no hot pixels from frames 180-239).
- Confirmed warm fire pixels persist through most of the clip but are no longer present at 7s.
- Confirmed representative 1s and 4s frames show live clustered fire, while the 7s frame shows smoke over the dark burn footprint.

### G. Add Finite Flame and Ash Lifecycle

- Implemented finite `flame_end_frame` values per source, staggered across the visible render window.
- Implemented a smolder tail where source smoke can continue at reduced strength after bright flame ends.
- Implemented persistent ash/burn-scar rendering under smoke and active fire.
- Kept bright cores and hot front strokes limited to flame-live sources.

## Implemented Steps

### 1. Establish Fire-Only Baseline

Rendered and inspected existing fire maps at representative frames:

- Early: around 1s
- Mid: around 4s
- Late: around 7s

Used the existing output and fire overlay behavior only as baseline evidence. Smoke, terrain, camera, labels, and timing were not altered.

### 2. Tune `hybrid_fire_sources_rgba()`

Modified only the fire drawing internals:

- Kept the existing function signature stable.
- Replaced uniform source ellipses with layered fire marks:
  - small white-hot core
  - yellow middle flare
  - orange ember edge
  - soft localized bloom halo
- Added deterministic variation using source seeds, source index, cluster seeds, and `frame_index`.
- Made stronger/hotter clustered fronts flare brighter while preserving weaker red/orange embers.
- Reduced broad red/orange fill so bloom stays local to active fire.

### 3. Tune Existing Fire Passes Only

The final render path still uses two fire passes:

- Under-smoke glow-only bloom pass around `fire_bloom_map = hybrid_fire_sources_rgba(...)`
- Over-smoke final core pass around `fire_map = hybrid_fire_sources_rgba(...)`

Tuned only fire-specific arguments:

- Under-smoke pass: kept `glow_only=True`, with narrower and more source-local bloom.
- Over-smoke pass: increased hot-core contrast and reduced broad haze.

The pass order was not changed.

### 4. Add Fire-Only Tests

Added focused tests in `tests/test_california_cigar_smoke_hybrid.py`:

- `hybrid_fire_sources_rgba()` returns an RGBA layer of the requested shape and dtype.
- The layer contains white/yellow hot-core pixels.
- Orange/red bloom area is larger than the white core area but remains localized.
- Active source brightness varies across sources.
- `glow_only=True` suppresses hard cores while preserving grouped bloom.
- Warm connected components are larger than a single source dot.
- Hot connected strokes exist in the front rendering.
- Warm bloom/core pixels remain localized to the fire complex.
- Generated source lifetimes are finite, staggered, and warmup-aware.
- A burned-out source loses hot cores and leaves an ash/burn-scar footprint.
- Smolder smoke can continue after visible flame ends without producing hot fire cores.

Existing smoke tests were not changed.

### 5. Add Finite Flame and Ash Lifecycle

Added lifecycle-specific source behavior:

- `flame_end_frame` marks the end of bright source flame.
- `end_frame` marks the end of the reduced smolder/smoke tail.
- Shared lifecycle helpers compute flame, smolder, source-smoke, and burn-scar weights.
- `hybrid_fire_sources_rgba()` uses only flame-live sources for active fire fronts and cores.
- `hybrid_burn_scar_rgba()` draws stable irregular burnt-ground patches for burned sources.
- The final render path composites the burn-scar layer onto terrain before smoke and active fire.

### 6. Render and Compare

Rendered the 8-second video after the fire lifecycle change.

Compared all frames, including representative frames at approximately:

- 1s
- 4s
- 7s

Success criteria passed:

- Early and mid fire has visible white-yellow cores.
- Fire bloom is stronger but remains localized while sources are flame-live.
- Hotspots vary in size and brightness while active.
- Fire clusters look more irregular and front-like before burning out.
- Late frames no longer retain hot fire cores; they show smoke over the burn footprint.
- Smoke shape, smoke opacity, terrain, camera, labels, and timing are unchanged except for fire visibility through the existing smoke.

## Validation Completed

Existing cigar smoke hybrid suite:

```bash
pytest tests/test_california_cigar_smoke_hybrid.py -q
```

Result: 38 passed.

Rendered clip:

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 examples/california_cigar_smoke_demo.py
```

Result: wrote `examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4` and `examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.preview.png`.

Artifact metadata:

```bash
shasum -a 256 examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4
```

Result: SHA-256 `6f2fe9703366afb82c19c0f85cdb79be7c97466c17e3169739cdbcd08d62ee00`; 960x540, 30 FPS, 8.0 seconds, 240 frames.

Full-frame lifecycle metric audit:

```bash
ffmpeg -y -i examples/out/california_cigar_smoke/august_complex_cigar_smoke_8s.mp4 /private/tmp/cigar_lifecycle_full_0612/frame_%04d.png
```

Result: 240 frames decoded. Hot-core pixels were present through frame index 177 and absent from frames 180-239; representative frame metrics were 1s hot=23/warm=776, 4s hot=10/warm=298, and 7s hot=0/warm=0.

## Rollback

If the fire improvement changes smoke, terrain, camera, labels, timing, or any non-fire effect:

1. Revert the non-fire change.
2. Keep only changes inside `hybrid_fire_sources_rgba()` or fire-only call arguments.
3. Re-render and compare again.
