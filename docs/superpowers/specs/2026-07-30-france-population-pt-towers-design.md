# France Population PT Tower-Relief Plate — Design

Date: 2026-07-30
Status: approved (brainstorm complete)
Deliverable: `examples/population_ghsl/france_population_pt_3d.py` + final plate
`examples/out/france_population_pt/france_population_pt.png`

## Goal

A publication-grade poster in the Egypt/Romania register (steel-blue PROMETHEUS
relief, orange GHS_POP population, serif plate) for **metropolitan France +
Corsica**, with one signature PT-native effect: **population rendered as real
3D towers fused into the traced heightfield**, so population cores cast
genuine long soft shadows across the relief inside the converged path trace.
Accuracy is non-negotiable: real GHS_POP 2020 counts, the official national
projection, measured relief calibration, verified light direction.

## Decisions (user-approved)

1. **Register**: keep the Egypt steel-blue + orange series look; add one
   signature effect on top ("signature effect" option).
2. **Signature effect**: population towers inside the trace (not dusk
   dual-light, not composited glow).
3. **Shadows**: pure PT. The uncommitted CSM/MSM viewer shadow work is NOT
   used; a converged PROMETHEUS trace supplies one coherent light field with
   true soft penumbrae. No real-time shadow pass is mixed in.

## Data and projection

- **Population**: `D:/ghsl-population/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif`
  (GHSL R2023A, epoch 2020, 3 arcsec, residents per cell).
- **Extent**: metropolitan France + Corsica only (Natural Earth admin mask;
  exclude overseas territories and distant NE offshore slivers — reuse the
  Iberia offshore-territory filter lesson).
- **Projection**: **EPSG:2154 (RGF93 / Lambert-93)** — France's official
  conformal grid, standard parallels 44/49. No custom LCC needed.
- **DEM**: AWS Terrarium **z11** tiles (measured France ceiling), reprojected
  to Lambert-93 on a 4096-edge grid, country-masked.
- **Sea-leak clamp**: values below ~−15 m (below the true Camargue/polder
  minima) are generalized-coastline sea leaks → clamp to 0; genuine
  below-sea-level land is preserved. Exact floor confirmed against the masked
  DEM histogram before freezing.

## Signature effect: towers in the heightfield

- GHS_POP (≈90 m cells) → DEM grid (≈250 m Lambert-93 cells at 4096 edge)
  with **MAX resampling** (Egypt lesson: nearest drops hamlets).
- `height_traced = DEM_world + tower(pop)` where `tower(pop)` applies:
  - a **display floor** (persons threshold, initial ~1.0 like Egypt, swept)
    that kills GHS_POP disaggregation dust;
  - a **gamma** on normalized population (anti-carpet lesson: keeps villages
    visible without turning Paris into a mesa);
  - a world-height scale chosen so the tallest tower (Paris core) reads as a
    spire relative to RELIEF_WORLD, not a plateau.
- The same tower mask drives the orange overlay; terrain remains steel-blue.
  Overlay stays light-free and is modulated by the traced light field
  (established recipe).
- Floor/gamma/scale swept on cached heightfields with fast probe renders
  BEFORE any full-convergence trace.

## PT execution

- Entry point: `forge3d.path_tracing.hybrid_render_terrain_reference`.
- Egypt's **feather-blended overlapping mosaic** of quasi-ortho camera cells
  (FOV_Y 8°) over the single 4096-edge heightfield.
- **TDR discipline** (France-measured): per-tile subprocess isolation, spp=1
  accumulation frames.
- **Sun**: azimuth verified on THIS terrain with the 10-peak quadrant probe
  (compass = sun_azimuth + 90 convention); elevation low (14–18°) to stretch
  tower shadows; intensity in the Egypt band (~3.2), tuned by probe.
- **Relief**: `--measure-relief` land-only p90 world-slope probes; target the
  map-maker calibrated band while keeping the Alps sculpted and the Paris
  basin legible. Towers are excluded from the relief measurement (they are
  signal, not terrain slope).

## Composition

- Shared Romania/Egypt plate composer (title serif stack, caption, centring
  fix, Windows font fallback lesson).
- Title: `Population` / `FRANCE`. Caption: ©2026 Milos Popovic (milosgis.com)
  + GHSL R2023A epoch 2020 3 arcsec credit.
- Compose canvas 6144² (Iberia-proven 8K-class size).

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Paris agglomeration reads as a mesa | gamma + floor + scale sweep on cached heightfield probes first |
| Tower aliasing at ~250 m cells | 1-texel spires, accepted Turkey-needle precedent; check at 100% zoom |
| TDR on France-extent traces | subprocess-per-tile from the start; spp=1 frames |
| Alps crush the exaggeration budget | measured p90 calibration, probes at candidate RELIEF values |
| Sea leak flattens normalization | clamp + assert masked-DEM min ≥ clamp floor |

## Verification gates

- Masked population sum sanity vs INSEE metropolitan France (~68 M ± GHSL
  window slop).
- Sea-clamp assert on the masked DEM.
- Shadow-direction quadrant probe PASS before the production trace.
- Tracer convergence certificate (no silent fake convergence).
- Final visual review at 100% zoom on Paris, Lyon, Marseille, Mont Blanc,
  Corsica.
