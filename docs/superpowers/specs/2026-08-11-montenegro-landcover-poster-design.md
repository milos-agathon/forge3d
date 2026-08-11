# Montenegro 3D Land-Cover Poster Design

## Goal

Create a publication-style static PNG of Montenegro that matches the approved Iberia land-cover poster family: PT-lit terrain relief under a categorical Sentinel-2/Esri 2024 land-cover palette, on a warm cream paper field with a soft southeast object shadow and a right-side title/legend block.

## Deliverables

- `examples/out/montenegro_landcover_pt/montenegro_poster_review_2250x1800.png`
- Reproducible source under `examples/montenegro_landcover_pt/`
- Cached class raster, terrain DEM, PT light field, and composition intermediates alongside the output

## Data and projection

- Natural Earth 10m country geometry, filtered to `ADM0_A3 == "MNE"`.
- AWS Terrarium DEM, zoom 11, reprojected and clipped to EPSG:3035.
- Sentinel-2 10m land-use/land-cover 2024 from the Esri/Impact Observatory/Microsoft tile `34T_20240101-20241231.tif`; categorical resampling uses mode/nearest only.
- Render grid is rectangular and normalized from a sea-level clamp, with country mask retained as alpha.

## Rendering and composition

- Use the `map-maker` terrain/land-cover PT archetype: converged nadir path-traced light field, then palette modulation and class-aware grade.
- Use the approved display colors from the Iberia production poster, in this legend order: Trees, Rangeland, Crops, Built area, Bare ground, Water, Flooded vegetation, Snow/ice.
- Use a 2250x1800 review canvas with the map filling the left/center and a negative-space furniture column on the right. Furniture consists of the title `MONTENEGRO`, subtitle `LAND COVER 2024`, data credit, and a vertically stacked legend.
- Apply cream paper background, warm-neutral contact shadow plus soft southeast cast shadow, no oblique subject rotation, and embedded sRGB PNG output.

## Data flow

1. Download/cache Natural Earth, Terrarium tiles, and the 34T land-cover COG.
2. Build a clipped EPSG:3035 DEM and categorical class raster aligned to it.
3. Trace the DEM light field with the map-maker PT constants, deriving Montenegro's relief scale from the land-only p90 slope target rather than copying a country value.
4. Modulate the approved class albedos with the traced shade, preserve alpha, and compose the review poster.
5. Validate class IDs, palette values, alpha coverage, furniture bounds, output dimensions, and PNG readability; inspect the final image visually.

## Failure handling and scope

- Nodata, offshore DEM leakage, and class IDs outside the approved palette are masked or rejected.
- A PT device-loss exit is resumable from per-cell caches; no fallback to a fake light field is allowed for the final render.
- The initial scope is one static PNG and its reproducible pipeline; no interactive viewer or animation is required.
