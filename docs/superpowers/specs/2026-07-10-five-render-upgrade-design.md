# Five-render upgrade pass — design notes (2026-07-10)

## Goal
Redo five example renders with forge3d's best current capabilities so the results
surpass typical ArcGIS / Blender / QGIS output. Priority: terrain relief and
population spikes must read as the centerpiece of each image.

Targets:
1. `examples/population_ghsl/romania_builtup_cover_3d.py`
2. `examples/swiss_terrain_landcover_viewer.py`
3. `examples/forest_cover_copernicus/italy_forest_cover_3d.py`
4. `examples/population_ghsl/southeast_europe_population_spikes_enhanced_3d.py`
5. `examples/climate_bivariate/belgium_bivariate_climate_map.py`

## Approach (chosen)
Keep each script's data pipeline (GHSL, Copernicus, Sentinel-2 LULC,
TerraClimate, AWS terrarium DEM) — they are correct and cached. Invest in the
*render pass* and *composition*:

- Baseline first: run each script unmodified, inspect the PNG, list weaknesses.
- Upgrade pass per script: camera (phi/theta/fov/radius), sun geometry,
  PCSS shadows, HDRI IBL, height-AO, soft sun-visibility, ACES tonemap
  white point/exposure, normal strength, z-scale, supersampling, palette and
  post-compose grading.
- Iterate: view every render before/after; only keep changes that visibly
  improve relief/spike readability.

Alternatives considered and rejected:
- Full rewrite on `Scene`/`TerrainRenderer` offline path: more control but
  discards thousands of lines of working composition code; slower to a good
  result.
- Path-traced (`render_offline`) versions: highest ceiling but the spike/
  overlay pipelines are viewer-IPC based; porting them is out of scope for
  this pass.

## Constraints
- Use the existing release `_forge3d.pyd` (WIP `src/geo` changes in the tree
  are unrelated; do not rebuild on top of another session's Rust work).
- Use `.venv/Scripts/python.exe`.
- Background long renders; log to files, never `| tail`.
- Do not commit renders or caches; outputs go to `examples/out/`.

## Success criteria
Each final PNG: strong, legible relief or spikes; no blown highlights or
crushed shadows; clean silhouette against canvas; readable typography/legend;
≥4K output without aliasing artifacts.
