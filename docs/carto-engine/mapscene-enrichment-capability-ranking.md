# MapScene Enrichment Capability Ranking

## A. Scope Confirmation

Active recipes ranked:

First batch:

- `terrain_demo`
- `terrain_label`
- `landcover_esri_terrain_viewer`
- `climate_bivariate`
- `hydrology_river`
- `mapscene_showcases`

Next batch:

- `terrain_relief_rem`
- `population_spike_worldpop`
- `population_ghsl_3d`
- `builtup_cover_3d`
- `pointcloud_cog`
- `urban_osm_city`
- `luxembourg_rail_overlay`

Excluded from scoring:

- `labels_styles_picking` as standalone
- `wildfire_smoke`
- `satellite_timelapse`
- `osm_city_flood_daycycle`
- `humanity_globe_video`

Ambiguity found: `hydrology_river` includes a video variant, and `urban_osm_city` includes `osm_city_daycycle.py`. Static or roadmap-named active scripts were counted, and temporal/video-only behavior did not affect rankings except where the same script-local data or layer logic is reused.

No files were edited or staged during the analysis pass.

## B. Ranked Capability Table

| Rank | Capability | Proposed MapScene/API surface | Score | Scripts | Recipes | F/M/P/D | First-batch impact | Evidence paths | Remaining blockers | Risk |
|---:|---|---|---:|---:|---:|---|---|---|---|---|
| 1 | Recipe-family manifest/provenance | `RecipeManifest`, stable JSON serialization, optional `SceneRecipe.metadata["manifest_id"]` bridge | 43 | 43 | 13 | 0/0/43/0 | 19 scripts, 6 recipes | `docs/carto-engine/g-001-p1-1-recipe-manifest-implementation-plan.md:63`, `:91`, `:257`; `docs/carto-engine/golden-map-recipe-capability-audit.md:1902` | Rendering adapters, styles, blend, plate | low |
| 2 | Spatial alignment diagnostics bridge | `MapScene.validate_alignment()`, `RasterOverlay.align_to_terrain()`, vector/raster CRS/grid diagnostics | 37 | 36 | 11 | 0/1/35/0 | landcover, climate, hydrology | `docs/carto-engine/golden-map-recipe-capability-audit.md:1903`; `python/forge3d/gis.py:538`, `:547`, `:571` | Style/render glue remains | low-medium |
| 3 | Map plate composition | Render `MapFurnitureLayer`: title, legend, scale bar, north arrow, captions, keepouts | 33 | 33 | 10 | 0/0/33/0 | landcover, climate, hydrology, mapscene | `docs/carto-engine/golden-map-recipe-capability-audit.md:1910`; `python/forge3d/legend.py:50`; `python/forge3d/scale_bar.py:51`; `python/forge3d/north_arrow.py:28` | Data rendering/style still separate | medium |
| 4 | Population surface / spike height style | `PopulationSurfaceLayer` or `RasterHeightSurfaceStyle` for density-to-height/shade/spikes | 25 | 13 | 2 | 0/12/1/0 | none | `docs/carto-engine/g-001-p1-1-recipe-manifest-implementation-plan.md:273`, `:274`; population scripts below | Alignment, plate, bivariate variant | high |
| 5 | Multi-pass compositing / blend modes | `RenderPassSpec`, `MapScene.render_passes`, blend modes for color + relief + overlay | 23 | 15 | 6 | 0/8/7/0 | landcover, climate, hydrology | `docs/carto-engine/golden-map-recipe-capability-audit.md:1904`; climate/landcover scripts below | Style specs, map plate | medium |
| 6 | Recipe-level golden/diagnostic fixtures | Per-recipe fixture metadata, expected outputs, support diagnostics | 21.5 | 43 | 13 | 0/0/0/43 | all first-batch | `docs/carto-engine/golden-map-recipe-capability-audit.md:1905`, `:1947`, `:1951` | Does not render by itself | medium |
| 7 | Terrain camera / lighting / relief presets | `LightingPreset.cartographic`, `TerrainRecipePreset`, default camera/sun/PBR bundles | 16 | 15 | 5 | 0/1/14/0 | terrain, landcover, hydrology | `python/forge3d/terrain_demo.py:310`, `:870`, `:1001` | Data/style/plate still needed | medium |
| 8 | Categorical/thematic raster style | `CategoricalRasterStyle`, class palette, nodata, class legend, thematic validation | 13 | 8 | 2 | 0/5/3/0 | landcover | `docs/carto-engine/golden-map-recipe-capability-audit.md:1907`; landcover scripts below | Alignment, blend, plate | low-medium |
| 9 | Vector line stroke / drape helper | `VectorLineLayer.stroke(width, halo, cap, join, drape, z_offset)` | 12 | 7 | 3 | 0/5/2/0 | hydrology | `docs/carto-engine/golden-map-recipe-capability-audit.md:1908`; `python/forge3d/terrain_params.py:1543` | Raster alignment, plate | medium |
| 10 | Bivariate raster style | `BivariateRasterStyle`, two-variable bins, palette matrix, legend axes | 8 | 4 | 2 | 0/4/0/0 | climate | `docs/carto-engine/golden-map-recipe-capability-audit.md:1890`; climate scripts below | Alignment, blend, plate | medium |
| 11 | Building MapScene render adapter | Real or explicit adapter for `MapSceneBuildingLayer`/`BuildingLayer` | 7 | 5 | 2 | 0/2/3/0 | mapscene showcases | `python/forge3d/map_scene.py:1590`, `:1630`; `python/forge3d/buildings.py:271` | Urban context, materials, LOD | high |
| 12 | Docs-to-test support traceability | Recipe support matrix linking docs, fixtures, tests, diagnostics | 6.5 | 13 | 13 | 0/0/0/13 | all first-batch | `docs/carto-engine/golden-map-recipe-capability-audit.md:20`, `:1989` | Diagnostic only | low |
| 13 | Urban context layer builder | `UrbanContextLayer.from_osm(...)` for roads, water, rail, buildings, local CRS | 6 | 3 | 1 | 0/3/0/0 | none | `docs/carto-engine/golden-map-recipe-capability-audit.md:1909`; `python/forge3d/gis.py:696` | Building adapter, line style, plate | high |
| 14 | Label recipe defaults / font fixture | `LabelLayer.terrain_defaults()`, deterministic font atlas/style presets | 5 | 4 | 2 | 0/1/3/0 | terrain_label, mapscene | `python/forge3d/label_plan.py:750`; label scripts below | Full terrain label composition | low-medium |
| 15 | PointCloud + COG MapScene source adapters | `PointCloudLayer.from_path()`, `TerrainSource.from_cog()`, `RasterOverlay.from_cog()` | 4 | 2 | 1 | 0/2/0/0 | none | `python/forge3d/map_scene.py:1492`; `python/forge3d/pointcloud.py:594`; `python/forge3d/cog.py:346` | Camera/LOD/streaming UI | medium |

No single capability earns `Full` under the strict definition: every active nontrivial recipe needs at least two named pieces before it becomes MapScene-native.

## C. Capability Cards

### 1. Recipe-Family Manifest / Provenance

What it adds to MapScene: a stable recipe identity and layer/source/capability manifest so every active recipe can be described before rendering.

Existing code it should reuse: `SceneRecipe`, `MapScene.save_bundle`, and existing dataclass serialization patterns in `python/forge3d/map_scene.py`.

Scripts it unlocks: all active canonical scripts because the P1-1 plan explicitly scopes first batch and next batch at `docs/carto-engine/g-001-p1-1-recipe-manifest-implementation-plan.md:91` and fixture rows at `:257-278`. Representative script evidence includes `examples/terrain_viewer/terrain_demo.py`, landcover variants cited at `docs/carto-engine/golden-map-recipe-capability-audit.md:102`, climate at `:104`, hydrology at `:108`, urban at `:109`, Luxembourg rail at `:111`, terrain label at `:116`, and pointcloud/COG at `:117`.

Recipes it helps: all 13 active recipes.

What remains unsolved: rendering, styling, CRS alignment, and plate composition.

Smallest useful implementation slice: manifest dataclasses plus JSON round-trip and capability tokens only.

Smallest useful test/golden/diagnostic: one fixture manifest per first-batch recipe.

### 2. Spatial Alignment Diagnostics Bridge

What it adds to MapScene: MapScene-level CRS/grid/extent diagnostics for terrain, raster overlays, vector overlays, and COG-backed sources.

Existing code it should reuse: `assert_grid_compatible`, `align_raster_grid`, `align_raster_to`, and `reproject_raster` in `python/forge3d/gis.py:538-578`; existing CRS checks in `python/forge3d/map_scene.py:1331-1369`.

Scripts it unlocks: `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:483`, `examples/landcover_esri/romania_terrain_landcover_viewer.py:170`, `examples/landcover_esri/swiss_terrain_landcover_viewer.py:791`, `examples/landcover_esri/germany_terrain_landcover_viewer.py:154`, `examples/landcover_esri/poland_terrain_landcover_viewer.py:314`, `examples/climate_bivariate/europe_bivariate_temp_precip.py:696`, `examples/climate_bivariate/france_bivariate_temp_precip.py:790`, `examples/urban_osm/luxembourg_rail_overlay.py:244`, and `examples/pointcloud_cog/cog_streaming_demo.py:302`.

Recipes it helps: landcover, climate, hydrology, terrain relief, population, built-up, COG, Luxembourg rail, and urban.

What remains unsolved: styling and rendering still need their own adapters.

Smallest useful implementation slice: validation-only diagnostics, no automatic resampling.

Smallest useful test/golden/diagnostic: construct misaligned terrain/raster/vector layers and assert named diagnostic codes.

### 3. Map Plate Composition

What it adds to MapScene: actual rendering support for `MapFurnitureLayer`: legends, north arrow, scale bar, title, caption, and keepouts.

Existing code it should reuse: `legend.py`, `scale_bar.py`, `north_arrow.py`, and serialized `MapFurnitureLayer` in `python/forge3d/map_scene.py:901-916`.

Scripts it unlocks: landcover composition in `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:688`, `examples/landcover_esri/poland_terrain_landcover_viewer.py:62-144`, `examples/landcover_esri/swiss_terrain_landcover_viewer.py:635`; climate composition in `examples/climate_bivariate/europe_bivariate_temp_precip.py:751` and `:929-987`, `examples/climate_bivariate/belgium_bivariate_temp_precip.py:746` and `:924`, `examples/climate_bivariate/france_bivariate_temp_precip.py:1219` and `:1443`; hydrology furniture in `examples/hydrology_river_basins/pnoa_river_showcase.py:271`, `examples/hydrology_river_basins/poland_river_basins_forge3d.py:636`, `examples/hydrology_river_basins/iberian_peninsula_river_basins.py:1275`; population plates in `examples/population_spike_worldpop/poland_population_spikes.py:639`, `examples/population_spike_worldpop/france_population_height_shade.py:153`, `examples/population_spike_worldpop/germany_population_height_shade.py:105`, and `examples/population_spike_worldpop/turkey_population_height_shade.py:126`.

Recipes it helps: landcover, climate, hydrology, terrain relief, population, built-up, and mapscene showcases.

What remains unsolved: data rendering and style definitions.

Smallest useful implementation slice: render title, legend, north arrow, and scale bar for static PNG output.

Smallest useful test/golden/diagnostic: deterministic furniture-only render with fixed dimensions and keepouts.

### 4. Population Surface / Spike Height Style

What it adds to MapScene: a MapScene-native density raster style that converts population to height, shade, spike, or contour surfaces.

Existing code it should reuse: raster normalization helpers in `python/forge3d/gis.py`, terrain params, and existing population recipe formulas.

Scripts it unlocks: `examples/population_spike_worldpop/poland_population_spikes.py:13-15`, `:206`; `examples/population_spike_worldpop/poland_population_spikes_height_shade.py:47`, `:80`, `:171`; `examples/population_spike_worldpop/poland_population_contour_terrain.py:155`, `:225`, `:234`; `examples/population_spike_worldpop/france_population_height_shade.py:43`, `:121`, `:451`; `examples/population_spike_worldpop/germany_population_height_shade.py:36`, `:274`; `examples/population_spike_worldpop/turkey_population_height_shade.py:41`, `:272`; GHSL population rendering evidence at `docs/carto-engine/golden-map-recipe-capability-audit.md:644-729`.

Recipes it helps: `population_spike_worldpop`, `population_ghsl_3d`.

What remains unsolved: plate composition, alignment, and the bivariate population-temperature variant.

Smallest useful implementation slice: one height-shade style from single raster, no spikes.

Smallest useful test/golden/diagnostic: validate population domain, nodata mask, height scale, and output range.

### 5. Multi-Pass Compositing / Blend Modes

What it adds to MapScene: declarative color pass + relief pass + overlay blending.

Existing code it should reuse: current render output arrays and MapScene deterministic renderer extension points.

Scripts it unlocks: landcover color/relief passes in `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:661-688`, `examples/landcover_esri/romania_terrain_landcover_viewer.py:376-403`, `examples/landcover_esri/swiss_terrain_landcover_viewer.py:1123-1151`; climate blend in `examples/climate_bivariate/belgium_bivariate_temp_precip.py:879`, `examples/climate_bivariate/europe_bivariate_temp_precip.py:884`, `examples/climate_bivariate/france_bivariate_temp_precip.py:1398`; terrain relief REM in `examples/terrain_relief/platte_rem_forge3d.py:618-854`; hydrology two-pass evidence in `docs/carto-engine/golden-map-recipe-capability-audit.md:861-874`; built-up variants in `:719-729`.

Recipes it helps: landcover, climate, terrain relief, hydrology, built-up, and population.

What remains unsolved: categorical/bivariate style definitions and plate furniture.

Smallest useful implementation slice: `multiply`, `screen`, and alpha-over blend between two named passes.

Smallest useful test/golden/diagnostic: two tiny image passes with known pixel math.

### 6. Recipe-Level Golden / Diagnostic Fixtures

What it adds to MapScene: fixture expectations and explicit support diagnostics per recipe.

Existing code it should reuse: existing test/golden conventions and `MapScene.validate`.

Scripts it unlocks diagnostically: all active scripts listed in audit discovery at `docs/carto-engine/golden-map-recipe-capability-audit.md:99-117` and fixture rows at `:1951-1960`.

Recipes it helps: all 13.

What remains unsolved: no rendering capability.

Smallest useful implementation slice: one fixture manifest and one expected diagnostic per first-batch recipe.

Smallest useful test/golden/diagnostic: fixture loads, validates, and reports missing capability token.

### 7. Terrain Camera / Lighting / Relief Presets

What it adds to MapScene: named terrain/cartographic presets for camera, sun, exaggeration, PBR, and relief defaults.

Existing code it should reuse: `python/forge3d/terrain_demo.py` helpers and `python/forge3d/terrain_params.py`.

Scripts it unlocks: `python/forge3d/terrain_demo.py:310-357`, `:870-912`, `:1001-1083`; landcover terrain defaults at `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:135-148`; hydrology terrain/poster setup at `docs/carto-engine/golden-map-recipe-capability-audit.md:861-874`; urban day/night lighting in `examples/urban_osm/osm_city_daycycle.py:541-655`.

Recipes it helps: terrain demo, landcover, terrain relief, hydrology, and urban.

What remains unsolved: recipe-specific data sources and styling.

Smallest useful implementation slice: one `cartographic_relief` preset used by MapScene terrain render.

Smallest useful test/golden/diagnostic: preset serializes and produces stable camera/light fields.

### 8. Categorical / Thematic Raster Style

What it adds to MapScene: class-id to color mapping, class legend entries, nodata handling, and categorical validation.

Existing code it should reuse: `python/forge3d/style.py`, `python/forge3d/legend.py`, and `python/forge3d/gis.py::classify_raster`.

Scripts it unlocks: `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:72`, `:456`, `:567`, `:595`; `examples/landcover_esri/germany_terrain_landcover_viewer.py:154-170`; `examples/landcover_esri/poland_terrain_landcover_viewer.py:62-144`; `examples/landcover_esri/romania_terrain_landcover_viewer.py:211`, `:544`; `examples/landcover_esri/swiss_terrain_landcover_viewer.py:78-104`, `:219-240`, `:943`; built-up variants cited at `docs/carto-engine/golden-map-recipe-capability-audit.md:719-729`.

Recipes it helps: landcover and built-up cover.

What remains unsolved: CRS alignment, relief blending, and map plate.

Smallest useful implementation slice: palette + nodata + legend entries for integer rasters.

Smallest useful test/golden/diagnostic: class raster maps to expected RGBA and legend rows.

### 9. Vector Line Stroke / Drape Helper

What it adds to MapScene: high-level line rendering for rivers, roads, and rails: width, halo, cap/join, drape, and z offset.

Existing code it should reuse: `VectorOverlayConfig` in `python/forge3d/terrain_params.py:1543-1625` and vector style parsing in `python/forge3d/style.py:721-735`.

Scripts it unlocks: `examples/urban_osm/luxembourg_rail_overlay.py:156-170`, `:283-291`, `:724-736`, `:1258-1288`; `examples/hydrology_river_basins/poland_river_basins_forge3d.py:234`, `:535-608`; `examples/hydrology_river_basins/pnoa_river_showcase.py:229-260`; `examples/hydrology_river_basins/iberian_peninsula_river_basins.py:410`, `:424-450`; urban road/rail buffers in `examples/urban_osm/osm_city_demo.py:789-820`.

Recipes it helps: hydrology river, Luxembourg rail, and urban OSM city.

What remains unsolved: source fetching and map plate.

Smallest useful implementation slice: draped polyline overlay with width and halo, no joins beyond miter/bevel default.

Smallest useful test/golden/diagnostic: a two-segment line produces nonempty overlay and validates width.

### 10. Bivariate Raster Style

What it adds to MapScene: two-variable binning, palette matrix, axis labels, and bivariate legend.

Existing code it should reuse: `python/forge3d/gis.py::normalize_raster`, `python/forge3d/legend.py`, and style validation.

Scripts it unlocks: `examples/climate_bivariate/belgium_bivariate_temp_precip.py:217`, `:691`, `:746`, `:879`; `examples/climate_bivariate/europe_bivariate_temp_precip.py:220`, `:696`, `:751`, `:884`; `examples/climate_bivariate/france_bivariate_temp_precip.py:281`, `:296`, `:928-944`, `:1035`, `:1219`; population-temperature bivariate recipe cited in `docs/carto-engine/golden-map-recipe-capability-audit.md:1890`.

Recipes it helps: climate bivariate and population GHSL bivariate variant.

What remains unsolved: alignment, pass compositing, and plate.

Smallest useful implementation slice: fixed 3x3 bin matrix and legend output.

Smallest useful test/golden/diagnostic: two tiny rasters classify into expected palette cells.

### 11. Building MapScene Render Adapter

What it adds to MapScene: real rendering or explicit supported diagnostic path for `BuildingLayer`.

Existing code it should reuse: `python/forge3d/buildings.py:271-361`, `:383-502`, `:572-615`; existing placeholder in `python/forge3d/map_scene.py:1590-1636`.

Scripts it unlocks: `examples/mapscene_buildings_labels.py:20-41`, `:68-76`; `examples/mapscene_p1_assets_bundle_showcase.py:151-168`; `examples/urban_osm/buildings_viewer_interactive.py:171-244`, `:547-562`; `examples/urban_osm/osm_city_demo.py:634-648`, `:823-856`.

Recipes it helps: mapscene showcases and urban OSM city.

What remains unsolved: OSM context assembly and full materials/LOD.

Smallest useful implementation slice: render building footprints as extruded or diagnostic-safe vector overlay.

Smallest useful test/golden/diagnostic: one rectangular building validates and renders nonblank.

### 12. Docs-To-Test Support Traceability

What it adds to MapScene: explicit docs table connecting each active recipe to required capability tokens, fixture image, and test.

Existing code it should reuse: current roadmap/gap register and future manifest fixtures.

Scripts it unlocks diagnostically: one per active recipe family, based on active scope at `docs/carto-engine/golden-map-recipe-capability-audit.md:1966-1968`.

Recipes it helps: all 13.

What remains unsolved: no rendering.

Smallest useful implementation slice: support matrix generated from manifest fixtures.

Smallest useful test/golden/diagnostic: every active recipe row cites at least one fixture and one capability status.

### 13. Urban Context Layer Builder

What it adds to MapScene: `UrbanContextLayer.from_osm` to collect buildings, water, roads, rail, local projection, masks, and style defaults.

Existing code it should reuse: `gis.prepare_osm_scene` at `python/forge3d/gis.py:696-701`, `estimate_local_utm` at `:785-788`, and `python/forge3d/buildings.py`.

Scripts it unlocks: `examples/urban_osm/osm_city_demo.py:285-330`, `:449-467`, `:699-715`, `:725-732`, `:789-856`; `examples/urban_osm/osm_city_daycycle.py:28`, `:312-317`; `examples/urban_osm/buildings_viewer_interactive.py:95-168`, `:276-298`.

Recipes it helps: urban OSM city.

What remains unsolved: building render adapter, line stroke helper, and daycycle/video.

Smallest useful implementation slice: local GeoDataFrame-to-MapScene context, no network fetching.

Smallest useful test/golden/diagnostic: cached/sample OSM-like geometries produce roads, water, and buildings layers.

### 14. Label Recipe Defaults / Font Fixture

What it adds to MapScene: deterministic label presets, font fixture defaults, and terrain label style bundles.

Existing code it should reuse: `python/forge3d/label_plan.py:750-1091` and existing `LabelLayer`.

Scripts it unlocks: `examples/labels_styles_picking/fuji_labels_demo.py:60-61`, `:720-732`, `:743-807`, `:823-840`, `:906-961`, `:1054-1069`; `examples/mapscene_vector_labels.py:62-91`, `:106`, `:123-124`; mapscene bundle label/furniture examples at `docs/carto-engine/golden-map-recipe-capability-audit.md:1630`.

Recipes it helps: terrain label and mapscene showcases.

What remains unsolved: full cartographic plate and specialized callout drawing.

Smallest useful implementation slice: deterministic default font and style preset only.

Smallest useful test/golden/diagnostic: same label plan accepts/rejects same labels across runs.

### 15. PointCloud + COG MapScene Source Adapters

What it adds to MapScene: first-class MapScene source descriptors and adapters for point cloud and COG-backed rasters.

Existing code it should reuse: `python/forge3d/pointcloud.py:594-608`, `python/forge3d/cog.py:346-368`, and current `PointCloudLayer` placeholder at `python/forge3d/map_scene.py:1492-1530`.

Scripts it unlocks: `examples/pointcloud_cog/pointcloud_viewer_interactive.py:321-352`; `examples/pointcloud_cog/cog_streaming_demo.py:155-179`, `:275-282`, `:302-319`, `:349`.

Recipes it helps: pointcloud COG.

What remains unsolved: interactive streaming UI, LOD policy, and camera presets.

Smallest useful implementation slice: static point-cloud layer and COG raster tile source validation.

Smallest useful test/golden/diagnostic: descriptor validates and renders deterministic placeholder or sample points.

## D. Per-Recipe Matrix

Legend: `F` full, `M` major, `P` partial, `D` diagnostic, `-` none.

| Recipe | 1 Manifest | 2 Align | 3 Plate | 4 PopSurf | 5 Blend | 6 Goldens | 7 Terrain | 8 CatStyle | 9 Line | 10 Bivar | 11 Building | 12 Trace | 13 Urban | 14 Labels | 15 Pt/COG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| terrain_demo | P | - | - | - | - | D | M | - | - | - | - | D | - | - | - |
| terrain_label | P | - | P | - | - | D | P | - | - | - | - | D | - | M | - |
| landcover_esri_terrain_viewer | P | P | P | - | M | D | P | M | - | - | - | D | - | - | - |
| climate_bivariate | P | P | P | - | M | D | P | - | - | M | - | D | - | - | - |
| hydrology_river | P | P | P | - | M | D | P | - | M | - | - | D | - | - | - |
| mapscene_showcases | P | P | P | - | - | D | - | - | - | - | M | D | - | P | - |
| terrain_relief_rem | P | P | P | - | M | D | P | - | - | - | - | D | - | - | - |
| population_spike_worldpop | P | P | P | M | P | D | P | - | - | - | - | D | - | - | - |
| population_ghsl_3d | P | P | P | M | P | D | P | - | - | P | - | D | - | - | - |
| builtup_cover_3d | P | P | P | P | M | D | P | P | - | - | - | D | - | - | - |
| pointcloud_cog | P | P | - | - | - | D | - | - | - | - | - | D | - | - | M |
| urban_osm_city | P | P | P | - | - | D | P | - | P | - | P | D | M | - | - |
| luxembourg_rail_overlay | P | M | - | - | - | D | P | - | M | - | - | D | - | - | - |

Minimum MapScene-native sets:

| Recipe | Minimum capability set |
|---|---|
| terrain_demo | 1 + 7 + 6 |
| terrain_label | 1 + 14 + 6, plus existing label primitives |
| landcover_esri_terrain_viewer | 1 + 2 + 5 + 8 + 3 + 6 |
| climate_bivariate | 1 + 2 + 10 + 5 + 3 + 6 |
| hydrology_river | 1 + 2 + 9 + 3 + 6; add 5 for two-pass variants |
| mapscene_showcases | 1 + 6; add 11 for building showcase, 3 for rendered furniture |
| terrain_relief_rem | 1 + 2 + 5 + 3 + 7 + 6 |
| population_spike_worldpop | 1 + 2 + 4 + 3 + 6 |
| population_ghsl_3d | 1 + 2 + 4 + 3 + 6; add 10 for population-temperature bivariate |
| builtup_cover_3d | 1 + 2 + 8 + 5 + 3 + 6 |
| pointcloud_cog | 1 + 15 + 2 + 6 |
| urban_osm_city | 1 + 13 + 11 + 9 + 3 + 6 |
| luxembourg_rail_overlay | 1 + 2 + 9 + 6 |

## E. Final Recommendation

### 1. Build the recipe manifest/provenance layer first

Why it is high leverage: it touches every active recipe with low blast radius and gives later MapScene enrichment work a stable target.

Which recipes/scripts it unlocks: all 43 active scripts across all 13 active recipes, partially.

Likely files touched: `python/forge3d/recipe_manifest.py`, a `.pyi` stub, manifest fixture JSON files, manifest tests, and the carto-engine schema docs.

What should explicitly not be built yet: rendering behavior, top-level exports, automatic recipe execution, or new data loaders.

### 2. Add MapScene spatial alignment diagnostics using existing GIS helpers

Why it is high leverage: alignment is the repeated hidden blocker behind landcover, climate, hydrology, population, built-up, COG, and Luxembourg rail. The repo already has most primitives.

Which recipes/scripts it unlocks: 36 scripts across 11 recipes, mostly partial but high leverage.

Likely files touched: `python/forge3d/map_scene.py`, maybe thin helpers around `python/forge3d/gis.py`, validation tests, and fixture manifests.

What should explicitly not be built yet: a new reprojection engine, new dependencies, or automatic destructive resampling.

### 3. Render MapScene map furniture / plate composition

Why it is high leverage: many scripts already render the data but still use local PIL/matplotlib glue for legends, title blocks, scale bars, north arrows, and captions.

Which recipes/scripts it unlocks: 33 scripts across 10 recipes, including first-batch landcover, climate, hydrology, and mapscene showcases.

Likely files touched: `python/forge3d/map_scene.py`, `python/forge3d/_map_scene_render.py`, `python/forge3d/legend.py`, `python/forge3d/scale_bar.py`, `python/forge3d/north_arrow.py`, and plate/golden tests.

What should explicitly not be built yet: HTML reports, PDF export, video layout, or a general-purpose layout engine.

## Out of Scope Appendix

The following recipes were intentionally excluded from ranking and score calculations: `labels_styles_picking` as a standalone recipe, `wildfire_smoke`, `satellite_timelapse`, `osm_city_flood_daycycle`, and `humanity_globe_video`.
