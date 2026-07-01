# 1. Executive summary

forge3d is already a capable cartographic rendering engine for terrain-first map production: the repository proves heightfield rendering, viewer snapshots, raster draping, vector overlays, labels, map furniture, Cloud Optimized GeoTIFF access, point-cloud rendering, typed MapScene recipes, validation, bundles, support diagnostics, and visual-golden testing. The strongest proven capabilities are terrain rendering and camera control (`README.md:44-48`, `python/forge3d/__init__.pyi:160-171`), raster overlays (`python/forge3d/terrain_params.py:1403-1462`, `python/forge3d/__init__.pyi:398`), labels and vector overlays (`examples/labels_styles_picking/fuji_labels_demo.py:536-1081`, `python/forge3d/label_plan.py:214-507`), COG and point-cloud support (`python/forge3d/cog.py:37-368`, `python/forge3d/pointcloud.py:1-608`), and public MapScene/SceneRecipe behavior (`python/forge3d/__init__.py:431-451`, `python/forge3d/map_scene.py:1010-1044`, `python/forge3d/map_scene.py:1231-1867`). The weakest proven gaps are not raw rendering features and not absence of a typed scene API; they are missing evidence-derived recipe-family manifests/helpers for gallery-style recipes, missing reusable alignment/styling/compositing helpers, thin public coverage for example-only workflows, and insufficient golden fixtures for the polished recipe outputs already present in `examples/`. I inspected 1,743 files by inventory, including 130 non-cache files under `examples/` and 117 runnable example scripts, and kept 23 recipe families from repo evidence. The biggest difference between "an example can produce a beautiful output" and "forge3d exposes this as reusable engine capability" is that scripts repeatedly hardcode data acquisition, reprojection, rasterization, palettes, blend modes, camera settings, lighting, labels, and final poster layout, while forge3d exposes lower-level rendering plus selected typed scene pieces. Top 5 forge3d-only next actions: create recipe-family manifests from existing examples, add reusable raster/vector alignment and drape helpers, promote repeated palette and bivariate/categorical styling logic, add deterministic golden fixtures for representative recipes, and harden MapScene render adapters plus diagnostics for buildings, point clouds, overlays, labels, and map furniture.

Counts used by this audit:

| Metric | Count | Evidence |
| --- | ---: | --- |
| Total files inspected by inventory | 1,743 | Non-cache files under `examples/`, `docs/`, `tests/`, `python/forge3d/`, `src/`, `scripts/`, `specs/`, `assets/`, `data/`, standard root config files, and generated/golden roots |
| Runnable examples inspected | 117 | `Get-ChildItem examples -Recurse -File -Force` filtered for non-cache `.py` files containing `__main__` |
| Support files under examples | 7 | Non-cache files under `examples/` excluding runnable `.py`, notebooks, and `examples/out/` |
| Notebooks inspected | 3 | Non-cache `.ipynb` files under `examples/` |
| Docs inspected | 105 | Non-cache files under `docs/` plus root project docs |
| Tests inspected | 230 | Non-cache files under `tests/` |
| Generated/golden assets inspected | 93 | Non-cache files under `tests/golden/`, `docs/gallery/images/`, `docs/tutorials/images/`, `examples/out/`, `output/`, `terrain_tv12_output/`, `terrain_tv2_output/`, and `logs/gallery-regen-20260315/` |
| Examples inspected, total non-cache | 130 | 117 runnable examples, 7 support files, 3 notebooks, 3 generated outputs under `examples/out/` |
| Recipe families discovered | 23 | Section 3 |
| Proven capabilities in the matrix | 19 | Section 5 rows with `proven_in_forge3d` after current-main reconciliation |
| Critical/high gaps | 11 | Section 6 rows G-001 through G-011 |

## Current-main reconciliation notes

- Claims kept from the original report: bottom-up recipe discovery from `examples/`, 23 recipe families, example-only status for most polished gallery-style scripts, and the need for recipe-level golden fixtures.
- Claims corrected: high-level recipe API is not `not_found`; `SceneRecipe`, `MapScene`, validation, render, save/load bundle, support matrices, and diagnostics exist as public forge3d evidence.
- Paths changed: canonical MapScene example references now use root-level MapScene example files; duplicate nested MapScene copies were verified but are no longer treated as canonical evidence paths.
- Counts changed: total inventory moved from 1,004 to 1,743 files; runnable example count is now explicit at 117; tests moved to 230; docs moved to 105; generated/golden assets moved to 93.
- Capability statuses changed because current main already has `SceneRecipe`, `MapScene.render`, `MapScene.validate`, `MapScene.save_bundle`/`load_bundle`, `tests/test_mapscene_recipe_contract.py`, `tests/test_mapscene_render_png.py`, `tests/test_mapscene_save_bundle.py`, and `tests/test_mapscene_support_status.py`.

# 2. Repository evidence inventory

Rows that name directories summarize the counted inventory. Rows that name files are the line-cited evidence used for recipe and capability findings.

| Path | Type | Relevant symbols / sections | Why it matters | Evidence quality |
| --- | --- | --- | --- | --- |
| `examples/` | examples inventory | 130 non-cache files: 117 runnable examples, 7 support files, 3 notebooks, 3 generated outputs under `examples/out/` | Primary recipe discovery source; includes root examples plus topic folders | high |
| `examples/terrain_viewer/` | examples directory | `terrain_demo.py`, `terrain_viewer_interactive.py`, camera and atmosphere demos | Terrain, viewer, camera, atmosphere recipe evidence | high |
| `examples/terrain_relief/` | examples directory | `colorado_rem_forge3d.py`, `platte_rem_forge3d.py` | Repeated REM terrain-plus-overlay pattern | high |
| `examples/landcover_esri/` | examples directory | Bosnia, Germany, Poland, Romania, Swiss landcover viewers | Repeated categorical raster over terrain workflow | high |
| `examples/climate_bivariate/` | examples directory | Belgium, France, Europe bivariate climate maps | Repeated bivariate style and legend workflow | high |
| `examples/population_spike_worldpop/` | examples directory | Poland population spike scripts | Spike and height-shade population workflow | high |
| `examples/population_ghsl/` | examples directory | GHSL population, built-up, bivariate population-temperature scripts | Regional population and built-up recipes | high |
| `examples/population_global_gpw/humanity_globe_video.py` | example | `classify_density`, `population_exaggeration`, `render_frame_density`, `render_video` (`:132-607`) | Globe video pipeline with population density classification and animation | high |
| `examples/hydrology_river_basins/` | examples directory | river basin, PNOA river, and river video scripts | River basin and waterways recipe family | high |
| `examples/wildfire_smoke/` | examples directory | California smoke effect and video scripts | Smoke/density overlay and animation recipes | high |
| `examples/satellite_timelapse/` | examples directory | Bryce Canyon, Khumbu, Mount Hood timelapse scripts | Temporal raster/terrain recipe evidence | high |
| `examples/urban_osm/` | examples directory | OSM city, daycycle, flood, rail, solar, lighthouse, travel-time, Helsinki scripts | Urban recipes with roads, water, buildings, labels, transport, flood, solar | high |
| `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`, `examples/mapscene_bundled_datasets_showcase.py`, `examples/mapscene_p1_assets_bundle_showcase.py`, `examples/mapscene_buildings_labels.py` | examples | canonical root-level MapScene examples; duplicate nested copies exist but are not used as canonical paths here | Typed scene grammar, render, validation, and bundle examples | high |
| `examples/labels_styles_picking/` | examples directory | `fuji_labels_demo.py`, `style_viewer_interactive.py`, picking demos | Label, style, vector, depth, halo, picking recipes | high |
| `examples/pointcloud_cog/` | examples directory | `pointcloud_viewer_interactive.py`, `cog_streaming_demo.py` | Point cloud and COG streaming recipes | high |
| `README.md` | docs | viewer snapshot (`:44-48`), feature list (`:55-87`) | Public documented capability surface | high |
| `docs/gallery/index.md` | docs | recipe-shaped gallery note (`:3`) | Confirms gallery is organized around rendered outputs | medium |
| `docs/tutorials/gis-track/02-drape-overlays-on-terrain.md` | docs | raster drape workflow (`:3-15`, `:61-69`) | Documents high-level raster drape and raw IPC boundary | high |
| `docs/tutorials/gis-track/03-build-a-map-plate.md` | docs | `Legend`, `ScaleBar`, `NorthArrow` (`:29-58`) | Map furniture and plate composition evidence | high |
| `docs/tutorials/python-track/03-point-clouds.md` | docs | point cloud load/snapshot (`:1-30`) | Public point-cloud workflow evidence | high |
| `docs/guides/offline_3d_map_rendering.md` | docs | MapScene support status (`:19-22`) | Documents typed scene maturity and underdeveloped paths | high |
| `docs/guides/data_and_scene_workflows.md` | docs | overlay/label/building/bundle workflows (`:62-181`) | Connects examples to scene and data APIs | high |
| `docs/guides/feature_map.md` | docs | typed scene features (`:19`) | Evidence of intended scene grammar | medium |
| `docs/guides/label_support_matrix.md` | docs | label support matrix (`:5-18`) | Differentiates supported and underdeveloped label behavior | high |
| `docs/guides/style_support_matrix.md` | docs | supported style layers (`:8-13`) | Shows style support is local/provided, not full streaming parity | high |
| `docs/guides/large_scene_support.md` | docs | memory/cache/LOD support (`:8-18`) | Scale limitations for large recipes | medium |
| `python/forge3d/__init__.pyi` | Python API | `Scene`, camera, render, GI, water, overlays (`:160-398`) | Public API contract for core rendering | high |
| `python/forge3d/map_scene.py` | Python API | typed scene classes and render/save/load (`:1`, `:223-1044`, `:1231-1867`) | Main forge3d-owned SceneRecipe / MapScene grammar | high |
| `python/forge3d/__init__.py` | Python API | public MapScene imports and `__all__` (`:431-451`, `:646-663`) | Confirms SceneRecipe and MapScene are public API, not hidden internals | high |
| `python/forge3d/terrain_demo.py` | Python API | colormap, camera/sun, render config (`:310-1083`) | Terrain demo pipeline and warnings | high |
| `python/forge3d/terrain_params.py` | Python API | overlay/vector/TerrainParams (`:1403-2237`) | Structured terrain, raster, and vector settings | high |
| `python/forge3d/style.py` | Python API | style support and vector configs (`:1-826`) | Local style support and validation | high |
| `python/forge3d/label_plan.py` | Python API | label candidates and export payloads (`:214-1150`) | Reusable label planning capability | high |
| `python/forge3d/legend.py` | Python API | `LegendConfig`, `Legend.render` (`:33-110`) | Map furniture capability | high |
| `python/forge3d/scale_bar.py` | Python API | `ScaleBarConfig`, `ScaleBar.render` (`:18-88`) | Map furniture capability | high |
| `python/forge3d/north_arrow.py` | Python API | `NorthArrowConfig`, `NorthArrow.render` (`:16-35`) | Map furniture capability | high |
| `python/forge3d/cog.py` | Python API | COG availability/open (`:37-368`) | COG terrain and raster ingest capability | high |
| `python/forge3d/pointcloud.py` | Python API | COPC/EPT support and renderer (`:1-608`) | Point-cloud recipe capability | high |
| `python/forge3d/buildings.py` | Python API | building helpers and support validation (`:1-613`) | Building footprint/extrusion support | high |
| `src/pointcloud/renderer.rs` | Rust source | `MemoryReport`, loading/cache (`:147-341`) | Native point-cloud memory/cache evidence | high |
| `src/colormap/mod.rs` | Rust source | built-in colormaps (`:13-41`) | Native colormap support | high |
| `src/bundle/manifest.rs` | Rust source | terrain metadata and camera bookmarks (`:53-117`) | Bundle manifest support | high |
| `src/bundle/mod.rs` | Rust source | bundle subdirectories (`:5-36`) | Scene asset packaging support | high |
| `tests/test_terrain_visual_goldens.py` | tests | terrain golden save/compare (`:149-283`) | Visual regression foundation | high |
| `tests/golden/terrain/` | fixtures | terrain golden PNGs | Existing golden image assets | high |
| `tests/test_mapscene_recipe_contract.py` | tests | public SceneRecipe/MapScene constructor and serialization (`:78-140`) | Public typed recipe API contract evidence | high |
| `tests/test_mapscene_render_png.py` | tests | MapScene render tests (`:10-254`) | Typed scene render behavior | high |
| `tests/test_mapscene_save_bundle.py` | tests | deterministic review bundle tests (`:68-164`) | Bundle evidence for typed scene recipes | high |
| `tests/test_mapscene_support_status.py` | tests | support diagnostics (`:106-169`) | Underdeveloped adapter diagnostics | high |
| `tests/test_vector_drape.py` | tests | vector drape settings/rendering (`:21-366`) | Vector overlay rendering evidence | high |
| `tests/test_map_plate_layout.py` | tests | legend/scale/map plate tests (`:156-305`) | Map furniture layout evidence | high |
| `tests/test_cog_streaming.py` | tests | COG API tests (`:81-294`) | COG public behavior evidence | high |
| `docs/gallery/images/` | generated assets | gallery PNG outputs `01` through `10` | Existing rendered-output candidates | medium |
| `output/` | generated assets | COG and vector SVG/PNG outputs | Generated evidence; provenance varies by file | medium |
| `terrain_tv12_output/`, `terrain_tv2_output/` | generated assets | terrain tile outputs | Generated terrain evidence | low |
| `examples/notebooks/` | notebooks | inspected as examples inventory | No stronger recipe evidence than scripts found | low |
| `logs/` | generated logs | audit JSON logs | Not relevant to cartographic recipe capability | not_relevant_after_inspection |

# 3. Discovered recipe families

| Recipe family | Name source | Representative files | Core visual/cartographic pattern | Required inputs | Main forge3d capabilities used | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `terrain_demo` | repo_file_name | `examples/terrain_viewer/terrain_demo.py`, `python/forge3d/terrain_demo.py` | Heightfield terrain with colormap, lighting, camera, PNG export | DEM / heightfield, palette, camera, lighting | `Scene`, terrain params, colormap, render PNG | `proven_in_forge3d` |
| `terrain_viewer_interactive` | repo_file_name | `examples/terrain_viewer/terrain_viewer_interactive.py`, `README.md:44-48` | Interactive terrain viewer, orbit camera, snapshots | DEM, camera parameters, viewer commands | viewer IPC, camera, snapshot | `proven_in_forge3d` |
| `terrain_relief_rem` | clustered_from_multiple_examples | `examples/terrain_relief/colorado_rem_forge3d.py`, `examples/terrain_relief/platte_rem_forge3d.py` | REM terrain, overlay, two-pass relief/poster composition | DEM, river/valley mask, overlay texture, palette, camera, lighting | terrain, overlay, snapshot | `exists_only_as_example_or_script_logic` |
| `landcover_esri_terrain_viewer` | clustered_from_multiple_examples | `examples/landcover_esri/bosnia_terrain_landcover_viewer.py`, Swiss/Germany/Poland/Romania variants | Categorical landcover draped on terrain with PBR lighting | DEM, categorical raster, class palette, camera, lighting | terrain, raster overlay, snapshot | `exists_only_as_example_or_script_logic` |
| `forest_cover_copernicus_italy` | repo_file_name | `examples/forest_cover_copernicus/italy_forest_cover_3d.py` | Forest-cover raster over terrain with vegetation palette | DEM, continuous/categorical vegetation raster, palette, camera | terrain, overlay, snapshot | `exists_only_as_example_or_script_logic` |
| `climate_bivariate` | clustered_from_multiple_examples | `examples/climate_bivariate/europe_bivariate_climate_map.py`, Belgium/France variants | Bivariate climate matrix with 2D legend plus 3D terrain render | DEM, two climate rasters, boundary mask, bivariate palette, legend | terrain, overlay, snapshot, map furniture by script | `exists_only_as_example_or_script_logic` |
| `population_spike_worldpop` | repo_file_name | `examples/population_spike_worldpop/poland_population_spikes.py`, `poland_population_spikes_height_shade.py` | Population spikes or height-shaded density over terrain/poster | population raster, optional DEM, palette, labels, camera | viewer IPC, overlay, snapshot | `exists_only_as_example_or_script_logic` |
| `population_ghsl_3d` | clustered_from_multiple_examples | `examples/population_ghsl/southeast_europe_population_3d.py`, `southeast_europe_population_spikes_enhanced_3d.py`, regional variants | GHSL population extrusion/spikes with reference overlay and final plate | GHSL population raster, boundaries, labels, palette, camera | terrain/overlay/snapshot plus script compositing | `exists_only_as_example_or_script_logic` |
| `builtup_cover_3d` | clustered_from_multiple_examples | `examples/population_ghsl/iberia_builtup_cover_3d.py`, Romania/Turkey variants | Built-up cover intensity over terrain | built-up raster, DEM, boundary mask, palette, camera | terrain, overlay, snapshot | `exists_only_as_example_or_script_logic` |
| `humanity_globe_video` | repo_file_name | `examples/population_global_gpw/humanity_globe_video.py`, root duplicate | Rotating globe video from population-density classes | GPW density raster, class palette, frame count, text labels | helper modules and image export; core globe recipe not public | `exists_only_as_example_or_script_logic` |
| `hydrology_river_basins` | repo_file_name | `examples/hydrology_river_basins/poland_river_basins_forge3d.py`, `pnoa_river_showcase.py`, `turkiye_river_basins_3d.py` | Basin polygons and river lines over terrain/poster | DEM, basin polygons, river lines, palette, camera, map text | terrain, overlay, snapshot; river styling in scripts | `exists_only_as_example_or_script_logic` |
| `wildfire_smoke` | repo_file_name | `examples/wildfire_smoke/california_fire_smoke_effect.py`, `california_wildfire_smoke_video.py`, `california_cigar_smoke_demo.py` | Smoke density plume overlay/animation | terrain/raster base, smoke density, wind params, palette, frame count | overlay and image/video export by scripts | `exists_only_as_example_or_script_logic` |
| `satellite_timelapse` | repo_file_name | `examples/satellite_timelapse/mount_hood_white_cloud_timelapse.py`, `khumbu_icefall_sentinel_timelapse.py`, `bryce_canyon_storm_timelapse.py` | Time-sequenced satellite/cloud/terrain frames | raster frames, DEM, timestamps, sun/camera parameters | terrain, overlay, snapshot; sequence control in scripts | `exists_only_as_example_or_script_logic` |
| `urban_osm_city` | clustered_from_multiple_examples | `examples/urban_osm/osm_city_demo.py`, `osm_city_daycycle.py`, `buildings_viewer_interactive.py` | OSM water/roads/rail/buildings/labels in city-scale render | OSM vectors, building footprints/heights, roads, water, labels, sun/camera | building helpers, water methods, vector overlay, snapshots | `exists_only_as_example_or_script_logic` |
| `osm_city_flood_daycycle` | repo_file_name | `examples/urban_osm/osm_city_flood_daycycle.py` | Flood extent animation over lightweight urban scene | waterway geometry, flood cells, terrain/urban surface, frame count, labels | image rendering helpers; flood model script-only | `exists_only_as_example_or_script_logic` |
| `rotterdam_solar_potential_shadow_study` | repo_file_name | `examples/urban_osm/rotterdam_solar_potential_shadow_study.py` | 3D building roof solar-potential and shadow study | building roof surfaces, sun position, irradiance reference, OSM context, palette | building mesh concepts, lighting/shadow concepts; solar analysis script-only | `exists_only_as_example_or_script_logic` |
| `luxembourg_rail_overlay` | repo_file_name | `examples/urban_osm/luxembourg_rail_overlay.py` | Rail lines converted to draped quads over terrain | vector lines, DEM, rail palette, z offset, camera | terrain, vector overlay, snapshot | `partially_proven` |
| `barcelona_travel_time_3d` | repo_file_name | `examples/urban_osm/barcelona_travel_time_3d.py`, root duplicate | Drive-time road network over 3D terrain poster | OSM roads, origin, speed model, DEM, road overlay, camera, poster text | terrain, raster overlay, snapshot; routing/rasterization script-only | `exists_only_as_example_or_script_logic` |
| `helsinki_transit_daycycle` | repo_file_name | `examples/urban_osm/helsinki_transit_daycycle.py`, root duplicate | Daily transit and traffic animation over OSM city surface | GTFS trips, traffic counts, buildings, roads, rails, camera, frame count | projection helpers and frame export in script; forge3d rendering limited | `exists_only_as_example_or_script_logic` |
| `uk_ireland_lighthouse_map` | repo_file_name | `examples/urban_osm/uk_ireland_lighthouse_map.py` | Lighthouse points and glow over terrain/poster | DEM/Terrarium samples, lighthouse points, labels, glow style, layout | terrain/sample helpers and image composition by script | `exists_only_as_example_or_script_logic` |
| `mapscene_showcases` | clustered_from_multiple_examples | `examples/mapscene_terrain_raster.py`, `examples/mapscene_vector_labels.py`, `examples/mapscene_bundled_datasets_showcase.py`, `examples/mapscene_p1_assets_bundle_showcase.py`, `examples/mapscene_buildings_labels.py` | Typed scene validation, deterministic render, bundles | synthetic DEM/raster/vector/labels/buildings, output spec | `MapScene`, `TerrainSource`, `RasterOverlay`, `LabelLayer`, bundles | `partially_proven` |
| `labels_styles_picking` | repo_file_name | `examples/labels_styles_picking/fuji_labels_demo.py`, `style_viewer_interactive.py`, picking demos | Labels, styled vector overlays, picking, depth/halo | DEM, points/lines/polygons, style JSON, font atlas, labels, camera | labels, style, vector overlays, picking | `proven_in_forge3d` |
| `pointcloud_cog` | repo_file_name | `examples/pointcloud_cog/pointcloud_viewer_interactive.py`, `cog_streaming_demo.py` | Point-cloud viewer plus COG terrain tile access | LAS/LAZ/COPC/EPT or COG, camera, cache budget, snapshot | point-cloud API, COG API, viewer snapshot | `proven_in_forge3d` |

# 4. Golden recipe cards

## Recipe: `terrain_demo`

### Name provenance

Derived from `examples/terrain_viewer/terrain_demo.py` and the matching Python API module `python/forge3d/terrain_demo.py`.

### Representative files

- `examples/terrain_viewer/terrain_demo.py`
- `python/forge3d/terrain_demo.py:310-1083`
- `python/forge3d/__init__.pyi:160-171`
- `tests/test_terrain_visual_goldens.py:149-283`

### Intended cartographic purpose

Render a DEM/heightfield as a shaded 3D terrain image with controlled palette, camera, sun, and PNG export.

### Required input data

- DEM / heightfield
- palette or colormap
- camera parameters
- lighting parameters
- export dimensions

### Evidence found in forge3d

- `python/forge3d/terrain_demo.py:310-357` builds terrain colormaps.
- `python/forge3d/terrain_demo.py:870-912` warns about camera/sun alignment.
- `python/forge3d/terrain_demo.py:1001-1083` constructs render config and renders.
- `python/forge3d/__init__.pyi:160-171` exposes `Scene`, `set_camera_look_at`, `set_height_from_r32f`, `render_png`, and `render_rgba`.
- `tests/test_terrain_visual_goldens.py:149-283` saves and compares terrain golden images.

### Current forge3d support status

`proven_in_forge3d`

### Pipeline reconstruction from evidence only

1. data ingest: heightfield data enters through `Scene.set_height_from_r32f` and terrain demo helpers.
2. CRS / projection / alignment, if present: `not_found` in this simple demo recipe.
3. clipping / masking, if present: `not_found`.
4. DEM / heightfield preprocessing: colormap and terrain parameter preparation in `python/forge3d/terrain_demo.py:310-1083`.
5. raster preprocessing: `not_found`.
6. vector preprocessing: `not_found`.
7. terrain generation: native `Scene` terrain path exposed in `python/forge3d/__init__.pyi:160-171`.
8. raster draping / overlay: `not_found`.
9. vector rendering / extrusion / overlay: `not_found`.
10. styling / palette / material assignment: colormap builder in `python/forge3d/terrain_demo.py:310-357`.
11. lighting / shading: camera/sun alignment handling in `python/forge3d/terrain_demo.py:870-912`.
12. camera / view: `set_camera_look_at` in `python/forge3d/__init__.pyi:160-171`.
13. labels / annotations / legend, if present: `not_found`.
14. export: `render_png` and `render_rgba` in `python/forge3d/__init__.pyi:160-171`.

### Hidden cartographic decisions

Colormap choices, camera presets, sun direction, terrain exaggeration, export size, and warnings around camera/sun alignment are embedded in terrain demo helpers rather than a recipe manifest.

### Reusable forge3d capability candidates

- candidate: `Terrain.from_heightmap(...)`
- candidate: `Terrain.style_with_colormap(...)`
- candidate: `Scene.set_lighting(...)`
- candidate: `Scene.export(...)`

### Gaps and risks

The recipe is proven but not represented as a named, validated forge3d recipe with declared inputs, defaults, and golden fixture metadata.

### Minimal acceptance criteria

- A deterministic terrain recipe fixture renders through public forge3d APIs.
- The fixture has a visual golden image and numeric tolerance.
- Camera, light, colormap, and export defaults are declared in data, not hidden in example code.

## Recipe: `terrain_viewer_interactive`

### Name provenance

Derived from `examples/terrain_viewer/terrain_viewer_interactive.py` and README viewer usage.

### Representative files

- `examples/terrain_viewer/terrain_viewer_interactive.py`
- `README.md:44-48`
- `README.md:81-87`

### Intended cartographic purpose

Open terrain interactively, set an orbit camera, and produce screenshots for inspection or map output.

### Required input data

- DEM / heightfield
- camera parameters
- optional overlay
- export dimensions

### Evidence found in forge3d

- `README.md:44-48` shows `open_viewer_async`, `set_orbit_camera`, and `snapshot("rainier.png")`.
- `README.md:81-87` lists interactive terrain, raster overlays, labels, picking, point-cloud/building viewers, camera animation, COG, and offscreen examples.

### Current forge3d support status

`proven_in_forge3d`

### Pipeline reconstruction from evidence only

1. data ingest: viewer loads terrain from file or helper path.
2. CRS / projection / alignment, if present: `not_found`.
3. clipping / masking, if present: `not_found`.
4. DEM / heightfield preprocessing: delegated to terrain loader.
5. raster preprocessing: optional overlay only.
6. vector preprocessing: optional vector overlay only.
7. terrain generation: viewer terrain command path.
8. raster draping / overlay: documented in README and GIS tutorial for raster overlays.
9. vector rendering / extrusion / overlay: supported by examples but not required by the minimal viewer recipe.
10. styling / palette / material assignment: viewer terrain params.
11. lighting / shading: viewer terrain/PBR commands.
12. camera / view: `set_orbit_camera` from `README.md:44-48`.
13. labels / annotations / legend, if present: optional.
14. export: `snapshot` from `README.md:44-48`.

### Hidden cartographic decisions

Orbit radius, look-at target, terrain z scale, overlay opacity, and snapshot dimensions are selected by scripts or user commands.

### Reusable forge3d capability candidates

- candidate: `ViewerRecipe.open_terrain(...)`
- candidate: `Scene.set_camera(...)`
- candidate: `Scene.snapshot(...)`

### Gaps and risks

The interactive path is strong, but defaults and command sequences are not captured as reusable recipes.

### Minimal acceptance criteria

- A viewer recipe can load a small DEM, apply a fixed camera, and write a deterministic PNG.
- The sequence is covered by docs and at least one smoke test that does not require manual interaction.

## Recipe: `terrain_relief_rem`

### Name provenance

Clustered from `examples/terrain_relief/colorado_rem_forge3d.py` and `examples/terrain_relief/platte_rem_forge3d.py`; both use REM terrain/poster logic.

### Representative files

- `examples/terrain_relief/colorado_rem_forge3d.py:1-11`
- `examples/terrain_relief/colorado_rem_forge3d.py:896-953`
- `examples/terrain_relief/colorado_rem_forge3d.py:1261-1382`
- `examples/terrain_relief/platte_rem_forge3d.py`

### Intended cartographic purpose

Render relative-elevation or river-valley terrain as a high-impact poster with terrain relief and overlay composition.

### Required input data

- DEM / heightfield
- river or valley mask
- overlay texture
- palette
- camera parameters
- lighting parameters
- map furniture / poster text

### Evidence found in forge3d

- `examples/terrain_relief/colorado_rem_forge3d.py:1-11` describes the RiverREM workflow.
- `examples/terrain_relief/colorado_rem_forge3d.py:896-953` builds REM and overlay images.
- `examples/terrain_relief/colorado_rem_forge3d.py:1261-1382` renders with forge3d viewer, terrain PBR, overlay, snapshot, and map furniture.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: script reads DEM and hydro/context data.
2. CRS / projection / alignment, if present: script-level alignment.
3. clipping / masking, if present: script-level basin/river masks.
4. DEM / heightfield preprocessing: REM computation in script.
5. raster preprocessing: overlay construction in script.
6. vector preprocessing: river/context geometry converted to masks in script.
7. terrain generation: forge3d viewer terrain path.
8. raster draping / overlay: viewer overlay load and preserve-color commands.
9. vector rendering / extrusion / overlay: vector information is rasterized before forge3d.
10. styling / palette / material assignment: hardcoded palette and blend constants in script.
11. lighting / shading: terrain PBR commands in script.
12. camera / view: camera constants in script.
13. labels / annotations / legend, if present: poster/map furniture in script.
14. export: viewer snapshot plus script composition.

### Hidden cartographic decisions

REM normalization, river mask width, terrain exaggeration, palette, overlay opacity, two-pass blend, camera angle, PBR values, title placement, and export dimensions are script constants.

### Reusable forge3d capability candidates

- candidate: `Terrain.compute_relative_elevation(...)`
- candidate: `RasterLayer.drape_on_terrain(...)`
- candidate: `Scene.render_passes(...)`
- candidate: `MapPlate.compose(...)`

### Gaps and risks

The terrain and snapshot parts are reusable, but REM generation, mask construction, pass combination, and poster layout are not public forge3d recipe capabilities.

### Minimal acceptance criteria

- A tiny REM fixture renders from DEM plus river mask through a forge3d recipe helper.
- The recipe writes color and relief outputs or a declared composite output.
- Golden comparison catches palette, blend, camera, and relief regressions.

## Recipe: `landcover_esri_terrain_viewer`

### Name provenance

Clustered from the landcover viewer files in `examples/landcover_esri/`; the family name follows the directory and file suffix.

### Representative files

- `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:123-153`
- `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:253-310`
- `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:456-582`
- `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:688-768`
- `examples/landcover_esri/swiss_terrain_landcover_viewer.py:992-1162`

### Intended cartographic purpose

Drape categorical landcover classes over terrain with lighting and export a map-like 3D view.

### Required input data

- DEM / heightfield
- categorical raster
- class palette
- camera parameters
- lighting parameters
- optional labels / annotations

### Evidence found in forge3d

- Bosnia example defines camera/terrain/PBR/relief constants at `:123-153`.
- Bosnia example builds DEM at `:253-310`.
- Bosnia example builds landcover class overlays at `:456-582`.
- Bosnia example composes and renders two passes at `:688-768`.
- Swiss example configures and renders terrain/landcover at `:992-1162`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: scripts load DEM and landcover raster.
2. CRS / projection / alignment, if present: handled in scripts.
3. clipping / masking, if present: region masks in scripts.
4. DEM / heightfield preprocessing: script DEM build/resampling.
5. raster preprocessing: class raster to RGBA overlay.
6. vector preprocessing: `not_found` for core recipe.
7. terrain generation: forge3d viewer terrain path.
8. raster draping / overlay: forge3d overlay load.
9. vector rendering / extrusion / overlay: `not_found`.
10. styling / palette / material assignment: class colors hardcoded in scripts.
11. lighting / shading: PBR constants in scripts.
12. camera / view: camera constants in scripts.
13. labels / annotations / legend, if present: script-level final composition.
14. export: viewer snapshot and optional postprocess.

### Hidden cartographic decisions

Class palette, class inclusion, alpha, land/water masks, relief blend, PBR values, camera bearing, export size, and final crop are script constants.

### Reusable forge3d capability candidates

- candidate: `CategoricalRasterLayer.from_classes(...)`
- candidate: `CategoricalRasterLayer.style_with_palette(...)`
- candidate: `RasterLayer.align_to_terrain(...)`
- candidate: `Scene.render_landcover_terrain(...)`

### Gaps and risks

Categorical styling, class legends, and raster alignment are not first-class forge3d recipe capabilities, so each region script repeats cartographic decisions.

### Minimal acceptance criteria

- A tiny categorical raster can be declared with class colors and draped on a DEM.
- A golden image verifies class colors, alpha, and terrain shading.
- Diagnostics fail when raster extent, shape, or class palette is incomplete.

## Recipe: `forest_cover_copernicus_italy`

### Name provenance

Derived from `examples/forest_cover_copernicus/italy_forest_cover_3d.py`.

### Representative files

- `examples/forest_cover_copernicus/italy_forest_cover_3d.py`
- `examples/forest_cover_copernicus/`

### Intended cartographic purpose

Render forest-cover intensity or class information for Italy over 3D terrain.

### Required input data

- DEM / heightfield
- forest-cover raster
- vegetation palette
- boundary mask
- camera parameters
- lighting parameters

### Evidence found in forge3d

- The topic folder and script name identify a Copernicus forest-cover 3D recipe.
- The shared landcover and raster-overlay APIs provide the forge3d rendering substrate (`python/forge3d/terrain_params.py:1403-1462`).

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: forest-cover and terrain inputs are loaded by the example script.
2. CRS / projection / alignment, if present: script-level handling.
3. clipping / masking, if present: boundary mask handled by script.
4. DEM / heightfield preprocessing: script-level DEM preparation.
5. raster preprocessing: forest-cover raster converted to styled overlay.
6. vector preprocessing: boundary mask only if supplied by script.
7. terrain generation: forge3d terrain path.
8. raster draping / overlay: forge3d raster overlay path.
9. vector rendering / extrusion / overlay: `not_found`.
10. styling / palette / material assignment: vegetation palette in script.
11. lighting / shading: camera/light constants in script.
12. camera / view: camera constants in script.
13. labels / annotations / legend, if present: script composition.
14. export: snapshot or image output from script.

### Hidden cartographic decisions

Forest palette, opacity, mask handling, terrain exaggeration, camera angle, and output crop are not declared in a reusable forge3d object.

### Reusable forge3d capability candidates

- candidate: `RasterLayer.style_with_colormap(...)`
- candidate: `RasterLayer.mask_to_boundary(...)`
- candidate: `Scene.recipe_from_layers(...)`

### Gaps and risks

The recipe depends on script-only raster preprocessing and palette decisions; no specific forest-cover fixture validates the path.

### Minimal acceptance criteria

- A small forest-cover raster fixture renders with a declared palette.
- A validation report confirms DEM/raster shape and extent compatibility.
- A golden catches palette and alpha regressions.

## Recipe: `climate_bivariate`

### Name provenance

Clustered from `examples/climate_bivariate/europe_bivariate_climate_map.py`, Belgium, and France bivariate examples.

### Representative files

- `examples/climate_bivariate/europe_bivariate_climate_map.py:1-11`
- `examples/climate_bivariate/europe_bivariate_climate_map.py:356`
- `examples/climate_bivariate/europe_bivariate_climate_map.py:476`
- `examples/climate_bivariate/europe_bivariate_climate_map.py:696`
- `examples/climate_bivariate/europe_bivariate_climate_map.py:751-987`
- `examples/climate_bivariate/europe_bivariate_climate_map.py:1173-1256`

### Intended cartographic purpose

Show two climate variables in a bivariate color matrix, with a 2D map/legend and a 3D terrain view.

### Required input data

- DEM / heightfield
- two continuous rasters
- boundary mask
- bivariate palette
- legend labels
- camera parameters
- lighting parameters

### Evidence found in forge3d

- File docstring documents TerraClimate, clipping, smoothing, 4x4 bivariate classes, BlueGold matrix, and 2D plus 3D output at `:1-11`.
- `_build_terrarium_dem` is at `:356`.
- `_build_terraclimate_climatology` is at `:476`.
- `_build_texture` is at `:696`.
- Legend and 2D composition are at `:751-987`.
- `_render_3d` is at `:1173-1256`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: TerraClimate and Terrarium/DEM data loaded by script.
2. CRS / projection / alignment, if present: clipping/alignment in script.
3. clipping / masking, if present: region clipping and mask in script.
4. DEM / heightfield preprocessing: `_build_terrarium_dem`.
5. raster preprocessing: climatology, smoothing, binning.
6. vector preprocessing: boundary geometry/mask in script.
7. terrain generation: forge3d 3D render path.
8. raster draping / overlay: bivariate texture draped as overlay.
9. vector rendering / extrusion / overlay: boundary/context in image composition.
10. styling / palette / material assignment: 4x4 matrix and legend in script.
11. lighting / shading: render constants in script.
12. camera / view: render constants in script.
13. labels / annotations / legend, if present: legend code in script.
14. export: 2D and 3D outputs by script.

### Hidden cartographic decisions

Bin count, smoothing, matrix colors, legend geometry, mask edge treatment, terrain exaggeration, overlay opacity, camera, and final layout are hardcoded.

### Reusable forge3d capability candidates

- candidate: `BivariateRasterStyle.from_matrix(...)`
- candidate: `RasterLayer.bin_quantiles(...)`
- candidate: `Legend.bivariate(...)`
- candidate: `Scene.render_bivariate_terrain(...)`

### Gaps and risks

Bivariate classification and legend rendering exist only in scripts; forge3d has colormap support but not a first-class bivariate raster style.

### Minimal acceptance criteria

- A tiny two-raster fixture renders with a declared bivariate matrix and legend.
- Validation checks equal shape/extents and bin configuration.
- Golden catches swapped axes, palette drift, and legend layout regressions.

## Recipe: `population_spike_worldpop`

### Name provenance

Derived from `examples/population_spike_worldpop/` and the Poland population spike scripts.

### Representative files

- `examples/population_spike_worldpop/poland_population_spikes.py:13-15`
- `examples/population_spike_worldpop/poland_population_spikes.py:206-207`
- `examples/population_spike_worldpop/poland_population_spikes_height_shade.py:1-10`
- `examples/population_spike_worldpop/poland_population_spikes_height_shade.py:80-203`
- `examples/population_spike_worldpop/poland_population_spikes_height_shade.py:215-362`

### Intended cartographic purpose

Render gridded population density as spikes or height-shaded density with final map composition.

### Required input data

- population raster
- boundary mask
- optional DEM / heightfield
- palette
- camera parameters
- labels / annotations
- export dimensions

### Evidence found in forge3d

- `poland_population_spikes.py:13-15` describes direct IPC, 4K snapshot, and final plate composition.
- `poland_population_spikes.py:206-207` defines `render_spike_map`.
- `poland_population_spikes_height_shade.py:1-10` documents height-shade workflow.
- `poland_population_spikes_height_shade.py:80-203` renders height-shade map.
- `poland_population_spikes_height_shade.py:215-362` cleans and composes the final map.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: population raster loaded by script.
2. CRS / projection / alignment, if present: script-level region alignment.
3. clipping / masking, if present: boundary mask in script.
4. DEM / heightfield preprocessing: optional terrain/height values in script.
5. raster preprocessing: density normalization and spike/height conversion in script.
6. vector preprocessing: boundary/reference vectors in script.
7. terrain generation: viewer/IPC path when used.
8. raster draping / overlay: optional overlay path.
9. vector rendering / extrusion / overlay: population spike geometry is script-generated.
10. styling / palette / material assignment: palette and density scaling in script.
11. lighting / shading: script render settings.
12. camera / view: script constants.
13. labels / annotations / legend, if present: final plate composition in script.
14. export: snapshot and final PNG.

### Hidden cartographic decisions

Density scaling, spike height, color ramp, mask cleanup, title placement, crop, and output size are embedded in scripts.

### Reusable forge3d capability candidates

- candidate: `RasterLayer.to_spikes(...)`
- candidate: `RasterLayer.height_shade(...)`
- candidate: `Scene.add_population_spikes(...)`
- candidate: `MapPlate.compose_population(...)`

### Gaps and risks

Population-specific spike generation is not a public forge3d layer; visual quality depends on script constants and cleanup.

### Minimal acceptance criteria

- A small population raster renders deterministic spikes or height shade through a forge3d helper.
- The helper declares scaling and palette.
- Golden comparison covers spike height, color, and map plate output.

## Recipe: `population_ghsl_3d`

### Name provenance

Clustered from population-focused GHSL files such as `examples/population_ghsl/southeast_europe_population_3d.py`, `examples/population_ghsl/southeast_europe_population_spikes_enhanced_3d.py`, and `examples/population_ghsl/iberian_peninsula_population_3d.py`.

### Representative files

- `examples/population_ghsl/southeast_europe_population_3d.py:172-191`
- `examples/population_ghsl/southeast_europe_population_3d.py:307-561`
- `examples/population_ghsl/southeast_europe_population_spikes_enhanced_3d.py:41-105`
- `examples/population_ghsl/southeast_europe_population_spikes_enhanced_3d.py:282-323`

### Intended cartographic purpose

Render regional GHSL population surfaces with enhanced palettes, overlays, labels, and final map plates.

### Required input data

- GHSL population raster
- boundary mask
- optional DEM / relief raster
- reference overlay
- labels / annotations
- palette
- camera and lighting parameters

### Evidence found in forge3d

- `southeast_europe_population_3d.py:172-191` modifies shared spike settings.
- `southeast_europe_population_3d.py:307-561` builds reference overlay, labels, and final plate.
- `southeast_europe_population_spikes_enhanced_3d.py:41-105` declares color, relief, and PBR constants.
- `southeast_europe_population_spikes_enhanced_3d.py:282-323` renders the enhanced spike map.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: GHSL raster loaded by script.
2. CRS / projection / alignment, if present: script-level alignment.
3. clipping / masking, if present: region masks in script.
4. DEM / heightfield preprocessing: relief values and terrain state in script.
5. raster preprocessing: population scaling and texture/spike generation in script.
6. vector preprocessing: boundaries and labels in script.
7. terrain generation: forge3d viewer terrain where used.
8. raster draping / overlay: viewer overlay or image composition.
9. vector rendering / extrusion / overlay: script-generated overlays.
10. styling / palette / material assignment: hardcoded palette/PBR constants.
11. lighting / shading: script PBR and relief constants.
12. camera / view: script camera constants.
13. labels / annotations / legend, if present: final plate in script.
14. export: snapshot/final PNG from script.

### Hidden cartographic decisions

Population thresholds, spike scaling, relief mix, label placement, reference overlay opacity, camera, and final layout are not declared in reusable forge3d data.

### Reusable forge3d capability candidates

- candidate: `PopulationLayer.from_raster(...)`
- candidate: `PopulationLayer.style_spikes(...)`
- candidate: `LabelLayer.from_features(...)`

### Gaps and risks

Regional GHSL recipes repeat population and layout logic. Public forge3d supports rendering pieces, not the reusable population recipe.

### Minimal acceptance criteria

- A small GHSL-like raster fixture renders with declared scaling and labels.
- Validation catches missing mask, invalid scale, and missing palette.
- Golden catches spike scale and plate layout regressions.

## Recipe: `builtup_cover_3d`

### Name provenance

Clustered from `examples/population_ghsl/iberia_builtup_cover_3d.py`, `romania_builtup_cover_3d.py`, and `turkey_builtup_cover_3d.py`.

### Representative files

- `examples/population_ghsl/iberia_builtup_cover_3d.py`
- `examples/population_ghsl/romania_builtup_cover_3d.py`
- `examples/population_ghsl/turkey_builtup_cover_3d.py`

### Intended cartographic purpose

Show built-up cover intensity or class values as a 3D terrain/map visualization.

### Required input data

- built-up cover raster
- DEM / heightfield
- boundary mask
- palette
- camera parameters
- lighting parameters

### Evidence found in forge3d

- The GHSL folder contains three built-up cover 3D examples.
- The same terrain, overlay, camera, and snapshot substrate is proven by `python/forge3d/terrain_params.py:1403-2237` and `README.md:55-87`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: built-up raster loaded in scripts.
2. CRS / projection / alignment, if present: script-level handling.
3. clipping / masking, if present: region mask in scripts.
4. DEM / heightfield preprocessing: script-level terrain preparation.
5. raster preprocessing: built-up values styled into overlay.
6. vector preprocessing: boundaries only if used by script.
7. terrain generation: forge3d terrain path.
8. raster draping / overlay: forge3d overlay path.
9. vector rendering / extrusion / overlay: `not_found`.
10. styling / palette / material assignment: built-up palette in scripts.
11. lighting / shading: script constants.
12. camera / view: script constants.
13. labels / annotations / legend, if present: script composition.
14. export: snapshot/final PNG.

### Hidden cartographic decisions

Value scaling, opacity, palette, boundary treatment, terrain exaggeration, and export crop are hidden in scripts.

### Reusable forge3d capability candidates

- candidate: `RasterLayer.style_continuous(...)`
- candidate: `RasterLayer.apply_boundary_mask(...)`
- candidate: `Scene.render_builtup_cover(...)`

### Gaps and risks

The built-up workflow has no typed recipe or golden fixture, so changes to overlay blending or palette logic can drift unnoticed.

### Minimal acceptance criteria

- A tiny built-up raster fixture renders through a declared forge3d recipe.
- Validation checks raster shape, mask, and palette.
- Golden catches intensity-to-color regressions.

## Recipe: `humanity_globe_video`

### Name provenance

Derived from `examples/population_global_gpw/humanity_globe_video.py`.

### Representative files

- `examples/population_global_gpw/humanity_globe_video.py:2-27`
- `examples/population_global_gpw/humanity_globe_video.py:132-167`
- `examples/population_global_gpw/humanity_globe_video.py:331-607`

### Intended cartographic purpose

Render a rotating population-density globe video with density classes, palette, lighting, text overlay, and MP4 export.

### Required input data

- GPW population-density raster
- class palette
- frame count / FPS
- camera/orbit parameters
- text labels / annotations

### Evidence found in forge3d

- File docstring identifies the Humanity Globe population-density video and GPW data at `:2-27`.
- `classify_density`, `population_exaggeration`, and `roma_class_palette` are at `:132-167`.
- `render_frame_density`, text composition, frame writing, MP4 encoding, preview, and `render_video` are at `:331-607`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: GPW raster loaded by script.
2. CRS / projection / alignment, if present: latitude/longitude sampling in script.
3. clipping / masking, if present: `not_found`.
4. DEM / heightfield preprocessing: `not_found`.
5. raster preprocessing: density classification and exaggeration.
6. vector preprocessing: `not_found`.
7. terrain generation: globe mesh/raster sampling in script, not a public forge3d globe layer.
8. raster draping / overlay: population layer painted in script.
9. vector rendering / extrusion / overlay: `not_found`.
10. styling / palette / material assignment: density class palette in script.
11. lighting / shading: `_light` in script.
12. camera / view: orbit longitude and frame sampling in script.
13. labels / annotations / legend, if present: `compose_frame_text`.
14. export: PNG frames and MP4 encoding in script.

### Hidden cartographic decisions

Class thresholds, population exaggeration, globe lighting, orbit path, text placement, preview size, frame rate, and MP4 command are script constants.

### Reusable forge3d capability candidates

- candidate: `GlobeLayer.from_latlon_raster(...)`
- candidate: `RasterLayer.classify(...)`
- candidate: `Scene.render_frames(...)`

### Gaps and risks

forge3d does not expose this as a reusable globe or animation recipe; the script implements classification, rendering, and encoding.

### Minimal acceptance criteria

- A tiny lat/lon raster renders one deterministic globe preview through a forge3d-owned helper.
- The helper declares class thresholds, palette, orbit, and output frame size.
- A golden still frame catches palette, class, and lighting regressions.

## Recipe: `hydrology_river_basins`

### Name provenance

Derived from `examples/hydrology_river_basins/` and repeated river basin scripts.

### Representative files

- `examples/hydrology_river_basins/poland_river_basins_forge3d.py:95-114`
- `examples/hydrology_river_basins/poland_river_basins_forge3d.py:212-276`
- `examples/hydrology_river_basins/poland_river_basins_forge3d.py:408-608`
- `examples/hydrology_river_basins/poland_river_basins_forge3d.py:652-737`
- `examples/hydrology_river_basins/pnoa_river_showcase.py:62-397`
- `examples/hydrology_river_basins/turkiye_river_basins_3d.py:1-17`

### Intended cartographic purpose

Render river basins, waterways, and terrain as 3D-style regional maps or posters.

### Required input data

- DEM / heightfield
- basin polygons
- river / waterway lines
- palette
- line widths
- camera parameters
- map furniture / text

### Evidence found in forge3d

- Poland script parses render dimensions and prepare-only mode at `:95-114`.
- Basin and river loading/clipping are at `:212-276`.
- DEM and overlay construction are at `:408-608`.
- Forge3d render path loads terrain, PBR, overlay, and snapshot at `:652-737`.
- PNOA river showcase renders waterways and map furniture at `:62-397`.
- Turkiye helper describes poster aspect, camera defaults, river styling, and cache invalidation at `:1-17`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: DEM, basin polygons, and river lines loaded by scripts.
2. CRS / projection / alignment, if present: scripts reproject/clip to region.
3. clipping / masking, if present: country/basin clipping in scripts.
4. DEM / heightfield preprocessing: Terrarium/DEM sampling and masking in scripts.
5. raster preprocessing: overlay image creation in scripts.
6. vector preprocessing: basin and river line clipping/rasterization in scripts.
7. terrain generation: forge3d viewer terrain.
8. raster draping / overlay: preserve-color overlay in forge3d.
9. vector rendering / extrusion / overlay: vector data rasterized to overlay before forge3d.
10. styling / palette / material assignment: palette and river widths in scripts.
11. lighting / shading: PBR constants in scripts.
12. camera / view: camera constants in scripts.
13. labels / annotations / legend, if present: map text/furniture in scripts.
14. export: viewer snapshot and final image.

### Hidden cartographic decisions

Basin palette, river width by order, river lightening, mask treatment, map furniture placement, camera radius, z scale, and output dimensions are script constants.

### Reusable forge3d capability candidates

- candidate: `VectorLayer.clip_to_boundary(...)`
- candidate: `VectorLayer.style_by_attribute(...)`
- candidate: `VectorLayer.rasterize_for_drape(...)`
- candidate: `HydrologyRecipe.from_basins_and_rivers(...)`

### Gaps and risks

Vector-to-overlay conversion and hydrology styling are script-only, so repeated basin maps cannot be validated or reused consistently.

### Minimal acceptance criteria

- A tiny basin polygon plus river line fixture renders over a DEM.
- Attribute-driven river widths and basin colors are declared.
- Golden catches line width, color, clipping, and drape regressions.

## Recipe: `wildfire_smoke`

### Name provenance

Derived from `examples/wildfire_smoke/` and root wildfire smoke examples.

### Representative files

- `examples/wildfire_smoke/california_fire_smoke_effect.py:1-2`
- `examples/wildfire_smoke/california_fire_smoke_effect.py:350-662`
- `examples/wildfire_smoke/california_fire_smoke_effect.py:885-955`
- `examples/wildfire_smoke/california_cigar_smoke_demo.py:10524-10759`

### Intended cartographic purpose

Render smoke plumes or density fields over a terrain/context map, including animated video output.

### Required input data

- terrain or base raster
- smoke density field
- wind parameters
- palette / alpha ramp
- frame count
- camera parameters

### Evidence found in forge3d

- Smoke overlay purpose is documented in `california_fire_smoke_effect.py:1-2`.
- Wind, smoke simulation, and density-to-RGBA conversion are at `:350-662`.
- `render_smoke_frame_with_forge3d` is at `:885-955`.
- Cigar smoke demo video loop is at `california_cigar_smoke_demo.py:10524-10759`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: base data and smoke parameters loaded by scripts.
2. CRS / projection / alignment, if present: script-level alignment.
3. clipping / masking, if present: smoke/map masks in script.
4. DEM / heightfield preprocessing: terrain prep where used.
5. raster preprocessing: smoke density simulation and RGBA conversion.
6. vector preprocessing: fire/source geometry in script if present.
7. terrain generation: forge3d terrain path where used.
8. raster draping / overlay: smoke texture over render/base.
9. vector rendering / extrusion / overlay: `not_found`.
10. styling / palette / material assignment: density alpha/color ramp in script.
11. lighting / shading: script camera/light settings.
12. camera / view: script constants.
13. labels / annotations / legend, if present: script overlay text.
14. export: still frame and video loop in scripts.

### Hidden cartographic decisions

Wind field, diffusion, plume opacity, color ramp, temporal easing, frame size, and video output are hardcoded.

### Reusable forge3d capability candidates

- candidate: `RasterLayer.style_density_alpha(...)`
- candidate: `Scene.render_overlay_sequence(...)`
- candidate: `AtmosphericOverlay.smoke(...)`

### Gaps and risks

Smoke simulation and temporal overlay export are not reusable forge3d capabilities; changes are not protected by recipe goldens.

### Minimal acceptance criteria

- A tiny smoke density fixture renders as a draped transparent overlay.
- Declared alpha/color ramp produces a stable golden frame.
- Diagnostics reject invalid density shape or alpha range.

## Recipe: `satellite_timelapse`

### Name provenance

Derived from `examples/satellite_timelapse/`.

### Representative files

- `examples/satellite_timelapse/mount_hood_white_cloud_timelapse.py:194-300`
- `examples/satellite_timelapse/khumbu_icefall_sentinel_timelapse.py`
- `examples/satellite_timelapse/bryce_canyon_storm_timelapse.py`

### Intended cartographic purpose

Render time-sequenced satellite or cloud/terrain frames with stable camera and changing overlays.

### Required input data

- DEM / heightfield
- raster frames
- timestamps
- overlay opacity
- camera parameters
- lighting parameters
- frame count / FPS

### Evidence found in forge3d

- Mount Hood timelapse handles overlay, scene, sun state, and terrain IPC at `:194-300`.
- The topic folder contains Sentinel, cloud, and storm timelapse variants.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: raster frame inputs loaded by scripts.
2. CRS / projection / alignment, if present: script-level alignment.
3. clipping / masking, if present: script-level masks.
4. DEM / heightfield preprocessing: terrain prep in script.
5. raster preprocessing: per-frame texture/overlay creation.
6. vector preprocessing: `not_found`.
7. terrain generation: forge3d terrain IPC path.
8. raster draping / overlay: per-frame overlay commands.
9. vector rendering / extrusion / overlay: `not_found`.
10. styling / palette / material assignment: script color/opacity settings.
11. lighting / shading: sun state per frame.
12. camera / view: fixed camera in script.
13. labels / annotations / legend, if present: timestamp/title in script.
14. export: PNG frames and optional video.

### Hidden cartographic decisions

Frame pacing, overlay opacity, sun interpolation, timestamp placement, camera lock, and video encoding are script-only.

### Reusable forge3d capability candidates

- candidate: `Scene.render_sequence(...)`
- candidate: `RasterLayerSequence.from_frames(...)`
- candidate: `Scene.set_sun_for_frame(...)`

### Gaps and risks

Temporal recipe control is not a first-class forge3d concept; animated outputs rely on ad hoc loops.

### Minimal acceptance criteria

- A two-frame raster overlay fixture renders deterministic frames with fixed camera.
- A test verifies frame naming and metadata.
- Golden stills catch overlay alignment and lighting drift.

## Recipe: `urban_osm_city`

### Name provenance

Clustered from `examples/urban_osm/osm_city_demo.py`, `osm_city_daycycle.py`, and `buildings_viewer_interactive.py`.

### Representative files

- `examples/urban_osm/osm_city_demo.py:49-77`
- `examples/urban_osm/osm_city_demo.py:257-267`
- `examples/urban_osm/osm_city_demo.py:699-714`
- `examples/urban_osm/osm_city_demo.py:780-859`
- `examples/urban_osm/osm_city_demo.py:1828-1966`
- `examples/urban_osm/buildings_viewer_interactive.py:524-574`
- `python/forge3d/buildings.py:268-613`

### Intended cartographic purpose

Render an urban context map with buildings, water, roads, rails, labels, daycycle lighting, and optional viewer interaction.

### Required input data

- building footprints
- building heights
- roads
- water polygons
- rail lines
- labels / annotations
- palette
- sun/camera parameters

### Evidence found in forge3d

- OSM city colors are declared at `osm_city_demo.py:49-77`.
- Building height inference/binning is at `:257-267`.
- Buildings, water, roads, and rails parsing is at `:699-714`.
- Water, roads, and buildings layers are at `:780-859`.
- Water surface, AO, and roof outline masks are at `:1828-1966`.
- Building viewer sets terrain/PBR/vector overlay/snapshot at `buildings_viewer_interactive.py:524-574`.
- Public building helpers and support validation are in `python/forge3d/buildings.py:268-613`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: OSM/building vectors loaded by scripts.
2. CRS / projection / alignment, if present: local metric projection in scripts.
3. clipping / masking, if present: AOI filtering in scripts.
4. DEM / heightfield preprocessing: optional terrain in viewer path.
5. raster preprocessing: masks and context layers in scripts.
6. vector preprocessing: roads/water/rails/buildings parsed and localized in scripts.
7. terrain generation: viewer path for building overlay examples.
8. raster draping / overlay: optional base/surface layers.
9. vector rendering / extrusion / overlay: building meshes and road/water layers in scripts, building helpers in API.
10. styling / palette / material assignment: hardcoded colors and heights.
11. lighting / shading: daycycle/sun state in scripts.
12. camera / view: script camera constants.
13. labels / annotations / legend, if present: labels in scripts.
14. export: snapshots or image/video frames.

### Hidden cartographic decisions

Road widths, water colors, building height defaults, roof outlines, AO amount, sun schedule, label rules, and camera are script-level decisions.

### Reusable forge3d capability candidates

- candidate: `BuildingLayer.extrude(...)`
- candidate: `UrbanContextLayer.from_osm_features(...)`
- candidate: `VectorLayer.style_roads(...)`
- candidate: `Scene.set_daycycle_lighting(...)`

### Gaps and risks

forge3d has building helpers, but the urban recipe still depends on script-only OSM parsing, styling, layer ordering, and layout.

### Minimal acceptance criteria

- A tiny building/water/road fixture renders with declared layer order and styles.
- Diagnostics report missing height fields and default extrusion behavior.
- Golden catches road width, water color, building extrusion, and camera regressions.

## Recipe: `osm_city_flood_daycycle`

### Name provenance

Derived from `examples/urban_osm/osm_city_flood_daycycle.py`.

### Representative files

- `examples/urban_osm/osm_city_flood_daycycle.py:158-224`
- `examples/urban_osm/osm_city_flood_daycycle.py:530-743`
- `examples/urban_osm/osm_city_flood_daycycle.py:746-989`
- `examples/urban_osm/osm_city_flood_daycycle.py:1047-1173`
- `examples/urban_osm/osm_city_flood_daycycle.py:1205-1270`

### Intended cartographic purpose

Animate a connected HAND-style flood extent over an urban surface with water level labels.

### Required input data

- waterway geometry
- urban surface / lightweight scene
- flood cell size
- flood start/end levels
- camera/projection parameters
- frame count
- labels / annotations

### Evidence found in forge3d

- Argparse documents flood-only OSM city timelapse and connected HAND-style model at `:158-224`.
- Flood activation and lightweight scene construction are at `:530-743`.
- Flood cell projection and scene preparation are at `:746-989`.
- Flood mask/surface/frame rendering is at `:1047-1173`.
- Main frame loop and encoding path are at `:1205-1270`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: OSM city/waterway data loaded by script.
2. CRS / projection / alignment, if present: local metric context in script.
3. clipping / masking, if present: AOI filtering and flood source geometry.
4. DEM / heightfield preprocessing: lightweight surface model in script.
5. raster preprocessing: flood mask generation.
6. vector preprocessing: waterway and surface geometry projection.
7. terrain generation: script renders lightweight scene, not public terrain recipe.
8. raster draping / overlay: flood surface layer composited in script.
9. vector rendering / extrusion / overlay: city surfaces from script meshes.
10. styling / palette / material assignment: water/flood colors in script.
11. lighting / shading: script triangle shading.
12. camera / view: projection functions in script.
13. labels / annotations / legend, if present: flood overlay labels.
14. export: PNG frames and video.

### Hidden cartographic decisions

Flood cell size, activation model, transition width, water alpha, label content, progress bar, frame pacing, and video encoding are script constants.

### Reusable forge3d capability candidates

- candidate: `FloodLayer.from_activation_grid(...)`
- candidate: `RasterLayer.animate_threshold(...)`
- candidate: `Scene.render_frames(...)`

### Gaps and risks

Flood modeling and animation are script-only. Without fixtures, flood extent or overlay alpha changes can silently alter the map.

### Minimal acceptance criteria

- A tiny activation grid renders two flood levels over a simple surface.
- Declared water color, alpha, and transition width are validated.
- Golden frames catch flood extent and label regressions.

## Recipe: `rotterdam_solar_potential_shadow_study`

### Name provenance

Derived from `examples/urban_osm/rotterdam_solar_potential_shadow_study.py`.

### Representative files

- `examples/urban_osm/rotterdam_solar_potential_shadow_study.py:7-10`
- `examples/urban_osm/rotterdam_solar_potential_shadow_study.py:167-254`
- `examples/urban_osm/rotterdam_solar_potential_shadow_study.py:341-363`
- `examples/urban_osm/rotterdam_solar_potential_shadow_study.py:664-752`
- `examples/urban_osm/rotterdam_solar_potential_shadow_study.py:770-936`
- `examples/urban_osm/rotterdam_solar_potential_shadow_study.py:986-1140`

### Intended cartographic purpose

Render 3D roof solar potential and shadow risk on building surfaces with urban context.

### Required input data

- roof/building geometry
- OSM context layers
- sun position / date-time
- irradiance reference
- solar palette
- camera parameters
- labels / summary text

### Evidence found in forge3d

- File comments list OSM context and warn solar values are planning estimates at `:7-10`.
- Data classes define context, assumptions, roof surfaces, mesh stats, summary, and layout at `:167-254`.
- Solar color/bin functions are at `:341-363`.
- Roof surface extraction is at `:664-752`.
- Sun, orientation, shadow, and roof evaluation are at `:770-936`.
- Mesh layers and solar-color surfaces are built at `:986-1140`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: building/roof/context data loaded by script.
2. CRS / projection / alignment, if present: metric localization in script.
3. clipping / masking, if present: AOI clipping in script.
4. DEM / heightfield preprocessing: `not_found` for main roof recipe.
5. raster preprocessing: `not_found`.
6. vector preprocessing: roof surfaces and context layers built in script.
7. terrain generation: urban mesh generation in script.
8. raster draping / overlay: `not_found`.
9. vector rendering / extrusion / overlay: roof meshes built and styled in script.
10. styling / palette / material assignment: solar palette/binning in script.
11. lighting / shading: sun/shadow logic in script.
12. camera / view: layout/camera constants in script.
13. labels / annotations / legend, if present: summary/layout in script.
14. export: still image and optional video.

### Hidden cartographic decisions

Solar min/max, bin count, shadow penalty, roof fallback rules, palette, context colors, sun fallback, mesh layer order, and summary layout are hardcoded.

### Reusable forge3d capability candidates

- candidate: `BuildingLayer.roof_surfaces(...)`
- candidate: `BuildingLayer.style_by_metric(...)`
- candidate: `Scene.set_sun_from_datetime(...)`
- candidate: `BuildingLayer.shadow_study(...)`

### Gaps and risks

Building mesh rendering exists in parts, but solar evaluation, roof extraction, color binning, and shadow-study semantics are not reusable forge3d capabilities.

### Minimal acceptance criteria

- A tiny roof-surface fixture renders with a declared metric palette.
- A deterministic sun position produces stable shadow/lighting.
- Golden catches roof color, context layer, and summary layout regressions.

## Recipe: `luxembourg_rail_overlay`

### Name provenance

Derived from `examples/urban_osm/luxembourg_rail_overlay.py`.

### Representative files

- `examples/urban_osm/luxembourg_rail_overlay.py:1-25`
- `examples/urban_osm/luxembourg_rail_overlay.py:156-275`
- `tests/test_vector_drape.py:21-366`

### Intended cartographic purpose

Drape rail lines over terrain as visible line/quads with controlled width, color, and z offset.

### Required input data

- DEM / heightfield
- vector lines
- line width
- color / palette
- z offset / drape settings
- camera parameters

### Evidence found in forge3d

- File docstring describes rail overlay workflow at `:1-25`.
- GeoPackage lines are converted to triangle quads at `:156-275`.
- Vector drape rendering/settings tests are in `tests/test_vector_drape.py:21-366`.

### Current forge3d support status

`partially_proven`

### Pipeline reconstruction from evidence only

1. data ingest: rail vector file loaded by script.
2. CRS / projection / alignment, if present: script handles projection/localization.
3. clipping / masking, if present: script clips/filter lines.
4. DEM / heightfield preprocessing: terrain loaded for drape.
5. raster preprocessing: `not_found`.
6. vector preprocessing: lines converted to quads.
7. terrain generation: forge3d terrain path.
8. raster draping / overlay: `not_found`.
9. vector rendering / extrusion / overlay: forge3d vector overlay path, with script-generated quads.
10. styling / palette / material assignment: rail width/color in script.
11. lighting / shading: terrain lighting in script.
12. camera / view: camera constants in script.
13. labels / annotations / legend, if present: `not_found`.
14. export: snapshot.

### Hidden cartographic decisions

Rail width, cap/join handling, z offset, depth bias, color, terrain exaggeration, and camera are script-level constants.

### Reusable forge3d capability candidates

- candidate: `VectorLineLayer.from_lines(...)`
- candidate: `VectorLineLayer.stroke(width=..., join=...)`
- candidate: `VectorLayer.drape_on_terrain(offset=...)`

### Gaps and risks

Vector drape is tested, but line-to-quad conversion and cartographic stroke styling are not exposed as a simple public recipe helper.

### Minimal acceptance criteria

- A small line fixture renders as a draped line without manual triangulation in user code.
- Tests cover width, z offset, depth bias, and color.
- Golden catches line placement and occlusion regressions.

## Recipe: `barcelona_travel_time_3d`

### Name provenance

Derived from `examples/urban_osm/barcelona_travel_time_3d.py`.

### Representative files

- `examples/urban_osm/barcelona_travel_time_3d.py:2-9`
- `examples/urban_osm/barcelona_travel_time_3d.py:744-937`
- `examples/urban_osm/barcelona_travel_time_3d.py:1063-1233`
- `examples/urban_osm/barcelona_travel_time_3d.py:1475-1644`
- `examples/urban_osm/barcelona_travel_time_3d.py:1700-1993`
- `examples/urban_osm/barcelona_travel_time_3d.py:1999-2147`

### Intended cartographic purpose

Render estimated drive-time road reachability on 3D terrain as a poster.

### Required input data

- OSM roads
- origin point
- road speeds / maxspeed tags
- DEM / heightfield
- road overlay texture
- palette
- camera parameters
- poster text

### Evidence found in forge3d

- Docstring lists OSM roads, travel-time ramp, terrain drape, and forge3d snapshot at `:2-9`.
- Speed parsing, graph building, Dijkstra, and segment selection are at `:744-937`.
- Heightmap and road overlay construction are at `:1063-1233`.
- Blend/composite logic is at `:1475-1644`.
- forge3d snapshot, overlay commands, two render passes, and fallback settings are at `:1700-1993`.
- Main workflow is at `:1999-2147`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: OSM roads, boundary/coastline, and terrain tiles loaded by script.
2. CRS / projection / alignment, if present: script UTM/local extent logic.
3. clipping / masking, if present: Spain/coast/land masks in script.
4. DEM / heightfield preprocessing: Terrarium heightmap construction.
5. raster preprocessing: road overlay rasterization and blur/glow.
6. vector preprocessing: roads parsed, graph built, travel times computed.
7. terrain generation: forge3d viewer terrain.
8. raster draping / overlay: road overlay loaded and preserved.
9. vector rendering / extrusion / overlay: vector roads rasterized before forge3d.
10. styling / palette / material assignment: sequential travel-time ramp in script.
11. lighting / shading: PBR and relief pass settings in script.
12. camera / view: camera radius/pitch settings in script.
13. labels / annotations / legend, if present: poster composition in script.
14. export: raw snapshot and final poster output.

### Hidden cartographic decisions

Speed defaults, travel-time cutoff, line width, glow blur, origin marker size, terrain relief scaling, blend mode, camera, memory-safe fallback, and poster layout are hardcoded.

### Reusable forge3d capability candidates

- candidate: `NetworkLayer.estimate_travel_time(...)`
- candidate: `VectorLineLayer.style_by_value(...)`
- candidate: `RasterLayer.screen_blend(...)`
- candidate: `MapPlate.compose_travel_time(...)`

### Gaps and risks

Road network analysis, line styling, overlay rasterization, and two-pass compositing are script-only.

### Minimal acceptance criteria

- A tiny road graph fixture produces deterministic travel-time lines and draped output.
- Validation checks origin, speed defaults, cutoff, and terrain/overlay alignment.
- Golden catches road color ramp, glow, and terrain blend regressions.

## Recipe: `helsinki_transit_daycycle`

### Name provenance

Derived from `examples/urban_osm/helsinki_transit_daycycle.py`.

### Representative files

- `examples/urban_osm/helsinki_transit_daycycle.py:155-190`
- `examples/urban_osm/helsinki_transit_daycycle.py:565-758`
- `examples/urban_osm/helsinki_transit_daycycle.py:986-1124`
- `examples/urban_osm/helsinki_transit_daycycle.py:1156-1235`
- `examples/urban_osm/helsinki_transit_daycycle.py:1321-1347`
- `examples/urban_osm/helsinki_transit_daycycle.py:1499-1596`

### Intended cartographic purpose

Render a day-long transit and road-traffic animation over Helsinki city context.

### Required input data

- GTFS trips
- road traffic counts
- OSM roads / rails / water
- building footprints / heights
- camera/projection parameters
- route colors
- frame count / FPS
- labels / legend

### Evidence found in forge3d

- Parser description says daily transit timelapse with OSM basemap layers at `:155-190`.
- OSM surface and building layers are built at `:565-758`.
- GTFS service/trip loading and active positions are at `:986-1124`.
- Transit and car overlays are projected at `:1156-1235`.
- Legend/header helpers are at `:1321-1347`.
- Main frame loop adds car/transit/text overlays at `:1499-1596`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: GTFS, traffic counts, and OSM/building data loaded by script.
2. CRS / projection / alignment, if present: metric context and localization in script.
3. clipping / masking, if present: AOI radius filters.
4. DEM / heightfield preprocessing: `not_found` for main recipe.
5. raster preprocessing: grayscale/base overlays in script.
6. vector preprocessing: roads, rails, buildings, trips, traffic paths.
7. terrain generation: city surface rendered by script helpers.
8. raster draping / overlay: image overlays in script.
9. vector rendering / extrusion / overlay: moving points/flows projected in script.
10. styling / palette / material assignment: route colors and dark map palette in script.
11. lighting / shading: daycycle settings in script.
12. camera / view: prepared scene projection.
13. labels / annotations / legend, if present: header and legend in script.
14. export: PNG frames and MP4.

### Hidden cartographic decisions

Route type colors, vehicle density, active point size, dark-map brightness, legend order, frame schedule, camera, and traffic hour shares are hardcoded.

### Reusable forge3d capability candidates

- candidate: `TemporalPointLayer.from_tracks(...)`
- candidate: `TrafficLayer.from_counts(...)`
- candidate: `Scene.render_time_window(...)`

### Gaps and risks

Temporal transport rendering is script-only; forge3d does not yet expose reusable animated point/flow layers.

### Minimal acceptance criteria

- A tiny GTFS-like fixture renders two time steps with fixed route colors.
- Validation catches missing stop times, route colors, and projection mismatch.
- Golden frames catch point placement and legend regressions.

## Recipe: `uk_ireland_lighthouse_map`

### Name provenance

Derived from `examples/urban_osm/uk_ireland_lighthouse_map.py`.

### Representative files

- `examples/urban_osm/uk_ireland_lighthouse_map.py:642-723`
- `examples/urban_osm/uk_ireland_lighthouse_map.py:1026-1554`

### Intended cartographic purpose

Render lighthouse locations over UK/Ireland terrain with glow, labels, and poster layout.

### Required input data

- DEM / heightfield
- lighthouse points
- labels / annotations
- glow palette
- map layout
- camera parameters

### Evidence found in forge3d

- Terrarium sampling is at `:642-723`.
- Terrain map, glow, layout, and main workflow are at `:1026-1554`.

### Current forge3d support status

`exists_only_as_example_or_script_logic`

### Pipeline reconstruction from evidence only

1. data ingest: DEM/Terrarium and lighthouse points loaded by script.
2. CRS / projection / alignment, if present: script coordinate sampling.
3. clipping / masking, if present: region mask in script.
4. DEM / heightfield preprocessing: Terrarium sampling.
5. raster preprocessing: glow layers and background image.
6. vector preprocessing: lighthouse points.
7. terrain generation: terrain/render map in script.
8. raster draping / overlay: glow/composite in script.
9. vector rendering / extrusion / overlay: point symbols in script.
10. styling / palette / material assignment: glow and terrain palette in script.
11. lighting / shading: script shading.
12. camera / view: layout constants in script.
13. labels / annotations / legend, if present: labels and layout in script.
14. export: final image.

### Hidden cartographic decisions

Glow radius, label size, point symbol, terrain palette, mask edge, title placement, and output dimensions are script constants.

### Reusable forge3d capability candidates

- candidate: `PointLayer.symbolize(...)`
- candidate: `PointLayer.glow(...)`
- candidate: `LabelLayer.place_points(...)`

### Gaps and risks

Point symbol/glow and poster layout are not first-class forge3d capabilities for this recipe.

### Minimal acceptance criteria

- A tiny point fixture renders glow and labels over terrain.
- Validation catches missing label fields and invalid glow radius.
- Golden catches glow radius, label placement, and point color regressions.

## Recipe: `mapscene_showcases`

### Name provenance

Clustered from canonical root-level MapScene example files.

### Representative files

- `examples/mapscene_terrain_raster.py:1-105`
- `examples/mapscene_vector_labels.py:1-140`
- `examples/mapscene_bundled_datasets_showcase.py:4-198`
- `examples/mapscene_p1_assets_bundle_showcase.py:26-228`
- `python/forge3d/map_scene.py:1-1867`
- `tests/test_mapscene_render_png.py:10-250`

### Intended cartographic purpose

Declare map scenes as typed data, validate them, render deterministic PNGs, and package assets into bundles.

### Required input data

- DEM / heightfield
- raster overlay
- vector polygons/lines/points
- labels / annotations
- optional buildings / point clouds
- output spec
- bundle assets

### Evidence found in forge3d

- `mapscene_terrain_raster.py:1-105` creates terrain+raster scene and render/bundle output.
- `mapscene_vector_labels.py:1-140` creates terrain+vector+labels scene.
- Bundle showcases use label planning, deterministic render, and bundles.
- `python/forge3d/map_scene.py:1044` defines `MapScene`; `:1231` `load_bundle`; `:1249` `validate`; `:1771` `render`; `:1867` `save_bundle`.
- `python/forge3d/map_scene.py:1527` and `:1631` identify placeholder fallback diagnostics for point cloud and building render adapters.

### Current forge3d support status

`partially_proven`

### Pipeline reconstruction from evidence only

1. data ingest: typed sources reference files or generated fixture assets.
2. CRS / projection / alignment, if present: scene metadata supports dimensions; full CRS alignment is underdeveloped.
3. clipping / masking, if present: layer metadata; clipping is underdeveloped.
4. DEM / heightfield preprocessing: synthetic or source DEM via `TerrainSource`.
5. raster preprocessing: `RasterOverlay`.
6. vector preprocessing: `VectorOverlay` / feature layers.
7. terrain generation: render path supports fixture-backed terrain.
8. raster draping / overlay: terrain+raster example.
9. vector rendering / extrusion / overlay: vector labels example; some adapters partial.
10. styling / palette / material assignment: layer style data and support matrices.
11. lighting / shading: scene/render metadata.
12. camera / view: scene view fields.
13. labels / annotations / legend, if present: `LabelLayer` and map furniture classes.
14. export: `OutputSpec`, `render`, `save_bundle`.

### Hidden cartographic decisions

Support levels, placeholder rendering, fallback diagnostics, layer support, and small fixture defaults are encoded in MapScene code and examples rather than recipe-level declarations.

### Reusable forge3d capability candidates

- candidate: `MapScene.from_recipe(...)`
- candidate: `MapScene.validate_recipe_support(...)`
- candidate: `MapScene.render_fixture(...)`

### Gaps and risks

MapScene is the right forge3d-owned direction, but key adapters are partial and many recipe families still bypass it.

### Minimal acceptance criteria

- MapScene can represent one terrain+raster+vector+label recipe with deterministic render.
- Diagnostics are stable for unsupported building/point-cloud paths.
- Bundled scene reloads and renders the same output.

## Recipe: `labels_styles_picking`

### Name provenance

Derived from `examples/labels_styles_picking/`.

### Representative files

- `examples/labels_styles_picking/fuji_labels_demo.py:2-15`
- `examples/labels_styles_picking/fuji_labels_demo.py:536-1081`
- `examples/labels_styles_picking/style_viewer_interactive.py:2-36`
- `examples/labels_styles_picking/style_viewer_interactive.py:444-550`
- `examples/labels_styles_picking/picking_demo.py:2-9`
- `python/forge3d/style.py:281-826`
- `python/forge3d/label_plan.py:214-1150`

### Intended cartographic purpose

Render terrain labels, line labels, callouts, style-driven vector overlays, and interactive feature picking.

### Required input data

- DEM / heightfield
- points / lines / polygons
- style JSON
- labels / annotations
- font atlas
- z offsets
- camera parameters

### Evidence found in forge3d

- Fuji labels docstring and snapshot usage are at `fuji_labels_demo.py:2-15`.
- Fuji loads terrain, PBR, font atlas, labels, line labels, and snapshot at `:536-1081`.
- Style viewer documents style JSON vector overlays at `style_viewer_interactive.py:2-36` and loads terrain/overlay/snapshot at `:444-550`.
- Picking demo documents feature picking at `picking_demo.py:2-9`.
- `python/forge3d/style.py:281-826` validates style support and builds vector/label configs.
- `python/forge3d/label_plan.py:214-1150` plans label candidates and payloads.

### Current forge3d support status

`proven_in_forge3d`

### Pipeline reconstruction from evidence only

1. data ingest: DEM and vector/style/label inputs loaded by examples.
2. CRS / projection / alignment, if present: terrain-local coordinate transforms in examples.
3. clipping / masking, if present: nodata mask in style viewer.
4. DEM / heightfield preprocessing: terrain load.
5. raster preprocessing: optional nodata mask overlay.
6. vector preprocessing: GPKG/style vectors converted to overlay payloads.
7. terrain generation: viewer terrain path.
8. raster draping / overlay: optional mask overlay.
9. vector rendering / extrusion / overlay: vector overlay API and style configs.
10. styling / palette / material assignment: style support module and example args.
11. lighting / shading: PBR, sky, denoise options in Fuji example.
12. camera / view: script camera and terrain settings.
13. labels / annotations / legend, if present: labels, line labels, callouts, font atlas.
14. export: viewer snapshot.

### Hidden cartographic decisions

Font atlas limits, ASCII stripping, label offsets, z offsets, halo width, depth bias, style support subset, and terrain-local transforms are scattered across examples and helper modules.

### Reusable forge3d capability candidates

- candidate: `LabelLayer.from_points(...)`
- candidate: `VectorLayer.style_from_spec(...)`
- candidate: `Scene.pick_features(...)`

### Gaps and risks

The capability is proven, but recipe-level defaults and fixture coverage should lock down label placement, halo, depth, and style behavior.

### Minimal acceptance criteria

- A deterministic Fuji-sized label fixture renders with point and line labels.
- Style validation reports unsupported layers with stable codes.
- Golden catches label anchor, halo, and depth regressions.

## Recipe: `pointcloud_cog`

### Name provenance

Derived from `examples/pointcloud_cog/`.

### Representative files

- `examples/pointcloud_cog/pointcloud_viewer_interactive.py:2-42`
- `examples/pointcloud_cog/pointcloud_viewer_interactive.py:235-386`
- `examples/pointcloud_cog/cog_streaming_demo.py:4-27`
- `examples/pointcloud_cog/cog_streaming_demo.py:118-285`
- `python/forge3d/cog.py:37-368`
- `python/forge3d/pointcloud.py:1-608`
- `src/pointcloud/renderer.rs:147-341`
- `tests/test_cog_streaming.py:81-294`

### Intended cartographic purpose

Load point clouds and COG terrain/raster tiles for interactive viewing, cache-aware inspection, and snapshots.

### Required input data

- LAS / LAZ / COPC / EPT point cloud
- COG URL/path
- cache budget
- camera parameters
- rendering parameters
- snapshot path

### Evidence found in forge3d

- Point-cloud viewer docstring says load LAZ/LAS, orbit camera, and take snapshots at `:2-42`.
- Viewer CLI and snapshot path are at `:235-386`.
- COG demo docstring says streaming terrain from COG without pre-tiling at `:4-27`.
- COG benchmark/tile/heightmap functions are at `:118-285`.
- `python/forge3d/cog.py:37-368` exposes COG availability and opening.
- `python/forge3d/pointcloud.py:1-608` exposes COPC/EPT support and renderer.
- `src/pointcloud/renderer.rs:147-341` reports memory and handles loading/cache.
- `tests/test_cog_streaming.py:81-294` tests COG API behavior.

### Current forge3d support status

`proven_in_forge3d`

### Pipeline reconstruction from evidence only

1. data ingest: point cloud or COG opened through forge3d APIs.
2. CRS / projection / alignment, if present: point/COG metadata available; full map recipe alignment is limited.
3. clipping / masking, if present: tile window or point subset logic.
4. DEM / heightfield preprocessing: COG heightmap tile/grid path.
5. raster preprocessing: tile fetch and mosaic in COG demo.
6. vector preprocessing: `not_found`.
7. terrain generation: COG terrain visualization path.
8. raster draping / overlay: `not_found` for point-cloud core; COG terrain uses heightmap.
9. vector rendering / extrusion / overlay: point cloud renderer, not vector overlay.
10. styling / palette / material assignment: point color/size and terrain visualization settings.
11. lighting / shading: viewer settings.
12. camera / view: orbit camera and viewer commands.
13. labels / annotations / legend, if present: `not_found`.
14. export: snapshot.

### Hidden cartographic decisions

Cache budget, tile grid selection, point size, color fallback, camera, and memory policy thresholds are partly APIs and partly example defaults.

### Reusable forge3d capability candidates

- candidate: `PointCloudLayer.from_path(...)`
- candidate: `CogTerrainSource.open(...)`
- candidate: `Scene.set_cache_budget(...)`

### Gaps and risks

Core point-cloud and COG APIs are proven, but MapScene render integration and cross-layer alignment need stronger fixtures.

### Minimal acceptance criteria

- A tiny point-cloud/COG fixture renders a deterministic snapshot or reports stable diagnostics when rendering is unavailable.
- Cache/memory report values are testable.
- MapScene can represent point-cloud and COG layers with clear support status.

# 5. Capability audit matrix

| Capability | Required by recipe families | Current status | Evidence | What exists today | What is missing | Recommended forge3d abstraction | Priority | Acceptance criteria |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Heightfield ingest/render | terrain, landcover, hydrology, travel-time, COG | `proven_in_forge3d` | `python/forge3d/__init__.pyi:160-171`, `tests/test_terrain_visual_goldens.py:149-283` | `Scene` heightfield render and PNG paths | Recipe metadata around terrain defaults | `Terrain.from_heightmap(...)` | `P0_foundation` | Small DEM renders deterministic PNG |
| Terrain mesh generation | terrain and all terrain-overlay recipes | `proven_in_forge3d` | `python/forge3d/terrain_params.py:1767-2237` | Structured terrain params | Recipe-level z-scale defaults | `TerrainParams` recipe wrapper | `P0_foundation` | Terrain dimensions/z scale validated |
| Terrain PBR / relief shading | terrain, landcover, REM, hydrology, urban | `proven_in_forge3d` | `python/forge3d/terrain_demo.py:870-912`, `python/forge3d/__init__.pyi:184-208` | Lighting/GI/PBR settings | Named cartographic lighting presets | `Scene.set_lighting(...)` | `P1_core_recipe_support` | Lighting defaults reproduce golden |
| Raster drape overlay | landcover, climate, travel-time, smoke, satellite | `proven_in_forge3d` | `python/forge3d/terrain_params.py:1403-1462`, `docs/tutorials/gis-track/02-drape-overlays-on-terrain.md:3-69` | Overlay settings and viewer overlay path | Alignment validation and recipe fixture | `RasterLayer.drape_on_terrain(...)` | `P0_foundation` | Misaligned raster emits diagnostic |
| Continuous colormaps | terrain, built-up, climate, solar | `proven_in_forge3d` | `src/colormap/mod.rs:13-41`, `python/forge3d/terrain_demo.py:310-357` | Built-in colormaps and demo builder | Data-aware normalization metadata | `RasterLayer.style_with_colormap(...)` | `P1_core_recipe_support` | Min/max and palette are declared/tested |
| Vector polygon overlay | hydrology, urban, MapScene, labels | `proven_in_forge3d` | `python/forge3d/style.py:764-826`, `tests/test_vector_drape.py:21-366` | Vector overlay configs and tests | Attribute-driven recipe styling | `VectorLayer.style_by_attribute(...)` | `P1_core_recipe_support` | Polygon fixture renders with style |
| Vector line overlay | rail, hydrology, travel-time, labels | `proven_in_forge3d` | `tests/test_vector_drape.py:21-366`, `examples/urban_osm/luxembourg_rail_overlay.py:156-275` | Drape/render support | Public line stroke helper without manual quads | `VectorLineLayer.stroke(...)` | `P1_core_recipe_support` | Line fixture renders width/z offset |
| Vector z/depth/halo | rail, labels, picking | `proven_in_forge3d` | `examples/labels_styles_picking/fuji_labels_demo.py:327-336`, `tests/test_vector_drape.py:21-366` | Depth, bias, halo settings | Golden coverage across terrain slopes | `VectorLayer.drape_settings(...)` | `P2_quality_and_scale` | Halo/depth golden passes |
| Labels and annotations | labels, lighthouse, population, Helsinki | `proven_in_forge3d` | `docs/guides/label_support_matrix.md:5-18`, `examples/labels_styles_picking/fuji_labels_demo.py:720-1081` | Point/line/callout label APIs | Recipe defaults and font fixture | `LabelLayer.from_features(...)` | `P1_core_recipe_support` | Label fixture renders deterministic text |
| Label planning/export | labels and MapScene | `proven_in_forge3d` | `python/forge3d/label_plan.py:214-1150` | Candidate planning and payloads | Wider recipe integration | `LabelPlan.for_scene(...)` | `P2_quality_and_scale` | Overlap result is stable |
| Map furniture | map plates, climate, hydrology, population | `proven_in_forge3d` | `docs/tutorials/gis-track/03-build-a-map-plate.md:29-58`, `tests/test_map_plate_layout.py:156-305` | Legend, scale bar, north arrow | Unified plate recipe and bivariate legend | `MapPlate` / `Legend.bivariate(...)` | `P1_core_recipe_support` | Furniture fixture has stable layout |
| Camera controls | all rendered recipes | `proven_in_forge3d` | `README.md:44-48`, `python/forge3d/__init__.pyi:160-171` | Orbit/look-at/snapshot path | Camera presets tied to recipe manifests | `Scene.set_camera(...)` | `P0_foundation` | Camera hash/render size stable |
| Lighting / GI controls | terrain, urban, solar, smoke, timelapse | `proven_in_forge3d` | `python/forge3d/__init__.pyi:184-208`, `README.md:55-67` | SSGI/SSR/bloom/PBR controls | Cartographic presets and diagnostics | `Scene.set_lighting_preset(...)` | `P2_quality_and_scale` | Invalid values warn or clamp |
| Image export | all still-image recipes | `proven_in_forge3d` | `python/forge3d/__init__.pyi:160-171`, `README.md:44-48` | PNG/RGBA render and viewer snapshot | Recipe output metadata | `Scene.export(...)` | `P0_foundation` | Output size/path tested |
| COG ingest / tile IO | COG, terrain streaming | `proven_in_forge3d` | `python/forge3d/cog.py:37-368`, `tests/test_cog_streaming.py:81-294` | COG availability/open and tests | Recipe alignment with terrain/vector layers | `CogTerrainSource.open(...)` | `P2_quality_and_scale` | Tiny COG fixture opens and samples |
| Point cloud load/render/cache | pointcloud_cog | `proven_in_forge3d` | `python/forge3d/pointcloud.py:1-608`, `src/pointcloud/renderer.rs:147-341` | Point-cloud renderer and memory reports | MapScene render integration | `PointCloudLayer.from_path(...)` | `P2_quality_and_scale` | Memory report and snapshot/diagnostic stable |
| Scene serialization / bundles | MapScene showcases | `proven_in_forge3d` | `python/forge3d/map_scene.py:1231-1867`, `src/bundle/manifest.rs:53-117` | Bundle load/save and manifest | Recipe-family bundle examples | `MapScene.save_bundle(...)` | `P1_core_recipe_support` | Bundle reload preserves render metadata |
| Diagnostics and visual-golden foundation | all reusable recipes | `proven_in_forge3d` | `tests/test_terrain_visual_goldens.py:149-283`, `tests/test_mapscene_support_status.py:106-169` | Terrain goldens and support diagnostics | Recipe-level golden set | `RecipeValidationReport` | `P0_foundation` | Each core recipe has stable diagnostic/golden |
| Typed SceneRecipe / MapScene API | MapScene showcases, reusable recipe foundation | `proven_in_forge3d` | `python/forge3d/__init__.py:431-451`, `python/forge3d/map_scene.py:1010-1044`, `tests/test_mapscene_recipe_contract.py:78-140`, `tests/test_mapscene_render_png.py:58-254`, `tests/test_mapscene_save_bundle.py:68-164` | Public `SceneRecipe`, `MapScene`, validation, deterministic PNG render path, and bundle save/load | Evidence-derived gallery recipe-family manifests and full adapter coverage for every discovered family | Existing `SceneRecipe` / `MapScene` plus thin family metadata | `P0_foundation` | Public constructor, validation, render, and bundle tests pass |
| Scene grammar coverage | all first-class recipes | `partially_proven` | `python/forge3d/map_scene.py:223-1044`, `docs/guides/offline_3d_map_rendering.md:19-22`, `tests/test_mapscene_support_status.py:106-169` | MapScene classes and validation exist | Full coverage for discovered recipes and partial adapters | `MapScene` support reports per family | `P0_foundation` | One manifest per recipe validates or emits stable diagnostics |
| Raster alignment | landcover, climate, hydrology, travel-time | `partially_proven` | `docs/tutorials/gis-track/02-drape-overlays-on-terrain.md:61-69` | High-level drape path and raw IPC boundary | Explicit extent/CRS/shape diagnostics | `RasterLayer.align_to_terrain(...)` | `P0_foundation` | Alignment mismatch produces code |
| Categorical palettes | landcover, built-up | `exists_only_as_example_or_script_logic` | `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:456-582` | Class-to-RGBA in scripts | Public categorical style object | `CategoricalRasterStyle` | `P1_core_recipe_support` | Class legend and colors tested |
| Bivariate styling | climate, population-temperature | `exists_only_as_example_or_script_logic` | `examples/climate_bivariate/europe_bivariate_climate_map.py:1-11`, `:751-987` | Matrix styling and legend in scripts | Public bivariate style/legend | `BivariateRasterStyle` | `P1_core_recipe_support` | Two-raster fixture renders matrix |
| Data-aware terrain blending | REM, travel-time, landcover | `exists_only_as_example_or_script_logic` | `examples/urban_osm/barcelona_travel_time_3d.py:1475-1644` | Custom blend functions in scripts | Declared blend modes | `RasterLayer.blend(...)` | `P1_core_recipe_support` | Blend output matches golden |
| Building footprint/extrusion recipe | urban, solar, MapScene buildings | `partially_proven` | `python/forge3d/buildings.py:268-613`, `python/forge3d/map_scene.py:1631` | Building helper API and diagnostics | Deterministic MapScene render adapter | `BuildingLayer.extrude(...)` | `P1_core_recipe_support` | Building fixture renders or reports stable support |
| Roads / water / context layers | urban, travel-time, Helsinki | `exists_only_as_example_or_script_logic` | `examples/urban_osm/osm_city_demo.py:699-859` | OSM parsing/styling in scripts | Reusable context layer builder | `UrbanContextLayer` | `P1_core_recipe_support` | Road/water fixture renders by style |
| Temporal frame sequencing | smoke, satellite, Helsinki, flood | `exists_only_as_example_or_script_logic` | `examples/urban_osm/helsinki_transit_daycycle.py:1499-1596` | Per-script frame loops | Public sequence render helper | `Scene.render_sequence(...)` | `P2_quality_and_scale` | Two-frame fixture output deterministic |
| Globe population layer | humanity_globe_video | `exists_only_as_example_or_script_logic` | `examples/population_global_gpw/humanity_globe_video.py:331-607` | Script renderer | Public globe raster layer | `GlobeLayer.from_latlon_raster(...)` | `P3_polish` | One-frame globe fixture rendered |
| Evidence-derived recipe-family manifests/helpers | all discovered gallery-style recipe families | `not_found` | Section 3; examples encode family-specific decisions manually despite existing `SceneRecipe` / `MapScene` | Root examples and typed scene primitives exist | Per-family manifests/defaults/helpers derived from examples | `RecipeManifest` as metadata feeding existing `SceneRecipe` / `MapScene` | `P0_foundation` | Existing examples can emit or validate family metadata without replacing MapScene |

# 6. Gap register

| Gap ID | Capability | Recipe affected | Evidence | Severity | Why it matters | Recommended forge3d-only action | Acceptance criteria |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G-001 | Recipe-family manifests and provenance | all 23 families | Section 3; `SceneRecipe`/`MapScene` exist, but examples still encode family decisions in scripts | critical | Reusable gallery-style recipes cannot be validated, documented, or compared without declared inputs/defaults | Add a lightweight forge3d recipe-family manifest format that references existing examples and feeds existing MapScene pieces where applicable | At least five existing examples emit/validate manifests with source paths and defaults |
| G-002 | Raster/vector/terrain alignment diagnostics | landcover, climate, hydrology, rail, travel-time | `docs/tutorials/gis-track/02-drape-overlays-on-terrain.md:61-69` | critical | Beautiful overlays fail as reusable recipes when extent/shape/projection assumptions are hidden | Add alignment metadata and validation codes to raster/vector layer helpers | A mismatched tiny fixture fails with a stable diagnostic code |
| G-003 | Reusable pass compositing and blend modes | REM, Barcelona, landcover, hydrology | `examples/urban_osm/barcelona_travel_time_3d.py:1475-1644` | critical | Two-pass color/relief blends are repeated by hand and are not protected by tests | Add declared blend/pass composition helpers in forge3d | A two-pass fixture reproduces a known composite golden |
| G-004 | Recipe-level golden fixtures | all polished map outputs | `tests/test_terrain_visual_goldens.py:149-283` only covers terrain | critical | Current goldens do not protect the recipe outputs that motivated the audit | Add small deterministic recipe goldens under forge3d tests/fixtures | At least terrain+raster, categorical, bivariate, vector-line, label, building/diagnostic, point-cloud/diagnostic fixtures exist |
| G-005 | MapScene render adapters for partial layers | MapScene, buildings, point cloud | `python/forge3d/map_scene.py:1527`, `:1631` | critical | Typed scenes cannot become the reusable recipe surface while important layers render only placeholders or diagnostics | Harden MapScene adapters or make support status explicit and tested | Building and point-cloud layers either render deterministic fixtures or emit stable support diagnostics |
| G-006 | Categorical and bivariate raster styles | landcover, climate, built-up | `examples/landcover_esri/bosnia_terrain_landcover_viewer.py:456-582`, `examples/climate_bivariate/europe_bivariate_climate_map.py:751-987` | critical | Repeated class palettes and bivariate matrices remain script-only | Add forge3d style objects for categorical and bivariate rasters | Tiny categorical and bivariate fixtures validate and render with legends |
| G-007 | Public line-stroke layer helper | rail, hydrology, travel-time | `examples/urban_osm/luxembourg_rail_overlay.py:156-275` | high | Users must convert lines to quads manually for common map line recipes | Add line layer stroke helper backed by existing vector overlay path | Line fixture renders width, join, cap, z offset, and depth bias |
| G-008 | Urban context layer builder | urban OSM, Helsinki, Barcelona | `examples/urban_osm/osm_city_demo.py:699-859` | high | Roads, water, rails, and buildings repeat parsing/styling in scripts | Add forge3d-owned context layer helpers for already-used feature types | Tiny urban fixture renders roads/water/buildings with declared style |
| G-009 | Map plate composition | climate, population, hydrology, lighthouse | `docs/tutorials/gis-track/03-build-a-map-plate.md:29-58` and script layouts | high | Legends, scale bars, titles, and credits are scattered across scripts | Add a map plate helper that composes existing `Legend`, `ScaleBar`, and `NorthArrow` | Plate fixture validates positions and output size |
| G-010 | Temporal sequence rendering | smoke, satellite, flood, Helsinki | `examples/urban_osm/helsinki_transit_daycycle.py:1499-1596` | high | Animations cannot be reused without copying frame loops and encoding code | Add `Scene.render_sequence(...)` or equivalent forge3d-owned frame helper | Two-frame sequence fixture writes deterministic frame paths and metadata |
| G-011 | Documentation-to-test traceability | all first-class recipes | README/docs list capabilities, tests cover only some outputs | high | Docs can drift from actual support levels | Add recipe docs that link each capability to tests and fixture outputs | Each first-class recipe doc cites a passing test and fixture image |
| G-012 | Globe raster layer | humanity_globe_video | `examples/population_global_gpw/humanity_globe_video.py:331-607` | medium | The globe video is script-only and separate from Scene/MapScene | Decide whether globe raster rendering is in current forge3d scope; if yes, add a tiny fixture | One-frame globe output or explicit `not_found` support diagnostic |
| G-013 | Generated output provenance | gallery and `examples/out` images | `docs/gallery/images/`, `examples/out/` | low | Some generated assets are useful but not tied to exact scripts/inputs | Add metadata sidecars for selected canonical outputs | Selected outputs list source script, inputs, command, and expected hash/tolerance |

# 7. Proposed forge3d-only abstraction direction

1. low-level primitives

- Keep `Scene`, `TerrainParams`, raster overlays, vector overlays, labels, COG, point clouds, buildings, and bundles as the primitive base. These are already evidenced by `python/forge3d/__init__.pyi`, `terrain_params.py`, `map_scene.py`, `cog.py`, `pointcloud.py`, and `buildings.py`.
- Add small missing primitives only where examples repeatedly reimplement them: line stroking, raster alignment validation, categorical raster style, bivariate raster style, and pass compositing.

2. cartographic building blocks

- `RasterLayer.align_to_terrain(...)` for the repeated DEM/raster pairing in landcover, climate, built-up, travel-time, smoke, and satellite examples.
- `CategoricalRasterStyle` and `BivariateRasterStyle` for landcover and climate examples.
- `VectorLineLayer.stroke(...)` for rail, hydrology, and roads.
- `MapPlate` using existing `Legend`, `ScaleBar`, and `NorthArrow`.
- `UrbanContextLayer` for the OSM city scripts' repeated roads/water/rails/building context.

3. recipe-level helpers

- A minimal recipe manifest should describe source example, inputs, layers, camera, lighting, styling, output, and support status. It should not replace MapScene; it should make repeated example choices explicit and allow MapScene to be the typed scene representation where it fits.
- Example direction, derived from `barcelona_travel_time_3d.py`: a helper could declare DEM source, road graph output, travel-time overlay style, two render passes, camera, and final output path. The implementation can still call existing forge3d terrain/overlay/snapshot primitives.
- Example direction, derived from `landcover_esri`: a helper could declare DEM, class raster, class palette, overlay alpha, PBR, camera, and output.

4. validation / diagnostics

- Extend existing MapScene/support diagnostics to report missing extent, shape mismatch, missing palette class, unsupported line stroke option, unsupported building adapter, missing font atlas, and unsupported temporal render.
- Keep diagnostics stable enough for tests, following `tests/test_mapscene_support_status.py:106-169`.

5. visual regression fixtures

- Promote representative examples into tiny deterministic fixtures: terrain, landcover, bivariate climate, river lines, labels, point cloud/COG, building diagnostics, and one animated two-frame recipe.
- Build on the current terrain golden pattern in `tests/test_terrain_visual_goldens.py:149-283`.

# 8. Suggested golden fixtures

| Fixture name | Source example/script | Recipe family | Minimal input data needed | Expected visual properties | Failure modes it should catch | Repo currently contains assets for it |
| --- | --- | --- | --- | --- | --- | --- |
| `terrain_demo_rainier_small` | `examples/terrain_viewer/terrain_demo.py` | `terrain_demo` | tiny DEM, colormap, camera/light | shaded terrain with stable palette and camera | terrain mesh, colormap, lighting, export drift | partially; terrain goldens exist |
| `landcover_esri_tiny` | `examples/landcover_esri/bosnia_terrain_landcover_viewer.py` | `landcover_esri_terrain_viewer` | tiny DEM, 4-class raster, palette | categorical overlay draped on terrain | class-color swap, alpha, alignment | missing tiny canonical asset |
| `climate_bivariate_tiny` | `examples/climate_bivariate/europe_bivariate_climate_map.py` | `climate_bivariate` | tiny DEM, two rasters, 4x4 palette | bivariate matrix overlay and legend | axis swap, bin drift, legend mismatch | missing tiny canonical asset |
| `population_spike_tiny` | `examples/population_spike_worldpop/poland_population_spikes.py` | `population_spike_worldpop` | small density grid, mask, palette | deterministic spike/height-shade output | scale, threshold, color, camera drift | missing tiny canonical asset |
| `hydrology_line_drape_tiny` | `examples/hydrology_river_basins/poland_river_basins_forge3d.py` | `hydrology_river_basins` | tiny DEM, one basin polygon, one river line | colored basin plus draped river line | clipping, line width, color, drape offset | missing tiny canonical asset |
| `luxembourg_rail_line_tiny` | `examples/urban_osm/luxembourg_rail_overlay.py` | `luxembourg_rail_overlay` | tiny DEM and rail line | line stroke over terrain with depth/z offset | line triangulation, halo/depth, z offset | missing tiny canonical asset |
| `fuji_labels_tiny` | `examples/labels_styles_picking/fuji_labels_demo.py` | `labels_styles_picking` | tiny DEM, points/line labels, font atlas | point labels, line label, callout | anchor, glyph, halo, overlap drift | partially; label code exists |
| `mapscene_bundle_tiny` | `examples/mapscene_terrain_raster.py` | `mapscene_showcases` | synthetic DEM/raster/vector/labels | valid scene, render path, bundle reload | support diagnostics, bundle drift | yes, examples synthesize assets |
| `pointcloud_cog_tiny` | `examples/pointcloud_cog/pointcloud_viewer_interactive.py` | `pointcloud_cog` | tiny point cloud or stable diagnostic fixture | point colors, memory report, snapshot/diagnostic | cache, color fallback, support drift | partially; tests cover COG |
| `urban_building_tiny` | `examples/urban_osm/buildings_viewer_interactive.py` | `urban_osm_city` | few building footprints/heights, roads/water | extruded buildings with context layers | height defaults, layer order, camera | missing tiny canonical asset |
| `flood_two_frame_tiny` | `examples/urban_osm/osm_city_flood_daycycle.py` | `osm_city_flood_daycycle` | small activation grid and surface | two water levels with stable overlay | flood extent, alpha, label drift | missing tiny canonical asset |
| `smoke_frame_tiny` | `examples/wildfire_smoke/california_fire_smoke_effect.py` | `wildfire_smoke` | small smoke density grid and base terrain | translucent plume overlay | density-to-alpha/color, blend drift | missing tiny canonical asset |

# 9. Final prioritized forge3d-only roadmap

## Phase 1: evidence-to-recipe foundation

| Task | Exact objective | Affected files or likely file areas | Dependency | Acceptance criteria | Risk if skipped |
| --- | --- | --- | --- | --- | --- |
| P1-1 | Add a small recipe manifest schema for existing examples | `python/forge3d/map_scene.py`, new docs under `docs/carto-engine/`, selected examples metadata | none | Five representative examples validate manifests with source paths, inputs, camera, lighting, output, and status | Example logic remains undocumented and hard to test |
| P1-2 | Create recipe inventory docs from Section 3 | `docs/carto-engine/`, `docs/gallery/` | P1-1 | Docs list all 23 families and link evidence files | New work repeats stale or generic categories |
| P1-3 | Add alignment diagnostics for raster/vector/terrain recipes | `python/forge3d/terrain_params.py`, `python/forge3d/map_scene.py`, tests | P1-1 | Shape/extent mismatch produces stable diagnostic code | Overlay recipes remain fragile |
| P1-4 | Select canonical tiny inputs for golden fixtures | `tests/fixtures/`, `tests/golden/`, `docs/gallery/images/` | P1-2 | Fixture list has owner/source/example/expected output metadata | Goldens cannot be deterministic |

## Phase 2: reusable forge3d recipe API

| Task | Exact objective | Affected files or likely file areas | Dependency | Acceptance criteria | Risk if skipped |
| --- | --- | --- | --- | --- | --- |
| P2-1 | Promote categorical and bivariate raster styles | `python/forge3d/style.py`, `python/forge3d/terrain_params.py`, tests | P1-3 | Landcover and bivariate tiny fixtures render with declared styles and legends | Landcover/climate scripts keep duplicating palette logic |
| P2-2 | Add public line stroke/drape helper | `python/forge3d/style.py`, vector overlay modules, `tests/test_vector_drape.py` | P1-3 | Line fixture renders without user-side triangulation | Rail/hydrology/travel-time keep manual quad code |
| P2-3 | Add map plate helper around existing furniture | `python/forge3d/legend.py`, `scale_bar.py`, `north_arrow.py`, tests | P1-4 | Plate fixture positions legend, scale, north arrow, title deterministically | Poster outputs remain script-only |
| P2-4 | Harden MapScene adapters and support diagnostics | `python/forge3d/map_scene.py`, buildings/pointcloud modules, tests | P1-1 | Building/point-cloud layers render tiny fixtures or report stable support diagnostics | Typed scenes cannot cover important examples |
| P2-5 | Add minimal urban context layer helpers | `python/forge3d/buildings.py`, style/vector helpers, examples | P2-2 | Tiny urban fixture renders roads, water, and buildings with declared styles | Urban examples remain large one-offs |

## Phase 3: quality gates and hardening

| Task | Exact objective | Affected files or likely file areas | Dependency | Acceptance criteria | Risk if skipped |
| --- | --- | --- | --- | --- | --- |
| P3-1 | Add recipe-level visual goldens | `tests/golden/`, recipe fixture tests | P1-4, P2 tasks | At least eight recipe fixtures compare images or stable diagnostics | Visual regressions in flagship recipes go uncaught |
| P3-2 | Add sequence fixture support | sequence helpers/tests | P2-3 | Two-frame satellite/smoke/flood fixture writes deterministic frames | Animated recipes keep bespoke loops |
| P3-3 | Document support matrices per recipe | `docs/guides/`, `docs/carto-engine/` | P2 tasks | Each first-class recipe lists supported, partial, and unsupported capabilities with tests | Users misread examples as reusable APIs |
| P3-4 | Add command metadata for selected gallery outputs | `docs/gallery/`, generated-output sidecars | P3-1 | Canonical images list command, inputs, source script, and tolerance | Gallery images cannot serve as regression evidence |

# 10. Open questions for Milos

1. Which generated images under `docs/gallery/images/` and `examples/out/` should become canonical golden fixtures?
2. For duplicated root examples and topic-folder examples, which path should be considered authoritative when line evidence differs?
3. Are there private or omitted scripts for LA heat or other scenes that should be included in the recipe inventory?
4. Should `humanity_globe_video` become a forge3d-owned globe recipe, or should it stay documented as script-only example logic?
5. Which public API surface is considered stable for first-class recipes: `Scene`, viewer IPC helpers, `MapScene`, or a combination of those existing forge3d surfaces?
