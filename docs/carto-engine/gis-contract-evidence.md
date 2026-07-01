# GIS Contract Evidence Ledger

This ledger is the evidence backing `rust-gis-implementation-plan.md` and `gis-operation-api-crosswalk.md`. It is documentation only.

## Inventory Summary

| Metric | Count | Counting rule |
|---|---:|---|
| `examples/**/*.py` physical scripts inspected | 121 | Direct filesystem walk, excluding cache/output/log/temp folders. |
| Root-level Python files checked | 1 | `conftest.py`; not counted as an example GIS script. |
| GIS-relevant example scripts | 109 | Read-only inventory pass; terrain-only demos count when they load/render DEMs or geospatial rasters. |
| Content-distinct example scripts | 107 | SHA-256 content groups over the 121 physical scripts. |
| Contract pattern classes found | 89 | AST call/attribute/keyword pass plus narrow text evidence for CRS literals, remote data, and domain helpers. |
| Proposed API contract rows derived | 68 | Roll-up of the 89 pattern classes by forge3d API candidate. |
| Exact duplicate content groups | 14 | Root/topic mirrors are kept in physical counts but content-distinct counts are used for priority. |
| Basename duplicate groups | 50 | Root/topic mirrors and recipe variants. |
| AST parse failures | 0 | No fallback-only script parsing was needed. |
| Notebooks assessed | 3 | `map_plate.ipynb` has unique cartographic logic; `quickstart.ipynb` and `terrain_explorer.ipynb` repeat script-covered basics. |

Excluded paths and reasons:

| Path | Reason |
|---|---|
| `examples/.cache/` | Cache; no source scripts counted. |
| `examples/out/` | Generated output; no source scripts counted. |
| `examples/**/__pycache__/` | Bytecode cache. |
| `logs/` | Logs/generated artifacts; outside example-source scope. |
| Root `conftest.py` | Test/example mapping helper, not an example script. |

Exact duplicate content groups:

| Group |
|---|
| `examples/_png.py`; `examples/support/_png.py` |
| `examples/buildings_viewer_interactive.py`; `examples/urban_osm/buildings_viewer_interactive.py` |
| `examples/camera_animation_demo.py`; `examples/terrain_viewer/camera_animation_demo.py` |
| `examples/cog_streaming_demo.py`; `examples/pointcloud_cog/cog_streaming_demo.py` |
| `examples/fuji_labels_demo.py`; `examples/labels_styles_picking/fuji_labels_demo.py` |
| `examples/hydrology_river_basins/turkiye_river_basins_3d.py`; `examples/turkiye_river_basins_3d.py` |
| `examples/khumbu_icefall_sentinel_timelapse.py`; `examples/satellite_timelapse/khumbu_icefall_sentinel_timelapse.py` |
| `examples/labels_styles_picking/picking_demo.py`; `examples/picking_demo.py` |
| `examples/labels_styles_picking/picking_test_interactive.py`; `examples/picking_test_interactive.py` |
| `examples/labels_styles_picking/style_viewer_interactive.py`; `examples/style_viewer_interactive.py` |
| `examples/luxembourg_rail_overlay.py`; `examples/urban_osm/luxembourg_rail_overlay.py` |
| `examples/pointcloud_cog/pointcloud_viewer_interactive.py`; `examples/pointcloud_viewer_interactive.py` |
| `examples/terrain_demo.py`; `examples/terrain_viewer/terrain_demo.py` |
| `examples/terrain_viewer/terrain_viewer_interactive.py`; `examples/terrain_viewer_interactive.py` |

## Scripts Inspected

```text
examples/_import_shim.py
examples/_png.py
examples/barcelona_travel_time_3d.py
examples/belgium_bivariate_climate_map.py
examples/bosnia_terrain_landcover_viewer.py
examples/bryce_canyon_storm_timelapse.py
examples/buildings_viewer_interactive.py
examples/california_cigar_smoke_demo.py
examples/california_fire_smoke_effect.py
examples/california_wildfire_smoke_video.py
examples/camera_animation_demo.py
examples/climate_bivariate/belgium_bivariate_climate_map.py
examples/climate_bivariate/europe_bivariate_climate_map.py
examples/climate_bivariate/france_bivariate_climate_map.py
examples/cog_streaming_demo.py
examples/colorado_rem_forge3d.py
examples/core_sanity/diagnostics_support_matrices_demo.py
examples/core_sanity/png_numpy_roundtrip.py
examples/core_sanity/terrain_single_tile.py
examples/core_sanity/triangle_png.py
examples/diagnostics_support_matrices_demo.py
examples/forest_cover_copernicus/italy_forest_cover_3d.py
examples/fuji_labels_demo.py
examples/germany_terrain_landcover_viewer.py
examples/helsinki_transit_daycycle.py
examples/humanity_globe_video.py
examples/hydrology_river_basins/iberian_peninsula_river_basins.py
examples/hydrology_river_basins/pnoa_river_showcase.py
examples/hydrology_river_basins/pnoa_river_showcase_video.py
examples/hydrology_river_basins/poland_river_basins_forge3d.py
examples/hydrology_river_basins/turkiye_river_basins_3d.py
examples/khumbu_icefall_sentinel_timelapse.py
examples/label_api_truth_basic.py
examples/labels_styles_picking/fuji_labels_demo.py
examples/labels_styles_picking/label_api_truth_basic.py
examples/labels_styles_picking/picking_demo.py
examples/labels_styles_picking/picking_test_interactive.py
examples/labels_styles_picking/style_viewer_interactive.py
examples/landcover_esri/bosnia_terrain_landcover_viewer.py
examples/landcover_esri/germany_terrain_landcover_viewer.py
examples/landcover_esri/poland_terrain_landcover_viewer.py
examples/landcover_esri/romania_terrain_landcover_viewer.py
examples/landcover_esri/swiss_terrain_landcover_viewer.py
examples/luxembourg_rail_overlay.py
examples/mapscene/mapscene_buildings_labels.py
examples/mapscene/mapscene_bundled_datasets_showcase.py
examples/mapscene/mapscene_p1_assets_bundle_showcase.py
examples/mapscene/mapscene_terrain_raster.py
examples/mapscene/mapscene_vector_labels.py
examples/mapscene_buildings_labels.py
examples/mapscene_bundled_datasets_showcase.py
examples/mapscene_p1_assets_bundle_showcase.py
examples/mapscene_terrain_raster.py
examples/mapscene_vector_labels.py
examples/osm_city_daycycle.py
examples/osm_city_demo.py
examples/osm_city_flood_daycycle.py
examples/picking_demo.py
examples/picking_test_interactive.py
examples/platte_rem_forge3d.py
examples/png_numpy_roundtrip.py
examples/pnoa_river_showcase.py
examples/pnoa_river_showcase_video.py
examples/pointcloud_cog/cog_streaming_demo.py
examples/pointcloud_cog/pointcloud_viewer_interactive.py
examples/pointcloud_viewer_interactive.py
examples/poland_population_spikes_height_shade.py
examples/poland_river_basins_forge3d.py
examples/population_ghsl/egypt_population_3d.py
examples/population_ghsl/germany_population_3d.py
examples/population_ghsl/iberia_builtup_cover_3d.py
examples/population_ghsl/iberian_peninsula_bivariate_population_temperature.py
examples/population_ghsl/iberian_peninsula_population_3d.py
examples/population_ghsl/poland_population_3d.py
examples/population_ghsl/romania_builtup_cover_3d.py
examples/population_ghsl/southeast_europe_population_3d.py
examples/population_ghsl/southeast_europe_population_spikes_enhanced_3d.py
examples/population_ghsl/turkey_builtup_cover_3d.py
examples/population_global_gpw/humanity_globe_video.py
examples/population_spike_worldpop/france_population_spikes_height_shade.py
examples/population_spike_worldpop/germany_population_spikes_height_shade.py
examples/population_spike_worldpop/poland_population_contour_3d.py
examples/population_spike_worldpop/poland_population_spikes.py
examples/population_spike_worldpop/poland_population_spikes_height_shade.py
examples/population_spike_worldpop/turkey_population_spikes_height_shade.py
examples/romania_terrain_landcover_viewer.py
examples/rotterdam_solar_potential_shadow_study.py
examples/satellite_timelapse/bryce_canyon_storm_timelapse.py
examples/satellite_timelapse/khumbu_icefall_sentinel_timelapse.py
examples/satellite_timelapse/mount_hood_white_cloud_timelapse.py
examples/style_viewer_interactive.py
examples/support/_import_shim.py
examples/support/_png.py
examples/swiss_terrain_landcover_viewer.py
examples/terrain_atmosphere_path_demo.py
examples/terrain_camera_rigs_demo.py
examples/terrain_demo.py
examples/terrain_relief/colorado_rem_forge3d.py
examples/terrain_relief/platte_rem_forge3d.py
examples/terrain_single_tile.py
examples/terrain_viewer/camera_animation_demo.py
examples/terrain_viewer/terrain_atmosphere_path_demo.py
examples/terrain_viewer/terrain_camera_rigs_demo.py
examples/terrain_viewer/terrain_demo.py
examples/terrain_viewer/terrain_viewer_interactive.py
examples/terrain_viewer_interactive.py
examples/triangle_png.py
examples/turkiye_river_basins_3d.py
examples/uk_ireland_lighthouse_map.py
examples/urban_osm/barcelona_travel_time_3d.py
examples/urban_osm/buildings_viewer_interactive.py
examples/urban_osm/helsinki_transit_daycycle.py
examples/urban_osm/luxembourg_rail_overlay.py
examples/urban_osm/osm_city_daycycle.py
examples/urban_osm/osm_city_demo.py
examples/urban_osm/osm_city_flood_daycycle.py
examples/urban_osm/rotterdam_solar_potential_shadow_study.py
examples/urban_osm/uk_ireland_lighthouse_map.py
examples/wildfire_smoke/california_cigar_smoke_demo.py
examples/wildfire_smoke/california_fire_smoke_effect.py
examples/wildfire_smoke/california_wildfire_smoke_video.py
```

## Pattern Coverage

Counts report physical script paths and content-distinct scripts separately so duplicate mirrors do not silently inflate priority.

| Pattern class | Family | Proposed API | Phase | Priority | Occurrences | Script paths | Content-distinct scripts | Representative evidence |
|---|---|---|---|---|---:|---:|---:|---|
| `domain.building.heights` | Domain preparation | `extract_building_heights` | later | P1 | 3582 | 114 | 100 | `examples/_png.py`:27; `examples/_png.py`:38 |
| `domain.gridded.subset.sample` | Domain preparation | `subset_grid` | later | P2 | 5914 | 111 | 99 | `examples/barcelona_travel_time_3d.py`:2; `examples/barcelona_travel_time_3d.py`:5 |
| `overwrite.parent_path` | Raster reads/writes | `write_raster` | G-002a1 | P1 | 819 | 102 | 91 | `examples/barcelona_travel_time_3d.py`:455; `examples/barcelona_travel_time_3d.py`:457 |
| `domain.dem.preparation` | Domain preparation | `prepare_dem` | later | P1 | 2856 | 101 | 89 | `examples/barcelona_travel_time_3d.py`:8; `examples/barcelona_travel_time_3d.py`:50 |
| `clip.by.vector.geometry` | Raster windowing/clipping/masking | `mask_raster` | G-002c | P0 | 5002 | 95 | 87 | `examples/barcelona_travel_time_3d.py`:239; `examples/barcelona_travel_time_3d.py`:240 |
| `remote.cache.paths` | Remote/geodata helpers | `cache_geodata` | later | P1 | 1450 | 90 | 85 | `examples/barcelona_travel_time_3d.py`:45; `examples/barcelona_travel_time_3d.py`:247 |
| `cog.range.reads` | Remote/geodata helpers | `read_cog` | later | P2 | 758 | 94 | 81 | `examples/_png.py`:48; `examples/barcelona_travel_time_3d.py`:380 |
| `rasterize.fill.dtype` | Rasterization/vector-to-raster | `rasterize_vectors` | G-002c | P0 | 3468 | 81 | 78 | `examples/barcelona_travel_time_3d.py`:317; `examples/barcelona_travel_time_3d.py`:945 |
| `domain.terrain.derivatives` | Domain preparation | `prepare_terrain_derivatives` | later | P2 | 1498 | 86 | 77 | `examples/barcelona_travel_time_3d.py`:265; `examples/barcelona_travel_time_3d.py`:1515 |
| `nodata.fill.behavior` | Raster windowing/clipping/masking | `apply_nodata` | G-002b | P0 | 1012 | 78 | 72 | `examples/barcelona_travel_time_3d.py`:1352; `examples/barcelona_travel_time_3d.py`:1358 |
| `raster.width` | Raster metadata | `read_raster_info` | G-002a1 | P0 | 465 | 76 | 67 | `examples/barcelona_travel_time_3d.py`:1150; `examples/barcelona_travel_time_3d.py`:2140 |
| `vector.clip` | Vector transforms and operations | `clip_vector` | G-002c | P0 | 2044 | 66 | 64 | `examples/barcelona_travel_time_3d.py`:310; `examples/barcelona_travel_time_3d.py`:962 |
| `raster.height` | Raster metadata | `read_raster_info` | G-002a1 | P0 | 405 | 73 | 64 | `examples/barcelona_travel_time_3d.py`:1151; `examples/barcelona_travel_time_3d.py`:646 |
| `EPSG.integer` | CRS | `parse_crs` | G-002b | P0 | 257 | 67 | 64 | `examples/barcelona_travel_time_3d.py`:451; `examples/barcelona_travel_time_3d.py`:626 |
| `EPSG.string` | CRS | `parse_crs` | G-002b | P0 | 214 | 67 | 64 | `examples/barcelona_travel_time_3d.py`:626; `examples/barcelona_travel_time_3d.py`:2007 |
| `vector.feature.count` | Vector IO and metadata | `feature_count` | G-002c | P1 | 806 | 72 | 63 | `examples/barcelona_travel_time_3d.py`:166; `examples/barcelona_travel_time_3d.py`:1642 |
| `vector.overlay.intersection` | Vector transforms and operations | `intersect_vectors` | G-002c | P0 | 260 | 70 | 63 | `examples/barcelona_travel_time_3d.py`:1122; `examples/barcelona_travel_time_3d.py`:2064 |
| `cog.geotiff.assumptions` | Raster reads/writes | `read_cog` | later | P2 | 252 | 73 | 63 | `examples/belgium_bivariate_climate_map.py`:351; `examples/belgium_bivariate_climate_map.py`:1266 |
| `raster.shape` | Raster metadata | `read_raster_info` | G-002a1 | P0 | 901 | 65 | 61 | `examples/barcelona_travel_time_3d.py`:1027; `examples/barcelona_travel_time_3d.py`:1028 |
| `domain.osm.scene` | Domain preparation | `prepare_osm_scene` | later | P1 | 1324 | 64 | 58 | `examples/barcelona_travel_time_3d.py`:5; `examples/barcelona_travel_time_3d.py`:7 |
| `shape.extent.transform.match` | Raster reprojection/resampling/alignment | `assert_grid_compatible` | G-002b | P0 | 365 | 57 | 54 | `examples/barcelona_travel_time_3d.py`:205; `examples/barcelona_travel_time_3d.py`:641 |
| `remote.url.download` | Remote/geodata helpers | `fetch_remote_geodata` | later | P1 | 216 | 55 | 53 | `examples/barcelona_travel_time_3d.py`:459; `examples/barcelona_travel_time_3d.py`:510 |
| `wgs84.4326` | CRS | `parse_crs` | G-002b | P0 | 437 | 53 | 49 | `examples/barcelona_travel_time_3d.py`:620; `examples/barcelona_travel_time_3d.py`:625 |
| `rasterio.open` | Raster reads/writes | `read_raster_info` | G-002a1 | P0 | 220 | 56 | 49 | `examples/belgium_bivariate_climate_map.py`:393; `examples/belgium_bivariate_climate_map.py`:992 |
| `rasterio.mask.mask` | Raster windowing/clipping/masking | `mask_raster` | G-002c | P0 | 382 | 52 | 48 | `examples/barcelona_travel_time_3d.py`:735; `examples/barcelona_travel_time_3d.py`:1125 |
| `Resampling.method` | Raster reprojection/resampling/alignment | `resample_raster` | G-002b | P0 | 282 | 49 | 48 | `examples/belgium_bivariate_climate_map.py`:541; `examples/belgium_bivariate_climate_map.py`:543 |
| `categorical.continuous.resampling` | Raster reprojection/resampling/alignment | `resample_raster` | G-002b | P1 | 275 | 51 | 48 | `examples/barcelona_travel_time_3d.py`:316; `examples/barcelona_travel_time_3d.py`:874 |
| `dataset.read` | Raster reads/writes | `read_raster` | G-002a1 | P0 | 111 | 47 | 44 | `examples/barcelona_travel_time_3d.py`:460; `examples/barcelona_travel_time_3d.py`:511 |
| `raster.transform` | Affine / transform | `raster_transform` | G-002b | P0 | 169 | 45 | 43 | `examples/barcelona_travel_time_3d.py`:627; `examples/barcelona_travel_time_3d.py`:653 |
| `xyz.slippy.tile.math` | Remote/geodata helpers | `slippy_tile_index` | later | P1 | 610 | 49 | 41 | `examples/barcelona_travel_time_3d.py`:250; `examples/barcelona_travel_time_3d.py`:959 |
| `rasterize.target.grid` | Rasterization/vector-to-raster | `rasterize_vectors` | G-002c | P0 | 114 | 41 | 41 | `examples/barcelona_travel_time_3d.py`:737; `examples/barcelona_travel_time_3d.py`:738 |
| `domain.population` | Domain preparation | `prepare_population_raster` | later | P1 | 1543 | 43 | 39 | `examples/barcelona_travel_time_3d.py`:1792; `examples/barcelona_travel_time_3d.py`:1864 |
| `dataset.write` | Raster reads/writes | `write_raster` | G-002a1 | P0 | 104 | 38 | 37 | `examples/belgium_bivariate_climate_map.py`:406; `examples/belgium_bivariate_climate_map.py`:1005 |
| `remote.vector.fetch` | Remote/geodata helpers | `fetch_vector` | later | P2 | 310 | 41 | 35 | `examples/barcelona_travel_time_3d.py`:170; `examples/barcelona_travel_time_3d.py`:273 |
| `raster.nodata` | Raster metadata | `nodata_per_band` | G-002b | P0 | 145 | 34 | 32 | `examples/belgium_bivariate_climate_map.py`:403; `examples/belgium_bivariate_climate_map.py`:1002 |
| `vector.geometry.type` | Vector IO and metadata | `geometry_type` | G-002c | P1 | 107 | 34 | 32 | `examples/barcelona_travel_time_3d.py`:705; `examples/barcelona_travel_time_3d.py`:707 |
| `vector.bounds` | Vector transforms and operations | `vector_bounds` | G-002c | P0 | 88 | 37 | 32 | `examples/belgium_bivariate_climate_map.py`:356; `examples/belgium_bivariate_climate_map.py`:429 |
| `raster.bounds` | Raster metadata | `raster_bounds` | G-002b | P0 | 75 | 37 | 32 | `examples/belgium_bivariate_climate_map.py`:356; `examples/bosnia_terrain_landcover_viewer.py`:257 |
| `osm.overpass.nominatim` | Remote/geodata helpers | `query_osm_features` | later | P1 | 588 | 33 | 31 | `examples/barcelona_travel_time_3d.py`:5; `examples/barcelona_travel_time_3d.py`:46 |
| `domain.buildings.footprints` | Domain preparation | `load_building_footprints` | later | P1 | 586 | 32 | 30 | `examples/barcelona_travel_time_3d.py`:1242; `examples/barcelona_travel_time_3d.py`:1277 |
| `raster.crs` | CRS | `raster_crs` | G-002b | P0 | 130 | 32 | 29 | `examples/belgium_bivariate_climate_map.py`:444; `examples/belgium_bivariate_climate_map.py`:1000 |
| `vector.crs` | Vector IO and metadata | `vector_crs` | G-002c | P0 | 130 | 32 | 29 | `examples/belgium_bivariate_climate_map.py`:444; `examples/belgium_bivariate_climate_map.py`:1000 |
| `driver.creation_options` | Raster reads/writes | `write_raster` | G-002a1 | P1 | 106 | 29 | 29 | `examples/belgium_bivariate_climate_map.py`:396; `examples/belgium_bivariate_climate_map.py`:404 |
| `rotated.sheared.transform` | Affine / transform | `validate_transform` | G-002b | P1 | 193 | 31 | 28 | `examples/belgium_bivariate_climate_map.py`:768; `examples/belgium_bivariate_climate_map.py`:769 |
| `web_mercator.3857` | CRS | `web_mercator_bounds` | G-002b | P0 | 47 | 26 | 26 | `examples/belgium_bivariate_climate_map.py`:401; `examples/bosnia_terrain_landcover_viewer.py`:292 |
| `terrarium.decode` | Remote/geodata helpers | `decode_terrarium_dem` | later | P2 | 221 | 24 | 24 | `examples/barcelona_travel_time_3d.py`:8; `examples/barcelona_travel_time_3d.py`:50 |
| `vector.validity.empty` | Vector transforms and operations | `validate_geometry` | G-002c | P1 | 127 | 23 | 23 | `examples/barcelona_travel_time_3d.py`:1123; `examples/barcelona_travel_time_3d.py`:2070 |
| `warp.reproject` | Raster reprojection/resampling/alignment | `reproject_raster` | G-002b | P0 | 57 | 24 | 22 | `examples/belgium_bivariate_climate_map.py`:539; `examples/belgium_bivariate_climate_map.py`:543 |
| `from_bounds` | Affine / transform | `transform_from_bounds` | G-002b | P0 | 36 | 22 | 22 | `examples/barcelona_travel_time_3d.py`:738; `examples/belgium_bivariate_climate_map.py`:391 |
| `windows.from_bounds` | Raster windowing/clipping/masking | `window_from_bounds` | G-002b | P0 | 36 | 22 | 22 | `examples/barcelona_travel_time_3d.py`:738; `examples/belgium_bivariate_climate_map.py`:391 |
| `to_crs` | CRS | `reproject_vector` | G-002c | P0 | 39 | 23 | 21 | `examples/belgium_bivariate_climate_map.py`:292; `examples/bosnia_terrain_landcover_viewer.py`:217 |
| `geopandas.read_file` | Vector IO and metadata | `read_vector` | G-002c | P0 | 25 | 23 | 21 | `examples/belgium_bivariate_climate_map.py`:288; `examples/bosnia_terrain_landcover_viewer.py`:213 |
| `vector.buffer` | Vector transforms and operations | `buffer_geometry` | G-002c | P0 | 54 | 20 | 20 | `examples/barcelona_travel_time_3d.py`:719; `examples/california_wildfire_smoke_video.py`:874 |
| `vector.union` | Vector transforms and operations | `union_geometries` | G-002c | P0 | 44 | 19 | 19 | `examples/belgium_bivariate_climate_map.py`:299; `examples/belgium_bivariate_climate_map.py`:298 |
| `raster.profile` | Raster metadata | `read_raster_info` | G-002a1 | P0 | 36 | 21 | 19 | `examples/bosnia_terrain_landcover_viewer.py`:312; `examples/bosnia_terrain_landcover_viewer.py`:385 |
| `CRS.source.destination.pair` | CRS | `create_crs_transformer` | G-002b | P1 | 130 | 19 | 18 | `examples/bosnia_terrain_landcover_viewer.py`:330; `examples/bosnia_terrain_landcover_viewer.py`:332 |
| `features.geometry_mask` | Raster windowing/clipping/masking | `geometry_mask` | G-002c | P0 | 26 | 18 | 18 | `examples/barcelona_travel_time_3d.py`:735; `examples/barcelona_travel_time_3d.py`:1125 |
| `domain.landcover` | Domain preparation | `prepare_landcover_raster` | later | P1 | 259 | 15 | 14 | `examples/bosnia_terrain_landcover_viewer.py`:34; `examples/bosnia_terrain_landcover_viewer.py`:35 |
| `always_xy` | CRS | `create_crs_transformer` | G-002b | P0 | 68 | 15 | 14 | `examples/barcelona_travel_time_3d.py`:626; `examples/barcelona_travel_time_3d.py`:2007 |
| `pyproj.Transformer` | CRS | `create_crs_transformer` | G-002b | P0 | 34 | 15 | 14 | `examples/barcelona_travel_time_3d.py`:626; `examples/barcelona_travel_time_3d.py`:2007 |
| `vector.length.area` | Vector transforms and operations | `geometry_measure` | G-002c | P1 | 33 | 14 | 14 | `examples/barcelona_travel_time_3d.py`:714; `examples/colorado_rem_forge3d.py`:675 |
| `rasterize.all_touched` | Rasterization/vector-to-raster | `rasterize_vectors` | G-002c | P0 | 12 | 12 | 12 | `examples/barcelona_travel_time_3d.py`:740; `examples/california_wildfire_smoke_video.py`:1018 |
| `features.rasterize` | Rasterization/vector-to-raster | `rasterize_vectors` | G-002c | P0 | 20 | 11 | 11 | `examples/barcelona_travel_time_3d.py`:1125; `examples/climate_bivariate/france_bivariate_climate_map.py`:1012 |
| `missing.invalid.crs` | CRS | `inspect_crs` | G-002b | P1 | 15 | 15 | 11 | `examples/buildings_viewer_interactive.py`:390; `examples/forest_cover_copernicus/italy_forest_cover_3d.py`:767 |
| `windowed.read` | Raster windowing/clipping/masking | `read_raster_window` | G-002b | P0 | 14 | 11 | 11 | `examples/colorado_rem_forge3d.py`:540; `examples/humanity_globe_video.py`:239 |
| `vector.make_valid.buffer0` | Vector transforms and operations | `repair_geometry` | G-002c | P1 | 20 | 10 | 10 | `examples/california_wildfire_smoke_video.py`:872; `examples/california_wildfire_smoke_video.py`:874 |
| `set_crs.assign` | CRS | `assign_crs` | G-002b | P0 | 10 | 8 | 8 | `examples/belgium_bivariate_climate_map.py`:419; `examples/climate_bivariate/belgium_bivariate_climate_map.py`:428 |
| `gridded.xarray.rioxarray` | Remote/geodata helpers | `read_gridded_dataset` | later | P2 | 101 | 7 | 7 | `examples/belgium_bivariate_climate_map.py`:535; `examples/belgium_bivariate_climate_map.py`:539 |
| `local.utm.estimated` | CRS | `estimate_local_utm` | later | P2 | 18 | 8 | 7 | `examples/mapscene/mapscene_buildings_labels.py`:24; `examples/mapscene/mapscene_terrain_raster.py`:45 |
| `vector.total_bounds` | Vector IO and metadata | `vector_bounds` | G-002c | P0 | 13 | 9 | 7 | `examples/belgium_bivariate_climate_map.py`:429; `examples/climate_bivariate/belgium_bivariate_climate_map.py`:438 |
| `rasterize.burn.values` | Rasterization/vector-to-raster | `rasterize_vectors` | G-002c | P0 | 92 | 6 | 6 | `examples/california_cigar_smoke_demo.py`:317; `examples/california_cigar_smoke_demo.py`:385 |
| `transform_bounds` | Raster reprojection/resampling/alignment | `transform_bounds` | G-002b | P0 | 8 | 6 | 6 | `examples/colorado_rem_forge3d.py`:406; `examples/hydrology_river_basins/pnoa_river_showcase.py`:132 |
| `calculate_default_transform` | Raster reprojection/resampling/alignment | `calculate_default_transform` | G-002b | P0 | 7 | 7 | 6 | `examples/bosnia_terrain_landcover_viewer.py`:309; `examples/landcover_esri/bosnia_terrain_landcover_viewer.py`:318 |
| `CRS.object` | CRS | `inspect_crs` | G-002b | P0 | 11 | 6 | 5 | `examples/landcover_esri/swiss_terrain_landcover_viewer.py`:846; `examples/luxembourg_rail_overlay.py`:211 |
| `from_origin` | Affine / transform | `transform_from_origin` | G-002b | P0 | 5 | 5 | 5 | `examples/forest_cover_copernicus/italy_forest_cover_3d.py`:748; `examples/humanity_globe_video.py`:252 |
| `window_transform` | Affine / transform | `window_transform` | G-002b | P0 | 5 | 5 | 5 | `examples/belgium_bivariate_climate_map.py`:589; `examples/climate_bivariate/belgium_bivariate_climate_map.py`:598 |
| `vector.interpolate` | Vector transforms and operations | `interpolate_line` | G-002c | P2 | 8 | 4 | 4 | `examples/colorado_rem_forge3d.py`:681; `examples/osm_city_flood_daycycle.py`:503 |
| `vector.centroid` | Vector transforms and operations | `geometry_centroid` | G-002c | P1 | 4 | 4 | 4 | `examples/osm_city_demo.py`:570; `examples/rotterdam_solar_potential_shadow_study.py`:719 |
| `vector.schema.columns` | Vector IO and metadata | `vector_schema` | G-002c | P1 | 4 | 4 | 4 | `examples/climate_bivariate/france_bivariate_climate_map.py`:410; `examples/population_ghsl/southeast_europe_population_3d.py`:231 |
| `raster.count` | Raster metadata | `read_raster_info` | G-002a1 | P0 | 5 | 4 | 3 | `examples/satellite_timelapse/mount_hood_white_cloud_timelapse.py`:159; `examples/swiss_terrain_landcover_viewer.py`:324 |
| `pixel.center.corner` | Affine / transform | `pixel_convention` | G-002b | P1 | 6 | 2 | 2 | `examples/osm_city_flood_daycycle.py`:96; `examples/osm_city_flood_daycycle.py`:672 |
| `network.travel.time` | Deferred | `defer_routing` | defer | Defer | 4 | 2 | 2 | `examples/barcelona_travel_time_3d.py`:44; `examples/barcelona_travel_time_3d.py`:45 |
| `dataset.read_masks` | Raster metadata | `read_raster_mask` | G-002b | P1 | 3 | 3 | 2 | `examples/luxembourg_rail_overlay.py`:118; `examples/population_spike_worldpop/poland_population_spikes.py`:74 |
| `Affine` | Affine / transform | `affine_transform` | G-002b | P1 | 2 | 2 | 2 | `examples/population_ghsl/germany_population_3d.py`:518; `examples/population_ghsl/romania_builtup_cover_3d.py`:599 |
| `vector.simplify` | Vector transforms and operations | `simplify_geometry` | G-002c | P1 | 2 | 2 | 2 | `examples/osm_city_demo.py`:479; `examples/urban_osm/osm_city_demo.py`:488 |
| `vector.representative_point` | Vector transforms and operations | `representative_point` | G-002c | P2 | 2 | 2 | 2 | `examples/rotterdam_solar_potential_shadow_study.py`:1031; `examples/urban_osm/rotterdam_solar_potential_shadow_study.py`:1040 |
| `raster.resolution` | Raster metadata | `raster_resolution` | G-002b | P0 | 2 | 1 | 1 | `examples/satellite_timelapse/mount_hood_white_cloud_timelapse.py`:157; `examples/satellite_timelapse/mount_hood_white_cloud_timelapse.py`:156 |
| `xy.index` | Affine / transform | `xy_index` | G-002b | P1 | 1 | 1 | 1 | `examples/population_ghsl/southeast_europe_population_3d.py`:363 |
| `raster.block_shapes` | Raster metadata | `read_raster_info` | G-002a1 | P2 | 1 | 1 | 1 | `examples/swiss_terrain_landcover_viewer.py`:329 |

## Coverage Exceptions / Requires Human Confirmation

| Evidence | Mapped status | Reason |
|---|---|---|
| `examples/notebooks/map_plate.ipynb` cartographic `MapPlate` and `ScaleBar.compute_meters_per_pixel` logic | Deferred outside `G-002` GIS metadata foundation | It is cartographic composition, not raster/vector CRS preparation. It should be handled by map-plate roadmap work, not `src/gis/`. |
| Root/topic duplicate examples with same basename but non-identical content | Counted physically and content-distinct | Priority must use content-distinct counts when deciding implementation order. A human should choose canonical doc paths for future recipe docs. |
| Domain text classes such as building height, grid sampling, DEM prep, and cache handling | Mapped to later helper contracts | They are explicit in examples but often appear as comments, constants, and variable names. Contract APIs are justified, but exact option names require implementation-time review of each family. |
| `network.travel.time` in Barcelona travel-time scripts | Deferred to routing subsystem | It is graph/network analysis, not the GIS metadata/preparation foundation. |
| Live OSM, WFS, Terrarium, PVGIS, and remote raster fetches | Mapped to later remote/cache contracts | Tests must mock network payloads or use cached fixtures. The roadmap must not require live services. |

Coverage is complete under this plan's standard: every observed pattern class is mapped to an implemented/current API, a future planned GIS contract row, a deferred/later helper row, or an exception above.
