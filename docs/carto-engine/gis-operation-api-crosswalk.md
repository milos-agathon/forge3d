# GIS Operation API Crosswalk

This crosswalk converts GIS operations observed in existing example scripts into a Rust-first forge3d API roadmap. It is documentation only: no Rust, Python, example, shader, rendering, visual golden, runtime, or generated-output behavior changes here.

## Architecture Rules

- GIS backend behavior belongs in Rust under `src/gis/`.
- Python exposes thin wrappers over Rust bindings and must not implement backend GIS behavior with `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or R `terra`.
- Python wrappers may accept paths, numpy arrays, plain dicts, and Rust-backed metadata objects.
- Python wrappers must not require live `rasterio`, `geopandas`, `shapely`, `xarray`, `rioxarray`, or `terra` objects.
- CRS assignment is not reprojection. Missing CRS is reported as missing and is never guessed.
- Raster reprojection changes sampled values and must declare a resampling method.
- Vector reprojection and raster reprojection are separate operations with separate contracts.
- Bounds order is always explicit: `(left, bottom, right, top)`.
- Affine transform order is always explicit: `(a, b, c, d, e, f)` mapping `x = a * col + b * row + c`, `y = d * col + e * row + f`.
- Nodata is per-band where source formats allow it.
- Remote fetching must be explicit and cache-aware; metadata APIs must not hide downloads.

## Evidence Summary

| Metric | Count |
|---|---:|
| Example scripts inspected under `examples/**/*.py` | 121 |
| Root-level Python files checked | 1 |
| GIS-relevant example scripts | 109 |
| Content-distinct scripts | 107 |
| Pattern classes mapped | 89 |
| Proposed API contract rows | 68 |
| Exact duplicate content groups | 14 |
| Basename duplicate groups | 50 |

`docs/carto-engine/gis-contract-evidence.md` contains the full inventory, duplicate groups, pattern counts, line evidence, and coverage exceptions.

## Priority Method

| Priority | Meaning |
|---|---|
| P0 | Required by many examples or blocks multiple recipe families; implementation should happen first. |
| P1 | Common across examples or critical for correctness. |
| P2 | Less frequent but important for specific recipe families. |
| P3 | Rare, domain-specific, or later helper. |
| Defer | Outside the GIS metadata/preparation foundation or requires a separate subsystem. |

Priority uses occurrence count, distinct script count, content-distinct script count, recipe-family breadth, and correctness risk. Physical duplicate scripts are retained for inventory proof, but content-distinct counts prevent root/topic mirrors from silently inflating priority.

## Phase Summary

| Phase | Contract scope |
|---|---|
| `G-002a1` | Raster metadata/read/write foundation. |
| `G-002b` | Raster CRS, transform, nodata, bounds, windowing, resampling, alignment, reprojection, CRS transformer creation, Web Mercator extent, CRS assignment. |
| `G-002c` | Vector read/metadata, vector CRS/reprojection, union, buffer, clip/intersection, rasterization, geometry masking, raster clipping/masking, validation/repair, simplification, boundary union, raster normalization/classification. |
| `later` | Remote/cache/OSM/DEM/Terrarium/thematic/building/population/gridded helper contracts built on the core GIS primitives. |
| `defer` | Network travel-time/routing analysis. |

## Canonical Operation Crosswalk

| Canonical operation | Phase | Priority | Proposed forge3d API candidates | Evidence count | Distinct script paths | Reference behavior | Notes |
|---|---|---|---|---:|---:|---|---|
| `geospatial_raster_read` | G-002a1 | P0 | `read_raster_info`, `read_raster` | 220 `rasterio.open`; 111 reads | 56 open; 47 reads | `rasterio.open`, `DatasetReader.read`, `profile`, `terra::rast` | Metadata first; full value reads can follow once `RasterInfo` is stable. |
| `geospatial_raster_write` | G-002a1 | P0 | `write_raster` | 104 writes; 106 creation-option hits | 38 writes; 29 creation-option scripts | `rasterio.open(..., "w", **profile)`, `DatasetWriter.write`, `terra::writeRaster` | Preserve profile fields, creation options, overwrite behavior, and post-write metadata validation. |
| `affine_transform_handling` | G-002b | P0 | `raster_transform`, `affine_transform`, `transform_from_origin`, `transform_from_bounds`, `validate_transform`, `xy_index`, `pixel_convention` | 169 transform attrs; 36 `from_bounds`; 5 `from_origin` | 45 transform scripts | Rasterio `Affine`, `from_bounds`, `from_origin`; terra extent/resolution | Transform order and pixel center/corner semantics must be explicit. |
| `crs_inspection` | G-002b | P0 | `parse_crs`, `inspect_crs`, `raster_crs`, `vector_crs`, `estimate_local_utm` | 908 CRS literal/object hits; 130 raster CRS; 130 vector CRS | 75 CRS-literal scripts | `rasterio.crs.CRS`, `pyproj.CRS`, GeoPandas `.crs`, terra `crs` | Missing CRS is a diagnostic, not a guess. |
| `nodata_handling` | G-002b | P0 | `nodata_per_band`, `apply_nodata`, `read_raster_mask` | 1012 fill/nodata behavior; 145 nodata attrs | 78 fill scripts; 34 nodata scripts | Rasterio `nodata`, `nodatavals`, masked reads; terra `NAflag` | Must cover `-9999`, `0`, `NaN`, masks, per-band mismatches, and dtype compatibility. |
| `bounds_extent_extraction` | G-002b/G-002c | P0 | `raster_bounds`, `vector_bounds`, `raster_resolution`, `array_bounds` | 75 raster bounds; 101 vector bounds | 37 raster-bound scripts | Rasterio `bounds`/`res`, GeoPandas `total_bounds`, Shapely bounds, terra `ext` | Bounds order is `(left,bottom,right,top)`. |
| `raster_resampling` | G-002b | P0 | `resample_raster` | 557 resampling/category hits | 59 scripts | Rasterio `Resampling`, `read(out_shape=...)`, `warp.reproject`, terra `resample` | Method is required; nearest/mode for categorical, bilinear/cubic for continuous. |
| `raster_grid_alignment` | G-002b | P0 | `assert_grid_compatible`, `align_raster_grid` | 365 shape/extent/transform match hits | 57 scripts | Rasterio profile matching, `aligned_target`, `reproject_match`, terra align/project | Must diagnose CRS, shape, extent, transform, resolution, and nodata mismatches. |
| `raster_reprojection` | G-002b | P0 | `reproject_raster`, `calculate_default_transform`, `warped_vrt` | 57 `reproject`; 7 default-transform hits | 24 reproject scripts | Rasterio `warp.reproject`, `calculate_default_transform`, terra `project` | Separate from vector reprojection; resampling required. |
| `crs_transformer_creation` | G-002b | P0 | `create_crs_transformer` | 232 transformer/source-destination hits | 34 scripts | `pyproj.Transformer.from_crs(..., always_xy=True)` | Axis-order diagnostics are required; `always_xy` is evidence-backed. |
| `web_mercator_extent_calculation` | G-002b | P0 | `web_mercator_bounds`, `transform_bounds` | 47 Web Mercator hits; 8 `transform_bounds` | 26 Web Mercator scripts | Rasterio `transform_bounds`, pyproj transforms | Must reject invalid latitude ranges and document antimeridian behavior. |
| `crs_assignment_or_definition` | G-002b | P0 | `assign_crs` | 10 assignment calls | 8 scripts | GeoPandas `set_crs`, rioxarray `write_crs`, terra `crs(x)<-` | Assignment updates metadata only; no coordinate changes. |
| `raster_windowing_by_bounds` | G-002b | P0 | `window_from_bounds`, `read_raster_window`, `window_transform` | 36 window-from-bounds; 14 windowed reads; 5 window transforms | 22 window scripts | Rasterio `windows.from_bounds`, `dataset.window`, `window_transform` | Boundless behavior and CRS of bounds must be explicit. |
| `vector_data_read` | G-002c | P0 | `read_vector`, `geometry_type`, `vector_schema`, `feature_count` | 25 `read_file`; 806 feature-count hits | 23 read-file scripts; 72 count scripts | GeoPandas `read_file`, Fiona/GDAL layer metadata, terra `vect` | Must expose driver, layers, schema, geometry type, feature count, CRS, bounds. |
| `vector_reprojection` | G-002c | P0 | `reproject_vector` | 39 `to_crs` hits | 23 scripts | GeoPandas `to_crs`, Shapely transform with pyproj, terra `project` | Error on missing source CRS unless explicitly supplied. |
| `vector_geometry_union` | G-002c | P0 | `union_geometries`, `dissolve_vector` | 44 union hits | 19 scripts | Shapely `union_all`/`unary_union`, GeoPandas dissolve, terra union/aggregate | Must handle invalid geometry, mixed output, and empty unions. |
| `vector_buffering` | G-002c | P0 | `buffer_geometry` | 54 buffer hits | 20 scripts | GeoPandas/Shapely `buffer`, terra `buffer` | Warn or error for distance units in geographic CRS. |
| `vector_clip_or_intersection` | G-002c | P0 | `clip_vector`, `intersect_vectors`, `geometry_measure`, `geometry_centroid`, `representative_point`, `interpolate_line` | 2044 clip hits; 260 intersection hits | 66 clip scripts; 70 intersection scripts | GeoPandas `clip`/`overlay`, Shapely `intersection`, terra `crop`/`intersect` | Empty output is valid with a warning; CRS mismatch is an error. |
| `vector_rasterization` | G-002c | P0 | `rasterize_vectors` | 3706 rasterize/target-grid hits | 81 scripts | Rasterio `features.rasterize`, GDAL rasterize, terra `rasterize` | Must cover burn values, fill, dtype, target shape/transform, all_touched, merge_alg. |
| `geometry_masking` | G-002c | P0 | `geometry_mask` | 26 geometry-mask calls | 18 scripts | Rasterio `features.geometry_mask`, terra `mask` | Mask polarity (`invert`) must be explicit and tested. |
| `raster_clip_or_mask` | G-002c | P0 | `mask_raster` | 5384 clip/mask text and calls | 95 scripts | `rasterio.mask.mask`, `geometry_mask`, terra `crop`/`mask` | Needs nodata fill behavior, crop behavior, outside-raster behavior. |
| `geometry_validation_repair` | G-002c | P1 | `validate_geometry`, `repair_geometry` | 127 validity/empty hits; 20 repair hits | 23 validity scripts; 10 repair scripts | Shapely `is_valid`, `is_empty`, `make_valid`, `buffer(0)` | Repair may change type/count; diagnostics must report that. |
| `polygon_boundary_loading_or_union` | G-002c | P1 | `load_boundary` | Covered by read/vector/union/filter patterns | 4 baseline scripts | GeoPandas read/filter/union, terra `vect`/`union` | Keep as a helper over `read_vector`, filtering, reprojection, and union. |
| `geometry_simplification` | G-002c | P1 | `simplify_geometry` | 2 simplify calls | 2 scripts | Shapely `simplify(preserve_topology=...)`, terra `simplifyGeom` | Units and topology preservation must be explicit. |
| `raster_normalization_or_classification` | G-002c | P1 | `normalize_raster`, `classify_raster` | 50 baseline classification/normalization uses | 28 baseline scripts | Numpy/Rasterio workflows; terra `classify` | Must exclude nodata/masked pixels and report class counts/breaks. |
| `remote_geospatial_data_fetch` | later | P1 | `fetch_remote_geodata`, `cache_geodata`, `read_cog` | 216 URL hits; 1450 cache hits | 55 URL scripts; 90 cache scripts | Requests/urllib in examples; GDAL URL support; terra URL reads | Must use explicit cache, timeout, user-agent, checksum/size diagnostics. |
| `slippy_tile_indexing` | later | P1 | `slippy_tile_index` | 610 tile math hits | 49 scripts | Custom XYZ math; Web Mercator tile conventions | Must reject invalid zoom/latitude and state antimeridian policy. |
| `osm_feature_query_or_loading` | later | P1 | `query_osm_features` | 588 OSM/Overpass/Nominatim hits | 33 scripts | Overpass JSON, GeoPandas local extracts | Tests must mock live services. |
| `context_vector_loading` | later | P1 | `load_context_vectors` | Covered by vector IO plus OSM/context patterns | 6 baseline scripts | GeoPandas `read_file`, local GeoJSON/GPKG/WFS extracts | Road/water/rail/building layer loading should stay above core vector IO. |
| `osm_feature_parsing` | later | P1 | `parse_osm_features` | Covered by OSM parsing patterns | 6 baseline scripts | OSM JSON to geometries | Must handle incomplete ways and unsupported relations. |
| `terrain_derivative_preparation` | later | P2 | `prepare_terrain_derivatives` | 1498 derivative hits | 86 scripts | Numpy DEM derivative workflows; terra `terrain` | Needs resolution/units diagnostics. |
| `dem_heightfield_preparation` | later | P1 | `prepare_dem` | 2856 DEM prep hits | 101 scripts | Rasterio read/resample/nodata workflows; terra `rast` | Should build on `read_raster_info`, `read_raster`, `apply_nodata`, and alignment. |
| `gridded_dataset_read` | later | P2 | `read_gridded_dataset` | 101 xarray/rioxarray hits | 7 scripts | `xarray.open_dataset`, `rioxarray.open_rasterio`, terra subdatasets | Defer broad multidimensional support; start with metadata and variable selection. |
| `terrarium_dem_decoding` | later | P2 | `decode_terrarium_dem` | 221 Terrarium hits | 24 scripts | Custom RGB Terrarium decode | Must reject non-RGB or unsupported dtype inputs. |
| `terrarium_dem_tile_fetch_or_mosaic` | later | P2 | `build_terrarium_dem` | Covered by Terrarium and tile/fetch patterns | 10 baseline scripts | Custom tile fetch/mosaic | Build only after fetch/cache/tile math contracts are stable. |
| `landcover_raster_preparation` | later | P1 | `prepare_landcover_raster` | 259 landcover hits | 15 scripts | Rasterio read/resample/class palette; terra classify/project | Nearest/mode resampling for categorical data. |
| `building_height_extraction` | later | P1 | `extract_building_heights` | 3582 height/building hits | 114 scripts | OSM/GeoJSON attribute parsing | Text evidence is broad; exact parsing options need family-level review. |
| `population_raster_preparation` | later | P1 | `prepare_population_raster` | 1543 population hits | 43 scripts | Rasterio read/mask/log/percentile workflows; terra classify/global | Empty valid pixels and negative population values require diagnostics. |
| `building_footprint_loading` | later | P1 | `load_building_footprints` | 586 footprint/building hits | 32 scripts | GeoPandas/GeoJSON/CityJSON/GPKG | Must report unsupported geometry, invalid rings, missing CRS. |
| `gridded_dataset_subsetting_or_sampling` | later | P2 | `subset_grid` | 5914 grid/sample hits | 111 scripts | xarray `.sel`/`.interp`, raster sampling, terra extract | Text evidence is broad; exact axis/time semantics need implementation-time narrowing. |
| `osm_scene_geodata_preparation` | later | P1 | `prepare_osm_scene` | 1324 OSM scene hits | 64 scripts | Combined Overpass/GeoPandas/Shapely script logic | Build as a domain helper over lower-level vector/fetch/OSM primitives. |
| `remote_vector_data_fetch` | later | P2 | `fetch_vector` | 310 GeoJSON/GPKG/WFS hits | 41 scripts | Requests/urllib, GeoPandas URL reads | Must validate content type, GeoJSON shape, CRS presence. |
| `network_travel_time_analysis` | defer | Defer | `defer_routing` | 4 travel-time hits | 2 scripts | NetworkX/custom graph analysis | Separate routing subsystem, not `src/gis/` foundation. |

## First Contracts

`read_raster_info` and `write_raster` remain the first implementation contracts from the original roadmap. This upgrade extends the roadmap from operation-level planning to contract-level planning for every observed future GIS function. Exact per-function contracts live in `rust-gis-implementation-plan.md`.
