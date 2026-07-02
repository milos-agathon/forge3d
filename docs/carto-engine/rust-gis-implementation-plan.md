# Rust GIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the Rust-first GIS contract roadmap required by existing forge3d example scripts before implementing new `forge3d.gis` helpers.

**Architecture:** GIS operations are implemented in Rust under `src/gis/`. Python exposes thin wrappers over Rust bindings and may use `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or R `terra` only as reference behavior in docs/tests, never as the forge3d backend. The roadmap is contract-level: every future function, attribute, diagnostic, error, and test requirement below is derived from existing examples or explicitly listed as an exception.

**Tech Stack:** Rust, PyO3, numpy arrays at the Python boundary, `gdal`/`proj` for broad geospatial support, `geo`/`geo-types`/`geojson` for vector data, `tiff`/`image`/existing forge3d COG code where narrower native support is enough, and `ndarray` for array validation and data movement.

## Global Constraints

- Documentation-only change: do not implement Rust, Python wrappers, examples, shaders, rendering behavior, visual goldens, or runtime behavior while editing this roadmap.
- GIS backend behavior belongs in Rust under `src/gis/`.
- Python wrappers remain thin and must not use `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, or `terra` as backend implementation.
- Missing CRS is never guessed.
- CRS assignment is not reprojection.
- Raster reprojection requires an explicit resampling method.
- Raster and vector reprojection are separate APIs.
- Bounds order is `(left, bottom, right, top)`.
- Affine transform order is `(a, b, c, d, e, f)` mapping `x = a * col + b * row + c`, `y = d * col + e * row + f`.
- Nodata is per-band where supported.
- Remote fetching is explicit and cache-aware.
- Network routing is deferred outside `src/gis/`.

---

## Evidence Baseline

The previous roadmap planned operations. This upgrade derives contracts from example-script evidence:

| Metric | Count |
|---|---:|
| `examples/**/*.py` scripts inspected | 121 |
| Root-level Python files checked | 1 |
| GIS-relevant example scripts found | 109 |
| Content-distinct example scripts | 107 |
| GIS pattern classes found | 89 |
| Evidence-derived API rollups | 68 |
| Additional explicitly implied helper rows retained from the prior roadmap | 9 |
| Total contract rows in this plan | 77 |

Full inventory, duplicates, pattern counts, and coverage exceptions are in `docs/carto-engine/gis-contract-evidence.md`.

## Prioritization Method

| Priority | Rule |
|---|---|
| P0 | Required by many examples or blocks multiple recipe families; implement first. |
| P1 | Common across examples or critical for correctness. |
| P2 | Less frequent but important for a specific recipe family. |
| P3 | Rare, domain-specific, or polish. |
| Defer | Outside GIS metadata/preparation foundation or requires a separate subsystem. |

Priority combines occurrence count, distinct physical scripts, distinct content scripts, recipe-family breadth, and correctness risk. Duplicate root/topic scripts remain in inventory, but content-distinct counts govern priority.

## Rust Module Layout

| Module | Responsibility | First phase |
|---|---|---|
| `src/gis/mod.rs` | Public Rust module exports and feature gating | G-002a1 |
| `src/gis/error.rs` | `GisError`, diagnostic codes, warning struct | G-002a1 |
| `src/gis/types.rs` | `RasterInfo`, `VectorInfo`, `CrsInfo`, `AffineTransform`, `RasterWindow`, dtype and shape helpers | G-002a1 |
| `src/gis/raster_info.rs` | Raster metadata, read contracts, masks | G-002a1 |
| `src/gis/raster_write.rs` | Raster write contracts and post-write metadata validation | G-002a1 |
| `src/gis/crs.rs` | CRS parsing, inspection, assignment, transformer creation | G-002b |
| `src/gis/affine.rs` | Transform math, bounds/resolution derivation, pixel convention, window math | G-002b |
| `src/gis/warp.rs` | Raster resampling, alignment, reprojection | G-002b |
| `src/gis/vector.rs` | Vector metadata/read/reproject/clip/union/buffer/simplify contracts | G-002c |
| `src/gis/rasterize.rs` | Vector rasterization, geometry masks, raster masking | G-002c |
| `src/gis/thematic.rs` | Raster normalization/classification and thematic preparation contracts | G-002c |
| `src/gis/remote.rs` | Explicit fetch/cache contracts | later |
| `src/gis/tiles.rs` | Web Mercator and XYZ tile indexing | later |
| `src/gis/osm.rs` | OSM query/load/parse contracts | later |
| `src/gis/terrarium.rs` | Terrarium decode/fetch/mosaic | later |
| `src/gis/domain.rs` | DEM, landcover, population, building, and scene-prep helpers built from primitives | later |

`src/geo/` already contains limited PROJ-backed coordinate reprojection. Reuse or re-export it when implementing `src/gis/crs.rs`; do not duplicate it.

## Shared Output Types

`RasterInfo` fields required by the examples: `path`, `driver`, `width`, `height`, `band_count`, `shape`, `dtype_per_band`, `crs_wkt`, `crs_authority`, `transform`, `bounds`, `resolution`, `nodata_per_band`, `mask_flags`, `block_size`, `tiling`, `compression`, `profile`, `tags`, `overviews`, `is_georeferenced`, `warnings`.

`VectorInfo` fields required by the examples: `path`, `driver`, `layers`, `layer_name`, `feature_count`, `geometry_type`, `schema`, `columns`, `crs_wkt`, `crs_authority`, `bounds`, `is_empty`, `invalid_geometry_count`, `warnings`.

`CrsInfo` fields required by the examples: `input`, `wkt`, `authority_name`, `authority_code`, `axis_order`, `is_geographic`, `is_projected`, `area_of_use`, `warnings`.

`GeometryOperationInfo` fields required by vector examples: `input_count`, `output_count`, `dropped_count`, `invalid_input_count`, `repaired_count`, `empty_output`, `geometry_type`, `bounds`, `crs_wkt`, `warnings`.

`RemoteDatasetInfo` fields required by remote examples: `url`, `cache_path`, `status`, `content_type`, `byte_size`, `checksum`, `etag`, `last_modified`, `from_cache`, `warnings`.

## Diagnostic And Error Vocabulary

Stable diagnostics: `missing_crs`, `invalid_crs`, `axis_order_ambiguous`, `assignment_not_reprojection`, `missing_transform`, `invalid_transform`, `rotated_or_sheared_transform`, `pixel_convention_explicit`, `bounds_order_invalid`, `not_georeferenced`, `nodata_dtype_mismatch`, `per_band_nodata_mismatch`, `mask_polarity_explicit`, `shape_mismatch`, `crs_mismatch`, `grid_mismatch`, `resampling_required`, `categorical_resampling_warning`, `empty_geometry`, `invalid_geometry`, `geometry_repaired`, `geometry_repair_changed_type`, `empty_raster`, `unsupported_dtype`, `unsupported_driver`, `unsupported_option`, `unsupported_geometry_type`, `missing_layer`, `empty_feature_set`, `network_timeout`, `cache_miss`, `cache_stale`, `checksum_mismatch`, `malformed_payload`, `metadata_unavailable`.

Stable errors: `NotFound`, `Io`, `AlreadyExists`, `InvalidArgument`, `UnsupportedDriver`, `UnsupportedDType`, `UnsupportedCreationOption`, `InvalidRaster`, `InvalidShape`, `InvalidTransform`, `InvalidCrs`, `MissingCrs`, `CrsMismatch`, `ShapeMismatch`, `GridMismatch`, `InvalidBounds`, `InvalidNodata`, `UnsupportedGeometry`, `InvalidGeometry`, `EmptyGeometry`, `EmptyRaster`, `Network`, `Cache`, `MalformedPayload`, `WriteFailed`, `PostWriteValidationFailed`.

## Test Bundles

Every contract row below names one or more bundles. A bundle is explicit test scope.

| Bundle | Required tests |
|---|---|
| `T-raster-meta` | Happy-path GeoTIFF fixture; missing path; unsupported driver; zero width/height/bands; missing CRS warning; invalid CRS error; missing transform warning; rotated/sheared transform warning; block/tiling/compression metadata; representative fixture linked to raster metadata examples. |
| `T-raster-read` | Happy array read; missing path; unsupported driver; unsupported dtype; windowed read; boundless option if supported; nodata/mask compatibility; shape and band order validation. |
| `T-raster-write` | Happy GeoTIFF write; parent missing; existing path without overwrite; unsupported dtype; unsupported driver; unsupported creation option; invalid CRS; invalid transform; shape mismatch; nodata compatibility; post-write metadata validation. |
| `T-crs` | EPSG string; EPSG integer; WKT/PROJ when supported; invalid CRS; missing CRS; source/destination pair; always-XY axis-order check; assignment distinct from reprojection. |
| `T-affine` | Six-coefficient happy path; non-finite coefficient; zero pixel size; bounds derivation; resolution derivation; pixel center/corner convention; row/col and xy/index round trip; rotated/sheared diagnostics. |
| `T-window` | Bounds-to-window happy path; invalid bounds order; CRS mismatch; outside-raster bounds; boundless read; window transform; clipped output metadata. |
| `T-reproject` | Raster reprojection happy path; missing source CRS; invalid destination CRS; resampling required; nearest vs bilinear; categorical warning; nodata preserved; output shape/transform/CRS validation. |
| `T-align` | Matching grid happy path; CRS mismatch; shape mismatch; transform mismatch; resolution mismatch; nodata mismatch; post-alignment metadata validation. |
| `T-vector-io` | Read GeoJSON/GPKG fixture; missing path; unsupported driver; missing layer; empty layer; missing CRS warning; invalid CRS error; schema/columns/feature count/geometry type metadata. |
| `T-vector-crs` | Vector reprojection happy path; missing source CRS; invalid destination CRS; CRS mismatch across inputs; output bounds/CRS validation. |
| `T-vector-geom` | Union/buffer/clip/intersection/simplify happy paths; invalid geometry; repair by make-valid/buffer(0) equivalent; empty geometry; mixed geometry type; geographic-distance warning; output metadata. |
| `T-rasterize-mask` | Rasterize happy path; burn values; `all_touched`; `merge_alg`; fill value; dtype overflow; target shape/transform; geometry mask polarity; raster clip/mask crop/nodata behavior; CRS mismatch. |
| `T-thematic` | Normalize/classify happy path; empty valid pixels; unsupported dtype; non-finite bins; nodata/mask exclusion; categorical class table and unknown-class diagnostics. |
| `T-remote` | Mocked URL fetch; cache hit; cache miss; stale-cache fallback; timeout; unsupported scheme/content type; checksum mismatch; malformed payload; atomic cache write. |
| `T-osm` | Mocked Overpass JSON; empty result; incomplete way; unsupported relation; tag filtering; cache behavior; CRS/bounds metadata. |
| `T-domain` | Representative fixture for DEM, landcover, population, building, Terrarium, and OSM-scene helpers; missing input; CRS/grid mismatch; unsupported dtype/format; empty valid data; post-operation metadata validation. |

## Contract Matrix: G-002a1 Raster Foundation

| API | Phase/Priority | Rust module / Python wrapper | Required inputs | Required output attributes and metadata | Diagnostics/errors | Tests | Evidence and reference behavior |
|---|---|---|---|---|---|---|---|
| `read_raster_info` | G-002a1/P0 | `raster_info.rs`; `forge3d.gis.read_raster_info(path)` | Local readable raster path | Full `RasterInfo`: path, driver, width, height, band_count, shape, dtype_per_band, CRS, transform, bounds, resolution, nodata, masks, blocks, tiling, compression, profile, tags, overviews, warnings | `NotFound`, `Io`, `UnsupportedDriver`, `InvalidRaster`, `invalid_crs`, `missing_crs`, `missing_transform`, `metadata_unavailable` | `T-raster-meta` | 2033 rollup hits; 86 scripts; `rasterio.open`/profile/attrs; terra `rast`. |
| `read_raster` | G-002a1/P0 | `raster_info.rs`; `forge3d.gis.read_raster(path, bands=None, window=None, masked=False)` | Path plus optional bands/window/masked flag | Array, `RasterInfo`, selected bands, mask/nodata summary | `NotFound`, `UnsupportedDriver`, `UnsupportedDType`, `InvalidShape`, `InvalidNodata`, `InvalidBounds` | `T-raster-read`, `T-window` | 111 read hits; 47 scripts; Rasterio `DatasetReader.read`. |
| `write_raster` | G-002a1/P0 | `raster_write.rs`; `forge3d.gis.write_raster(path, array, *, crs=None, transform=None, nodata=None, driver="GTiff", overwrite=False, creation_options=None, like_path=None, like_info=None)` | Output path, `(height,width)` or `(bands,height,width)` array, optional CRS/transform/nodata/driver/options/like source | Post-write `RasterInfo` read from disk, not assembled from request | `AlreadyExists`, `NotFound`, `UnsupportedDType`, `UnsupportedDriver`, `UnsupportedCreationOption`, `InvalidCrs`, `InvalidTransform`, `InvalidNodata`, `ShapeMismatch`, `WriteFailed`, `PostWriteValidationFailed` | `T-raster-write`, `T-raster-meta` | 1029 rollup hits; 102 scripts; Rasterio writer/profile behavior; terra `writeRaster`. |

## Contract Matrix: G-002b Raster CRS, Transform, Alignment, Reprojection

| API | Phase/Priority | Rust module / Python wrapper | Required inputs | Required output attributes and metadata | Diagnostics/errors | Tests | Evidence and reference behavior |
|---|---|---|---|---|---|---|---|
| `parse_crs` | G-002b/P0 | `crs.rs`; `forge3d.gis.parse_crs(value)` | EPSG string/int, WKT, PROJ string, or CRS dict | `CrsInfo` with WKT, authority, axis order, projection kind | `InvalidCrs`, `missing_crs`, `axis_order_ambiguous` | `T-crs` | 908 rollup hits; 75 scripts; pyproj/rasterio CRS. |
| `inspect_crs` | G-002b/P0 | `crs.rs`; `forge3d.gis.inspect_crs(source)` | Raster/vector info, path, or CRS value | `CrsInfo` or `None` plus warnings | `InvalidCrs`, `missing_crs`, `metadata_unavailable` | `T-crs`, `T-raster-meta`, `T-vector-io` | 26 rollup hits; 18 scripts; examples branch on missing CRS. |
| `raster_crs` | G-002b/P0 | `crs.rs`; `RasterInfo.crs` / `forge3d.gis.raster_crs(info_or_path)` | Raster path or `RasterInfo` | Raster CRS WKT/authority and warnings | `InvalidCrs`, `missing_crs` | `T-crs`, `T-raster-meta` | 130 hits; 32 scripts; Rasterio `dataset.crs`. |
| `assign_crs` | G-002b/P0 | `crs.rs`; `forge3d.gis.assign_crs(target, crs, overwrite=False)` | Metadata/path plus CRS | Updated metadata/path info; no coordinate changes | `InvalidCrs`, `AlreadyExists` when CRS exists and overwrite false, `assignment_not_reprojection` warning | `T-crs`, `T-raster-write`, `T-vector-crs` | 10 hits; 8 scripts; GeoPandas `set_crs`, rioxarray `write_crs`. |
| `create_crs_transformer` | G-002b/P0 | `crs.rs`; `forge3d.gis.CrsTransform.from_crs(src, dst, always_xy=True)` | Source and destination CRS | Transformer metadata, source/destination `CrsInfo`, axis-order policy | `InvalidCrs`, `MissingCrs`, `axis_order_ambiguous` | `T-crs` | 232 rollup hits; 34 scripts; `always_xy=True` appears 68 times. |
| `web_mercator_bounds` | G-002b/P0 | `crs.rs`/`tiles.rs`; `forge3d.gis.web_mercator_bounds(bounds, src_crs)` | Bounds and source CRS | EPSG:3857 bounds, optional lon/lat bounds, transform metadata | `InvalidBounds`, `InvalidCrs`, latitude/antimeridian warnings | `T-crs`, `T-affine` | 47 hits; 26 scripts; `EPSG:3857`/Web Mercator patterns. |
| `transform_bounds` | G-002b/P0 | `crs.rs`; `forge3d.gis.transform_bounds(src_crs, dst_crs, bounds, densify=...)` | Source CRS, destination CRS, bounds | Transformed bounds and CRS metadata | `InvalidCrs`, `InvalidBounds`, antimeridian warning | `T-crs` | 8 hits; 6 scripts; Rasterio `transform_bounds`. |
| `raster_transform` | G-002b/P0 | `affine.rs`; `RasterInfo.transform` / `forge3d.gis.raster_transform(info_or_path)` | Raster path or `RasterInfo` | Six coefficients, pixel convention, rotation/shear flag | `InvalidTransform`, `missing_transform`, `rotated_or_sheared_transform` | `T-affine`, `T-raster-meta` | 169 hits; 45 scripts; Rasterio `dataset.transform`. |
| `affine_transform` | G-002b/P1 | `affine.rs`; `forge3d.gis.AffineTransform(a,b,c,d,e,f)` | Six finite coefficients | Transform object, determinant, pixel scale, rotation/shear flags | `InvalidTransform` | `T-affine` | 2 explicit `Affine` hits; 2 scripts; still required by transform contract. |
| `transform_from_origin` | G-002b/P0 | `affine.rs`; `forge3d.gis.transform_from_origin(west, north, xsize, ysize)` | Origin and pixel size | Six-coefficient transform, resolution, bounds helper compatibility | `InvalidTransform`, `InvalidArgument` | `T-affine` | 5 hits; 5 scripts; Rasterio `from_origin`. |
| `transform_from_bounds` | G-002b/P0 | `affine.rs`; `forge3d.gis.transform_from_bounds(bounds, width, height)` | Bounds, width, height | Transform and resolution | `InvalidBounds`, `InvalidShape`, `InvalidTransform` | `T-affine` | 36 hits; 22 scripts; Rasterio `from_bounds`. |
| `array_bounds` | G-002b/P1 | `affine.rs`; `forge3d.gis.array_bounds(height, width, transform)` | Shape and transform | Bounds `(left,bottom,right,top)` | `InvalidShape`, `InvalidTransform` | `T-affine` | Explicitly required by examples/audit even when folded into bounds calls. |
| `raster_bounds` | G-002b/P0 | `affine.rs`; `RasterInfo.bounds` / `forge3d.gis.raster_bounds(info_or_path)` | Raster info/path | Bounds with CRS and pixel convention | `missing_transform`, `missing_crs`, `InvalidTransform` | `T-affine`, `T-raster-meta` | 75 hits; 37 scripts; Rasterio bounds. |
| `raster_resolution` | G-002b/P0 | `affine.rs`; `RasterInfo.resolution` | Raster info/path | Positive `(pixel_width, pixel_height)` | `InvalidTransform`, `missing_transform` | `T-affine` | 2 hits; 1 script; Rasterio `res`. |
| `validate_transform` | G-002b/P1 | `affine.rs`; `forge3d.gis.validate_transform(transform, *, require_north_up=False)` | Transform and validation options | Validity status, rotation/shear flags, resolution | `InvalidTransform`, `rotated_or_sheared_transform` | `T-affine` | 193 rotation/shear/north-up hits; 31 scripts. |
| `pixel_convention` | G-002b/P1 | `affine.rs`; `forge3d.gis.pixel_convention(transform)` | Transform or `RasterInfo` | Pixel center/corner convention note | `pixel_convention_explicit` | `T-affine` | 6 hits; 2 scripts; prevents row/col off-by-half ambiguity. |
| `rowcol` | G-002b/P2 | `affine.rs`; `forge3d.gis.rowcol(transform, x, y)` | Transform and coordinates | Row/col indices, bounds status | `InvalidTransform`, `InvalidBounds` | `T-affine` | Required by affine contract; reference Rasterio `rowcol`. |
| `xy_index` | G-002b/P1 | `affine.rs`; `forge3d.gis.xy(transform, row, col)` and `index(transform, x, y)` | Transform plus row/col or coordinates | Coordinate/index pair, pixel convention | `InvalidTransform`, `InvalidBounds` | `T-affine` | 1 hit; 1 script; Rasterio `xy`/`index`. |
| `nodata_per_band` | G-002b/P0 | `raster_info.rs`; `RasterInfo.nodata_per_band` | Raster info/path | Per-band nodata list | `InvalidNodata`, `per_band_nodata_mismatch`, `nodata_dtype_mismatch` | `T-raster-meta`, `T-raster-write` | 145 hits; 34 scripts; Rasterio `nodata`/`nodatavals`. |
| `apply_nodata` | G-002b/P0 | `raster_info.rs`; `forge3d.gis.apply_nodata(array, nodata, mask=None)` | Array, nodata, optional mask | Masked/filled array, valid count, nodata summary | `InvalidNodata`, `UnsupportedDType`, `empty_raster` | `T-raster-read`, `T-thematic` | 1012 hits; 78 scripts; examples use `0`, `-9999`, `NaN`, masks. |
| `read_raster_mask` | G-002b/P1 | `raster_info.rs`; `forge3d.gis.read_raster_mask(path, band=None)` | Raster path and optional band | Mask array, mask flags, nodata metadata | `NotFound`, `InvalidNodata`, `mask_polarity_explicit` | `T-raster-read` | 3 hits; 3 scripts; Rasterio `read_masks`. |
| `resample_raster` | G-002b/P0 | `warp.rs`; `forge3d.gis.resample_raster(source, shape_or_resolution, method)` | Raster array/info or path, target shape/resolution, method | Output array, `RasterInfo`, method, nodata summary | `resampling_required`, `UnsupportedOption`, `categorical_resampling_warning`, `ShapeMismatch` | `T-reproject`, `T-align` | 557 hits; 59 scripts; Rasterio `Resampling`. |
| `assert_grid_compatible` | G-002b/P0 | `warp.rs`; `forge3d.gis.assert_grid_compatible(a, b, *, compare_nodata=True)` | Two raster/vector target grid infos | Compatibility status and diff fields | `crs_mismatch`, `shape_mismatch`, `grid_mismatch`, `nodata_dtype_mismatch` | `T-align` | 365 hits; 57 scripts; repeated shape/extent/transform checks. |
| `align_raster_grid` | G-002b/P1 | `warp.rs`; `forge3d.gis.align_raster_grid(source, target_info, method)` | Source raster, target grid, resampling method | Aligned array/info, diff diagnostics | `crs_mismatch`, `resampling_required`, `grid_mismatch` | `T-align`, `T-reproject` | Explicitly required by roadmap; reference Rasterio `aligned_target`/rioxarray `reproject_match`. |
| `reproject_raster` | G-002b/P0 | `warp.rs`; `forge3d.gis.reproject_raster(source, dst_crs, resampling, *, dst_grid=None)` | Source raster, destination CRS, resampling method, optional grid | Reprojected array, output `RasterInfo`, CRS transform metadata | `MissingCrs`, `InvalidCrs`, `resampling_required`, `InvalidTransform`, `InvalidNodata` | `T-reproject` | 57 hits; 24 scripts; Rasterio `warp.reproject`. |
| `calculate_default_transform` | G-002b/P0 | `warp.rs`; `forge3d.gis.calculate_default_transform(src_info, dst_crs, resolution=None)` | Source info, destination CRS, optional resolution | Transform, width, height, bounds, CRS | `MissingCrs`, `InvalidCrs`, `InvalidTransform` | `T-reproject`, `T-affine` | 7 hits; 7 scripts; Rasterio equivalent. |
| `warped_vrt` | later/P2 | `warp.rs`; `forge3d.gis.warped_vrt_info(source, dst_crs, resampling)` | Source raster, destination CRS, resampling | Virtual raster metadata only | `MissingCrs`, `InvalidCrs`, `resampling_required` | `T-reproject` | Required by task inventory; no current high-priority detector count, so keep later. |
| `window_from_bounds` | G-002b/P0 | `affine.rs`; `forge3d.gis.window_from_bounds(info, bounds, boundless=False)` | Raster info, bounds, boundless flag | Pixel window, clipped bounds, output transform/shape | `InvalidBounds`, `CrsMismatch`, `InvalidTransform` | `T-window` | 36 hits; 22 scripts; Rasterio `windows.from_bounds`. |
| `read_raster_window` | G-002b/P0 | `raster_info.rs`; `forge3d.gis.read_raster_window(path, bounds_or_window, boundless=False)` | Path plus bounds/window | Array, mask, window transform, output `RasterInfo` | `InvalidBounds`, `CrsMismatch`, `ShapeMismatch`, `InvalidNodata` | `T-window`, `T-raster-read` | 14 hits; 11 scripts; windowed Rasterio reads. |
| `window_transform` | G-002b/P0 | `affine.rs`; `forge3d.gis.window_transform(info, window)` | Raster info and window | Transform for windowed output | `InvalidTransform`, `InvalidBounds` | `T-window`, `T-affine` | 5 hits; 5 scripts; Rasterio `window_transform`. |

## Contract Matrix: G-002c Vector, Mask, Rasterization, Classification

| API | Phase/Priority | Rust module / Python wrapper | Required inputs | Required output attributes and metadata | Diagnostics/errors | Tests | Evidence and reference behavior |
|---|---|---|---|---|---|---|---|
| `read_vector` | G-002c/P0 | `vector.rs`; `forge3d.gis.read_vector(path, layer=None, columns=None)` | Local vector path, optional layer/columns | Features plus `VectorInfo` | `NotFound`, `UnsupportedDriver`, `missing_layer`, `empty_feature_set`, `missing_crs`, `invalid_crs` | `T-vector-io` | 25 hits; 23 scripts; GeoPandas `read_file`, terra `vect`. |
| `geometry_type` | G-002c/P1 | `vector.rs`; `VectorInfo.geometry_type` | Vector path/info/features | Geometry type or mixed marker | `UnsupportedGeometry`, `empty_feature_set` | `T-vector-io` | 107 hits; 34 scripts. |
| `vector_schema` | G-002c/P1 | `vector.rs`; `VectorInfo.schema` / `columns` | Vector path/info | Column names, dtype/schema summary | `missing_layer`, `metadata_unavailable` | `T-vector-io` | 4 hits; 4 scripts; examples branch on columns. |
| `feature_count` | G-002c/P1 | `vector.rs`; `VectorInfo.feature_count` | Vector path/info/features | Feature count and empty flag | `empty_feature_set` warning | `T-vector-io` | 806 hits; 72 scripts; examples use `len(...)`. |
| `vector_crs` | G-002c/P0 | `crs.rs`; `VectorInfo.crs` / `forge3d.gis.vector_crs(info_or_path)` | Vector info/path | CRS WKT/authority | `missing_crs`, `invalid_crs` | `T-crs`, `T-vector-io` | 130 hits; 32 scripts; GeoPandas `.crs`. |
| `vector_bounds` | G-002c/P0 | `vector.rs`; `forge3d.gis.vector_bounds(source)` | Vector info/features/geometry | Bounds, CRS, empty flag | `empty_geometry`, `missing_crs` warning | `T-vector-io`, `T-vector-geom` | 101 rollup hits; 37 scripts; GeoPandas `total_bounds`. |
| `reproject_vector` | G-002c/P0 | `vector.rs`; `forge3d.gis.reproject_vector(input, dst_crs, src_crs=None)` | Vector features/info, destination CRS, optional supplied source CRS | Reprojected features, source/destination CRS, bounds, feature count | `MissingCrs`, `InvalidCrs`, `invalid_geometry`, `geometry_repaired` optional | `T-vector-crs` | 39 hits; 23 scripts; GeoPandas `to_crs`. |
| `union_geometries` | G-002c/P0 | `vector.rs`; `forge3d.gis.union_geometries(geometries, crs=None, repair=False)` | Geometry collection/features and optional CRS | Union geometry, bounds, input/output counts | `InvalidGeometry`, `empty_geometry`, `crs_mismatch`, `geometry_repair_changed_type` | `T-vector-geom` | 44 hits; 19 scripts; Shapely `union_all`/`unary_union`. |
| `dissolve_vector` | G-002c/P2 | `vector.rs`; `forge3d.gis.dissolve_vector(features, by=None)` | Vector features and optional group key | Dissolved features, counts, schema summary | `missing_column`, `InvalidGeometry`, `empty_feature_set` | `T-vector-geom`, `T-vector-io` | Lower-frequency union variant; GeoPandas `dissolve`. |
| `buffer_geometry` | G-002c/P0 | `vector.rs`; `forge3d.gis.buffer_geometry(geometry, distance, *, crs=None)` | Geometry, distance, optional CRS | Buffered geometry, CRS, bounds, empty flag | `InvalidGeometry`, `empty_geometry`, geographic-distance warning | `T-vector-geom` | 54 hits; 20 scripts; Shapely/GeoPandas `buffer`. |
| `clip_vector` | G-002c/P0 | `vector.rs`; `forge3d.gis.clip_vector(input, aoi)` | Input features and AOI geometry/features | Clipped features, dropped count, bounds, CRS | `CrsMismatch`, `InvalidGeometry`, `empty_feature_set`, `empty_geometry` | `T-vector-geom`, `T-vector-crs` | 2044 hits; 66 scripts; GeoPandas `clip`. |
| `intersect_vectors` | G-002c/P0 | `vector.rs`; `forge3d.gis.intersect_vectors(a, b)` | Two geometry/vector inputs | Intersection output, dropped/empty counts, bounds | `CrsMismatch`, `InvalidGeometry`, `empty_geometry` warning | `T-vector-geom` | 260 hits; 70 scripts; Shapely `intersection`, GeoPandas `overlay`. |
| `geometry_measure` | G-002c/P1 | `vector.rs`; `forge3d.gis.geometry_measure(geometry, crs=None)` | Geometry and optional CRS | Length/area, units, CRS warning | geographic-units warning, `InvalidGeometry`, `empty_geometry` | `T-vector-geom` | 33 hits; 14 scripts; Shapely `.length`/`.area`. |
| `geometry_centroid` | G-002c/P1 | `vector.rs`; `forge3d.gis.geometry_centroid(geometry)` | Geometry | Centroid geometry and source bounds | `InvalidGeometry`, `empty_geometry` | `T-vector-geom` | 4 hits; 4 scripts. |
| `representative_point` | G-002c/P2 | `vector.rs`; `forge3d.gis.representative_point(geometry)` | Geometry | Interior representative point | `InvalidGeometry`, `empty_geometry` | `T-vector-geom` | 2 hits; 2 scripts. |
| `interpolate_line` | G-002c/P2 | `vector.rs`; `forge3d.gis.interpolate_line(line, distance, normalized=False)` | Line geometry and distance | Point geometry, distance metadata | `UnsupportedGeometry`, `InvalidArgument`, `empty_geometry` | `T-vector-geom` | 8 hits; 4 scripts. |
| `validate_geometry` | G-002c/P1 | `vector.rs`; `forge3d.gis.validate_geometry(geometry)` | Geometry/features | Validity status, reason, empty flag | `invalid_geometry`, `empty_geometry` | `T-vector-geom` | 127 hits; 23 scripts; Shapely `is_valid`/`is_empty`. |
| `repair_geometry` | G-002c/P1 | `vector.rs`; `forge3d.gis.repair_geometry(geometry, method="make_valid")` | Invalid geometry and method | Repaired geometry, type/count change metadata | `InvalidGeometry`, `geometry_repaired`, `geometry_repair_changed_type`, `UnsupportedOption` | `T-vector-geom` | 20 hits; 10 scripts; Shapely `make_valid` or `buffer(0)`. |
| `simplify_geometry` | G-002c/P1 | `vector.rs`; `forge3d.gis.simplify_geometry(geometry, tolerance, preserve_topology=True)` | Geometry, tolerance, topology flag | Simplified geometry, before/after vertex counts | `InvalidArgument`, geographic-units warning, `empty_geometry` | `T-vector-geom` | 2 hits; 2 scripts; Shapely `simplify`. |
| `load_boundary` | G-002c/P1 | `vector.rs`; `forge3d.gis.load_boundary(path, filter=None, union=True, dst_crs=None)` | Vector path, optional filter, union flag, destination CRS | Boundary geometry, `VectorInfo`, filter metadata, union count | `NotFound`, `missing_layer`, `empty_feature_set`, `CrsMismatch`, `InvalidGeometry` | `T-vector-io`, `T-vector-crs`, `T-vector-geom` | Explicitly required by roadmap; built from read/filter/reproject/union examples. |
| `rasterize_vectors` | G-002c/P0 | `rasterize.rs`; `forge3d.gis.rasterize_vectors(geometries, target_info, burn_values=1, fill=0, dtype="uint8", all_touched=False, merge_alg="replace")` | Geometries, target grid, burn/fill/dtype/options | Raster array, target `RasterInfo`, burn schema, counts | `CrsMismatch`, `InvalidTransform`, `InvalidShape`, `UnsupportedGeometry`, `UnsupportedDType`, dtype overflow | `T-rasterize-mask` | 3706 rollup hits; 81 scripts; Rasterio `features.rasterize`. |
| `geometry_mask` | G-002c/P0 | `rasterize.rs`; `forge3d.gis.geometry_mask(geometries, target_info, invert=False, all_touched=False)` | Geometries, target grid, polarity flags | Boolean mask, polarity metadata, target transform/shape | `CrsMismatch`, `InvalidTransform`, `InvalidShape`, `InvalidGeometry`, `mask_polarity_explicit` | `T-rasterize-mask` | 26 hits; 18 scripts; Rasterio `geometry_mask`. |
| `mask_raster` | G-002c/P0 | `rasterize.rs`; `forge3d.gis.mask_raster(source, geometries, crop=False, filled=True, nodata=None)` | Raster source, geometries, crop/fill/nodata flags | Output array, mask, output `RasterInfo`, crop bounds | `CrsMismatch`, `InvalidGeometry`, `empty_geometry`, `InvalidNodata`, `empty_raster` | `T-rasterize-mask`, `T-window` | 5384 rollup hits; 95 scripts; Rasterio `mask.mask`. |
| `normalize_raster` | G-002c/P1 | `thematic.rs`; `forge3d.gis.normalize_raster(array, mask=None, nodata=None, method="linear")` | Array, optional mask/nodata, method | Normalized array, min/max/percentile metadata, valid count | `UnsupportedDType`, `InvalidNodata`, `empty_raster`, non-finite diagnostic | `T-thematic` | Required by normalization/classification examples; reference Numpy/Rasterio workflows. |
| `classify_raster` | G-002c/P1 | `thematic.rs`; `forge3d.gis.classify_raster(array, bins, nodata=None, labels=None)` | Array, bins, optional nodata/labels | Class array, class table, counts, nodata-excluded count | `UnsupportedDType`, `InvalidArgument`, non-finite bins, `empty_raster` | `T-thematic` | 50 baseline uses across 28 scripts; terra `classify`. |

## Contract Matrix: Later Domain And Remote Helpers

| API | Phase/Priority | Rust module / Python wrapper | Required inputs | Required output attributes and metadata | Diagnostics/errors | Tests | Evidence and reference behavior |
|---|---|---|---|---|---|---|---|
| `fetch_remote_geodata` | later/P1 | `remote.rs`; `forge3d.gis.fetch_remote_geodata(url, cache=None, timeout=None, checksum=None)` | URL and explicit cache/fetch options | `RemoteDatasetInfo`, local path, cache status | `Network`, `Cache`, `network_timeout`, `checksum_mismatch`, `malformed_payload` | `T-remote` | 216 URL hits; 55 scripts; requests/urllib examples. |
| `cache_geodata` | later/P1 | `remote.rs`; `forge3d.gis.cache_geodata(key_or_url, cache_dir, refresh=False)` | Cache key/URL and directory | Cache path/status, age, atomic-write metadata | `cache_miss`, `cache_stale`, `Io`, `checksum_mismatch` | `T-remote` | 1450 cache hits; 90 scripts. |
| `fetch_vector` | later/P2 | `remote.rs`; `forge3d.gis.fetch_vector(url, cache=None)` | URL and cache policy | Local vector path or parsed GeoJSON metadata, CRS warning | `Network`, `UnsupportedDriver`, `malformed_payload`, `missing_crs` | `T-remote`, `T-vector-io` | 310 hits; 41 scripts; GeoJSON/GPKG/WFS patterns. |
| `read_cog` | later/P2 | `raster_info.rs`/existing COG; `forge3d.gis.read_cog(path_or_url, window=None, overview=None)` | Local/remote COG, optional window/overview | Tile/window array or metadata, overview/tile info, range-read diagnostics | `UnsupportedDriver`, `InvalidBounds`, `Network`, `metadata_unavailable` | `T-raster-read`, `T-remote` | 1010 rollup hits; 102 scripts; existing forge3d COG examples. |
| `slippy_tile_index` | later/P1 | `tiles.rs`; `forge3d.gis.slippy_tile_index(bounds, zoom, crs="EPSG:4326")` | Bounds, zoom, CRS | Tile x/y/z list, tile bounds, Web Mercator bounds | `InvalidBounds`, invalid zoom/latitude, antimeridian warning | `T-crs`, `T-domain` | 610 hits; 49 scripts. |
| `query_osm_features` | later/P1 | `osm.rs`; `forge3d.gis.query_osm_features(aoi, tags, cache=None)` | AOI, tag filters, cache/network options | OSM JSON/features, bounds, cache metadata | `Network`, `empty_feature_set`, `malformed_payload`, rate-limit diagnostic | `T-osm`, `T-remote` | 588 hits; 33 scripts; Overpass/Nominatim examples. |
| `parse_osm_features` | later/P1 | `osm.rs`; `forge3d.gis.parse_osm_features(osm_json, tags=None)` | OSM JSON payload and optional tag filters | Feature groups, tag schema, skipped/invalid counts | `malformed_payload`, incomplete way, unsupported relation, `InvalidGeometry` | `T-osm`, `T-vector-geom` | Explicitly required by roadmap; mocked OSM fixtures only. |
| `load_context_vectors` | later/P1 | `domain.rs`; `forge3d.gis.load_context_vectors(path_or_features, layers=None)` | Local vector source/features and layer selection | Layer info, feature counts, CRS, schema, bounds | `missing_layer`, `UnsupportedGeometry`, `CrsMismatch` | `T-vector-io`, `T-vector-crs` | Explicitly required by urban/context examples. |
| `prepare_osm_scene` | later/P1 | `domain.rs`; `forge3d.gis.prepare_osm_scene(aoi, tags=None, cache=None)` | AOI, tags, cache/network options | Roads/water/building/context layers, CRS, diagnostics | Empty layers, invalid geometries, service/cache failure, height fallback diagnostics | `T-osm`, `T-domain` | 1324 hits; 64 scripts. |
| `prepare_dem` | later/P1 | `domain.rs`; `forge3d.gis.prepare_dem(source, target_info=None, nodata=None)` | DEM path/array/info, optional target grid | Heightfield array, `RasterInfo`, valid mask, scale metadata | `NotFound`, `UnsupportedDType`, `MissingCrs`, `GridMismatch`, `empty_raster` | `T-domain`, `T-raster-meta`, `T-align` | 2856 hits; 101 scripts. |
| `prepare_terrain_derivatives` | later/P2 | `domain.rs`; `forge3d.gis.prepare_terrain_derivatives(dem, derivatives=("slope","hillshade"))` | DEM array/info and derivative list | Derivative arrays, units, transform, CRS | Missing resolution, geographic-units warning, `UnsupportedOption`, `empty_raster` | `T-domain`, `T-affine` | 1498 hits; 86 scripts. |
| `read_gridded_dataset` | later/P2 | `domain.rs`; `forge3d.gis.read_gridded_dataset(path, variable=None)` | NetCDF/OPeNDAP/raster-like path and variable | Variables, dimensions, CRS if available, bounds, nodata | Ambiguous variable, unsupported layout, `missing_crs`, `metadata_unavailable` | `T-domain` | 101 hits; 7 scripts; xarray/rioxarray examples. |
| `subset_grid` | later/P2 | `domain.rs`; `forge3d.gis.subset_grid(source, bounds_or_coords, variable=None)` | Grid source, spatial/coordinate subset, optional variable | Subset array, coords, CRS/time/variable metadata | Ambiguous axes, unsupported interpolation, missing CRS for spatial subset | `T-domain` | 5914 broad grid/sample hits; 111 scripts. |
| `decode_terrarium_dem` | later/P2 | `terrarium.rs`; `forge3d.gis.decode_terrarium_dem(rgb_array_or_path)` | RGB tile/path | Elevation array in meters, nodata policy, min/max | `UnsupportedDType`, `InvalidShape`, Terrarium encoding warning | `T-domain` | 221 hits; 24 scripts. |
| `build_terrarium_dem` | later/P2 | `terrarium.rs`; `forge3d.gis.build_terrarium_dem(bounds, zoom, cache=None)` | Bounds, zoom, cache/network options | DEM mosaic array, `RasterInfo`, tile manifest | `Network`, `cache_miss`, missing tile, partial mosaic warning | `T-domain`, `T-remote` | Required by Terrarium tile fetch/mosaic examples. |
| `prepare_landcover_raster` | later/P1 | `domain.rs`; `forge3d.gis.prepare_landcover_raster(source, target_info, classes=None)` | Landcover raster and target grid/class table | Class raster, class table, `RasterInfo`, nodata | `GridMismatch`, `categorical_resampling_warning`, unknown class diagnostic | `T-domain`, `T-thematic`, `T-align` | 259 hits; 15 scripts. |
| `prepare_population_raster` | later/P1 | `domain.rs`; `forge3d.gis.prepare_population_raster(source, target_info=None, normalization=None)` | Population raster and optional grid/normalization | Prepared array, transform/CRS, nodata, normalization metadata | Empty valid pixels, negative population, `GridMismatch`, `missing_crs` | `T-domain`, `T-thematic` | 1543 hits; 43 scripts. |
| `load_building_footprints` | later/P1 | `domain.rs`; `forge3d.gis.load_building_footprints(path_or_features, dst_crs=None)` | Building vector/GeoJSON/CityJSON/GPKG source | Footprint features, CRS, bounds, height attribute schema | `UnsupportedGeometry`, invalid rings, `missing_crs`, `empty_feature_set` | `T-domain`, `T-vector-io` | 586 hits; 32 scripts. |
| `extract_building_heights` | later/P1 | `domain.rs`; `forge3d.gis.extract_building_heights(features, defaults=None)` | Building features with `height`, `building:levels`, roof/storey attributes | Per-feature heights, units, source attribute, fallback count | Invalid unit, negative/zero height, missing attribute fallback diagnostic | `T-domain` | 3582 broad height/building hits; 114 scripts. |
| `estimate_local_utm` | later/P2 | `crs.rs`; `forge3d.gis.estimate_local_utm(bounds_or_geometry)` | Bounds or geometry with CRS | Estimated UTM CRS and confidence metadata | Missing CRS, invalid bounds, polar/zone warning | `T-crs`, `T-domain` | 18 hits; 8 scripts; GeoPandas `estimate_utm_crs`. |

## Deferred Contract

| API | Phase/Priority | Rust module / Python wrapper | Required inputs | Required output attributes and metadata | Diagnostics/errors | Tests | Evidence and reference behavior |
|---|---|---|---|---|---|---|---|
| `defer_routing` | defer/Defer | none under `src/gis/`; possible future `forge3d.routing` | Network nodes/edges/cost model if later accepted | Routing graph, travel-time surfaces, reachable areas | Separate graph data model and cost diagnostics required | No `G-002` tests; future routing fixture required if accepted | 4 hits; 2 scripts; Barcelona travel-time examples use graph/network analysis outside GIS prep. |

## Top 20 Highest-Priority Attributes And Functions By Evidence

| Rank | API or attribute | Phase | Priority | Evidence |
|---:|---|---|---|---|
| 1 | `write_raster` path/overwrite/profile creation options | G-002a1 | P0 | 1029 rollup hits; 102 scripts. |
| 2 | `mask_raster` clip/crop/fill/nodata behavior | G-002c | P0 | 5384 rollup hits; 95 scripts. |
| 3 | `rasterize_vectors` fill/dtype/target grid/burn values | G-002c | P0 | 3706 rollup hits; 81 scripts. |
| 4 | `read_raster_info` width/height/shape/profile/dtype | G-002a1 | P0 | 2033 rollup hits; 86 scripts. |
| 5 | `apply_nodata` and mask semantics | G-002b | P0 | 1012 rollup hits; 78 scripts. |
| 6 | `parse_crs` EPSG/WGS84 parsing | G-002b | P0 | 908 rollup hits; 75 scripts. |
| 7 | `clip_vector` | G-002c | P0 | 2044 rollup hits; 66 scripts. |
| 8 | `intersect_vectors` | G-002c | P0 | 260 rollup hits; 70 scripts. |
| 9 | `resample_raster` | G-002b | P0 | 557 rollup hits; 59 scripts. |
| 10 | `assert_grid_compatible` | G-002b | P0 | 365 rollup hits; 57 scripts. |
| 11 | `read_raster` | G-002a1 | P0 | 111 read hits; 47 scripts. |
| 12 | `raster_transform` | G-002b | P0 | 169 hits; 45 scripts. |
| 13 | `create_crs_transformer(always_xy=True)` | G-002b | P0 | 232 rollup hits; 34 scripts. |
| 14 | `nodata_per_band` | G-002b | P0 | 145 hits; 34 scripts. |
| 15 | `vector_bounds` | G-002c | P0 | 101 rollup hits; 37 scripts. |
| 16 | `raster_bounds` | G-002b | P0 | 75 hits; 37 scripts. |
| 17 | `raster_crs` and `vector_crs` | G-002b/G-002c | P0 | 130 hits each; 32 scripts each. |
| 18 | `web_mercator_bounds` | G-002b | P0 | 47 hits; 26 scripts. |
| 19 | `reproject_raster` | G-002b | P0 | 57 hits; 24 scripts. |
| 20 | `read_vector` and `reproject_vector` | G-002c | P0 | 25 read hits and 39 reprojection hits; 23 scripts each. |

## Coverage Proof

Coverage is complete under the stated standard: every explicitly observed GIS operation, attribute, error condition, and testable assumption in the example scripts is mapped to an existing/current API, a future planned API contract, a deferred/later helper, or a coverage exception.

Proof sources:

- Every inspected example script is listed in `gis-contract-evidence.md`.
- Every detected GIS pattern class is listed in `gis-contract-evidence.md`.
- Every pattern maps to a proposed API, phase, priority, occurrence count, script count, and representative evidence.
- The contract matrix above maps each proposed API to inputs, output attributes, metadata fields, diagnostics/errors, tests, source evidence, reference behavior, implementation notes, and exceptions where relevant.
- Coverage exceptions are limited to map-plate notebook cartographic composition, duplicate canonical-path choice, broad domain text classes needing implementation-time option review, deferred routing, and live network service behavior.

## Implementation Phases

### G-002a1: Raster Read/Write Foundation

- [ ] Add `src/gis/` module skeleton with `error.rs`, `types.rs`, `raster_info.rs`, and `raster_write.rs`.
- [ ] Add Rust `RasterInfo`, `AffineTransform`, and error/warning types.
- [ ] Implement `read_raster_info`.
- [ ] Implement `read_raster` only after metadata tests pass.
- [ ] Implement `write_raster`.
- [ ] Add thin PyO3 wrappers and `.pyi` stubs.
- [ ] Add fixture tests from `T-raster-meta`, `T-raster-read`, and `T-raster-write`.

### G-002b: Raster CRS, Transform, Alignment, And Reprojection

- [ ] Add `crs.rs` and `affine.rs` using existing `src/geo/reproject.rs` where possible.
- [ ] Add CRS parsing/inspection/assignment and transformer contracts.
- [ ] Add transform/bounds/resolution/window helpers.
- [ ] Add nodata/mask helpers.
- [ ] Add alignment diagnostics for shape/CRS/transform/nodata mismatch.
- [ ] Add raster reprojection only with explicit resampling.
- [ ] Add fixture tests from `T-crs`, `T-affine`, `T-window`, `T-reproject`, and `T-align`.

### G-002c: Vector, Mask, Rasterization, Classification

- [ ] Add `vector.rs`, `rasterize.rs`, and `thematic.rs`.
- [ ] Add vector metadata/read before full feature mutation helpers.
- [ ] Add vector reprojection separate from raster reprojection.
- [ ] Add geometry validation/repair before union/clip/intersection helpers.
- [ ] Add masks/rasterization against explicit `RasterInfo` target grids.
- [ ] Add raster normalization/classification with nodata-aware tests.
- [ ] Add fixture tests from `T-vector-io`, `T-vector-crs`, `T-vector-geom`, `T-rasterize-mask`, and `T-thematic`.

### Later: Domain Helpers And Remote Data

- [ ] Build remote/cache contracts only after local raster/vector metadata contracts are stable.
- [ ] Build OSM/Terrarium/slippy-tile helpers only on top of explicit remote/cache and CRS contracts.
- [ ] Build DEM/landcover/population/building/gridded helpers as small wrappers over core raster/vector primitives.
- [ ] Add fixture tests from `T-remote`, `T-osm`, and `T-domain`.

### Defer

- `network_travel_time_analysis` stays outside `G-002`. It needs a graph/routing data model, cost semantics, and fixtures before a public API is planned.

## Reference Library Comparison

| Concern | Rasterio | pyproj | GeoPandas/Shapely | xarray/rioxarray | R terra | forge3d decision |
|---|---|---|---|---|---|---|
| Raster metadata | `dataset.profile`, attrs, `bounds`, `res`, `transform`, `crs`, `nodata` | CRS parsing only | Not raster data | `rio` metadata where present | `rast`, `ext`, `res`, `crs`, `NAflag` | Normalize into `RasterInfo`; missing metadata becomes stable diagnostics. |
| Raster write | Writer profile and creation options | CRS validation | Not raster write | `rio.to_raster` reference only | `writeRaster` | Explicit fields or `like_*`, post-write `RasterInfo` validation. |
| CRS | `dataset.crs` | `CRS`, `Transformer.from_crs(always_xy=True)` | `.crs`, `to_crs`, Shapely geometries carry no CRS | `rio.write_crs`, `rio.reproject` | `crs`, `project` | Missing CRS is never guessed; assignment and reprojection are separate. |
| Raster reprojection | `warp.reproject`, `calculate_default_transform`, `Resampling` | Supplies transforms | Not raster | `rio.reproject_match` | `project` | Resampling method is required; categorical warnings are explicit. |
| Vector operations | Rasterize/mask interop | Coordinate transforms | `read_file`, `to_crs`, `clip`, `overlay`, `union_all`, `buffer`, `simplify`, `make_valid` | Not primary vector backend | `vect`, `project`, `intersect`, `union`, `buffer` | Use Rust geometry/GDAL/PROJ; return metadata and stable diagnostics. |
| Remote data | GDAL URL support where configured | Not a fetcher | `read_file(url)` possible | OPeNDAP/netCDF possible | URLs where GDAL supports them | No hidden downloads; explicit cache/fetch contracts and mocked tests. |
