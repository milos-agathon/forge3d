# G-002b GIS Raster Support Matrix

G-002b is Rust-first and local TIFF/GeoTIFF-backed. Python wrappers in
`forge3d.gis` marshal values into PyO3 only, including `os.PathLike` to
`os.fspath`; backend GIS behavior and CRS/path disambiguation live under
`src/gis/`.

## Implemented APIs

| Area | Public API | Status | Notes |
|---|---|---|---|
| Raster read/write foundation | `read_raster_info`, `read_raster`, `write_raster` | Implemented | `read_raster` is the public G-002a1 Rust-backed read API: `read_raster(path, bands=None, window=None, masked=False)`. Bands are 1-based, output arrays are always band-first `(bands, height, width)`, pixel windows are `(col_off, row_off, width, height)` and in-bounds only, result `info` is a serialized `RasterInfo` dict, nodata is per selected band, and masks are explicit `true_valid` when requested. |
| CRS parsing and inspection | `parse_crs`, `inspect_crs`, `raster_crs` | Implemented subset | EPSG integers/strings, WKT-shaped strings, and `{"name": "EPSG", "code": "..."}` dictionaries are supported. Existing paths inspect raster metadata; missing dataset-looking paths raise `NotFound`; plain invalid CRS strings raise `InvalidCrs`. Missing CRS is reported, never guessed. |
| CRS assignment | `assign_crs` | Implemented | Path input mutates local TIFF CRS metadata only. `RasterInfo` input returns updated metadata without mutating the file. Neither path changes pixel values, dimensions, dtype, transform, bounds, resolution, or nodata. Existing CRS requires `overwrite=True`. Diagnostic: `assignment_not_reprojection`. |
| CRS transforms | `create_crs_transformer`, `CrsTransform.from_crs`, `transform_bounds`, `web_mercator_bounds` | Partial | `create_crs_transformer` and `CrsTransform.from_crs` accept supported CRS strings, EPSG integers, and `{"name": "EPSG", "code": "..."}` dictionaries. Built-in coordinate transforms cover EPSG:4326 <-> EPSG:3857 and same-CRS passthrough. Other valid CRS pairs fail with stable `BackendUnavailable` unless a future PROJ-backed path is enabled. Axis order is explicit `always_xy`. `transform_bounds(densify=None or 0)` is supported; nonzero densify fails with `unsupported_option` rather than being ignored. |
| Affine transforms | `AffineTransform`, `raster_transform`, `transform_from_origin`, `transform_from_bounds`, `array_bounds`, `raster_bounds`, `raster_resolution`, `validate_transform`, `pixel_convention`, `rowcol`, `xy`, `index` | Implemented | `AffineTransform` exposes `coefficients`, `resolution`, and `rotated_or_sheared`. Affine order is `(a, b, c, d, e, f)`. Bounds order is `(left, bottom, right, top)`. `xy` uses pixel centers by default; affine coefficients map pixel corners. Non-finite coefficients and zero pixel size are rejected. Rotation/shear is diagnosed. |
| Nodata and masks | `RasterInfo.nodata_per_band`, `apply_nodata`, `read_raster_mask` | Implemented | Per-band nodata is authoritative. Nodata application supports scalar/per-band nodata, explicit true-valid masks, NaN invalidation for floating arrays, and `empty_raster` diagnostics. Raster masks return true-valid polarity and mask flags. |
| Resampling | `resample_raster` | Implemented subset | Supports `nearest` and `bilinear`; method is required. Accepted targets are `{"shape": (height, width)}`, `{"resolution": scalar_or_pair}`, integer tuple `(height, width)` as shape, and float scalar/tuple as resolution when source transform exists. Dicts containing both `shape` and `resolution` fail with `InvalidArgument`. Array sources support shape targets only and reject resolution targets with `missing_transform`. Local path/`RasterInfo` sources support shape and resolution targets when an affine transform is present. Result `info` is a serialized `RasterInfo` dict. Nodata is excluded from nearest/bilinear sampling and used as the fill value where available. |
| Alignment | `assert_grid_compatible`, `align_raster_grid` | Implemented subset | Target grid is explicit. Compatibility reports shape, CRS, transform, resolution, and nodata differences. Alignment does not reproject; CRS mismatch fails. Marked categorical sources reject non-nearest resampling with a stable categorical diagnostic. Alignment result `info` is a serialized `RasterInfo` dict. |
| Reprojection | `reproject_raster`, `calculate_default_transform` | Partial | Source CRS and explicit resampling are required. Reprojection result `info` is a serialized `RasterInfo` dict with transform, shape, bounds, CRS, and nodata. `calculate_default_transform` returns transform/grid/CRS metadata directly and does not include an `info` field. Built-in reprojection is limited to EPSG:4326 <-> EPSG:3857; arbitrary GDAL/PROJ-grade warping is not claimed. |
| Windowing | `window_from_bounds`, `read_raster_window`, `window_transform` | Implemented subset | Bounds-to-window conversion supports clipping and `boundless=True`. Window reads return array, window transform, and `info` as a serialized `RasterInfo` dict. With `masked=True`, `mask` is a boolean true-valid array and `mask_polarity` is `"true_valid"`; with `masked=False`, both are `None`. |

## Compatibility Aliases

| Alias | Canonical API | Status |
|---|---|---|
| `bounds` | `raster_bounds` | Retained; tests assert equivalent behavior. |
| `align_raster_to` | `align_raster_grid` | Retained; tests assert equivalent behavior. |

## Backend Features And Limits

- Required default backend: Rust `tiff`, existing raster metadata/write code, `ndarray`/NumPy at the Python boundary.
- Optional external backend: the existing Cargo `proj` feature pulls `proj-sys`; it is not required for default G-002b validation.
- GDAL is not required or used by the default G-002b implementation.
- Broad CRS database lookup, arbitrary CRS transforms, cubic/lanczos resampling, and GDAL/PROJ-grade warping are deferred until those external backend paths are explicitly wired and tested.
- The G-002a1 local raster foundation is sealed for public read/write: `read_raster_info`, `read_raster`, `write_raster`, `read_raster_mask`, `read_raster_window`, and G-002b raster operations share Rust TIFF read internals.

## Out Of Scope

G-002b does not implement or claim vector read/reprojection, vector union,
buffering, clipping/intersection, geometry masking, raster clip/mask, vector
rasterization, thematic normalization/classification, OSM, Terrarium, remote
fetch/cache, slippy tiles, domain helpers, recipe manifests, gallery goldens,
or MapScene recipe-family work.
