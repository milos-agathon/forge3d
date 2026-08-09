//! P3.1-P3.2: COG HeightReader implementation.

use super::cache::CogTileCache;
use super::error::CogError;
use super::ifd_parser::{
    parse_cog_header, CogHeader, COMPRESSION_DEFLATE, COMPRESSION_DEFLATE_ALT, COMPRESSION_F3DZ,
    COMPRESSION_LZW, COMPRESSION_NONE, SAMPLE_FORMAT_FLOAT, SAMPLE_FORMAT_INT, SAMPLE_FORMAT_UINT,
    TIFF_PREDICTOR_HORIZONTAL, TIFF_PREDICTOR_NONE,
};
use super::range_reader::RangeReader;
use crate::terrain::page_table::{HeightRead, HeightReader};
use crate::terrain::tiling::TileBounds;
use glam::Vec2;
use std::path::PathBuf;
use std::sync::Arc;

/// COG-based height reader implementing the HeightReader trait.
pub struct CogHeightReader {
    reader: Arc<RangeReader>,
    header: CogHeader,
    cache: Arc<CogTileCache>,
    runtime: Arc<tokio::runtime::Runtime>,
    planetary_bounds: (f64, f64, f64, f64),
    source_crs: crate::gis::raster_write::CrsSpec,
    wgs84_crs: crate::gis::raster_write::CrsSpec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeightTileRequest {
    pub tile_id: crate::terrain::tiling::TileId,
    pub output_width: u32,
    pub output_height: u32,
}

fn local_file_url_path(url: &str) -> String {
    let raw = url.strip_prefix("file://").unwrap_or(url);
    #[cfg(windows)]
    let raw = if raw.as_bytes().get(2) == Some(&b':') && raw.starts_with('/') {
        &raw[1..]
    } else {
        raw
    };
    raw.to_string()
}

#[cfg(test)]
mod local_path_tests {
    use super::local_file_url_path;

    #[test]
    fn standard_windows_file_uri_becomes_drive_path() {
        #[cfg(windows)]
        assert_eq!(local_file_url_path("file:///D:/data/terrain.tif"), "D:/data/terrain.tif");
    }

}

fn validate_planetary_geography(
    header: &CogHeader,
) -> Result<
    (
        (f64, f64, f64, f64),
        crate::gis::raster_write::CrsSpec,
        crate::gis::raster_write::CrsSpec,
    ),
    CogError,
> {
    let ifd = header
        .full_resolution()
        .ok_or_else(|| CogError::InvalidIfd("COG contains no full-resolution IFD".to_string()))?;
    let georef = header.geo_reference()?;
    let source = crate::gis::raster_write::CrsSpec::from_string(format!("EPSG:{}", georef.epsg))
        .map_err(|error| CogError::InvalidIfd(format!("unsupported COG CRS: {error}")))?;
    let wgs84 = crate::gis::raster_write::CrsSpec::from_string("EPSG:4326".to_string())
        .map_err(|error| CogError::InvalidIfd(error.to_string()))?;
    crate::gis::crs::transform_pair_supported(&source, &wgs84)
        .map_err(|error| CogError::InvalidIfd(format!("unsupported planetary COG CRS: {error}")))?;
    let mut min_lon = f64::INFINITY;
    let mut min_lat = f64::INFINITY;
    let mut max_lon = f64::NEG_INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    // Projected image edges can curve in WGS84 and rotated affine datasets do
    // not have axis-aligned model-space corners. Densify every image edge so
    // the advertised planetary bounds retain interior extrema.
    for step in 0..=32 {
        let t = f64::from(step) / 32.0;
        for (pixel_x, pixel_y) in [
            (t * f64::from(ifd.width), 0.0),
            (t * f64::from(ifd.width), f64::from(ifd.height)),
            (0.0, t * f64::from(ifd.height)),
            (f64::from(ifd.width), t * f64::from(ifd.height)),
        ] {
            let (model_x, model_y) = georef.pixel_to_model(pixel_x, pixel_y)?;
            let (lon, lat) =
                crate::gis::crs::transform_point(model_x, model_y, &source, &wgs84).map_err(
                    |error| {
                        CogError::InvalidIfd(format!(
                            "COG georeference cannot map to WGS84: {error}"
                        ))
                    },
                )?;
            if !lon.is_finite() || !lat.is_finite() || !(-90.0..=90.0).contains(&lat) {
                return Err(CogError::InvalidIfd(
                    "COG georeference produced invalid planetary longitude/latitude".to_string(),
                ));
            }
            min_lon = min_lon.min(lon);
            min_lat = min_lat.min(lat);
            max_lon = max_lon.max(lon);
            max_lat = max_lat.max(lat);
        }
    }
    Ok(((min_lon, min_lat, max_lon, max_lat), source, wgs84))
}

impl CogHeightReader {
    fn full_resolution_pixel_bounds(
        &self,
        lonlat_bounds: (f64, f64, f64, f64),
    ) -> Result<Option<(f64, f64, f64, f64)>, CogError> {
        let (lon_min, lat_min, lon_max, lat_max) = lonlat_bounds;
        let georef = self.header.geo_reference()?;
        let mut min_px = f64::INFINITY;
        let mut min_py = f64::INFINITY;
        let mut max_px = f64::NEG_INFINITY;
        let mut max_py = f64::NEG_INFINITY;
        // Densify all edges so projected CRSs do not use corner-only extrema.
        for step in 0..=32 {
            let t = step as f64 / 32.0;
            for (lon, lat) in [
                (lon_min + (lon_max - lon_min) * t, lat_min),
                (lon_min + (lon_max - lon_min) * t, lat_max),
                (lon_min, lat_min + (lat_max - lat_min) * t),
                (lon_max, lat_min + (lat_max - lat_min) * t),
            ] {
                let (model_x, model_y) = crate::gis::crs::transform_point(
                    lon,
                    lat,
                    &self.wgs84_crs,
                    &self.source_crs,
                )
                .map_err(|error| {
                    CogError::InvalidIfd(format!(
                        "global height tile cannot transform into source CRS: {error}"
                    ))
                })?;
                let (pixel_x, pixel_y) = georef.model_to_pixel(model_x, model_y)?;
                min_px = min_px.min(pixel_x);
                min_py = min_py.min(pixel_y);
                max_px = max_px.max(pixel_x);
                max_py = max_py.max(pixel_y);
            }
        }
        let full = self
            .header
            .full_resolution()
            .ok_or_else(|| CogError::InvalidIfd("COG contains no image levels".to_string()))?;
        let clipped = (
            min_px.max(0.0),
            min_py.max(0.0),
            max_px.min(f64::from(full.width)),
            max_py.min(f64::from(full.height)),
        );
        if clipped.0 >= clipped.2 || clipped.1 >= clipped.3 {
            Ok(None)
        } else {
            Ok(Some(clipped))
        }
    }

    /// Create a new COG height reader from a URL.
    pub async fn new(url: &str, cache_size_mb: u32) -> Result<Self, CogError> {
        Self::new_with_cache_options(url, cache_size_mb, None, cache_size_mb).await
    }

    /// Create a new COG height reader with explicit range-cache options.
    pub async fn new_with_cache_options(
        url: &str,
        cache_size_mb: u32,
        cache_dir: Option<PathBuf>,
        range_cache_budget_mb: u32,
    ) -> Result<Self, CogError> {
        let range_cache_budget_bytes = range_cache_budget_mb as u64 * 1024 * 1024;
        let reader = if url.starts_with("file://") {
            let path = local_file_url_path(url);
            RangeReader::new_local_with_cache_options(
                &path,
                range_cache_budget_bytes,
                cache_dir,
                range_cache_budget_bytes,
            )?
        } else {
            RangeReader::new_with_cache_options(
                url,
                range_cache_budget_bytes,
                cache_dir,
                range_cache_budget_bytes,
            )
            .await?
        };

        let reader = Arc::new(reader);
        let header = parse_cog_header(&reader).await?;
        let (planetary_bounds, source_crs, wgs84_crs) = validate_planetary_geography(&header)?;
        let cache = Arc::new(CogTileCache::new(cache_size_mb));

        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_name("forge3d-cog-reader-io")
                .enable_all()
                .build()
                .map_err(|error| CogError::HttpError(error.to_string()))?,
        );

        Ok(Self {
            reader,
            header,
            cache,
            runtime,
            planetary_bounds,
            source_crs,
            wgs84_crs,
        })
    }

    /// Create with an existing tokio runtime handle.
    pub async fn new_with_runtime(
        url: &str,
        cache_size_mb: u32,
        runtime: Arc<tokio::runtime::Runtime>,
    ) -> Result<Self, CogError> {
        Self::new_with_runtime_and_cache_options(url, cache_size_mb, runtime, None, cache_size_mb)
            .await
    }

    /// Create with an existing tokio runtime handle and cache options.
    pub async fn new_with_runtime_and_cache_options(
        url: &str,
        cache_size_mb: u32,
        runtime: Arc<tokio::runtime::Runtime>,
        cache_dir: Option<PathBuf>,
        range_cache_budget_mb: u32,
    ) -> Result<Self, CogError> {
        let range_cache_budget_bytes = range_cache_budget_mb as u64 * 1024 * 1024;
        let reader = if url.starts_with("file://") {
            let path = local_file_url_path(url);
            RangeReader::new_local_with_cache_options(
                &path,
                range_cache_budget_bytes,
                cache_dir,
                range_cache_budget_bytes,
            )?
        } else {
            RangeReader::new_with_cache_options(
                url,
                range_cache_budget_bytes,
                cache_dir,
                range_cache_budget_bytes,
            )
            .await?
        };

        let reader = Arc::new(reader);
        let header = parse_cog_header(&reader).await?;
        let (planetary_bounds, source_crs, wgs84_crs) = validate_planetary_geography(&header)?;
        let cache = Arc::new(CogTileCache::new(cache_size_mb));

        Ok(Self {
            reader,
            header,
            cache,
            runtime,
            planetary_bounds,
            source_crs,
            wgs84_crs,
        })
    }

    /// Get geographic bounds (from first IFD).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        self.planetary_bounds
    }

    /// Get number of overview levels.
    pub fn overview_count(&self) -> usize {
        self.header.ifds.len()
    }

    /// Get the COG header for inspection.
    pub fn header(&self) -> &CogHeader {
        &self.header
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> super::cache::CogCacheStats {
        let mut stats = self.cache.stats();
        stats.byte_cache_used_bytes = self.reader.stats().cached_bytes();
        stats.byte_cache_budget_bytes = self.reader.byte_cache_budget_bytes();
        stats.disk_cache_used_bytes = self.reader.stats().disk_cached_bytes();
        stats.disk_cache_budget_bytes = self.reader.disk_cache_budget_bytes();
        stats
    }

    /// Read a specific tile at given LOD.
    pub fn read_tile(&self, tile_x: u32, tile_y: u32, lod: u32) -> Result<Vec<f32>, CogError> {
        let ifd = self.header.select_ifd_for_lod(lod)?;

        let cache_key = (ifd.overview_level, tile_x, tile_y);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        let tile_idx = ifd
            .tile_index(tile_x, tile_y)
            .ok_or(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            })?;

        if tile_idx >= ifd.tile_offsets.len() || tile_idx >= ifd.tile_byte_counts.len() {
            return Err(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            });
        }

        let offset = ifd.tile_offsets[tile_idx];
        let byte_count = ifd.tile_byte_counts[tile_idx];

        let reader = self.reader.clone();
        let compression = ifd.compression;
        let bits_per_sample = ifd.bits_per_sample;
        let sample_format = ifd.sample_format;
        let predictor = ifd.predictor;
        let tile_width = ifd.tile_width;
        let tile_height = ifd.tile_height;

        let heights = self.runtime.block_on(async move {
            let compressed = reader.read_range(offset, byte_count).await?;
            if compression == COMPRESSION_F3DZ
                && (bits_per_sample != 32 || sample_format != SAMPLE_FORMAT_FLOAT)
            {
                return Err(CogError::InvalidIfd(
                    "F3DZ TIFF tiles require 32-bit floating-point sample metadata".into(),
                ));
            }
            let decompressed =
                decompress_tile(&compressed, compression, Some((tile_width, tile_height)))?;
            decode_heights(
                &decompressed,
                bits_per_sample,
                sample_format,
                tile_width,
                tile_height,
                if compression == COMPRESSION_F3DZ {
                    TIFF_PREDICTOR_NONE
                } else {
                    predictor
                },
            )
        })?;

        let tile_size = (tile_width as usize)
            .checked_mul(tile_height as usize)
            .ok_or_else(|| CogError::InvalidIfd("tile element count overflow".into()))?;
        let memory_bytes = tile_size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CogError::InvalidIfd("tile byte size overflow".into()))?;
        self.cache.insert(cache_key, heights.clone(), memory_bytes);

        Ok(heights)
    }

    /// Read tile async.
    pub async fn read_tile_async(
        &self,
        tile_x: u32,
        tile_y: u32,
        lod: u32,
    ) -> Result<Vec<f32>, CogError> {
        let ifd = self.header.select_ifd_for_lod(lod)?;

        let cache_key = (ifd.overview_level, tile_x, tile_y);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        let tile_idx = ifd
            .tile_index(tile_x, tile_y)
            .ok_or(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            })?;

        if tile_idx >= ifd.tile_offsets.len() || tile_idx >= ifd.tile_byte_counts.len() {
            return Err(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            });
        }

        let offset = ifd.tile_offsets[tile_idx];
        let byte_count = ifd.tile_byte_counts[tile_idx];

        let compressed = self.reader.read_range(offset, byte_count).await?;
        if ifd.compression == COMPRESSION_F3DZ
            && (ifd.bits_per_sample != 32 || ifd.sample_format != SAMPLE_FORMAT_FLOAT)
        {
            return Err(CogError::InvalidIfd(
                "F3DZ TIFF tiles require 32-bit floating-point sample metadata".into(),
            ));
        }
        let decompressed = decompress_tile(
            &compressed,
            ifd.compression,
            Some((ifd.tile_width, ifd.tile_height)),
        )?;
        let heights = decode_heights(
            &decompressed,
            ifd.bits_per_sample,
            ifd.sample_format,
            ifd.tile_width,
            ifd.tile_height,
            if ifd.compression == COMPRESSION_F3DZ {
                TIFF_PREDICTOR_NONE
            } else {
                ifd.predictor
            },
        )?;

        let tile_size = (ifd.tile_width as usize)
            .checked_mul(ifd.tile_height as usize)
            .ok_or_else(|| CogError::InvalidIfd("tile element count overflow".into()))?;
        let memory_bytes = tile_size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CogError::InvalidIfd("tile byte size overflow".into()))?;
        self.cache.insert(cache_key, heights.clone(), memory_bytes);

        Ok(heights)
    }

    /// Read one ORBIS quadtree footprint. Quadtree LOD is a normalized
    /// geographic subdivision of the dataset, never a TIFF overview index.
    pub fn read_height_tile(&self, request: HeightTileRequest) -> Result<Vec<f32>, CogError> {
        self.runtime.block_on(self.read_height_tile_async(request))
    }

    pub async fn read_height_tile_async(
        &self,
        request: HeightTileRequest,
    ) -> Result<Vec<f32>, CogError> {
        Ok(self.read_height_tile_covered_async(request).await?.heights)
    }

    pub async fn read_height_tile_covered_async(
        &self,
        request: HeightTileRequest,
    ) -> Result<HeightRead, CogError> {
        if request.output_width == 0 || request.output_height == 0 {
            return Err(CogError::InvalidIfd(
                "height tile output dimensions must be non-zero".to_string(),
            ));
        }
        let tile_bounds = crate::terrain::planetary_tiles::global_tile_lonlat_bounds(
            request.tile_id,
        )
        .map_err(CogError::InvalidIfd)?;
        let covered_bounds = (
            tile_bounds.0.max(self.planetary_bounds.0),
            tile_bounds.1.max(self.planetary_bounds.1),
            tile_bounds.2.min(self.planetary_bounds.2),
            tile_bounds.3.min(self.planetary_bounds.3),
        );
        let output_len = (request.output_width as usize)
            .checked_mul(request.output_height as usize)
            .ok_or_else(|| CogError::InvalidIfd("height tile output size overflow".to_string()))?;
        if covered_bounds.0 >= covered_bounds.2 || covered_bounds.1 >= covered_bounds.3 {
            // Outside the dataset's geographic footprint is deliberate no-data, not
            // a fabricated substitute for a failed source read.
            return Ok(HeightRead {
                heights: vec![0.0; output_len],
                coverage: vec![0; output_len],
            });
        }
        let Some((full_x0, full_y0, full_x1, full_y1)) =
            self.full_resolution_pixel_bounds(covered_bounds)?
        else {
            return Ok(HeightRead {
                heights: vec![0.0; output_len],
                coverage: vec![0; output_len],
            });
        };
        let full = self
            .header
            .full_resolution()
            .ok_or_else(|| CogError::InvalidIfd("COG contains no image levels".to_string()))?;
        let full_span_x = full_x1 - full_x0;
        let full_span_y = full_y1 - full_y0;
        let covered_output_width = ((covered_bounds.2 - covered_bounds.0)
            / (tile_bounds.2 - tile_bounds.0)
            * f64::from(request.output_width))
        .ceil()
        .max(1.0);
        let covered_output_height = ((covered_bounds.3 - covered_bounds.1)
            / (tile_bounds.3 - tile_bounds.1)
            * f64::from(request.output_height))
        .ceil()
        .max(1.0);
        let ifd = self
            .header
            .ifds
            .iter()
            .filter(|ifd| {
                full_span_x * f64::from(ifd.width) / f64::from(full.width)
                    >= covered_output_width
                    && full_span_y * f64::from(ifd.height) / f64::from(full.height)
                        >= covered_output_height
            })
            .min_by_key(|ifd| u64::from(ifd.width) * u64::from(ifd.height))
            .or_else(|| self.header.ifds.iter().max_by_key(|ifd| ifd.width))
            .ok_or_else(|| CogError::InvalidIfd("COG contains no image levels".to_string()))?
            .clone();

        let scale_x = f64::from(ifd.width) / f64::from(full.width);
        let scale_y = f64::from(ifd.height) / f64::from(full.height);
        let x0 = (full_x0 * scale_x).floor() as u32;
        let y0 = (full_y0 * scale_y).floor() as u32;
        let x1 = ((full_x1 * scale_x).ceil() as u32)
            .clamp(x0 + 1, ifd.width);
        let y1 = ((full_y1 * scale_y).ceil() as u32)
            .clamp(y0 + 1, ifd.height);
        let window_width = x1 - x0;
        let window_height = y1 - y0;
        let mut window = vec![0.0f32; (window_width * window_height) as usize];
        let tile_x0 = x0 / ifd.tile_width;
        let tile_y0 = y0 / ifd.tile_height;
        let tile_x1 = (x1 - 1) / ifd.tile_width;
        let tile_y1 = (y1 - 1) / ifd.tile_height;
        for physical_y in tile_y0..=tile_y1 {
            for physical_x in tile_x0..=tile_x1 {
                let physical = self
                    .read_physical_tile_async(ifd.overview_level, physical_x, physical_y)
                    .await?;
                let physical_origin_x = physical_x * ifd.tile_width;
                let physical_origin_y = physical_y * ifd.tile_height;
                let copy_x0 = x0.max(physical_origin_x);
                let copy_y0 = y0.max(physical_origin_y);
                let copy_x1 = x1.min((physical_origin_x + ifd.tile_width).min(ifd.width));
                let copy_y1 = y1.min((physical_origin_y + ifd.tile_height).min(ifd.height));
                for image_y in copy_y0..copy_y1 {
                    let source_y = image_y - physical_origin_y;
                    let destination_y = image_y - y0;
                    for image_x in copy_x0..copy_x1 {
                        let source_x = image_x - physical_origin_x;
                        let destination_x = image_x - x0;
                        window[(destination_y * window_width + destination_x) as usize] = physical
                            [(source_y * ifd.tile_width + source_x) as usize];
                    }
                }
            }
        }
        let georef = self.header.geo_reference()?;
        let mut output = vec![0.0f32; output_len];
        let mut coverage = vec![0u8; output_len];
        for output_y in 0..request.output_height {
            let y_fraction = if request.output_height > 1 {
                f64::from(output_y) / f64::from(request.output_height - 1)
            } else {
                0.5
            };
            let lat = tile_bounds.3 - y_fraction * (tile_bounds.3 - tile_bounds.1);
            for output_x in 0..request.output_width {
                let x_fraction = if request.output_width > 1 {
                    f64::from(output_x) / f64::from(request.output_width - 1)
                } else {
                    0.5
                };
                let lon = tile_bounds.0 + x_fraction * (tile_bounds.2 - tile_bounds.0);
                if lon < self.planetary_bounds.0
                    || lon > self.planetary_bounds.2
                    || lat < self.planetary_bounds.1
                    || lat > self.planetary_bounds.3
                {
                    continue;
                }
                let (model_x, model_y) = crate::gis::crs::transform_point(
                    lon,
                    lat,
                    &self.wgs84_crs,
                    &self.source_crs,
                )
                .map_err(|error| {
                    CogError::InvalidIfd(format!(
                        "global height sample cannot transform into source CRS: {error}"
                    ))
                })?;
                let (full_pixel_x, full_pixel_y) = georef.model_to_pixel(model_x, model_y)?;
                if !full_pixel_x.is_finite()
                    || !full_pixel_y.is_finite()
                    || full_pixel_x < -1.0e-6
                    || full_pixel_y < -1.0e-6
                    || full_pixel_x > f64::from(full.width) + 1.0e-6
                    || full_pixel_y > f64::from(full.height) + 1.0e-6
                {
                    continue;
                }
                let image_x = (full_pixel_x * scale_x)
                    .clamp(0.0, f64::from(ifd.width.saturating_sub(1)));
                let image_y = (full_pixel_y * scale_y)
                    .clamp(0.0, f64::from(ifd.height.saturating_sub(1)));
                let local_x = (image_x - f64::from(x0))
                    .clamp(0.0, f64::from(window_width.saturating_sub(1)));
                let local_y = (image_y - f64::from(y0))
                    .clamp(0.0, f64::from(window_height.saturating_sub(1)));
                let sx0 = local_x.floor() as u32;
                let sy0 = local_y.floor() as u32;
                let sx1 = (sx0 + 1).min(window_width - 1);
                let sy1 = (sy0 + 1).min(window_height - 1);
                let tx = (local_x - f64::from(sx0)) as f32;
                let ty = (local_y - f64::from(sy0)) as f32;
                let h00 = window[(sy0 * window_width + sx0) as usize];
                let h10 = window[(sy0 * window_width + sx1) as usize];
                let h01 = window[(sy1 * window_width + sx0) as usize];
                let h11 = window[(sy1 * window_width + sx1) as usize];
                output[(output_y * request.output_width + output_x) as usize] =
                    (h00 * (1.0 - tx) + h10 * tx) * (1.0 - ty)
                        + (h01 * (1.0 - tx) + h11 * tx) * ty;
                coverage[(output_y * request.output_width + output_x) as usize] = u8::MAX;
            }
        }
        Ok(HeightRead {
            heights: output,
            coverage,
        })
    }

    async fn read_physical_tile_async(
        &self,
        overview_level: u32,
        tile_x: u32,
        tile_y: u32,
    ) -> Result<Vec<f32>, CogError> {
        let ifd = self
            .header
            .overview(overview_level)
            .ok_or_else(|| CogError::InvalidIfd(format!("missing IFD {overview_level}")))?
            .clone();
        let cache_key = (overview_level, tile_x, tile_y);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }
        let index = ifd
            .tile_index(tile_x, tile_y)
            .ok_or(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod: overview_level,
            })?;
        let compressed = self
            .reader
            .read_range(ifd.tile_offsets[index], ifd.tile_byte_counts[index])
            .await?;
        let decompressed = decompress_tile(
            &compressed,
            ifd.compression,
            Some((ifd.tile_width, ifd.tile_height)),
        )?;
        let heights = decode_heights(
            &decompressed,
            ifd.bits_per_sample,
            ifd.sample_format,
            ifd.tile_width,
            ifd.tile_height,
            if ifd.compression == COMPRESSION_F3DZ {
                TIFF_PREDICTOR_NONE
            } else {
                ifd.predictor
            },
        )?;
        self.cache.insert(
            cache_key,
            heights.clone(),
            heights.len() * std::mem::size_of::<f32>(),
        );
        Ok(heights)
    }
}

impl HeightReader for CogHeightReader {
    fn read_result(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: crate::terrain::tiling::TileId,
        width: u32,
        height: u32,
    ) -> Result<Vec<f32>, String> {
        let heights = self
            .read_height_tile(HeightTileRequest {
            tile_id,
            output_width: width,
            output_height: height,
            })
            .map_err(|error| error.to_string())?;
        if heights.len() == (width * height) as usize {
            Ok(heights)
        } else {
            Ok(resample_tile(&heights, width, height))
        }
    }

    fn read_covered_result(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: crate::terrain::tiling::TileId,
        width: u32,
        height: u32,
    ) -> Result<HeightRead, String> {
        self.runtime
            .block_on(self.read_height_tile_covered_async(HeightTileRequest {
                tile_id,
                output_width: width,
                output_height: height,
            }))
            .map_err(|error| error.to_string())
    }
}

fn decompress_tile(
    data: &[u8],
    compression: u16,
    expected_dimensions: Option<(u32, u32)>,
) -> Result<Vec<u8>, CogError> {
    match compression {
        COMPRESSION_NONE => Ok(data.to_vec()),
        COMPRESSION_DEFLATE | COMPRESSION_DEFLATE_ALT => {
            use flate2::read::ZlibDecoder;
            use std::io::Read;

            let mut decoder = ZlibDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| CogError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
        COMPRESSION_LZW => decompress_lzw(data),
        COMPRESSION_F3DZ => {
            let decoded = crate::codec::f3dz::decode_dem(data, None)
                .map_err(|error| CogError::DecompressionError(error.to_string()))?;
            if let Some((width, height)) = expected_dimensions {
                if decoded.width != width || decoded.height != height {
                    return Err(CogError::DecompressionError(format!(
                        "F3DZ grid {}x{} does not match TIFF tile {}x{}",
                        decoded.width, decoded.height, width, height
                    )));
                }
            }
            let mut bytes = Vec::with_capacity(decoded.values.len() * 4);
            for value in decoded.values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(bytes)
        }
        other => Err(CogError::UnsupportedCompression(other)),
    }
}

fn decompress_lzw(data: &[u8]) -> Result<Vec<u8>, CogError> {
    const CLEAR_CODE: u16 = 256;
    const EOI_CODE: u16 = 257;

    let mut output = Vec::new();
    let mut table: Vec<Vec<u8>> = (0u16..256).map(|i| vec![i as u8]).collect();
    table.push(Vec::new()); // CLEAR_CODE placeholder
    table.push(Vec::new()); // EOI_CODE placeholder

    let mut bit_reader = LzwBitReader::new(data);
    let mut code_size = 9u8;
    let mut prev_code: Option<u16> = None;

    while let Some(code) = bit_reader.read_bits(code_size) {
        if code == EOI_CODE {
            break;
        }

        if code == CLEAR_CODE {
            table.truncate(258);
            code_size = 9;
            prev_code = None;
            continue;
        }

        let entry = if (code as usize) < table.len() {
            table[code as usize].clone()
        } else if code as usize == table.len() {
            if let Some(pc) = prev_code {
                let mut e = table[pc as usize].clone();
                e.push(e[0]);
                e
            } else {
                return Err(CogError::DecompressionError(
                    "LZW: invalid code sequence".into(),
                ));
            }
        } else {
            return Err(CogError::DecompressionError(format!(
                "LZW: code {} out of range (table size {})",
                code,
                table.len()
            )));
        };

        output.extend_from_slice(&entry);

        if let Some(pc) = prev_code {
            if table.len() < 4096 {
                let mut new_entry = table[pc as usize].clone();
                new_entry.push(entry[0]);
                table.push(new_entry);

                if table.len() == (1 << code_size) && code_size < 12 {
                    code_size += 1;
                }
            }
        }

        prev_code = Some(code);
    }

    Ok(output)
}

struct LzwBitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> LzwBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bits(&mut self, count: u8) -> Option<u16> {
        let mut result: u32 = 0;
        let mut bits_read = 0u8;

        while bits_read < count {
            if self.byte_pos >= self.data.len() {
                return None;
            }

            let bits_available = 8 - self.bit_pos;
            let bits_needed = count - bits_read;
            let bits_to_read = bits_available.min(bits_needed);

            let mask = ((1u16 << bits_to_read) - 1) as u8;
            let shift = 8 - self.bit_pos - bits_to_read;
            let bits = (self.data[self.byte_pos] >> shift) & mask;

            result = (result << bits_to_read) | (bits as u32);
            bits_read += bits_to_read;
            self.bit_pos += bits_to_read;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Some(result as u16)
    }
}

fn decode_heights(
    data: &[u8],
    bits_per_sample: u16,
    sample_format: u16,
    tile_width: u32,
    tile_height: u32,
    predictor: u16,
) -> Result<Vec<f32>, CogError> {
    let pixel_count = (tile_width as usize)
        .checked_mul(tile_height as usize)
        .ok_or_else(|| CogError::InvalidIfd("tile element count overflow".into()))?;
    let mut heights = Vec::with_capacity(pixel_count);
    let bytes_per_sample = (bits_per_sample as usize + 7) / 8;
    let data = apply_predictor(data, predictor, bytes_per_sample, tile_width, tile_height)?;
    let data = data.as_slice();

    match (bits_per_sample, sample_format) {
        (32, SAMPLE_FORMAT_FLOAT) => {
            let needed = pixel_count
                .checked_mul(4)
                .ok_or_else(|| CogError::InvalidIfd("f32 tile byte size overflow".into()))?;
            if data.len() < needed {
                return Err(CogError::InvalidIfd(format!(
                    "Data too short: {} < {}",
                    data.len(),
                    needed
                )));
            }
            for i in 0..pixel_count {
                heights.push(f32::from_le_bytes(read_le_bytes4(data, i * 4)));
            }
        }
        (64, SAMPLE_FORMAT_FLOAT) => {
            if data.len()
                < pixel_count
                    .checked_mul(8)
                    .ok_or_else(|| CogError::InvalidIfd("f64 tile byte size overflow".into()))?
            {
                return Err(CogError::InvalidIfd("Data too short for f64".into()));
            }
            for i in 0..pixel_count {
                heights.push(f64::from_le_bytes(read_le_bytes8(data, i * 8)) as f32);
            }
        }
        (16, SAMPLE_FORMAT_UINT) => {
            if data.len()
                < pixel_count
                    .checked_mul(2)
                    .ok_or_else(|| CogError::InvalidIfd("u16 tile byte size overflow".into()))?
            {
                return Err(CogError::InvalidIfd("Data too short for u16".into()));
            }
            for i in 0..pixel_count {
                let val = u16::from_le_bytes(read_le_bytes2(data, i * 2));
                heights.push(val as f32);
            }
        }
        (16, SAMPLE_FORMAT_INT) => {
            if data.len()
                < pixel_count
                    .checked_mul(2)
                    .ok_or_else(|| CogError::InvalidIfd("i16 tile byte size overflow".into()))?
            {
                return Err(CogError::InvalidIfd("Data too short for i16".into()));
            }
            for i in 0..pixel_count {
                let val = i16::from_le_bytes(read_le_bytes2(data, i * 2));
                heights.push(val as f32);
            }
        }
        (32, SAMPLE_FORMAT_INT) => {
            if data.len()
                < pixel_count
                    .checked_mul(4)
                    .ok_or_else(|| CogError::InvalidIfd("i32 tile byte size overflow".into()))?
            {
                return Err(CogError::InvalidIfd("Data too short for i32".into()));
            }
            for i in 0..pixel_count {
                let val = i32::from_le_bytes(read_le_bytes4(data, i * 4));
                heights.push(val as f32);
            }
        }
        (8, _) => {
            for &byte in data.iter().take(pixel_count) {
                heights.push(byte as f32);
            }
        }
        _ => {
            return Err(CogError::UnsupportedSampleFormat {
                bits: bits_per_sample,
                format: sample_format,
            });
        }
    }

    while heights.len() < pixel_count {
        heights.push(0.0);
    }

    Ok(heights)
}

fn resample_tile(src: &[f32], dst_width: u32, dst_height: u32) -> Vec<f32> {
    let src_side = (src.len() as f32).sqrt() as u32;
    if src_side == 0 {
        return vec![0.0f32; (dst_width * dst_height) as usize];
    }

    let mut dst = Vec::with_capacity((dst_width * dst_height) as usize);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = (x as f32 / dst_width as f32 * src_side as f32) as u32;
            let src_y = (y as f32 / dst_height as f32 * src_side as f32) as u32;
            let src_idx = (src_y.min(src_side - 1) * src_side + src_x.min(src_side - 1)) as usize;
            dst.push(src.get(src_idx).copied().unwrap_or(0.0));
        }
    }
    dst
}

fn apply_predictor(
    data: &[u8],
    predictor: u16,
    bytes_per_sample: usize,
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<u8>, CogError> {
    if predictor == TIFF_PREDICTOR_NONE {
        return Ok(data.to_vec());
    }
    if predictor != TIFF_PREDICTOR_HORIZONTAL {
        return Err(CogError::InvalidIfd(format!(
            "Unsupported TIFF predictor: {}",
            predictor
        )));
    }
    if !matches!(bytes_per_sample, 1 | 2 | 4 | 8) {
        return Err(CogError::InvalidIfd(format!(
            "Unsupported predictor sample width: {}",
            bytes_per_sample
        )));
    }

    let row_bytes = (tile_width as usize)
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| CogError::InvalidIfd("predictor row size overflow".into()))?;
    let needed = row_bytes
        .checked_mul(tile_height as usize)
        .ok_or_else(|| CogError::InvalidIfd("predictor payload size overflow".into()))?;
    if data.len() < needed {
        return Err(CogError::InvalidIfd(format!(
            "Data too short for predictor: {} < {}",
            data.len(),
            needed
        )));
    }

    let mut out = data.to_vec();
    for row in 0..tile_height as usize {
        let row_start = row * row_bytes;
        for col in 1..tile_width as usize {
            let prev = row_start + (col - 1) * bytes_per_sample;
            let cur = row_start + col * bytes_per_sample;
            match bytes_per_sample {
                1 => out[cur] = out[cur].wrapping_add(out[prev]),
                2 => {
                    let a = u16::from_le_bytes(read_le_bytes2(&out, prev));
                    let b = u16::from_le_bytes(read_le_bytes2(&out, cur));
                    out[cur..cur + 2].copy_from_slice(&b.wrapping_add(a).to_le_bytes());
                }
                4 => {
                    let a = u32::from_le_bytes(read_le_bytes4(&out, prev));
                    let b = u32::from_le_bytes(read_le_bytes4(&out, cur));
                    out[cur..cur + 4].copy_from_slice(&b.wrapping_add(a).to_le_bytes());
                }
                8 => {
                    let a = u64::from_le_bytes(read_le_bytes8(&out, prev));
                    let b = u64::from_le_bytes(read_le_bytes8(&out, cur));
                    out[cur..cur + 8].copy_from_slice(&b.wrapping_add(a).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(out)
}

fn read_le_bytes2(data: &[u8], offset: usize) -> [u8; 2] {
    [data[offset], data[offset + 1]]
}

fn read_le_bytes4(data: &[u8], offset: usize) -> [u8; 4] {
    [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]
}

fn read_le_bytes8(data: &[u8], offset: usize) -> [u8; 8] {
    [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_heights_applies_horizontal_predictor_to_u16_rows() {
        let encoded: Vec<u8> = [10u16, 2, 3, 20, 4, 5]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();

        let decoded = decode_heights(
            &encoded,
            16,
            SAMPLE_FORMAT_UINT,
            3,
            2,
            TIFF_PREDICTOR_HORIZONTAL,
        )
        .unwrap();

        assert_eq!(decoded, vec![10.0, 12.0, 15.0, 20.0, 24.0, 29.0]);
    }

    #[test]
    fn decode_heights_rejects_short_f32_payload_without_panic() {
        let err = decode_heights(
            &[0, 0, 128],
            32,
            SAMPLE_FORMAT_FLOAT,
            1,
            1,
            TIFF_PREDICTOR_NONE,
        )
        .unwrap_err();

        assert!(matches!(err, CogError::InvalidIfd(message) if message.contains("Data too short")));
    }

    #[test]
    fn decode_heights_rejects_short_f64_payload_without_panic() {
        let err = decode_heights(
            &[0, 0, 0, 0, 0, 0, 0],
            64,
            SAMPLE_FORMAT_FLOAT,
            1,
            1,
            TIFF_PREDICTOR_NONE,
        )
        .unwrap_err();

        assert!(
            matches!(err, CogError::InvalidIfd(message) if message.contains("Data too short for f64"))
        );
    }

    #[test]
    fn predictor_rejects_short_horizontal_payload_without_panic() {
        let err = apply_predictor(&[0, 1, 2], TIFF_PREDICTOR_HORIZONTAL, 2, 2, 1).unwrap_err();

        assert!(matches!(err, CogError::InvalidIfd(message) if message.contains("predictor")));
    }

    #[test]
    fn private_f3dz_compression_branch_decodes_f32_tile_bytes() {
        let source = vec![10.0f32, 10.1, f32::NAN, 10.3];
        let stream = crate::codec::f3dz::encode_dem(
            &source,
            2,
            2,
            &crate::codec::f3dz::EncodeOptions::new(0.05),
        )
        .unwrap();
        let bytes = decompress_tile(&stream, COMPRESSION_F3DZ, Some((2, 2))).unwrap();
        let decoded =
            decode_heights(&bytes, 32, SAMPLE_FORMAT_FLOAT, 2, 2, TIFF_PREDICTOR_NONE).unwrap();
        assert_eq!(decoded.len(), source.len());
        assert!(decoded[2].is_nan());
        assert!(decoded
            .iter()
            .zip(source)
            .filter(|(_, source)| !source.is_nan())
            .all(|(decoded, source)| (*decoded - source).abs() <= 0.05));

        assert!(matches!(
            decompress_tile(&stream, COMPRESSION_F3DZ, Some((4, 1))),
            Err(CogError::DecompressionError(message)) if message.contains("does not match")
        ));
    }
}
