//! Raster metadata (`RasterInfo`) and the public `read_raster` entry point.
//! Full-image decoding lives in `raster_read`, windowed strip/tile decoding in
//! `raster_window`, TIFF tag/CRS readers in `raster_tags`, and shared
//! pixel-value utilities in `raster_values`; the moved items are re-exported
//! here so `raster_info::` remains the stable crate-internal path.

use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use tiff::decoder::{ChunkType, Decoder};

use crate::gis::affine::PixelWindow;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_tags::{
    compression_name, dtype_per_band, nodata_per_band, read_crs, read_transform, sample_count,
};
use crate::gis::raster_write::RasterArray;
use crate::gis::types::{
    AffineTransform, RasterInfo, RasterWarning, WARNING_METADATA_UNAVAILABLE, WARNING_MISSING_CRS,
    WARNING_MISSING_TRANSFORM, WARNING_NOT_GEOREFERENCED, WARNING_PER_BAND_NODATA_MISMATCH,
    WARNING_ROTATED_OR_SHEARED,
};

pub(crate) use crate::gis::raster_read::read_raster_data;
pub(crate) use crate::gis::raster_values::{
    copy_window, f64_to_raster_data, raster_to_f64, valid_mask,
};
#[cfg(feature = "cog_streaming")]
pub(crate) use crate::gis::raster_window::read_window_from_decoder;

#[derive(Debug, Clone)]
pub(crate) struct LoadedRaster {
    pub(crate) array: RasterArray,
    pub(crate) info: RasterInfo,
}

#[derive(Debug, Clone)]
pub struct RasterReadResult {
    pub array: RasterArray,
    pub info: RasterInfo,
    pub bands: Vec<u16>,
    pub window: Option<PixelWindow>,
    pub window_transform: Option<AffineTransform>,
    pub mask: Option<Vec<bool>>,
    pub nodata_per_band: Vec<Option<f64>>,
    pub warnings: Vec<RasterWarning>,
}

pub fn read_raster_info(path: impl AsRef<Path>) -> GisResult<RasterInfo> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(GisError::NotFound(path.to_path_buf()));
    }
    validate_tiff_path(path)?;
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?;
    raster_info_from_decoder(&mut decoder, normalized_path(path))
}

/// Build a `RasterInfo` from an already-open decoder over any `Read + Seek`
/// source (a local file or a range-backed remote reader). Only the TIFF header
/// and metadata are read here; no pixel data is decoded.
pub(crate) fn raster_info_from_decoder<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    path_label: PathBuf,
) -> GisResult<RasterInfo> {
    let (width, height) = decoder.dimensions()?;
    let band_count = sample_count(decoder)?;
    let mut info = RasterInfo::new(path_label, width, height, band_count);

    info.dtype_per_band = dtype_per_band(decoder, band_count)?;
    info.nodata_per_band = nodata_per_band(decoder, band_count)?;
    // MENSURA M-03: read back the persisted vertical datum, if any. An ABSENT
    // tag leaves the honest "unspecified" default; a PRESENT tag that fails to
    // decode or names an unknown height system is rejected — silently coercing
    // a declared-but-unrecognized vertical datum to "unspecified" would erase
    // the declaration and invite an orthometric/ellipsoidal mix-up downstream.
    if let Some(value) = decoder.find_tag(tiff::tags::Tag::Unknown(
        crate::gis::raster_write::FORGE3D_HEIGHT_SYSTEM_TAG,
    ))? {
        let raw = value.into_string().map_err(|err| {
            GisError::InvalidRaster(format!(
                "invalid_height_system: persisted height-system tag {} is not a readable \
                 ASCII value: {err}",
                crate::gis::raster_write::FORGE3D_HEIGHT_SYSTEM_TAG
            ))
        })?;
        let tag = raw.trim_matches(char::from(0)).trim();
        if !crate::gis::types::is_valid_height_system(tag) {
            return Err(GisError::InvalidRaster(format!(
                "invalid_height_system: persisted height-system tag value {tag:?} is not a \
                 recognized vertical datum; expected one of unspecified, ellipsoidal, \
                 orthometric_egm96, chart_datum"
            )));
        }
        info.height_system = tag.to_string();
    }
    info.block_size = Some(vec![decoder.chunk_dimensions(); band_count as usize]);
    info.tiling = Some(match decoder.get_chunk_type() {
        ChunkType::Strip => "striped".to_string(),
        ChunkType::Tile => "tiled".to_string(),
    });
    match compression_name(decoder) {
        Ok(Some(compression)) => info.compression = Some(compression),
        Ok(None) | Err(_) => info.warnings.push(RasterWarning::new(
            WARNING_METADATA_UNAVAILABLE,
            "TIFF compression metadata is unavailable",
            Some("compression"),
        )),
    }

    if let Some(transform) = read_transform(decoder)? {
        if transform.is_rotated_or_sheared() {
            info.warnings.push(RasterWarning::new(
                WARNING_ROTATED_OR_SHEARED,
                "transform contains rotation or shear terms",
                Some("transform"),
            ));
        }
        info.transform = Some(transform.tuple());
        info.bounds = Some(transform.bounds(width, height).tuple());
        info.resolution = Some(transform.resolution());
    } else {
        info.warnings.push(RasterWarning::new(
            WARNING_MISSING_TRANSFORM,
            "raster has no GeoTIFF transform metadata",
            Some("transform"),
        ));
    }

    let crs = read_crs(decoder)?;
    if let Some((wkt, authority)) = crs {
        info.crs_wkt = wkt;
        info.crs_authority = authority;
    }
    if info.crs_wkt.is_none() && info.crs_authority.is_some() {
        // G-002a1 uses a TIFF-only backend and does not synthesize full WKT from GeoKeys.
        info.warnings.push(RasterWarning::new(
            WARNING_METADATA_UNAVAILABLE,
            "CRS authority was read, but this TIFF-only backend could not extract full CRS WKT",
            Some("crs_wkt"),
        ));
    }
    let has_crs = info.crs_wkt.is_some() || info.crs_authority.is_some();
    if !has_crs {
        info.warnings.push(RasterWarning::new(
            WARNING_MISSING_CRS,
            "raster has no CRS metadata",
            Some("crs_wkt"),
        ));
    }

    if nodata_values_differ(&info.nodata_per_band) {
        info.warnings.push(RasterWarning::new(
            WARNING_PER_BAND_NODATA_MISMATCH,
            "bands advertise different nodata values",
            Some("nodata_per_band"),
        ));
    }

    info.is_georeferenced = has_crs && info.transform.is_some();
    if !info.is_georeferenced {
        info.warnings.push(RasterWarning::new(
            WARNING_NOT_GEOREFERENCED,
            "raster is not fully georeferenced",
            None,
        ));
    }

    Ok(info)
}

pub fn read_raster(
    path: impl AsRef<Path>,
    bands: Option<Vec<u16>>,
    window: Option<PixelWindow>,
    masked: bool,
) -> GisResult<RasterReadResult> {
    let loaded = read_raster_data(path)?;
    let selected_bands = validate_bands(bands, loaded.info.band_count)?;
    let window = window
        .map(|window| validate_read_window(window, loaded.info.width, loaded.info.height))
        .transpose()?;
    let windowed = match window {
        Some(window) => copy_window(&loaded.array, &loaded.info.nodata_per_band, window)?,
        None => loaded.array.clone(),
    };
    let array = select_bands(&windowed, &selected_bands)?;
    let nodata_per_band = select_band_metadata(&loaded.info.nodata_per_band, &selected_bands);
    let window_transform = match window {
        Some(window) if loaded.info.transform.is_some() => {
            Some(crate::gis::affine::window_transform(&loaded.info, window)?)
        }
        _ => None,
    };
    let info = read_result_info(
        &loaded.info,
        &array,
        &selected_bands,
        window,
        window_transform,
        nodata_per_band.clone(),
    );
    let mut warnings = info.warnings.clone();
    let mask = if masked {
        let mask = valid_mask(&array, &nodata_per_band, None);
        if mask.iter().all(|valid| !*valid) {
            push_warning_if_absent(
                &mut warnings,
                "empty_raster",
                "nodata and masks leave no valid pixels",
                None,
            );
        }
        Some(mask)
    } else {
        None
    };
    Ok(RasterReadResult {
        array,
        info,
        bands: selected_bands,
        window,
        window_transform,
        mask,
        nodata_per_band,
        warnings,
    })
}

fn validate_bands(bands: Option<Vec<u16>>, band_count: u16) -> GisResult<Vec<u16>> {
    let bands = bands.unwrap_or_else(|| (1..=band_count).collect());
    if bands.is_empty() {
        return Err(GisError::InvalidArgument(
            "bands must not be empty".to_string(),
        ));
    }
    let mut seen = HashSet::with_capacity(bands.len());
    for band in &bands {
        if *band == 0 || *band > band_count {
            return Err(GisError::InvalidArgument(format!(
                "band {band} is out of range; bands are 1-based and raster has {band_count} bands"
            )));
        }
        if !seen.insert(*band) {
            return Err(GisError::InvalidArgument(
                "duplicate band selections are not supported".to_string(),
            ));
        }
    }
    Ok(bands)
}

pub(crate) fn validate_read_window(
    window: PixelWindow,
    source_width: u32,
    source_height: u32,
) -> GisResult<PixelWindow> {
    if window.width == 0 || window.height == 0 {
        return Err(GisError::InvalidArgument(
            "window width and height must be positive".to_string(),
        ));
    }
    if window.col_off < 0
        || window.row_off < 0
        || window.col_off + window.width as i64 > source_width as i64
        || window.row_off + window.height as i64 > source_height as i64
    {
        return Err(GisError::InvalidArgument(
            "window must be inside raster extent; read_raster does not support boundless windows"
                .to_string(),
        ));
    }
    Ok(window)
}

fn select_bands(source: &RasterArray, bands: &[u16]) -> GisResult<RasterArray> {
    if bands.len() == source.bands
        && bands
            .iter()
            .enumerate()
            .all(|(i, band)| *band as usize == i + 1)
    {
        return Ok(source.clone());
    }
    let source_values = raster_to_f64(source);
    let pixels = source.height * source.width;
    let mut out = Vec::with_capacity(bands.len() * pixels);
    for band in bands {
        let band_index = (*band as usize) - 1;
        let start = band_index * pixels;
        out.extend_from_slice(&source_values[start..start + pixels]);
    }
    f64_to_raster_data(
        source.dtype(),
        out,
        bands.len(),
        source.height,
        source.width,
    )
}

fn select_band_metadata<T: Clone>(values: &[T], bands: &[u16]) -> Vec<T> {
    bands
        .iter()
        .filter_map(|band| values.get((*band as usize).saturating_sub(1)).cloned())
        .collect()
}

pub(crate) fn read_result_info(
    source: &RasterInfo,
    array: &RasterArray,
    bands: &[u16],
    window: Option<PixelWindow>,
    window_transform: Option<AffineTransform>,
    nodata_per_band: Vec<Option<f64>>,
) -> RasterInfo {
    let mut info = source.clone();
    if let Some(window) = window {
        info.width = window.width;
        info.height = window.height;
        if let Some(transform) = window_transform {
            info.transform = Some(transform.tuple());
            info.bounds = Some(transform.bounds(window.width, window.height).tuple());
            info.resolution = Some(transform.resolution());
        } else {
            info.transform = None;
            info.bounds = None;
            info.resolution = None;
        }
    }
    info.band_count = array.bands as u16;
    info.dtype_per_band = select_band_metadata(&source.dtype_per_band, bands);
    if info.dtype_per_band.len() != array.bands {
        info.dtype_per_band = vec![array.dtype().name().to_string(); array.bands];
    }
    info.nodata_per_band = nodata_per_band;
    info.block_size = source
        .block_size
        .as_ref()
        .map(|values| select_band_metadata(values, bands));
    info.warnings
        .retain(|warning| warning.code != WARNING_PER_BAND_NODATA_MISMATCH);
    if nodata_values_differ(&info.nodata_per_band) {
        push_warning_if_absent(
            &mut info.warnings,
            WARNING_PER_BAND_NODATA_MISMATCH,
            "bands advertise different nodata values",
            Some("nodata_per_band"),
        );
    }
    let has_crs = info.crs_wkt.is_some() || info.crs_authority.is_some();
    info.is_georeferenced = has_crs && info.transform.is_some();
    if !has_crs {
        push_warning_if_absent(
            &mut info.warnings,
            WARNING_MISSING_CRS,
            "raster has no CRS metadata",
            Some("crs_wkt"),
        );
    }
    if info.transform.is_none() {
        push_warning_if_absent(
            &mut info.warnings,
            WARNING_MISSING_TRANSFORM,
            "raster has no GeoTIFF transform metadata",
            Some("transform"),
        );
    }
    if !info.is_georeferenced {
        push_warning_if_absent(
            &mut info.warnings,
            WARNING_NOT_GEOREFERENCED,
            "raster is not fully georeferenced",
            None,
        );
    }
    info
}

fn nodata_values_differ(values: &[Option<f64>]) -> bool {
    let Some(first) = values.first() else {
        return false;
    };
    for value in values.iter().skip(1) {
        if !nodata_option_eq(*first, *value) {
            return true;
        }
    }
    false
}

fn nodata_option_eq(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => float_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}

fn float_eq(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a == b)
}

fn push_warning_if_absent(
    warnings: &mut Vec<RasterWarning>,
    code: &'static str,
    message: impl Into<String>,
    field: Option<&'static str>,
) {
    if warnings.iter().any(|warning| warning.code == code) {
        return;
    }
    warnings.push(RasterWarning::new(code, message, field));
}

fn normalized_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

fn validate_tiff_path(path: &Path) -> GisResult<()> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("tif" | "tiff") => Ok(()),
        _ => Err(GisError::UnsupportedDriver(format!(
            "G-002a1 read_raster_info supports local TIFF/GeoTIFF only: {}",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::gis::raster_write::{
        write_raster, CreationOptions, CrsSpec, RasterData, WriteRasterOptions,
    };

    fn temp_tif(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "forge3d_{name}_{}_{}.tif",
            std::process::id(),
            stamp
        ))
    }

    #[test]
    fn read_raster_selects_window_band_metadata_and_true_valid_mask() {
        let path = temp_tif("read_raster");
        let array = RasterArray::new(
            RasterData::U8(vec![
                0, 1, 2, 3, 4, 5, 10, 99, 12, 13, 14, 99, 20, 21, 22, 23, 24, 25,
            ]),
            &[3, 2, 3],
        )
        .expect("test raster shape");
        write_raster(
            &path,
            array,
            WriteRasterOptions {
                crs: Some(CrsSpec::from_string("EPSG:4326".to_string()).expect("valid crs")),
                transform: Some(
                    AffineTransform::new([2.0, 0.0, 10.0, 0.0, -3.0, 50.0])
                        .expect("valid transform"),
                ),
                nodata: vec![Some(0.0), Some(99.0), None],
                driver: "GTiff".to_string(),
                overwrite: true,
                creation_options: CreationOptions::default(),
                creation_options_explicit: false,
                like_info: None,
                height_system: crate::gis::types::HEIGHT_SYSTEM_UNSPECIFIED.to_string(),
            },
        )
        .expect("write test raster");

        let result = read_raster(
            &path,
            Some(vec![2]),
            Some(PixelWindow {
                col_off: 1,
                row_off: 0,
                width: 2,
                height: 2,
            }),
            true,
        )
        .expect("read raster");

        assert_eq!(result.bands, vec![2]);
        assert_eq!(result.info.width, 2);
        assert_eq!(result.info.height, 2);
        assert_eq!(result.info.band_count, 1);
        assert_eq!(result.info.dtype_per_band, vec!["uint8"]);
        assert_eq!(result.nodata_per_band, vec![Some(99.0)]);
        assert_eq!(
            result.window_transform.map(AffineTransform::tuple),
            Some((2.0, 0.0, 12.0, 0.0, -3.0, 50.0))
        );
        assert_eq!(result.mask, Some(vec![false, true, true, false]));
        match result.array.data {
            RasterData::U8(values) => assert_eq!(values, vec![99, 12, 14, 99]),
            other => panic!("unexpected dtype: {:?}", other),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn read_raster_info_preserves_projected_epsg_outside_legacy_three_code_subset() {
        let path = temp_tif("epsg_2154_metadata");
        let array = RasterArray::new(RasterData::U8(vec![1, 2, 3, 4]), &[1, 2, 2])
            .expect("test raster shape");
        write_raster(
            &path,
            array,
            WriteRasterOptions {
                crs: Some(CrsSpec::from_string("EPSG:2154".to_string()).expect("valid crs")),
                transform: Some(
                    AffineTransform::new([2.0, 0.0, 700_000.0, 0.0, -2.0, 6_600_000.0])
                        .expect("valid transform"),
                ),
                nodata: vec![None],
                driver: "GTiff".to_string(),
                overwrite: true,
                creation_options: CreationOptions::default(),
                creation_options_explicit: false,
                like_info: None,
                height_system: crate::gis::types::HEIGHT_SYSTEM_UNSPECIFIED.to_string(),
            },
        )
        .expect("write metadata fixture");

        let info = read_raster_info(&path).expect("read arbitrary authority metadata");
        assert_eq!(
            info.crs_authority,
            Some(std::collections::HashMap::from([
                ("name".to_string(), "EPSG".to_string()),
                ("code".to_string(), "2154".to_string()),
            ]))
        );
        assert!(
            info.crs_wkt.is_none(),
            "metadata reader must not synthesize WKT"
        );

        let _ = fs::remove_file(path);
    }
}
