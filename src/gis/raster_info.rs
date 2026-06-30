use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::Tag;

use crate::gis::affine::PixelWindow;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_write::{RasterArray, RasterData};
use crate::gis::types::{
    AffineTransform, RasterDType, RasterInfo, RasterWarning, WARNING_METADATA_UNAVAILABLE,
    WARNING_MISSING_CRS, WARNING_MISSING_TRANSFORM, WARNING_NOT_GEOREFERENCED,
    WARNING_PER_BAND_NODATA_MISMATCH, WARNING_ROTATED_OR_SHEARED,
};

const FORGE3D_NODATA_PREFIX: &str = "forge3d:nodata_per_band=";

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
    let (width, height) = decoder.dimensions()?;
    let band_count = sample_count(&mut decoder)?;
    let mut info = RasterInfo::new(normalized_path(path), width, height, band_count);

    info.dtype_per_band = dtype_per_band(&mut decoder, band_count)?;
    info.nodata_per_band = nodata_per_band(&mut decoder, band_count)?;
    info.block_size = Some(vec![decoder.chunk_dimensions(); band_count as usize]);
    info.tiling = Some(match decoder.get_chunk_type() {
        ChunkType::Strip => "striped".to_string(),
        ChunkType::Tile => "tiled".to_string(),
    });
    match compression_name(&mut decoder) {
        Ok(Some(compression)) => info.compression = Some(compression),
        Ok(None) | Err(_) => info.warnings.push(RasterWarning::new(
            WARNING_METADATA_UNAVAILABLE,
            "TIFF compression metadata is unavailable",
            Some("compression"),
        )),
    }

    if let Some(transform) = read_transform(&mut decoder)? {
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

    let crs = read_crs(&mut decoder)?;
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

pub(crate) fn read_raster_data(path: impl AsRef<Path>) -> GisResult<LoadedRaster> {
    let path = path.as_ref();
    let info = read_raster_info(path)?;
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?;
    let image = match decoder.read_image() {
        Ok(image) => image,
        Err(err) => read_uncompressed_strips(path, &info).map_err(|_| GisError::from(err))?,
    };
    let bands = info.band_count as usize;
    let shape = [bands, info.height as usize, info.width as usize];
    let array = match image {
        DecodingResult::U8(data) => {
            RasterArray::new(RasterData::U8(deinterleave(data, bands)), &shape)
        }
        DecodingResult::U16(data) => {
            RasterArray::new(RasterData::U16(deinterleave(data, bands)), &shape)
        }
        DecodingResult::I16(data) => {
            RasterArray::new(RasterData::I16(deinterleave(data, bands)), &shape)
        }
        DecodingResult::U32(data) => {
            RasterArray::new(RasterData::U32(deinterleave(data, bands)), &shape)
        }
        DecodingResult::I32(data) => {
            RasterArray::new(RasterData::I32(deinterleave(data, bands)), &shape)
        }
        DecodingResult::F32(data) => {
            RasterArray::new(RasterData::F32(deinterleave(data, bands)), &shape)
        }
        DecodingResult::F64(data) => {
            RasterArray::new(RasterData::F64(deinterleave(data, bands)), &shape)
        }
        _ => Err(GisError::UnsupportedDType(
            "unsupported TIFF dtype for raster operation".to_string(),
        )),
    }?;
    Ok(LoadedRaster { array, info })
}

fn read_uncompressed_strips(path: &Path, info: &RasterInfo) -> GisResult<DecodingResult> {
    if info.tiling.as_deref() != Some("striped") || info.compression.as_deref() != Some("NONE") {
        return Err(GisError::InvalidRaster(
            "raw strip fallback only supports uncompressed striped TIFFs".to_string(),
        ));
    }
    let dtype = dtype_name_to_dtype(
        info.dtype_per_band
            .first()
            .ok_or_else(|| GisError::InvalidRaster("raster has no dtype metadata".to_string()))?,
    )?;
    if info
        .dtype_per_band
        .iter()
        .any(|name| dtype_name_to_dtype(name).ok() != Some(dtype))
    {
        return Err(GisError::UnsupportedDType(
            "mixed per-band TIFF dtypes are not supported".to_string(),
        ));
    }

    let mut decoder = Decoder::new(BufReader::new(File::open(path)?))?;
    let planar_config = decoder
        .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)?
        .unwrap_or(1);
    if planar_config != 1 {
        return Err(GisError::InvalidRaster(
            "raw strip fallback only supports chunky planar configuration".to_string(),
        ));
    }
    let offsets = decoder
        .find_tag_unsigned_vec::<u64>(Tag::StripOffsets)?
        .ok_or_else(|| GisError::InvalidRaster("TIFF StripOffsets tag is missing".to_string()))?;
    let byte_counts = decoder
        .find_tag_unsigned_vec::<u64>(Tag::StripByteCounts)?
        .ok_or_else(|| {
            GisError::InvalidRaster("TIFF StripByteCounts tag is missing".to_string())
        })?;
    if offsets.len() != byte_counts.len() {
        return Err(GisError::InvalidRaster(
            "TIFF strip offset/count metadata length mismatch".to_string(),
        ));
    }

    let expected_bytes = (info.width as usize)
        .checked_mul(info.height as usize)
        .and_then(|value| value.checked_mul(info.band_count as usize))
        .and_then(|value| value.checked_mul(dtype_size(dtype)))
        .ok_or_else(|| GisError::InvalidRaster("raster byte size is too large".to_string()))?;
    let little_endian = read_tiff_little_endian(path)?;
    let mut file = File::open(path)?;
    let mut bytes = Vec::with_capacity(expected_bytes);
    for (offset, byte_count) in offsets.into_iter().zip(byte_counts) {
        let byte_count = usize::try_from(byte_count).map_err(|_| {
            GisError::InvalidRaster("TIFF strip byte count exceeds usize range".to_string())
        })?;
        let start_len = bytes.len();
        bytes.resize(start_len + byte_count, 0);
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut bytes[start_len..])?;
    }
    if bytes.len() != expected_bytes {
        return Err(GisError::InvalidRaster(format!(
            "TIFF strip data has {} bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }

    match dtype {
        RasterDType::UInt8 => Ok(DecodingResult::U8(bytes)),
        RasterDType::Int16 => Ok(DecodingResult::I16(
            bytes_to_i16(&bytes, little_endian)?.into_iter().collect(),
        )),
        RasterDType::UInt16 => Ok(DecodingResult::U16(bytes_to_u16(&bytes, little_endian)?)),
        RasterDType::Int32 => Ok(DecodingResult::I32(bytes_to_i32(&bytes, little_endian)?)),
        RasterDType::UInt32 => Ok(DecodingResult::U32(bytes_to_u32(&bytes, little_endian)?)),
        RasterDType::Float32 => Ok(DecodingResult::F32(bytes_to_f32(&bytes, little_endian)?)),
        RasterDType::Float64 => Ok(DecodingResult::F64(bytes_to_f64(&bytes, little_endian)?)),
    }
}

fn dtype_name_to_dtype(name: &str) -> GisResult<RasterDType> {
    match name {
        "uint8" => Ok(RasterDType::UInt8),
        "int16" => Ok(RasterDType::Int16),
        "uint16" => Ok(RasterDType::UInt16),
        "int32" => Ok(RasterDType::Int32),
        "uint32" => Ok(RasterDType::UInt32),
        "float32" => Ok(RasterDType::Float32),
        "float64" => Ok(RasterDType::Float64),
        other => Err(GisError::UnsupportedDType(format!(
            "unsupported TIFF dtype {other:?}"
        ))),
    }
}

fn dtype_size(dtype: RasterDType) -> usize {
    match dtype {
        RasterDType::UInt8 => 1,
        RasterDType::Int16 | RasterDType::UInt16 => 2,
        RasterDType::Int32 | RasterDType::UInt32 | RasterDType::Float32 => 4,
        RasterDType::Float64 => 8,
    }
}

fn read_tiff_little_endian(path: &Path) -> GisResult<bool> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 2];
    file.read_exact(&mut header)?;
    match &header {
        b"II" => Ok(true),
        b"MM" => Ok(false),
        _ => Err(GisError::InvalidRaster(
            "TIFF byte order marker is invalid".to_string(),
        )),
    }
}

fn bytes_to_u16(bytes: &[u8], little_endian: bool) -> GisResult<Vec<u16>> {
    bytes_to_array::<2, u16>(bytes, |chunk| {
        if little_endian {
            u16::from_le_bytes(chunk)
        } else {
            u16::from_be_bytes(chunk)
        }
    })
}

fn bytes_to_i16(bytes: &[u8], little_endian: bool) -> GisResult<Vec<i16>> {
    bytes_to_array::<2, i16>(bytes, |chunk| {
        if little_endian {
            i16::from_le_bytes(chunk)
        } else {
            i16::from_be_bytes(chunk)
        }
    })
}

fn bytes_to_u32(bytes: &[u8], little_endian: bool) -> GisResult<Vec<u32>> {
    bytes_to_array::<4, u32>(bytes, |chunk| {
        if little_endian {
            u32::from_le_bytes(chunk)
        } else {
            u32::from_be_bytes(chunk)
        }
    })
}

fn bytes_to_i32(bytes: &[u8], little_endian: bool) -> GisResult<Vec<i32>> {
    bytes_to_array::<4, i32>(bytes, |chunk| {
        if little_endian {
            i32::from_le_bytes(chunk)
        } else {
            i32::from_be_bytes(chunk)
        }
    })
}

fn bytes_to_f32(bytes: &[u8], little_endian: bool) -> GisResult<Vec<f32>> {
    bytes_to_u32(bytes, little_endian)
        .map(|values| values.into_iter().map(f32::from_bits).collect())
}

fn bytes_to_f64(bytes: &[u8], little_endian: bool) -> GisResult<Vec<f64>> {
    bytes_to_array::<8, f64>(bytes, |chunk| {
        if little_endian {
            f64::from_le_bytes(chunk)
        } else {
            f64::from_be_bytes(chunk)
        }
    })
}

fn bytes_to_array<const N: usize, T>(
    bytes: &[u8],
    convert: impl Fn([u8; N]) -> T,
) -> GisResult<Vec<T>> {
    if bytes.len() % N != 0 {
        return Err(GisError::InvalidRaster(
            "TIFF strip byte count is not aligned to sample size".to_string(),
        ));
    }
    Ok(bytes
        .chunks_exact(N)
        .map(|chunk| {
            let mut array = [0u8; N];
            array.copy_from_slice(chunk);
            convert(array)
        })
        .collect())
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

fn validate_read_window(
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

fn read_result_info(
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

pub(crate) fn copy_window(
    source: &RasterArray,
    nodata: &[Option<f64>],
    window: PixelWindow,
) -> GisResult<RasterArray> {
    let source_values = raster_to_f64(source);
    let out_width = window.width as usize;
    let out_height = window.height as usize;
    let mut out = vec![0.0; source.bands * out_width * out_height];
    for band in 0..source.bands {
        let fill = nodata.get(band).copied().flatten().unwrap_or(0.0);
        for out_row in 0..out_height {
            for out_col in 0..out_width {
                let src_col = window.col_off + out_col as i64;
                let src_row = window.row_off + out_row as i64;
                let out_index = band * out_width * out_height + out_row * out_width + out_col;
                out[out_index] = if src_col >= 0
                    && src_row >= 0
                    && src_col < source.width as i64
                    && src_row < source.height as i64
                {
                    source_values[band * source.width * source.height
                        + src_row as usize * source.width
                        + src_col as usize]
                } else {
                    fill
                };
            }
        }
    }
    f64_to_raster_data(source.dtype(), out, source.bands, out_height, out_width)
}

pub(crate) fn valid_mask(
    array: &RasterArray,
    nodata: &[Option<f64>],
    explicit_mask: Option<&[bool]>,
) -> Vec<bool> {
    let values = raster_to_f64(array);
    let pixels = array.height * array.width;
    let mut mask = Vec::with_capacity(values.len());
    for band in 0..array.bands {
        let nodata = nodata.get(band).copied().flatten();
        for pixel in 0..pixels {
            let index = band * pixels + pixel;
            let value = values[index];
            let explicit_valid = explicit_mask
                .and_then(|mask| mask.get(index))
                .copied()
                .unwrap_or(true);
            mask.push(explicit_valid && !value.is_nan() && !nodata_matches(value, nodata));
        }
    }
    mask
}

pub(crate) fn raster_to_f64(array: &RasterArray) -> Vec<f64> {
    match &array.data {
        RasterData::U8(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::I16(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::U16(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::I32(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::U32(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::F32(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::F64(values) => values.clone(),
    }
}

pub(crate) fn f64_to_raster_data(
    dtype: RasterDType,
    values: Vec<f64>,
    bands: usize,
    height: usize,
    width: usize,
) -> GisResult<RasterArray> {
    let shape = [bands, height, width];
    let data = match dtype {
        RasterDType::UInt8 => RasterData::U8(
            values
                .iter()
                .map(|&value| value.round().clamp(0.0, u8::MAX as f64) as u8)
                .collect(),
        ),
        RasterDType::Int16 => RasterData::I16(
            values
                .iter()
                .map(|&value| value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
                .collect(),
        ),
        RasterDType::UInt16 => RasterData::U16(
            values
                .iter()
                .map(|&value| value.round().clamp(0.0, u16::MAX as f64) as u16)
                .collect(),
        ),
        RasterDType::Int32 => RasterData::I32(
            values
                .iter()
                .map(|&value| value.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
                .collect(),
        ),
        RasterDType::UInt32 => RasterData::U32(
            values
                .iter()
                .map(|&value| value.round().clamp(0.0, u32::MAX as f64) as u32)
                .collect(),
        ),
        RasterDType::Float32 => RasterData::F32(values.iter().map(|&value| value as f32).collect()),
        RasterDType::Float64 => RasterData::F64(values),
    };
    RasterArray::new(data, &shape)
}

fn nodata_matches(value: f64, nodata: Option<f64>) -> bool {
    nodata.is_some_and(|nodata| value == nodata || (value.is_nan() && nodata.is_nan()))
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

fn deinterleave<T: Copy>(data: Vec<T>, bands: usize) -> Vec<T> {
    if bands <= 1 {
        return data;
    }
    let pixels = data.len() / bands;
    let mut out = Vec::with_capacity(data.len());
    for band in 0..bands {
        for pixel in 0..pixels {
            out.push(data[pixel * bands + band]);
        }
    }
    out
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

fn sample_count<R: std::io::Read + std::io::Seek>(decoder: &mut Decoder<R>) -> GisResult<u16> {
    if let Some(samples) = decoder.find_tag_unsigned::<u16>(Tag::SamplesPerPixel)? {
        return Ok(samples);
    }
    Ok(1)
}

fn dtype_per_band<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    band_count: u16,
) -> GisResult<Vec<String>> {
    let bits = decoder
        .find_tag_unsigned_vec::<u16>(Tag::BitsPerSample)?
        .unwrap_or_else(|| vec![8; band_count as usize]);
    let sample_formats = decoder
        .find_tag_unsigned_vec::<u16>(Tag::SampleFormat)?
        .unwrap_or_else(|| vec![1; band_count as usize]);

    let mut dtypes = Vec::with_capacity(band_count as usize);
    for index in 0..band_count as usize {
        let bit_depth = bits.get(index).copied().unwrap_or_else(|| bits[0]);
        let sample_format = sample_formats.get(index).copied().unwrap_or(1);
        let dtype = dtype_from_tags(bit_depth, sample_format)?;
        dtypes.push(dtype.name().to_string());
    }
    Ok(dtypes)
}

fn dtype_from_tags(bits: u16, sample_format: u16) -> GisResult<RasterDType> {
    match (bits, sample_format) {
        (8, 1) => Ok(RasterDType::UInt8),
        (16, 2) => Ok(RasterDType::Int16),
        (16, 1) => Ok(RasterDType::UInt16),
        (32, 2) => Ok(RasterDType::Int32),
        (32, 1) => Ok(RasterDType::UInt32),
        (32, 3) => Ok(RasterDType::Float32),
        (64, 3) => Ok(RasterDType::Float64),
        _ => Err(GisError::UnsupportedDType(format!(
            "unsupported TIFF dtype bits_per_sample={bits}, sample_format={sample_format}"
        ))),
    }
}

fn nodata_per_band<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    band_count: u16,
) -> GisResult<Vec<Option<f64>>> {
    if let Some(value) = decoder.find_tag(Tag::ImageDescription)? {
        let description = value
            .into_string()
            .map_err(|err| GisError::InvalidRaster(err.to_string()))?;
        if let Some(values) = parse_forge3d_nodata(&description, band_count)? {
            return Ok(values);
        }
    }

    match decoder.find_tag(Tag::GdalNodata)? {
        Some(value) => {
            let raw = value
                .into_string()
                .map_err(|err| GisError::InvalidRaster(err.to_string()))?;
            let parsed = parse_nodata_string(&raw)?;
            Ok(vec![parsed; band_count as usize])
        }
        None => Ok(vec![None; band_count as usize]),
    }
}

fn parse_forge3d_nodata(raw: &str, band_count: u16) -> GisResult<Option<Vec<Option<f64>>>> {
    let Some(rest) = raw
        .trim_matches(char::from(0))
        .strip_prefix(FORGE3D_NODATA_PREFIX)
    else {
        return Ok(None);
    };
    let values = rest
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.eq_ignore_ascii_case("none") || part.is_empty() {
                Ok(None)
            } else if part.eq_ignore_ascii_case("nan") {
                Ok(Some(f64::NAN))
            } else {
                part.parse::<f64>().map(Some).map_err(|_| {
                    GisError::InvalidRaster(format!(
                        "invalid forge3d per-band nodata value: {part:?}"
                    ))
                })
            }
        })
        .collect::<GisResult<Vec<_>>>()?;
    if values.len() != band_count as usize {
        return Err(GisError::InvalidRaster(format!(
            "forge3d per-band nodata count {} does not match band count {band_count}",
            values.len()
        )));
    }
    Ok(Some(values))
}

fn parse_nodata_string(raw: &str) -> GisResult<Option<f64>> {
    let trimmed = raw.trim_matches(char::from(0)).trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.eq_ignore_ascii_case("nan") {
        return Ok(Some(f64::NAN));
    }
    trimmed
        .parse::<f64>()
        .map(Some)
        .map_err(|_| GisError::InvalidRaster(format!("invalid GDAL_NODATA tag value: {trimmed:?}")))
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

fn compression_name<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> GisResult<Option<String>> {
    Ok(decoder
        .find_tag_unsigned::<u16>(Tag::Compression)?
        .map(|value| match value {
            1 => "NONE".to_string(),
            5 => "LZW".to_string(),
            8 | 32946 => "DEFLATE".to_string(),
            32773 => "PACKBITS".to_string(),
            other => format!("UNKNOWN({other})"),
        }))
}

fn read_transform<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> GisResult<Option<AffineTransform>> {
    if let Some(values) = decoder.find_tag(Tag::ModelTransformationTag)? {
        let values = values
            .into_f64_vec()
            .map_err(|err| GisError::InvalidTransform(err.to_string()))?;
        if values.len() < 16 {
            return Err(GisError::InvalidTransform(
                "ModelTransformationTag must contain 16 doubles".to_string(),
            ));
        }
        return AffineTransform::new([
            values[0], values[1], values[3], values[4], values[5], values[7],
        ])
        .map(Some);
    }

    let pixel_scale = match decoder.find_tag(Tag::ModelPixelScaleTag)? {
        Some(value) => value
            .into_f64_vec()
            .map_err(|err| GisError::InvalidTransform(err.to_string()))?,
        None => return Ok(None),
    };
    let tiepoints = match decoder.find_tag(Tag::ModelTiepointTag)? {
        Some(value) => value
            .into_f64_vec()
            .map_err(|err| GisError::InvalidTransform(err.to_string()))?,
        None => return Ok(None),
    };
    if pixel_scale.len() < 2 || tiepoints.len() < 6 {
        return Err(GisError::InvalidTransform(
            "GeoTIFF pixel scale/tiepoint tags are incomplete".to_string(),
        ));
    }
    let scale_x = pixel_scale[0];
    let scale_y = pixel_scale[1];
    let raster_i = tiepoints[0];
    let raster_j = tiepoints[1];
    let model_x = tiepoints[3];
    let model_y = tiepoints[4];
    AffineTransform::new([
        scale_x,
        0.0,
        model_x - raster_i * scale_x,
        0.0,
        -scale_y,
        model_y + raster_j * scale_y,
    ])
    .map(Some)
}

fn read_crs<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> GisResult<Option<(Option<String>, Option<HashMap<String, String>>)>> {
    let geo_ascii = decoder
        .find_tag(Tag::GeoAsciiParamsTag)?
        .map(|value| {
            value
                .into_string()
                .map_err(|err| GisError::InvalidCrs(err.to_string()))
        })
        .transpose()?;
    let directory = match decoder.find_tag(Tag::GeoKeyDirectoryTag)? {
        Some(value) => value
            .into_u16_vec()
            .map_err(|err| GisError::InvalidCrs(err.to_string()))?,
        None => {
            return Ok(match geo_ascii.as_deref() {
                Some(ascii) => validated_wkt_from_ascii(ascii)?.map(|wkt| (Some(wkt), None)),
                None => None,
            })
        }
    };
    if directory.len() < 4 {
        return Err(GisError::InvalidCrs(
            "GeoKeyDirectoryTag is too short".to_string(),
        ));
    }
    let entry_count = directory[3] as usize;
    if directory.len() < 4 + entry_count * 4 {
        return Err(GisError::InvalidCrs(
            "GeoKeyDirectoryTag entry count exceeds tag length".to_string(),
        ));
    }

    let mut geographic_epsg = None;
    let mut projected_epsg = None;
    let mut wkt = match geo_ascii.as_deref() {
        Some(ascii) => validated_wkt_from_ascii(ascii)?,
        None => None,
    };
    for entry in directory[4..4 + entry_count * 4].chunks_exact(4) {
        let key = entry[0];
        let tag_location = entry[1];
        let count = entry[2];
        let value_offset = entry[3];
        if tag_location == Tag::GeoAsciiParamsTag.to_u16() {
            if let Some(ascii) = &geo_ascii {
                if matches!(key, 1026 | 2049 | 3073) {
                    if let Some(value) = extract_ascii_range(ascii, value_offset, count) {
                        if looks_like_wkt(&value) {
                            wkt = Some(validate_wkt_literal(&value)?);
                        }
                    }
                }
            }
            continue;
        }
        if tag_location != 0 || count != 1 {
            continue;
        }
        match key {
            2048 if value_offset != 0 && value_offset != 32767 => {
                geographic_epsg = Some(value_offset)
            }
            3072 if value_offset != 0 && value_offset != 32767 => {
                projected_epsg = Some(value_offset)
            }
            _ => {}
        }
    }

    let epsg = projected_epsg.or(geographic_epsg);
    if let Some(code) = epsg {
        // G-002a1 has no CRS database; validate only the supported EPSG subset.
        if !matches!(code, 4326 | 3857 | 32631) {
            return Err(GisError::InvalidCrs(format!(
                "unsupported EPSG code {code}; this TIFF-only backend supports EPSG:4326, EPSG:3857, and EPSG:32631"
            )));
        }
        let mut authority = HashMap::new();
        authority.insert("name".to_string(), "EPSG".to_string());
        authority.insert("code".to_string(), code.to_string());
        return Ok(Some((wkt, Some(authority))));
    }

    Ok(wkt.map(|wkt| (Some(wkt), None)))
}

fn extract_ascii_range(ascii: &str, offset: u16, count: u16) -> Option<String> {
    let start = offset as usize;
    let end = start.checked_add(count as usize)?;
    let bytes = ascii.as_bytes();
    if start >= bytes.len() || end > bytes.len() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&bytes[start..end])
            .trim_matches(char::from(0))
            .trim_matches('|')
            .trim()
            .to_string(),
    )
}

fn extract_wkt_from_ascii(ascii: &str) -> Option<String> {
    ascii
        .split('|')
        .map(str::trim)
        .find(|value| looks_like_wkt(value))
        .map(str::to_string)
}

fn validated_wkt_from_ascii(ascii: &str) -> GisResult<Option<String>> {
    extract_wkt_from_ascii(ascii)
        .map(|wkt| validate_wkt_literal(&wkt))
        .transpose()
}

fn validate_wkt_literal(value: &str) -> GisResult<String> {
    let trimmed = value.trim();
    if !looks_like_wkt(trimmed) {
        return Err(GisError::InvalidCrs(
            "CRS WKT must start with a supported WKT CRS token".to_string(),
        ));
    }
    let mut depth = 0i32;
    for ch in trimmed.chars() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth < 0 {
                    return Err(GisError::InvalidCrs(
                        "CRS WKT has unbalanced brackets".to_string(),
                    ));
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(GisError::InvalidCrs(
            "CRS WKT has unbalanced brackets".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

fn looks_like_wkt(value: &str) -> bool {
    let upper = value.trim_start().to_ascii_uppercase();
    ["GEOGCRS[", "PROJCRS[", "GEOGCS[", "PROJCS["]
        .iter()
        .any(|prefix| upper.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::gis::raster_write::{write_raster, CreationOptions, CrsSpec, WriteRasterOptions};

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
}
