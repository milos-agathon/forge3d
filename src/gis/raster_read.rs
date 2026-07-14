//! Full-image raster decoding: the whole-file read path plus the raw
//! uncompressed-strip fallback (with its byte-order conversion helpers) used
//! when the `tiff` crate's `read_image` rejects a file the metadata says we
//! can still assemble.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;

use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_info::{read_raster_info, LoadedRaster};
use crate::gis::raster_values::deinterleave;
use crate::gis::raster_write::{RasterArray, RasterData};
use crate::gis::types::{RasterDType, RasterInfo};

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
    if !bytes.chunks_exact(N).remainder().is_empty() {
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
