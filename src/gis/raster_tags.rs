//! GeoTIFF tag and GeoKey readers: per-band dtype/nodata metadata,
//! compression names, the affine transform, and CRS extraction (EPSG GeoKeys
//! and embedded WKT). Pure metadata — no pixel data is decoded here.

use std::collections::HashMap;

use tiff::decoder::Decoder;
use tiff::tags::Tag;

use crate::gis::error::{GisError, GisResult};
use crate::gis::types::{AffineTransform, RasterDType};

const FORGE3D_NODATA_PREFIX: &str = "forge3d:nodata_per_band=";

pub(crate) fn sample_count<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> GisResult<u16> {
    if let Some(samples) = decoder.find_tag_unsigned::<u16>(Tag::SamplesPerPixel)? {
        return Ok(samples);
    }
    Ok(1)
}

pub(crate) fn dtype_per_band<R: std::io::Read + std::io::Seek>(
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

pub(crate) fn nodata_per_band<R: std::io::Read + std::io::Seek>(
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

pub(crate) fn compression_name<R: std::io::Read + std::io::Seek>(
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

pub(crate) fn read_transform<R: std::io::Read + std::io::Seek>(
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

pub(crate) fn read_crs<R: std::io::Read + std::io::Seek>(
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
