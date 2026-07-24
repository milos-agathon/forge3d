//! CPU F3DZ encoder.

use super::format::{
    crc32, write_prefix, ContainerHeader, PageIndexEntry, PAGE_MAGIC, PREDICTOR_LORENZO,
    PREDICTOR_PLANE, VERSION,
};
use super::predict::{
    decode_residual_tokens, encode_residual_tokens, fit_least_squares_plane, quantize_source,
    reconstruct_values, PlaneModel,
};
use super::rans::RansEncoded;
use super::{F3dzError, F3dzResult};

pub(crate) const PAGE_HEADER_LEN: usize = 48;

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    pub epsilon: f32,
    pub progressive: bool,
    pub tile_size: u16,
    pub height_datum: String,
}

impl EncodeOptions {
    pub fn new(epsilon: f32) -> Self {
        Self {
            epsilon,
            progressive: false,
            tile_size: super::format::MAX_GPU_PAGE_SIZE,
            height_datum: String::new(),
        }
    }
}

pub fn encode_dem(
    values: &[f32],
    width: u32,
    height: u32,
    options: &EncodeOptions,
) -> F3dzResult<Vec<u8>> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| F3dzError::InvalidArgument("DEM element count overflow".to_string()))?;
    if values.len() != expected {
        return Err(F3dzError::InvalidArgument(format!(
            "DEM contains {} values, expected {expected}",
            values.len()
        )));
    }
    if options.progressive {
        return Err(F3dzError::InvalidArgument(
            "progressive encoding is implemented by the next codec layer".to_string(),
        ));
    }
    let header = ContainerHeader::new(
        width,
        height,
        options.tile_size,
        options.epsilon,
        false,
        options.height_datum.clone(),
    )?;
    let pages_x = width.div_ceil(u32::from(options.tile_size));
    let pages_y = height.div_ceil(u32::from(options.tile_size));
    let mut payloads = Vec::with_capacity(header.page_count as usize);
    let mut entries = Vec::with_capacity(header.page_count as usize);
    let mut payload_offset = header.payload_offset;
    for page_y in 0..pages_y {
        for page_x in 0..pages_x {
            let page_width =
                (width - page_x * u32::from(options.tile_size)).min(u32::from(options.tile_size));
            let page_height =
                (height - page_y * u32::from(options.tile_size)).min(u32::from(options.tile_size));
            let source = extract_page(
                values,
                width as usize,
                page_x as usize * options.tile_size as usize,
                page_y as usize * options.tile_size as usize,
                page_width as usize,
                page_height as usize,
            );
            let encoded = encode_page(
                &source,
                page_width as usize,
                page_height as usize,
                options.epsilon,
            )?;
            let payload_len = u32::try_from(encoded.payload.len()).map_err(|_| {
                F3dzError::InvalidArgument("page payload exceeds u32 length".to_string())
            })?;
            let index = PageIndexEntry {
                page_x,
                page_y,
                width: page_width as u16,
                height: page_height as u16,
                predictor_id: encoded.predictor_id,
                flags: 0,
                payload_offset,
                payload_len,
                base_layer_len: payload_len,
                crc32: crc32(&encoded.payload),
                max_abs_err: encoded.max_abs_err,
                base_max_abs_err: encoded.max_abs_err,
                sample_count: page_width * page_height,
                nan_count: source.iter().filter(|value| value.is_nan()).count() as u32,
            };
            payload_offset = payload_offset
                .checked_add(u64::from(payload_len))
                .ok_or_else(|| {
                    F3dzError::InvalidArgument("container payload overflow".to_string())
                })?;
            entries.push(index);
            payloads.push(encoded.payload);
        }
    }
    let mut out = write_prefix(&header, &entries)?;
    for payload in payloads {
        out.extend_from_slice(&payload);
    }
    Ok(out)
}

struct EncodedPage {
    predictor_id: u8,
    max_abs_err: f32,
    payload: Vec<u8>,
}

fn encode_page(
    source: &[f32],
    width: usize,
    height: usize,
    epsilon: f32,
) -> F3dzResult<EncodedPage> {
    let step = epsilon * 2.0;
    let quantized = quantize_source(source, step, epsilon)?;
    let plane = fit_least_squares_plane(&quantized, width, height);
    let lorenzo_tokens = encode_residual_tokens(
        &quantized,
        width,
        PREDICTOR_LORENZO,
        PlaneModel::default(),
        None,
    )?;
    let plane_tokens = encode_residual_tokens(&quantized, width, PREDICTOR_PLANE, plane, None)?;
    let lorenzo_layer = encode_token_layer(&lorenzo_tokens)?;
    let plane_layer = encode_token_layer(&plane_tokens)?;
    let (predictor_id, selected_plane, tokens, layer) = if plane_layer.len() < lorenzo_layer.len() {
        (PREDICTOR_PLANE, plane, plane_tokens, plane_layer)
    } else {
        (
            PREDICTOR_LORENZO,
            PlaneModel::default(),
            lorenzo_tokens,
            lorenzo_layer,
        )
    };
    let decoded_quantized =
        decode_residual_tokens(&tokens, width, predictor_id, selected_plane, None)?;
    let reconstructed = reconstruct_values(&decoded_quantized, step);
    let max_abs_err = checked_max_error(source, &reconstructed, epsilon)?;
    let mut payload = vec![0u8; PAGE_HEADER_LEN];
    payload[0..4].copy_from_slice(&PAGE_MAGIC);
    put_u16(&mut payload, 4, VERSION);
    payload[8] = predictor_id;
    payload[9] = predictor_id;
    put_i32(&mut payload, 12, selected_plane.x_slope);
    put_i32(&mut payload, 16, selected_plane.y_slope);
    put_i32(&mut payload, 20, selected_plane.intercept);
    put_u32(
        &mut payload,
        24,
        u32::try_from(layer.len())
            .map_err(|_| F3dzError::InvalidArgument("rANS layer too large".to_string()))?,
    );
    put_u32(&mut payload, 32, source.len() as u32);
    put_u32(&mut payload, 36, step.to_bits());
    put_u32(&mut payload, 40, step.to_bits());
    payload.extend_from_slice(&layer);
    Ok(EncodedPage {
        predictor_id,
        max_abs_err,
        payload,
    })
}

pub(crate) fn encode_token_layer(tokens: &[u32]) -> F3dzResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    RansEncoded::encode(&bytes)?.to_bytes()
}

fn extract_page(
    values: &[f32],
    full_width: usize,
    start_x: usize,
    start_y: usize,
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut page = Vec::with_capacity(width * height);
    for y in 0..height {
        let start = (start_y + y) * full_width + start_x;
        page.extend_from_slice(&values[start..start + width]);
    }
    page
}

pub(crate) fn checked_max_error(
    source: &[f32],
    reconstructed: &[f32],
    bound: f32,
) -> F3dzResult<f32> {
    if source.len() != reconstructed.len() {
        return Err(F3dzError::InvalidArgument(
            "source/reconstruction length mismatch".to_string(),
        ));
    }
    let mut max_error = 0.0f32;
    for (index, (&source, &reconstructed)) in source.iter().zip(reconstructed).enumerate() {
        if source.is_nan() {
            if !reconstructed.is_nan() {
                return Err(F3dzError::InvalidArgument(format!(
                    "nodata did not round-trip at sample {index}"
                )));
            }
            continue;
        }
        let error = (source - reconstructed).abs();
        if !error.is_finite() || error > bound {
            return Err(F3dzError::InvalidArgument(format!(
                "error bound violated at sample {index}: error={error} bound={bound}"
            )));
        }
        max_error = max_error.max(error);
    }
    Ok(max_error)
}

fn put_u16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(data: &mut [u8], offset: usize, value: i32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::f3dz::decode::decode_dem;

    fn terrain(width: usize, height: usize) -> Vec<f32> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    if x == 7 && y == 9 {
                        f32::NAN
                    } else {
                        1200.0 + x as f32 * 1.75 - y as f32 * 0.625
                            + ((x * y) % 17) as f32 * 0.03125
                    }
                })
            })
            .collect()
    }

    #[test]
    fn cpu_codec_is_deterministic_and_error_bounded() {
        let source = terrain(131, 70);
        let options = EncodeOptions::new(0.05);
        let first = encode_dem(&source, 131, 70, &options).unwrap();
        let second = encode_dem(&source, 131, 70, &options).unwrap();
        assert_eq!(first, second);
        let decoded = decode_dem(&first, Some(0.05)).unwrap();
        assert_eq!((decoded.width, decoded.height), (131, 70));
        assert_eq!(decoded.values.len(), source.len());
        checked_max_error(&source, &decoded.values, 0.05).unwrap();
        assert!(decoded.values[9 * 131 + 7].is_nan());
    }

    #[test]
    fn demanded_epsilon_must_match_stream_bits() {
        let source = terrain(8, 8);
        let encoded = encode_dem(&source, 8, 8, &EncodeOptions::new(0.1)).unwrap();
        assert!(matches!(
            decode_dem(&encoded, Some(0.2)),
            Err(F3dzError::EpsilonMismatch { .. })
        ));
    }

    #[test]
    fn corrupt_page_crc_fails_closed() {
        let source = terrain(8, 8);
        let mut encoded = encode_dem(&source, 8, 8, &EncodeOptions::new(0.1)).unwrap();
        let last = encoded.len() - 1;
        encoded[last] ^= 1;
        assert!(matches!(
            decode_dem(&encoded, None),
            Err(F3dzError::CorruptPage { reason, .. }) if reason.contains("CRC mismatch")
        ));
    }
}
