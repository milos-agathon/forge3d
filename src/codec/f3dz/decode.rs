//! Fail-closed CPU F3DZ decoder.

use super::encode::PAGE_HEADER_LEN;
use super::format::{
    crc32, parse_prefix, PAGE_MAGIC, PREDICTOR_LORENZO, PREDICTOR_ORDER_ZERO, PREDICTOR_PLANE,
    VERSION,
};
use super::predict::{decode_residual_tokens, reconstruct_values, PlaneModel};
use super::rans::RansEncoded;
use super::{F3dzError, F3dzResult};

#[derive(Clone, Debug)]
pub struct DecodedDem {
    pub width: u32,
    pub height: u32,
    pub epsilon: f32,
    pub height_datum: String,
    pub base_quality: bool,
    pub values: Vec<f32>,
}

pub fn decode_dem(data: &[u8], demanded_epsilon: Option<f32>) -> F3dzResult<DecodedDem> {
    let (header, entries) = parse_prefix(data)?;
    if let Some(demanded) = demanded_epsilon {
        if demanded.to_bits() != header.epsilon.to_bits() {
            return Err(F3dzError::EpsilonMismatch {
                requested_bits: demanded.to_bits(),
                stream_bits: header.epsilon.to_bits(),
            });
        }
    }
    if header.progressive() {
        return Err(F3dzError::InvalidHeader(
            "progressive decode is implemented by the next codec layer".to_string(),
        ));
    }
    let output_len = (header.width as usize)
        .checked_mul(header.height as usize)
        .ok_or_else(|| F3dzError::InvalidHeader("DEM element count overflow".to_string()))?;
    let mut values = vec![f32::NAN; output_len];
    let mut expected_payload_offset = header.payload_offset;
    for (page_index, entry) in entries.iter().enumerate() {
        if entry.payload_offset != expected_payload_offset {
            return Err(F3dzError::CorruptPage {
                page: page_index,
                reason: format!(
                    "payload offset is not canonical/contiguous: expected={expected_payload_offset} actual={}",
                    entry.payload_offset
                ),
            });
        }
        let start = usize::try_from(entry.payload_offset).map_err(|_| F3dzError::CorruptPage {
            page: page_index,
            reason: "payload offset exceeds usize".to_string(),
        })?;
        let end = start
            .checked_add(entry.payload_len as usize)
            .ok_or_else(|| F3dzError::CorruptPage {
                page: page_index,
                reason: "payload range overflow".to_string(),
            })?;
        if data.len() < end {
            return Err(F3dzError::Truncated {
                needed: end,
                available: data.len(),
            });
        }
        let payload = &data[start..end];
        let actual_crc = crc32(payload);
        if actual_crc != entry.crc32 {
            return Err(F3dzError::CorruptPage {
                page: page_index,
                reason: format!(
                    "CRC mismatch: stored=0x{:08x} actual=0x{actual_crc:08x}",
                    entry.crc32
                ),
            });
        }
        let page = decode_page(
            payload,
            entry.width as usize,
            entry.height as usize,
            header.epsilon * 2.0,
        )
        .map_err(|error| F3dzError::CorruptPage {
            page: page_index,
            reason: error.to_string(),
        })?;
        if page.predictor_id != entry.predictor_id {
            return Err(F3dzError::CorruptPage {
                page: page_index,
                reason: "payload predictor disagrees with page index".to_string(),
            });
        }
        if page.nan_count != entry.nan_count {
            return Err(F3dzError::CorruptPage {
                page: page_index,
                reason: format!(
                    "payload nodata count {} disagrees with page index {}",
                    page.nan_count, entry.nan_count
                ),
            });
        }
        copy_page(
            &page.values,
            &mut values,
            header.width as usize,
            entry.page_x as usize * header.tile_size as usize,
            entry.page_y as usize * header.tile_size as usize,
            entry.width as usize,
            entry.height as usize,
        );
        expected_payload_offset = expected_payload_offset
            .checked_add(u64::from(entry.payload_len))
            .ok_or_else(|| F3dzError::CorruptPage {
                page: page_index,
                reason: "payload offset overflow".to_string(),
            })?;
    }
    if expected_payload_offset != data.len() as u64 {
        return Err(F3dzError::InvalidHeader(format!(
            "container has trailing or missing payload bytes: indexed_end={expected_payload_offset} actual={}",
            data.len()
        )));
    }
    let base_quality = header.base_only();
    Ok(DecodedDem {
        width: header.width,
        height: header.height,
        epsilon: header.epsilon,
        height_datum: header.height_datum,
        base_quality,
        values,
    })
}

struct DecodedPage {
    predictor_id: u8,
    nan_count: u32,
    values: Vec<f32>,
}

fn decode_page(
    payload: &[u8],
    width: usize,
    height: usize,
    expected_step: f32,
) -> F3dzResult<DecodedPage> {
    let sample_count = width
        .checked_mul(height)
        .ok_or_else(|| F3dzError::InvalidArgument("page sample count overflow".to_string()))?;
    require_len(payload, PAGE_HEADER_LEN)?;
    if payload[0..4] != PAGE_MAGIC {
        return Err(F3dzError::InvalidArgument(
            "page payload magic must be F3PG".to_string(),
        ));
    }
    if get_u16(payload, 4) != VERSION {
        return Err(F3dzError::InvalidArgument(
            "page payload version mismatch".to_string(),
        ));
    }
    if get_u16(payload, 6) != 0
        || get_u16(payload, 10) != 0
        || get_u32(payload, 28) != 0
        || get_u32(payload, 44) != 0
    {
        return Err(F3dzError::InvalidArgument(
            "non-progressive page has invalid flags/reserved fields".to_string(),
        ));
    }
    let predictor_id = payload[8];
    if predictor_id != payload[9]
        || !matches!(
            predictor_id,
            PREDICTOR_LORENZO | PREDICTOR_PLANE | PREDICTOR_ORDER_ZERO
        )
    {
        return Err(F3dzError::InvalidArgument(
            "invalid non-progressive predictor ids".to_string(),
        ));
    }
    if get_u32(payload, 32) as usize != sample_count {
        return Err(F3dzError::InvalidArgument(
            "page token count disagrees with page index".to_string(),
        ));
    }
    let step = f32::from_bits(get_u32(payload, 36));
    if !step.is_finite() || step <= 0.0 || get_u32(payload, 40) != step.to_bits() {
        return Err(F3dzError::InvalidArgument(
            "invalid non-progressive quantization step".to_string(),
        ));
    }
    if step.to_bits() != expected_step.to_bits() {
        return Err(F3dzError::InvalidArgument(format!(
            "page quantization step disagrees with container epsilon: page_bits=0x{:08x} expected_bits=0x{:08x}",
            step.to_bits(),
            expected_step.to_bits()
        )));
    }
    let layer_len = get_u32(payload, 24) as usize;
    let layer_end = PAGE_HEADER_LEN
        .checked_add(layer_len)
        .ok_or_else(|| F3dzError::InvalidArgument("page layer overflow".to_string()))?;
    if layer_end != payload.len() {
        return Err(F3dzError::InvalidArgument(
            "page layer length does not consume the payload".to_string(),
        ));
    }
    let (layer, consumed) = RansEncoded::from_bytes(&payload[PAGE_HEADER_LEN..layer_end])?;
    if consumed != layer_len {
        return Err(F3dzError::InvalidArgument(
            "rANS layer contains trailing bytes".to_string(),
        ));
    }
    let decoded = layer.decode()?;
    let expected_bytes = sample_count
        .checked_mul(4)
        .ok_or_else(|| F3dzError::InvalidArgument("page token bytes overflow".to_string()))?;
    if decoded.len() != expected_bytes {
        return Err(F3dzError::InvalidArgument(format!(
            "rANS decoded {} bytes, expected {expected_bytes}",
            decoded.len()
        )));
    }
    let tokens = decoded
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    let plane = PlaneModel {
        x_slope: get_i32(payload, 12),
        y_slope: get_i32(payload, 16),
        intercept: get_i32(payload, 20),
    };
    if predictor_id != PREDICTOR_PLANE && plane != PlaneModel::default() {
        return Err(F3dzError::InvalidArgument(
            "non-plane predictor carries non-zero plane coefficients".to_string(),
        ));
    }
    let quantized = decode_residual_tokens(&tokens, width, predictor_id, plane, None)?;
    let nan_count = quantized.iter().filter(|value| value.is_none()).count() as u32;
    Ok(DecodedPage {
        predictor_id,
        nan_count,
        values: reconstruct_values(&quantized, step),
    })
}

fn copy_page(
    page: &[f32],
    output: &mut [f32],
    output_width: usize,
    start_x: usize,
    start_y: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let source_start = y * width;
        let output_start = (start_y + y) * output_width + start_x;
        output[output_start..output_start + width]
            .copy_from_slice(&page[source_start..source_start + width]);
    }
}

fn require_len(data: &[u8], needed: usize) -> F3dzResult<()> {
    if data.len() < needed {
        Err(F3dzError::Truncated {
            needed,
            available: data.len(),
        })
    } else {
        Ok(())
    }
}

fn get_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn get_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn get_i32(data: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}
