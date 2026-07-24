//! Fail-closed CPU F3DZ decoder.

use super::encode::PAGE_HEADER_LEN;
use super::format::{
    crc32, parse_prefix, PAGE_FLAG_BASE_ONLY, PAGE_FLAG_PROGRESSIVE, PAGE_MAGIC, PREDICTOR_LORENZO,
    PREDICTOR_ORDER_ZERO, PREDICTOR_PLANE, PREDICTOR_PREVIOUS_LOD, VERSION,
};
use super::predict::{
    decode_residual_tokens, denormalize_residual_tokens, reconstruct_values,
    unpack_residual_tokens, PlaneModel,
};
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
            if header.progressive() {
                header.epsilon * 8.0
            } else {
                header.epsilon * 2.0
            },
            header.progressive(),
            header.base_only(),
            entry.base_layer_len as usize,
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
    if base_quality {
        crate::core::degradation::record_degradation(
            "base_quality",
            "f3dz_unrefined_pages",
            "terrain heights were decoded from the progressive base layer at a declared 4*epsilon bound",
        );
    }
    crate::core::certificate::record_f3dz_pages(header.epsilon, header.page_count, base_quality);
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
    expected_fine_step: f32,
    expected_base_step: f32,
    progressive: bool,
    base_only: bool,
    indexed_base_layer_len: usize,
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
    let expected_flags = if progressive {
        PAGE_FLAG_PROGRESSIVE | if base_only { PAGE_FLAG_BASE_ONLY } else { 0 }
    } else {
        0
    };
    if get_u16(payload, 6) != u16::from(expected_flags)
        || get_u16(payload, 10) != 0
        || get_u32(payload, 52) != 0
    {
        return Err(F3dzError::InvalidArgument(
            "page payload flags/reserved fields disagree with its container".to_string(),
        ));
    }
    let predictor_id = payload[8];
    let enhancement_predictor = payload[9];
    if !matches!(
        predictor_id,
        PREDICTOR_LORENZO | PREDICTOR_PLANE | PREDICTOR_ORDER_ZERO
    ) || (progressive && enhancement_predictor != PREDICTOR_PREVIOUS_LOD)
        || (!progressive && enhancement_predictor != predictor_id)
    {
        return Err(F3dzError::InvalidArgument(
            "invalid base/enhancement predictor ids".to_string(),
        ));
    }
    if get_u32(payload, 32) as usize != sample_count {
        return Err(F3dzError::InvalidArgument(
            "page token count disagrees with page index".to_string(),
        ));
    }
    let fine_step = f32::from_bits(get_u32(payload, 36));
    let base_step = f32::from_bits(get_u32(payload, 40));
    let base_scale = get_u32(payload, 44);
    let enhancement_scale = get_u32(payload, 48);
    if !fine_step.is_finite() || fine_step <= 0.0 || !base_step.is_finite() || base_step <= 0.0 {
        return Err(F3dzError::InvalidArgument(
            "invalid page quantization steps".to_string(),
        ));
    }
    if fine_step.to_bits() != expected_fine_step.to_bits()
        || base_step.to_bits() != expected_base_step.to_bits()
    {
        return Err(F3dzError::InvalidArgument(format!(
            "page quantization steps disagree with container epsilon: fine=0x{:08x}/0x{:08x} base=0x{:08x}/0x{:08x}",
            fine_step.to_bits(),
            expected_fine_step.to_bits(),
            base_step.to_bits(),
            expected_base_step.to_bits()
        )));
    }
    if base_scale == 0
        || base_scale > i32::MAX as u32
        || enhancement_scale == 0
        || enhancement_scale > i32::MAX as u32
    {
        return Err(F3dzError::InvalidArgument(
            "page residual scales must be in 1..=i32::MAX".to_string(),
        ));
    }
    let base_layer_len = get_u32(payload, 24) as usize;
    let enhancement_layer_len = get_u32(payload, 28) as usize;
    let base_end = PAGE_HEADER_LEN
        .checked_add(base_layer_len)
        .ok_or_else(|| F3dzError::InvalidArgument("base page layer overflow".to_string()))?;
    let payload_end = base_end
        .checked_add(enhancement_layer_len)
        .ok_or_else(|| F3dzError::InvalidArgument("enhancement page layer overflow".to_string()))?;
    if base_end != indexed_base_layer_len || payload_end != payload.len() {
        return Err(F3dzError::InvalidArgument(
            "page layer lengths disagree with the index/payload".to_string(),
        ));
    }
    if (base_only || !progressive) && enhancement_layer_len != 0 {
        return Err(F3dzError::InvalidArgument(format!(
            "page quality mode forbids enhancement bytes: {enhancement_layer_len}"
        )));
    }
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
    let base_tokens = decode_token_layer(&payload[PAGE_HEADER_LEN..base_end], sample_count)?;
    let base_tokens = denormalize_residual_tokens(&base_tokens, base_scale)?;
    let base_quantized = decode_residual_tokens(&base_tokens, width, predictor_id, plane, None)?;
    if base_quantized.len() != sample_count {
        return Err(F3dzError::InvalidArgument(format!(
            "base predictor decoded {} samples, expected {sample_count}",
            base_quantized.len()
        )));
    }
    let quantized = if progressive && !base_only {
        if enhancement_layer_len == 0 {
            return Err(F3dzError::InvalidArgument(
                "refined progressive page is missing its enhancement layer".to_string(),
            ));
        }
        let enhancement_tokens = decode_token_layer(&payload[base_end..payload_end], sample_count)?;
        let enhancement_tokens =
            denormalize_residual_tokens(&enhancement_tokens, enhancement_scale)?;
        let decoded = decode_residual_tokens(
            &enhancement_tokens,
            width,
            enhancement_predictor,
            PlaneModel::default(),
            Some(&base_quantized),
        )?;
        if decoded.len() != sample_count {
            return Err(F3dzError::InvalidArgument(format!(
                "enhancement predictor decoded {} samples, expected {sample_count}",
                decoded.len()
            )));
        }
        decoded
    } else {
        base_quantized
    };
    let nan_count = quantized
        .iter()
        .filter(|value| matches!(value, super::predict::QuantizedSample::Nan(_)))
        .count() as u32;
    Ok(DecodedPage {
        predictor_id,
        nan_count,
        values: reconstruct_values(&quantized, if base_only { base_step } else { fine_step }),
    })
}

fn decode_token_layer(data: &[u8], sample_count: usize) -> F3dzResult<Vec<u32>> {
    let (layer, consumed) = RansEncoded::from_bytes(data)?;
    if consumed != data.len() {
        return Err(F3dzError::InvalidArgument(
            "rANS layer contains trailing bytes".to_string(),
        ));
    }
    let decoded = layer.decode()?;
    let maximum_bytes = sample_count
        .checked_mul(5)
        .ok_or_else(|| F3dzError::InvalidArgument("page escape bytes overflow".to_string()))?;
    if !(1..=maximum_bytes).contains(&decoded.len()) {
        return Err(F3dzError::InvalidArgument(format!(
            "rANS decoded {} bytes, expected 1..={maximum_bytes}",
            decoded.len(),
        )));
    }
    unpack_residual_tokens(&decoded, sample_count)
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
