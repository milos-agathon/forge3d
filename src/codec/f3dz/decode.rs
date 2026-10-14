//! Fail-closed CPU F3DZ decoder.

use super::format::{parse_prefix, validate_pages, ValidatedPage};
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
    let pages = validate_pages(data, &header, entries)?;
    let output_len = (header.width as usize)
        .checked_mul(header.height as usize)
        .ok_or_else(|| F3dzError::InvalidHeader("DEM element count overflow".to_string()))?;
    let mut values = vec![f32::NAN; output_len];
    for (page_index, validated) in pages.iter().enumerate() {
        let entry = &validated.entry;
        let page =
            decode_page(validated, header.progressive(), header.base_only()).map_err(|error| {
                F3dzError::CorruptPage {
                    page: page_index,
                    reason: error.to_string(),
                }
            })?;
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
    nan_count: u32,
    values: Vec<f32>,
}

fn decode_page(
    page: &ValidatedPage<'_>,
    progressive: bool,
    base_only: bool,
) -> F3dzResult<DecodedPage> {
    let width = usize::from(page.entry.width);
    let sample_count = page.entry.sample_count as usize;
    let predictor_id = page.entry.predictor_id;
    let plane = page.plane;
    let base_scale = page.base_scale;
    let enhancement_scale = page.enhancement_scale;
    let base_tokens = decode_token_layer(page.base_layer, sample_count)?;
    let base_tokens = denormalize_residual_tokens(&base_tokens, base_scale)?;
    let base_quantized = decode_residual_tokens(&base_tokens, width, predictor_id, plane, None)?;
    if base_quantized.len() != sample_count {
        return Err(F3dzError::InvalidArgument(format!(
            "base predictor decoded {} samples, expected {sample_count}",
            base_quantized.len()
        )));
    }
    let quantized = if progressive && !base_only {
        let enhancement_tokens = decode_token_layer(page.enhancement_layer, sample_count)?;
        let enhancement_tokens =
            denormalize_residual_tokens(&enhancement_tokens, enhancement_scale)?;
        let decoded = decode_residual_tokens(
            &enhancement_tokens,
            width,
            page.enhancement_predictor,
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
        nan_count,
        values: reconstruct_values(
            &quantized,
            if base_only {
                page.base_step
            } else {
                page.fine_step
            },
        ),
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
