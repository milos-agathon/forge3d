// src/core/hdr_tonemapping.rs
// CPU-side tone mapping implementation
// RELEVANT FILES: shaders/tonemap.wgsl

use super::hdr_types::ToneMappingOperator;
use glam::Vec3;

/// Apply CPU-side tone mapping to HDR data
pub fn apply_cpu_tone_mapping(
    hdr_data: &[f32],
    width: u32,
    height: u32,
    operator: ToneMappingOperator,
    exposure: f32,
    white_point: f32,
    gamma: f32,
) -> Vec<u8> {
    let mut ldr_data = Vec::with_capacity((width * height * 4) as usize);

    for chunk in hdr_data.chunks(4) {
        let hdr = Vec3::new(chunk[0], chunk[1], chunk[2]) * exposure;

        let tone_mapped = match operator {
            ToneMappingOperator::Reinhard => reinhard(hdr),
            ToneMappingOperator::ReinhardExtended => reinhard_extended(hdr, white_point),
            ToneMappingOperator::Aces => aces_filmic(hdr),
            ToneMappingOperator::Uncharted2 => uncharted2(hdr, white_point),
            ToneMappingOperator::Exposure => exposure_mapping(hdr),
        };

        // Apply gamma correction
        let gamma_corrected = tone_mapped.powf(1.0 / gamma);

        // Convert to 8-bit
        let r = (gamma_corrected.x.clamp(0.0, 1.0) * 255.0) as u8;
        let g = (gamma_corrected.y.clamp(0.0, 1.0) * 255.0) as u8;
        let b = (gamma_corrected.z.clamp(0.0, 1.0) * 255.0) as u8;
        let a = (chunk[3].clamp(0.0, 1.0) * 255.0) as u8;

        ldr_data.extend_from_slice(&[r, g, b, a]);
    }

    ldr_data
}

/// Reinhard tone mapping: color / (color + 1)
fn reinhard(hdr: Vec3) -> Vec3 {
    hdr / (hdr + Vec3::ONE)
}

/// Extended Reinhard: color * (1 + color/white²) / (1 + color)
fn reinhard_extended(hdr: Vec3, white_point: f32) -> Vec3 {
    let white_sq = white_point * white_point;
    hdr * (Vec3::ONE + hdr / white_sq) / (Vec3::ONE + hdr)
}

/// ACES filmic tone mapping approximation
fn aces_filmic(hdr: Vec3) -> Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    (hdr * (hdr * a + b)) / (hdr * (hdr * c + d) + e)
}

/// Uncharted 2 filmic tone mapping
fn uncharted2(hdr: Vec3, white_point: f32) -> Vec3 {
    fn uncharted2_tonemap_partial(x: Vec3) -> Vec3 {
        let a = 0.15;
        let b = 0.50;
        let c = 0.10;
        let d = 0.20;
        let e = 0.02;
        let f = 0.30;
        ((x * (x * a + Vec3::splat(c * b)) + Vec3::splat(d * e))
            / (x * (x * a + b) + Vec3::splat(d * f)))
            - Vec3::splat(e / f)
    }

    let curr = uncharted2_tonemap_partial(hdr);
    let white_scale = Vec3::ONE / uncharted2_tonemap_partial(Vec3::splat(white_point));
    curr * white_scale
}

/// Simple exposure mapping
fn exposure_mapping(hdr: Vec3) -> Vec3 {
    Vec3::ONE - (-hdr).exp()
}
