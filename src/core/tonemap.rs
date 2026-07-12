//! Authoritative CPU resolve shared by the adjudication render paths.

/// Resolve a linear HDR RGBA buffer (f32, RGBA interleaved) to sRGB-encoded RGBA8.
///
/// AUTHORITATIVE TONEMAP OPERATOR: Reinhard with pre-exposure, `x / (1 + x)` with
/// `x = rgb * exposure`, matching `tonemap_reinhard` in `src/shaders/pt_kernel.wgsl`.
/// Both the adjudication path-traced resolve and the adjudication raster resolve
/// MUST call this single function with the same exposure so dE2000 is computed on
/// matched signals (tonemap parity by construction). Output is encoded with the
/// piecewise IEC 61966-2-1 sRGB curve; alpha is forced to 255.
pub fn resolve_reference_hdr_to_rgba8(hdr_rgba: &[f32], exposure: f32) -> Vec<u8> {
    fn srgb_encode(c: f32) -> f32 {
        if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    }
    let mut out = Vec::with_capacity(hdr_rgba.len());
    for px in hdr_rgba.chunks_exact(4) {
        for &c in &px[..3] {
            let x = c.max(0.0) * exposure;
            let t = x / (1.0 + x);
            let s = srgb_encode(t).clamp(0.0, 1.0);
            out.push((s * 255.0 + 0.5) as u8);
        }
        out.push(255u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::resolve_reference_hdr_to_rgba8;

    #[test]
    fn resolve_black_and_saturation() {
        let out = resolve_reference_hdr_to_rgba8(&[0.0, 0.0, 0.0, 1.0], 1.0);
        assert_eq!(out, vec![0, 0, 0, 255]);
        let bright = resolve_reference_hdr_to_rgba8(&[1e6, 1e6, 1e6, 1.0], 1.0);
        assert_eq!(&bright[..3], &[255, 255, 255]);
    }

    #[test]
    fn resolve_midpoint_matches_reinhard_srgb() {
        let out = resolve_reference_hdr_to_rgba8(&[1.0, 1.0, 1.0, 1.0], 1.0);
        assert_eq!(&out[..3], &[188, 188, 188]);
    }

    #[test]
    fn resolve_monotone_in_exposure() {
        let lo = resolve_reference_hdr_to_rgba8(&[0.25, 0.25, 0.25, 1.0], 0.5);
        let hi = resolve_reference_hdr_to_rgba8(&[0.25, 0.25, 0.25, 1.0], 2.0);
        assert!(hi[0] > lo[0]);
    }
}
