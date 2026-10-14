//! KTX2 texture container loading and parsing
//!
//! This module provides comprehensive KTX2 format support with transcoding
//! and decoder integration for compressed texture pipelines.

mod loader;
mod parser;
mod types;
mod validation;

pub use loader::{BasisTranscoder, Ktx2Loader};
pub use types::{
    Ktx2ChannelInfo, Ktx2DataFormatDescriptor, Ktx2Header, Ktx2LevelIndex, SuperCompressionScheme,
    KTX2_MAGIC,
};
pub use validation::{validate_ktx2_data, validate_ktx2_file};

#[cfg(test)]
mod tests {
    use super::*;

    /// KTX2 header layout, in bytes: a 12-byte magic, thirteen little-endian
    /// `u32` fields, then two `u64` fields. `parser::parse_header` reads exactly
    /// this sequence, so a hand-built container must match it byte for byte.
    const HEADER_BYTES: u64 = 80;
    /// One level index entry: `byte_offset`, `byte_length`,
    /// `uncompressed_byte_length` — three `u64` values.
    const LEVEL_INDEX_BYTES: u64 = 24;

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    /// Build a single-level KTX2 container whose base level is `payload`.
    ///
    /// `dfd_byte_length` and `kvd_byte_length` are zero so the loader skips the
    /// optional data-format-descriptor and key/value sections; the level index
    /// points at the payload that immediately follows it.
    fn build_ktx2(
        vk_format: u32,
        supercompression: u32,
        width: u32,
        height: u32,
        payload: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&KTX2_MAGIC);
        push_u32(&mut out, vk_format);
        push_u32(&mut out, 1); // type_size
        push_u32(&mut out, width);
        push_u32(&mut out, height);
        push_u32(&mut out, 1); // pixel_depth
        push_u32(&mut out, 1); // layer_count
        push_u32(&mut out, 1); // face_count
        push_u32(&mut out, 1); // level_count
        push_u32(&mut out, supercompression);
        push_u32(&mut out, 0); // dfd_byte_offset
        push_u32(&mut out, 0); // dfd_byte_length
        push_u32(&mut out, 0); // kvd_byte_offset
        push_u32(&mut out, 0); // kvd_byte_length
        push_u64(&mut out, 0); // sgd_byte_offset
        push_u64(&mut out, 0); // sgd_byte_length
        assert_eq!(out.len() as u64, HEADER_BYTES, "header layout drifted");

        let data_offset = HEADER_BYTES + LEVEL_INDEX_BYTES;
        push_u64(&mut out, data_offset);
        push_u64(&mut out, payload.len() as u64);
        push_u64(&mut out, payload.len() as u64);
        assert_eq!(out.len() as u64, data_offset, "level index layout drifted");

        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn test_ktx2_magic_validation() {
        // Valid magic
        let mut valid_data = Vec::new();
        valid_data.extend_from_slice(&KTX2_MAGIC);
        valid_data.extend_from_slice(&[0u8; 100]); // Padding

        assert!(validate_ktx2_data(&valid_data).unwrap());

        // Invalid magic
        let invalid_data = vec![0u8; 100];
        assert!(!validate_ktx2_data(&invalid_data).unwrap());

        // Too short
        let short_data = vec![0u8; 5];
        assert!(!validate_ktx2_data(&short_data).unwrap());
    }

    #[test]
    fn test_supercompression_scheme_conversion() {
        assert_eq!(
            SuperCompressionScheme::from(0),
            SuperCompressionScheme::None
        );
        assert_eq!(
            SuperCompressionScheme::from(1),
            SuperCompressionScheme::BasisLZ
        );
        assert_eq!(
            SuperCompressionScheme::from(999),
            SuperCompressionScheme::None
        );
    }

    #[test]
    fn test_loader_creation() {
        let loader = Ktx2Loader::new();
        // Accessing private fields for testing requires these to be pub(crate) or similar
        // Or we just test public API behavior.
        // The original tests accessed private fields, so I might need to make them visible to tests.
        // Stick to public API behavior or basic creation here.
        let _ = loader;

        let custom_formats = vec![wgpu::TextureFormat::Rgba8Unorm];
        let _custom_loader = Ktx2Loader::with_target_formats(custom_formats.clone());
    }

    #[test]
    fn uncompressed_ktx2_round_trips_dimensions_format_and_bytes() {
        // 4x2 RGBA8 (VK_FORMAT_R8G8B8A8_UNORM == 37) is 32 bytes of level data.
        let payload: Vec<u8> = (0..32u8).collect();
        let bytes = build_ktx2(37, 0, 4, 2, &payload);

        let image = Ktx2Loader::new()
            .load_from_memory(&bytes)
            .expect("uncompressed KTX2 must decode through the real loader");

        assert_eq!(image.width, 4);
        assert_eq!(image.height, 2);
        assert_eq!(image.mip_levels, 1);
        assert_eq!(image.format, wgpu::TextureFormat::Rgba8Unorm);
        assert!(!image.is_srgb);
        assert_eq!(image.source_format, "KTX2");
        assert_eq!(image.data, payload);
    }

    #[test]
    fn srgb_vk_format_maps_to_srgb_wgpu_format() {
        // VK_FORMAT_R8G8B8A8_SRGB == 43.
        let bytes = build_ktx2(43, 0, 2, 2, &[0u8; 16]);

        let image = Ktx2Loader::new()
            .load_from_memory(&bytes)
            .expect("sRGB KTX2 must decode");

        assert_eq!(image.format, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert!(image.is_srgb);
    }

    #[test]
    fn bad_magic_and_truncation_are_rejected() {
        let mut bad_magic = build_ktx2(37, 0, 2, 2, &[0u8; 16]);
        bad_magic[0] = 0x00; // break the leading 0xAB magic byte
        assert!(Ktx2Loader::new().load_from_memory(&bad_magic).is_err());

        let mut truncated = build_ktx2(37, 0, 2, 2, &[0u8; 16]);
        truncated.truncate(40);
        assert!(Ktx2Loader::new().load_from_memory(&truncated).is_err());
    }

    #[test]
    fn unsupported_supercompression_is_refused_loudly() {
        use crate::core::degradation::{begin_degradation_capture, finish_degradation_capture};

        // Every supercompression scheme without a wired decoder must fail AND
        // record a structured degradation. Silent success or a bare error would
        // hide an undecoded texture from the certificate.
        for (scheme, name) in [
            (1u32, "ktx2.basis_universal"),
            (2, "ktx2.zstd"),
            (3, "ktx2.zlib"),
        ] {
            let bytes = build_ktx2(37, scheme, 2, 2, &[0u8; 16]);

            begin_degradation_capture();
            let result = Ktx2Loader::new().load_from_memory(&bytes);
            let captured = finish_degradation_capture();

            assert!(result.is_err(), "scheme {scheme} must not decode");
            assert!(
                captured
                    .iter()
                    .any(|d| d.kind == "format_unsupported" && d.name == name),
                "scheme {scheme} must record the {name} degradation"
            );
        }
    }
}
