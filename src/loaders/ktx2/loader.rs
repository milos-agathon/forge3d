use super::parser;
use super::types::*;
use super::vk_format::vk_format_to_wgpu;
use crate::core::compressed_textures::CompressedImage;
use crate::core::texture_layout::{mip_extent, TransferLayout};
use std::io::{Cursor, Read, Seek, SeekFrom};

/// KTX2 container loader.
pub struct Ktx2Loader;

impl Default for Ktx2Loader {
    fn default() -> Self {
        Self::new()
    }
}

impl Ktx2Loader {
    /// Create new KTX2 loader
    pub fn new() -> Self {
        Self
    }

    /// Load KTX2 file
    pub fn load_from_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<CompressedImage, String> {
        let data = std::fs::read(path).map_err(|e| format!("Failed to read KTX2 file: {}", e))?;

        self.load_from_memory(&data)
    }

    /// Load KTX2 from memory
    pub fn load_from_memory(&self, data: &[u8]) -> Result<CompressedImage, String> {
        let mut reader = Cursor::new(data);

        // Parse header
        let header = parser::parse_header(&mut reader)?;

        // This loader implements the single-layer 2D path only: array, cubemap
        // and volume containers index level data per layer/face and would be
        // misread by the mip walk below.
        if header.face_count != 1 || header.layer_count > 1 || header.pixel_depth != 1 {
            crate::core::degradation::record_degradation(
                "format_unsupported",
                "ktx2.texture_shape",
                "Only single-layer 2D KTX2 textures are supported; array, cubemap and 3D containers are refused",
            );
            return Err(format!(
                "KTX2 texture shape unsupported: depth {}, layers {}, faces {}",
                header.pixel_depth, header.layer_count, header.face_count
            ));
        }

        // Parse level indices
        let level_indices = parser::parse_level_indices(&mut reader, &header)?;

        // Parse data format descriptor
        let _dfd = if header.dfd_byte_length > 0 {
            Some(parser::parse_data_format_descriptor(&mut reader, &header)?)
        } else {
            None
        };

        // Parse key-value data
        let _kvd = if header.kvd_byte_length > 0 {
            Some(parser::parse_key_value_data(&mut reader, &header)?)
        } else {
            None
        };

        // Extract texture data
        let texture_data = self.extract_texture_data(&mut reader, &header, &level_indices)?;

        // Convert to WGPU format
        let wgpu_format = vk_format_to_wgpu(header.vk_format)?;

        Ok(CompressedImage {
            data: texture_data,
            width: header.pixel_width,
            height: header.pixel_height,
            mip_levels: header.level_count,
            format: wgpu_format,
            is_srgb: wgpu_format.is_srgb(),
            source_format: "KTX2".to_string(),
        })
    }

    /// Extract texture data with supercompression handling
    fn extract_texture_data(
        &self,
        reader: &mut Cursor<&[u8]>,
        header: &Ktx2Header,
        level_indices: &[Ktx2LevelIndex],
    ) -> Result<Vec<u8>, String> {
        if level_indices.is_empty() {
            return Err("No texture data found".to_string());
        }

        // Validate the supercompression scheme before doing any work so an
        // unknown scheme cannot silently masquerade as "no compression".
        let supercompression = SuperCompressionScheme::try_from(header.supercompression_scheme)
            .inspect_err(|e| {
                crate::core::degradation::record_degradation(
                    "format_unsupported",
                    "ktx2.unknown_supercompression",
                    e,
                );
            })?;

        // Map the container format up front so per-level payloads can be
        // validated against the expected compressed footprint.
        let wgpu_format = vk_format_to_wgpu(header.vk_format)?;
        let base_extent = wgpu::Extent3d {
            width: header.pixel_width,
            height: header.pixel_height,
            depth_or_array_layers: 1,
        };

        // Render every mip level into a single packed buffer. KTX2 stores
        // levels from highest resolution (level 0) downward, so we preserve
        // that order; each level is exactly its format footprint, which is the
        // contract `CompressedImage::upload_to_texture` relies on.
        let data = reader.get_ref();
        let data_len = data.len() as u64;

        let mut combined: Vec<u8> = Vec::new();
        for (mip, level) in level_indices.iter().enumerate() {
            let mip = mip as u32;
            let extent = mip_extent(base_extent, wgpu::TextureDimension::D2, mip);
            let expected = TransferLayout::queue_write(wgpu_format, wgpu::TextureAspect::All, extent)
                .map(|layout| layout.payload_bytes)
                .map_err(|e| e.to_string())?;

            // Overflow-safe span check against the actual container.
            let end = level
                .byte_offset
                .checked_add(level.byte_length)
                .ok_or_else(|| "KTX2 level byte span overflows u64".to_string())?;
            if end > data_len {
                return Err(format!(
                    "KTX2 level {} byte span ({}..{}) exceeds container size {}",
                    mip, level.byte_offset, end, data_len
                ));
            }

            // Tie declared sizes against the footprint BEFORE allocating:
            // uncompressed levels must be exactly one footprint, and
            // supercompressed levels must declare that many decoded bytes.
            // This is what stops a small file with an attacker-controlled
            // level length from driving a huge allocation.
            if supercompression == SuperCompressionScheme::None {
                if level.byte_length != expected {
                    return Err(format!(
                        "KTX2 level {} byte length {} does not match the {:?} footprint {} for {}x{}",
                        mip,
                        level.byte_length,
                        wgpu_format,
                        expected,
                        extent.width,
                        extent.height
                    ));
                }
            } else if level.uncompressed_byte_length != expected {
                return Err(format!(
                    "KTX2 level {} declares {} decoded bytes but the {:?} footprint for {}x{} is {}",
                    mip,
                    level.uncompressed_byte_length,
                    wgpu_format,
                    extent.width,
                    extent.height,
                    expected
                ));
            }

            // Convert the (now footprint-bounded) claimed length without
            // truncation on 32-bit targets.
            let claimed = usize::try_from(level.byte_length)
                .map_err(|_| "KTX2 level byte length exceeds addressable size".to_string())?;

            let mut level_data = vec![0u8; claimed];
            reader
                .seek(SeekFrom::Start(level.byte_offset))
                .map_err(|e| format!("Failed to seek to texture data: {}", e))?;
            reader
                .read_exact(&mut level_data)
                .map_err(|e| format!("Failed to read texture data: {}", e))?;

            // Supercompression is applied per level; for schemes without a
            // wired decoder this returns an error and records a degradation.
            let decoded = match supercompression {
                SuperCompressionScheme::None => level_data,
                SuperCompressionScheme::BasisLZ => {
                    self.transcode_basis_universal(level_data, header)?
                }
                SuperCompressionScheme::ZStandard => self.decompress_zstd(level_data)?,
                SuperCompressionScheme::ZLIB => self.decompress_zlib(level_data)?,
            };
            if decoded.len() as u64 != expected {
                return Err(format!(
                    "KTX2 level {} decoded to {} bytes but the {:?} footprint for {}x{} is {}",
                    mip,
                    decoded.len(),
                    wgpu_format,
                    extent.width,
                    extent.height,
                    expected
                ));
            }
            combined.extend_from_slice(&decoded);
        }

        Ok(combined)
    }

    /// Transcode Basis Universal data
    fn transcode_basis_universal(
        &self,
        _data: Vec<u8>,
        _header: &Ktx2Header,
    ) -> Result<Vec<u8>, String> {
        // Stub for Basis Universal transcoding; returns an error until integrated.
        crate::core::degradation::record_degradation(
            "format_unsupported",
            "ktx2.basis_universal",
            "Basis Universal transcoding is not implemented; the texture cannot be decoded",
        );
        Err("Basis Universal transcoding not implemented".to_string())
    }

    /// Decompress ZSTD data
    fn decompress_zstd(&self, _data: Vec<u8>) -> Result<Vec<u8>, String> {
        // Stub for ZSTD decompression; returns an error until integrated.
        crate::core::degradation::record_degradation(
            "format_unsupported",
            "ktx2.zstd",
            "ZSTD supercompression decoding is not implemented; the texture cannot be decoded",
        );
        Err("ZSTD decompression not implemented".to_string())
    }

    /// Decompress ZLIB data
    fn decompress_zlib(&self, _data: Vec<u8>) -> Result<Vec<u8>, String> {
        // Stub for ZLIB decompression; returns an error until integrated.
        crate::core::degradation::record_degradation(
            "format_unsupported",
            "ktx2.zlib",
            "ZLIB supercompression decoding is not implemented; the texture cannot be decoded",
        );
        Err("ZLIB decompression not implemented".to_string())
    }
}
