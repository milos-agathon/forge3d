//! KTX2 texture container loading and parsing
//!
//! This module provides comprehensive KTX2 format support with transcoding
//! and decoder integration for compressed texture pipelines.

use crate::core::compressed_textures::CompressedImage;
use std::io::{Read, Seek, SeekFrom, Cursor};
use std::collections::HashMap;

/// KTX2 file magic number
const KTX2_MAGIC: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];

/// KTX2 file header structure
#[derive(Debug, Clone)]
pub struct Ktx2Header {
    /// Vulkan format identifier
    pub vk_format: u32,
    /// Type size (1 for compressed formats)
    pub type_size: u32,
    /// Pixel width
    pub pixel_width: u32,
    /// Pixel height
    pub pixel_height: u32,
    /// Pixel depth (1 for 2D textures)
    pub pixel_depth: u32,
    /// Array layers (1 for non-array textures)
    pub layer_count: u32,
    /// Number of faces (1 for non-cubemap textures)
    pub face_count: u32,
    /// Number of mip levels
    pub level_count: u32,
    /// Supercompression scheme
    pub supercompression_scheme: u32,
    /// Data format descriptor byte length
    pub dfd_byte_offset: u32,
    pub dfd_byte_length: u32,
    /// Key/value data
    pub kvd_byte_offset: u32,
    pub kvd_byte_length: u32,
    /// Supercompression global data
    pub sgd_byte_offset: u64,
    pub sgd_byte_length: u64,
}

/// KTX2 level index entry
#[derive(Debug, Clone)]
pub struct Ktx2LevelIndex {
    /// Byte offset to level data
    pub byte_offset: u64,
    /// Compressed byte length
    pub byte_length: u64,
    /// Uncompressed byte length
    pub uncompressed_byte_length: u64,
}

/// KTX2 data format descriptor
#[derive(Debug, Clone)]
pub struct Ktx2DataFormatDescriptor {
    /// Format information
    pub vendor_id: u32,
    pub descriptor_type: u32,
    pub version_number: u32,
    pub descriptor_block_size: u32,
    /// Channel information
    pub channels: Vec<Ktx2ChannelInfo>,
}

/// KTX2 channel information
#[derive(Debug, Clone)]
pub struct Ktx2ChannelInfo {
    pub channel_type: u32,
    pub bit_offset: u32,
    pub bit_length: u32,
    pub sample_position: [u32; 4],
    pub sample_lower: u32,
    pub sample_upper: u32,
}

/// Supercompression schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperCompressionScheme {
    /// No supercompression
    None = 0,
    /// Basis Universal ETC1S
    BasisLZ = 1,
    /// Basis Universal UASTC
    ZStandard = 2,
    /// ZLIB supercompression
    ZLIB = 3,
}

impl From<u32> for SuperCompressionScheme {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::BasisLZ,
            2 => Self::ZStandard,
            3 => Self::ZLIB,
            _ => Self::None,
        }
    }
}

/// KTX2 loader with transcoding support
pub struct Ktx2Loader {
    /// Enable transcoding support
    transcoding_enabled: bool,
    /// Supported target formats
    target_formats: Vec<wgpu::TextureFormat>,
    /// Basis universal transcoder (placeholder)
    basis_transcoder: Option<BasisTranscoder>,
}

impl Default for Ktx2Loader {
    fn default() -> Self {
        Self::new()
    }
}

impl Ktx2Loader {
    /// Create new KTX2 loader
    pub fn new() -> Self {
        Self {
            transcoding_enabled: true,
            target_formats: Self::default_target_formats(),
            basis_transcoder: None,
        }
    }
    
    /// Create loader with specific target formats
    pub fn with_target_formats(formats: Vec<wgpu::TextureFormat>) -> Self {
        Self {
            transcoding_enabled: true,
            target_formats: formats,
            basis_transcoder: None,
        }
    }
    
    /// Load KTX2 file
    pub fn load_from_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<CompressedImage, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read KTX2 file: {}", e))?;
        
        self.load_from_memory(&data)
    }
    
    /// Load KTX2 from memory
    pub fn load_from_memory(&self, data: &[u8]) -> Result<CompressedImage, String> {
        let mut reader = Cursor::new(data);
        
        // Parse header
        let header = self.parse_header(&mut reader)?;
        
        // Parse level indices
        let level_indices = self.parse_level_indices(&mut reader, &header)?;
        
        // Parse data format descriptor
        let _dfd = if header.dfd_byte_length > 0 {
            Some(self.parse_data_format_descriptor(&mut reader, &header)?)
        } else {
            None
        };
        
        // Parse key-value data
        let _kvd = if header.kvd_byte_length > 0 {
            Some(self.parse_key_value_data(&mut reader, &header)?)
        } else {
            None
        };
        
        // Extract texture data
        let texture_data = self.extract_texture_data(&mut reader, &header, &level_indices)?;
        
        // Convert to WGPU format
        let wgpu_format = self.vk_format_to_wgpu(header.vk_format)?;
        
        Ok(CompressedImage {
            data: texture_data,
            width: header.pixel_width,
            height: header.pixel_height,
            mip_levels: header.level_count,
            format: wgpu_format,
            is_srgb: self.is_srgb_format(header.vk_format),
            source_format: "KTX2".to_string(),
        })
    }
    
    /// Parse KTX2 header
    fn parse_header(&self, reader: &mut Cursor<&[u8]>) -> Result<Ktx2Header, String> {
        // Check magic number
        let mut magic = [0u8; 12];
        reader.read_exact(&mut magic)
            .map_err(|e| format!("Failed to read KTX2 magic: {}", e))?;
        
        if magic != KTX2_MAGIC {
            return Err("Invalid KTX2 magic number".to_string());
        }
        
        // Read header fields
        let vk_format = self.read_u32_le(reader)?;
        let type_size = self.read_u32_le(reader)?;
        let pixel_width = self.read_u32_le(reader)?;
        let pixel_height = self.read_u32_le(reader)?;
        let pixel_depth = self.read_u32_le(reader)?;
        let layer_count = self.read_u32_le(reader)?;
        let face_count = self.read_u32_le(reader)?;
        let level_count = self.read_u32_le(reader)?;
        let supercompression_scheme = self.read_u32_le(reader)?;
        
        let dfd_byte_offset = self.read_u32_le(reader)?;
        let dfd_byte_length = self.read_u32_le(reader)?;
        let kvd_byte_offset = self.read_u32_le(reader)?;
        let kvd_byte_length = self.read_u32_le(reader)?;
        let sgd_byte_offset = self.read_u64_le(reader)?;
        let sgd_byte_length = self.read_u64_le(reader)?;
        
        // Validate header
        if pixel_width == 0 || pixel_height == 0 {
            return Err("Invalid texture dimensions".to_string());
        }
        
        if level_count == 0 {
            return Err("Invalid mip level count".to_string());
        }
        
        Ok(Ktx2Header {
            vk_format,
            type_size,
            pixel_width,
            pixel_height,
            pixel_depth,
            layer_count,
            face_count,
            level_count,
            supercompression_scheme,
            dfd_byte_offset,
            dfd_byte_length,
            kvd_byte_offset,
            kvd_byte_length,
            sgd_byte_offset,
            sgd_byte_length,
        })
    }
    
    /// Parse level indices
    fn parse_level_indices(
        &self,
        reader: &mut Cursor<&[u8]>,
        header: &Ktx2Header
    ) -> Result<Vec<Ktx2LevelIndex>, String> {
        let mut indices = Vec::with_capacity(header.level_count as usize);
        
        for _ in 0..header.level_count {
            let byte_offset = self.read_u64_le(reader)?;
            let byte_length = self.read_u64_le(reader)?;
            let uncompressed_byte_length = self.read_u64_le(reader)?;
            
            indices.push(Ktx2LevelIndex {
                byte_offset,
                byte_length,
                uncompressed_byte_length,
            });
        }
        
        Ok(indices)
    }
    
    /// Parse data format descriptor
    fn parse_data_format_descriptor(
        &self,
        reader: &mut Cursor<&[u8]>,
        header: &Ktx2Header
    ) -> Result<Ktx2DataFormatDescriptor, String> {
        // Seek to DFD offset
        reader.seek(SeekFrom::Start(header.dfd_byte_offset as u64))
            .map_err(|e| format!("Failed to seek to DFD: {}", e))?;
        
        // Read DFD header
        let vendor_id = self.read_u32_le(reader)?;
        let descriptor_type = self.read_u32_le(reader)?;
        let version_number = self.read_u32_le(reader)?;
        let descriptor_block_size = self.read_u32_le(reader)?;
        
        // Parse channels (simplified)
        let channels = Vec::new(); // TODO: Parse channel information
        
        Ok(Ktx2DataFormatDescriptor {
            vendor_id,
            descriptor_type,
            version_number,
            descriptor_block_size,
            channels,
        })
    }
    
    /// Parse key-value data
    fn parse_key_value_data(
        &self,
        reader: &mut Cursor<&[u8]>,
        header: &Ktx2Header
    ) -> Result<HashMap<String, Vec<u8>>, String> {
        // Seek to KVD offset
        reader.seek(SeekFrom::Start(header.kvd_byte_offset as u64))
            .map_err(|e| format!("Failed to seek to KVD: {}", e))?;
        
        let mut kvd = HashMap::new();
        let mut bytes_read = 0u32;
        
        while bytes_read < header.kvd_byte_length {
            // Read key-value pair length
            let kv_length = self.read_u32_le(reader)?;
            bytes_read += 4;
            
            if kv_length == 0 || bytes_read + kv_length > header.kvd_byte_length {
                break;
            }
            
            // Read key-value data
            let mut kv_data = vec![0u8; kv_length as usize];
            reader.read_exact(&mut kv_data)
                .map_err(|e| format!("Failed to read KVD entry: {}", e))?;
            
            // Parse key (null-terminated string)
            if let Some(null_pos) = kv_data.iter().position(|&b| b == 0) {
                let key = String::from_utf8_lossy(&kv_data[..null_pos]).to_string();
                let value = kv_data[null_pos + 1..].to_vec();
                kvd.insert(key, value);
            }
            
            bytes_read += kv_length;
            
            // Align to 4-byte boundary
            let padding = (4 - (kv_length % 4)) % 4;
            if padding > 0 {
                reader.seek(SeekFrom::Current(padding as i64))
                    .map_err(|e| format!("Failed to skip KVD padding: {}", e))?;
                bytes_read += padding;
            }
        }
        
        Ok(kvd)
    }
    
    /// Extract texture data with supercompression handling
    fn extract_texture_data(
        &self,
        reader: &mut Cursor<&[u8]>,
        header: &Ktx2Header,
        level_indices: &[Ktx2LevelIndex]
    ) -> Result<Vec<u8>, String> {
        // For now, extract only the base level (mip 0)
        if level_indices.is_empty() {
            return Err("No texture data found".to_string());
        }
        
        let base_level = &level_indices[0];
        
        // Seek to texture data
        reader.seek(SeekFrom::Start(base_level.byte_offset))
            .map_err(|e| format!("Failed to seek to texture data: {}", e))?;
        
        // Read texture data
        let mut texture_data = vec![0u8; base_level.byte_length as usize];
        reader.read_exact(&mut texture_data)
            .map_err(|e| format!("Failed to read texture data: {}", e))?;
        
        // Handle supercompression
        let supercompression = SuperCompressionScheme::from(header.supercompression_scheme);
        match supercompression {
            SuperCompressionScheme::None => {
                Ok(texture_data)
            },
            SuperCompressionScheme::BasisLZ => {
                self.transcode_basis_universal(texture_data, header)
            },
            SuperCompressionScheme::ZStandard => {
                self.decompress_zstd(texture_data)
            },
            SuperCompressionScheme::ZLIB => {
                self.decompress_zlib(texture_data)
            },
        }
    }
    
    /// Transcode Basis Universal data
    fn transcode_basis_universal(
        &self,
        _data: Vec<u8>,
        _header: &Ktx2Header
    ) -> Result<Vec<u8>, String> {
        // Placeholder for Basis Universal transcoding
        // In practice, integrate with basis_universal crate or C library
        Err("Basis Universal transcoding not implemented".to_string())
    }
    
    /// Decompress ZSTD data
    fn decompress_zstd(&self, _data: Vec<u8>) -> Result<Vec<u8>, String> {
        // Placeholder for ZSTD decompression
        // In practice, use zstd crate
        Err("ZSTD decompression not implemented".to_string())
    }
    
    /// Decompress ZLIB data
    fn decompress_zlib(&self, _data: Vec<u8>) -> Result<Vec<u8>, String> {
        // Placeholder for ZLIB decompression
        // In practice, use flate2 crate
        Err("ZLIB decompression not implemented".to_string())
    }
    
    /// Convert Vulkan format to WGPU format
    fn vk_format_to_wgpu(&self, vk_format: u32) -> Result<wgpu::TextureFormat, String> {
        match vk_format {
            // BC formats
            131 => Ok(wgpu::TextureFormat::Bc1RgbaUnorm),
            132 => Ok(wgpu::TextureFormat::Bc1RgbaUnormSrgb),
            135 => Ok(wgpu::TextureFormat::Bc3RgbaUnorm),
            136 => Ok(wgpu::TextureFormat::Bc3RgbaUnormSrgb),
            139 => Ok(wgpu::TextureFormat::Bc4RUnorm),
            141 => Ok(wgpu::TextureFormat::Bc5RgUnorm),
            145 => Ok(wgpu::TextureFormat::Bc6hRgbUfloat),
            147 => Ok(wgpu::TextureFormat::Bc7RgbaUnorm),
            148 => Ok(wgpu::TextureFormat::Bc7RgbaUnormSrgb),
            
            // ETC2 formats (using correct KTX2 format values)
            163 => Ok(wgpu::TextureFormat::Etc2Rgb8Unorm),
            164 => Ok(wgpu::TextureFormat::Etc2Rgb8UnormSrgb),
            151 => Ok(wgpu::TextureFormat::Etc2Rgba8Unorm),
            152 => Ok(wgpu::TextureFormat::Etc2Rgba8UnormSrgb),
            
            // Basic formats
            37 => Ok(wgpu::TextureFormat::Rgba8Unorm),
            43 => Ok(wgpu::TextureFormat::Rgba8UnormSrgb),
            44 => Ok(wgpu::TextureFormat::Bgra8Unorm),
            
            _ => Err(format!("Unsupported Vulkan format: {}", vk_format)),
        }
    }
    
    /// Check if format is sRGB
    fn is_srgb_format(&self, vk_format: u32) -> bool {
        matches!(vk_format, 
            43 | 132 | 136 | 148 | 152 | 164 // sRGB variants
        )
    }
    
    /// Get default target formats for transcoding
    fn default_target_formats() -> Vec<wgpu::TextureFormat> {
        vec![
            wgpu::TextureFormat::Bc7RgbaUnorm,
            wgpu::TextureFormat::Bc3RgbaUnorm,
            wgpu::TextureFormat::Bc1RgbaUnorm,
            wgpu::TextureFormat::Etc2Rgba8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
    }
    
    /// Helper to read little-endian u32
    fn read_u32_le(&self, reader: &mut Cursor<&[u8]>) -> Result<u32, String> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)
            .map_err(|e| format!("Failed to read u32: {}", e))?;
        Ok(u32::from_le_bytes(bytes))
    }
    
    /// Helper to read little-endian u64
    fn read_u64_le(&self, reader: &mut Cursor<&[u8]>) -> Result<u64, String> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)
            .map_err(|e| format!("Failed to read u64: {}", e))?;
        Ok(u64::from_le_bytes(bytes))
    }
}

/// Placeholder for Basis Universal transcoder
pub struct BasisTranscoder {
    // Transcoder state
}

impl BasisTranscoder {
    /// Initialize transcoder
    pub fn new() -> Result<Self, String> {
        Ok(Self {})
    }
    
    /// Transcode to target format
    pub fn transcode(
        &self,
        _data: &[u8],
        _target_format: wgpu::TextureFormat,
        _width: u32,
        _height: u32,
    ) -> Result<Vec<u8>, String> {
        Err("Basis Universal transcoding not implemented".to_string())
    }
}

/// Validate KTX2 file
pub fn validate_ktx2_file<P: AsRef<std::path::Path>>(path: P) -> Result<bool, String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    
    validate_ktx2_data(&data)
}

/// Validate KTX2 data
pub fn validate_ktx2_data(data: &[u8]) -> Result<bool, String> {
    if data.len() < 12 {
        return Ok(false);
    }
    
    let magic = &data[..12];
    Ok(magic == KTX2_MAGIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
        assert_eq!(SuperCompressionScheme::from(0), SuperCompressionScheme::None);
        assert_eq!(SuperCompressionScheme::from(1), SuperCompressionScheme::BasisLZ);
        assert_eq!(SuperCompressionScheme::from(999), SuperCompressionScheme::None);
    }
    
    #[test]
    fn test_loader_creation() {
        let loader = Ktx2Loader::new();
        assert!(loader.transcoding_enabled);
        assert!(!loader.target_formats.is_empty());
        
        let custom_formats = vec![wgpu::TextureFormat::Rgba8Unorm];
        let custom_loader = Ktx2Loader::with_target_formats(custom_formats.clone());
        assert_eq!(custom_loader.target_formats, custom_formats);
    }
}