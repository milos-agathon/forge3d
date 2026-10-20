/// KTX2 file magic number
pub const KTX2_MAGIC: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

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
