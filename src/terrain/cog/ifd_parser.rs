//! P3.2: TIFF IFD (Image File Directory) parser for COG overview detection.

use super::error::CogError;
use super::range_reader::RangeReader;

/// TIFF tag constants.
const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_SAMPLE_FORMAT: u16 = 339;
const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;

/// Compression constants.
pub const COMPRESSION_NONE: u16 = 1;
pub const COMPRESSION_LZW: u16 = 5;
pub const COMPRESSION_DEFLATE: u16 = 8;
pub const COMPRESSION_DEFLATE_ALT: u16 = 32946;

/// Sample format constants.
pub const SAMPLE_FORMAT_UINT: u16 = 1;
pub const SAMPLE_FORMAT_INT: u16 = 2;
pub const SAMPLE_FORMAT_FLOAT: u16 = 3;

/// Parsed IFD entry representing one image/overview level.
#[derive(Debug, Clone)]
pub struct IfdEntry {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub bits_per_sample: u16,
    pub sample_format: u16,
    pub compression: u16,
    pub tile_offsets: Vec<u64>,
    pub tile_byte_counts: Vec<u64>,
    pub overview_level: u32,
    pub tiles_across: u32,
    pub tiles_down: u32,
}

impl IfdEntry {
    /// Get total number of tiles in this IFD.
    pub fn tile_count(&self) -> usize {
        self.tile_offsets.len()
    }

    /// Get bytes per sample based on bits_per_sample.
    pub fn bytes_per_sample(&self) -> usize {
        (self.bits_per_sample as usize + 7) / 8
    }

    /// Check if this IFD uses float samples.
    pub fn is_float(&self) -> bool {
        self.sample_format == SAMPLE_FORMAT_FLOAT
    }

    /// Get tile index for given tile coordinates.
    pub fn tile_index(&self, tile_x: u32, tile_y: u32) -> Option<usize> {
        if tile_x >= self.tiles_across || tile_y >= self.tiles_down {
            return None;
        }
        Some((tile_y * self.tiles_across + tile_x) as usize)
    }
}

/// COG header information.
#[derive(Debug, Clone)]
pub struct CogHeader {
    pub is_big_endian: bool,
    pub is_bigtiff: bool,
    pub ifds: Vec<IfdEntry>,
}

impl CogHeader {
    /// Get the full-resolution IFD (first one).
    pub fn full_resolution(&self) -> Option<&IfdEntry> {
        self.ifds.first()
    }

    /// Get IFD for a specific overview level.
    pub fn overview(&self, level: u32) -> Option<&IfdEntry> {
        self.ifds.iter().find(|ifd| ifd.overview_level == level)
    }

    /// Select best IFD for requested LOD.
    pub fn select_ifd_for_lod(&self, lod: u32) -> &IfdEntry {
        self.ifds
            .iter()
            .filter(|ifd| ifd.overview_level <= lod)
            .max_by_key(|ifd| ifd.width)
            .unwrap_or_else(|| self.ifds.last().unwrap())
    }
}

/// Parse COG header and all IFDs.
pub async fn parse_cog_header(reader: &RangeReader) -> Result<CogHeader, CogError> {
    let header_bytes = reader.read_range(0, 16).await?;

    let (is_big_endian, is_bigtiff) = parse_tiff_header(&header_bytes)?;

    let first_ifd_offset = if is_bigtiff {
        if is_big_endian {
            u64::from_be_bytes(header_bytes[8..16].try_into().unwrap())
        } else {
            u64::from_le_bytes(header_bytes[8..16].try_into().unwrap())
        }
    } else {
        let offset = if is_big_endian {
            u32::from_be_bytes(header_bytes[4..8].try_into().unwrap())
        } else {
            u32::from_le_bytes(header_bytes[4..8].try_into().unwrap())
        };
        offset as u64
    };

    let mut ifds = Vec::new();
    let mut ifd_offset = first_ifd_offset;
    let mut overview_level = 0u32;

    while ifd_offset != 0 {
        let (ifd, next_offset) =
            parse_ifd(reader, ifd_offset, is_big_endian, is_bigtiff, overview_level).await?;
        ifds.push(ifd);
        ifd_offset = next_offset;
        overview_level += 1;

        if ifds.len() > 20 {
            break;
        }
    }

    Ok(CogHeader {
        is_big_endian,
        is_bigtiff,
        ifds,
    })
}

fn parse_tiff_header(header: &[u8]) -> Result<(bool, bool), CogError> {
    if header.len() < 8 {
        return Err(CogError::InvalidTiffHeader("Header too short".into()));
    }

    let is_big_endian = match &header[0..2] {
        b"II" => false,
        b"MM" => true,
        _ => return Err(CogError::InvalidTiffHeader("Invalid byte order".into())),
    };

    let magic = if is_big_endian {
        u16::from_be_bytes([header[2], header[3]])
    } else {
        u16::from_le_bytes([header[2], header[3]])
    };

    let is_bigtiff = match magic {
        42 => false,
        43 => true,
        _ => {
            return Err(CogError::InvalidTiffHeader(format!(
                "Invalid magic number: {}",
                magic
            )))
        }
    };

    Ok((is_big_endian, is_bigtiff))
}

async fn parse_ifd(
    reader: &RangeReader,
    offset: u64,
    big_endian: bool,
    bigtiff: bool,
    overview_level: u32,
) -> Result<(IfdEntry, u64), CogError> {
    let entry_size: u64 = if bigtiff { 20 } else { 12 };
    let count_size: u64 = if bigtiff { 8 } else { 2 };
    let next_size: u64 = if bigtiff { 8 } else { 4 };

    let count_bytes = reader.read_range(offset, count_size).await?;
    let entry_count = if bigtiff {
        read_u64(&count_bytes, 0, big_endian)
    } else {
        read_u16(&count_bytes, 0, big_endian) as u64
    };

    let entries_size = entry_count * entry_size;
    let ifd_data = reader
        .read_range(offset + count_size, entries_size + next_size)
        .await?;

    let mut width = 0u32;
    let mut height = 0u32;
    let mut tile_width = 256u32;
    let mut tile_height = 256u32;
    let mut bits_per_sample = 8u16;
    let mut sample_format = SAMPLE_FORMAT_UINT;
    let mut compression = COMPRESSION_NONE;
    let mut tile_offsets_info: Option<(u64, u64)> = None;
    let mut tile_byte_counts_info: Option<(u64, u64)> = None;

    for i in 0..entry_count {
        let entry_offset = (i * entry_size) as usize;
        let tag = read_u16(&ifd_data, entry_offset, big_endian);
        let field_type = read_u16(&ifd_data, entry_offset + 2, big_endian);
        let count = if bigtiff {
            read_u64(&ifd_data, entry_offset + 4, big_endian)
        } else {
            read_u32(&ifd_data, entry_offset + 4, big_endian) as u64
        };

        let value_offset = if bigtiff { 12 } else { 8 };
        let value = read_tag_value(&ifd_data, entry_offset + value_offset, field_type, big_endian);

        match tag {
            TAG_IMAGE_WIDTH => width = value as u32,
            TAG_IMAGE_LENGTH => height = value as u32,
            TAG_BITS_PER_SAMPLE => bits_per_sample = value as u16,
            TAG_COMPRESSION => compression = value as u16,
            TAG_SAMPLE_FORMAT => sample_format = value as u16,
            TAG_TILE_WIDTH => tile_width = value as u32,
            TAG_TILE_LENGTH => tile_height = value as u32,
            TAG_TILE_OFFSETS => {
                let data_offset = if count > 1 || type_size(field_type) * count as usize > 4 {
                    value
                } else {
                    offset + count_size + (i * entry_size) + value_offset as u64
                };
                tile_offsets_info = Some((data_offset, count));
            }
            TAG_TILE_BYTE_COUNTS => {
                let data_offset = if count > 1 || type_size(field_type) * count as usize > 4 {
                    value
                } else {
                    offset + count_size + (i * entry_size) + value_offset as u64
                };
                tile_byte_counts_info = Some((data_offset, count));
            }
            _ => {}
        }
    }

    let tiles_across = (width + tile_width - 1) / tile_width;
    let tiles_down = (height + tile_height - 1) / tile_height;

    let tile_offsets = if let Some((off, count)) = tile_offsets_info {
        read_offset_array(reader, off, count as usize, big_endian).await?
    } else {
        Vec::new()
    };

    let tile_byte_counts = if let Some((off, count)) = tile_byte_counts_info {
        read_offset_array(reader, off, count as usize, big_endian).await?
    } else {
        Vec::new()
    };

    let next_ifd_offset = if bigtiff {
        read_u64(&ifd_data, entries_size as usize, big_endian)
    } else {
        read_u32(&ifd_data, entries_size as usize, big_endian) as u64
    };

    Ok((
        IfdEntry {
            width,
            height,
            tile_width,
            tile_height,
            bits_per_sample,
            sample_format,
            compression,
            tile_offsets,
            tile_byte_counts,
            overview_level,
            tiles_across,
            tiles_down,
        },
        next_ifd_offset,
    ))
}

async fn read_offset_array(
    reader: &RangeReader,
    offset: u64,
    count: usize,
    big_endian: bool,
) -> Result<Vec<u64>, CogError> {
    let bytes = reader.read_range(offset, (count * 8) as u64).await?;
    let mut offsets = Vec::with_capacity(count);

    for i in 0..count {
        let val = if bytes.len() >= (i + 1) * 8 {
            read_u64(&bytes, i * 8, big_endian)
        } else if bytes.len() >= (i + 1) * 4 {
            read_u32(&bytes, i * 4, big_endian) as u64
        } else {
            0
        };
        offsets.push(val);
    }

    Ok(offsets)
}

fn read_u16(data: &[u8], offset: usize, big_endian: bool) -> u16 {
    if offset + 2 > data.len() {
        return 0;
    }
    if big_endian {
        u16::from_be_bytes([data[offset], data[offset + 1]])
    } else {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    }
}

fn read_u32(data: &[u8], offset: usize, big_endian: bool) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    if big_endian {
        u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap())
    } else {
        u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
    }
}

fn read_u64(data: &[u8], offset: usize, big_endian: bool) -> u64 {
    if offset + 8 > data.len() {
        return 0;
    }
    if big_endian {
        u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap())
    } else {
        u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
    }
}

fn read_tag_value(data: &[u8], offset: usize, field_type: u16, big_endian: bool) -> u64 {
    match field_type {
        1 | 2 => data.get(offset).copied().unwrap_or(0) as u64,
        3 => read_u16(data, offset, big_endian) as u64,
        4 => read_u32(data, offset, big_endian) as u64,
        16 => read_u64(data, offset, big_endian),
        _ => read_u32(data, offset, big_endian) as u64,
    }
}

fn type_size(field_type: u16) -> usize {
    match field_type {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 | 16 | 17 | 18 => 8,
        _ => 4,
    }
}
