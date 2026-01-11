use super::types::*;
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom};

/// Helper to read little-endian u32
fn read_u32_le(reader: &mut Cursor<&[u8]>) -> Result<u32, String> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("Failed to read u32: {}", e))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Helper to read little-endian u64
fn read_u64_le(reader: &mut Cursor<&[u8]>) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| format!("Failed to read u64: {}", e))?;
    Ok(u64::from_le_bytes(bytes))
}

/// Parse KTX2 header
pub fn parse_header(reader: &mut Cursor<&[u8]>) -> Result<Ktx2Header, String> {
    // Check magic number
    let mut magic = [0u8; 12];
    reader
        .read_exact(&mut magic)
        .map_err(|e| format!("Failed to read KTX2 magic: {}", e))?;

    if magic != KTX2_MAGIC {
        return Err("Invalid KTX2 magic number".to_string());
    }

    // Read header fields
    let vk_format = read_u32_le(reader)?;
    let type_size = read_u32_le(reader)?;
    let pixel_width = read_u32_le(reader)?;
    let pixel_height = read_u32_le(reader)?;
    let pixel_depth = read_u32_le(reader)?;
    let layer_count = read_u32_le(reader)?;
    let face_count = read_u32_le(reader)?;
    let level_count = read_u32_le(reader)?;
    let supercompression_scheme = read_u32_le(reader)?;

    let dfd_byte_offset = read_u32_le(reader)?;
    let dfd_byte_length = read_u32_le(reader)?;
    let kvd_byte_offset = read_u32_le(reader)?;
    let kvd_byte_length = read_u32_le(reader)?;
    let sgd_byte_offset = read_u64_le(reader)?;
    let sgd_byte_length = read_u64_le(reader)?;

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
pub fn parse_level_indices(
    reader: &mut Cursor<&[u8]>,
    header: &Ktx2Header,
) -> Result<Vec<Ktx2LevelIndex>, String> {
    let mut indices = Vec::with_capacity(header.level_count as usize);

    for _ in 0..header.level_count {
        let byte_offset = read_u64_le(reader)?;
        let byte_length = read_u64_le(reader)?;
        let uncompressed_byte_length = read_u64_le(reader)?;

        indices.push(Ktx2LevelIndex {
            byte_offset,
            byte_length,
            uncompressed_byte_length,
        });
    }

    Ok(indices)
}

/// Parse data format descriptor
pub fn parse_data_format_descriptor(
    reader: &mut Cursor<&[u8]>,
    header: &Ktx2Header,
) -> Result<Ktx2DataFormatDescriptor, String> {
    // Seek to DFD offset
    reader
        .seek(SeekFrom::Start(header.dfd_byte_offset as u64))
        .map_err(|e| format!("Failed to seek to DFD: {}", e))?;

    // Read DFD header
    let vendor_id = read_u32_le(reader)?;
    let descriptor_type = read_u32_le(reader)?;
    let version_number = read_u32_le(reader)?;
    let descriptor_block_size = read_u32_le(reader)?;

    // Channel metadata is currently ignored by the loader; keep it empty.
    let channels = Vec::new();

    Ok(Ktx2DataFormatDescriptor {
        vendor_id,
        descriptor_type,
        version_number,
        descriptor_block_size,
        channels,
    })
}

/// Parse key-value data
pub fn parse_key_value_data(
    reader: &mut Cursor<&[u8]>,
    header: &Ktx2Header,
) -> Result<HashMap<String, Vec<u8>>, String> {
    // Seek to KVD offset
    reader
        .seek(SeekFrom::Start(header.kvd_byte_offset as u64))
        .map_err(|e| format!("Failed to seek to KVD: {}", e))?;

    let mut kvd = HashMap::new();
    let mut bytes_read = 0u32;

    while bytes_read < header.kvd_byte_length {
        // Read key-value pair length
        let kv_length = read_u32_le(reader)?;
        bytes_read += 4;

        if kv_length == 0 || bytes_read + kv_length > header.kvd_byte_length {
            break;
        }

        // Read key-value data
        let mut kv_data = vec![0u8; kv_length as usize];
        reader
            .read_exact(&mut kv_data)
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
            reader
                .seek(SeekFrom::Current(padding as i64))
                .map_err(|e| format!("Failed to skip KVD padding: {}", e))?;
            bytes_read += padding;
        }
    }

    Ok(kvd)
}
