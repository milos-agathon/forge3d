use super::types::*;

/// Validate KTX2 file
pub fn validate_ktx2_file<P: AsRef<std::path::Path>>(path: P) -> Result<bool, String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;

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
