//! PNTS (Point Cloud) payload parser

use super::error::{Tiles3dError, Tiles3dResult};
use bytemuck::{Pod, Zeroable};
use std::path::Path;

/// PNTS file header (28 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PntsHeader {
    /// Magic bytes "pnts"
    pub magic: [u8; 4],
    /// Version (should be 1)
    pub version: u32,
    /// Total byte length of the file
    pub byte_length: u32,
    /// Feature table JSON byte length
    pub feature_table_json_byte_length: u32,
    /// Feature table binary byte length
    pub feature_table_binary_byte_length: u32,
    /// Batch table JSON byte length
    pub batch_table_json_byte_length: u32,
    /// Batch table binary byte length
    pub batch_table_binary_byte_length: u32,
}

/// Decoded PNTS payload
#[derive(Debug)]
pub struct PntsPayload {
    /// Header information
    pub header: PntsHeader,
    /// Number of points
    pub points_length: u32,
    /// Point positions (3 floats per point, relative to RTC_CENTER if present)
    pub positions: Vec<f32>,
    /// RGB colors (3 u8 per point, optional)
    pub colors: Option<Vec<u8>>,
    /// RGBA colors (4 u8 per point, optional)
    pub colors_rgba: Option<Vec<u8>>,
    /// Normals (3 floats per point, optional)
    pub normals: Option<Vec<f32>>,
    /// Batch IDs (optional)
    pub batch_ids: Option<Vec<u32>>,
    /// RTC (Relative-To-Center) offset, if present
    pub rtc_center: Option<[f64; 3]>,
    /// Quantized volume offset (for POSITION_QUANTIZED)
    pub quantized_volume_offset: Option<[f32; 3]>,
    /// Quantized volume scale (for POSITION_QUANTIZED)
    pub quantized_volume_scale: Option<[f32; 3]>,
    /// Feature table JSON (for additional properties)
    pub feature_table: serde_json::Value,
    /// Batch table JSON (optional)
    pub batch_table: Option<serde_json::Value>,
}

impl PntsPayload {
    /// Get number of points
    pub fn point_count(&self) -> usize {
        self.points_length as usize
    }

    /// Check if colors are available
    pub fn has_colors(&self) -> bool {
        self.colors.is_some() || self.colors_rgba.is_some()
    }

    /// Check if normals are available
    pub fn has_normals(&self) -> bool {
        self.normals.is_some()
    }

    /// Get world-space positions (applies RTC_CENTER if present)
    pub fn world_positions(&self) -> Vec<f32> {
        if let Some(rtc) = self.rtc_center {
            self.positions
                .chunks(3)
                .flat_map(|p| {
                    [
                        p[0] + rtc[0] as f32,
                        p[1] + rtc[1] as f32,
                        p[2] + rtc[2] as f32,
                    ]
                })
                .collect()
        } else {
            self.positions.clone()
        }
    }
}

/// Decode a PNTS file from bytes
pub fn decode_pnts(data: &[u8]) -> Tiles3dResult<PntsPayload> {
    if data.len() < 28 {
        return Err(Tiles3dError::InvalidPnts("File too small for header".into()));
    }

    let header: PntsHeader = *bytemuck::from_bytes(&data[0..28]);

    if &header.magic != b"pnts" {
        return Err(Tiles3dError::InvalidPnts(format!(
            "Invalid magic: {:?}",
            header.magic
        )));
    }

    if header.version != 1 {
        return Err(Tiles3dError::InvalidPnts(format!(
            "Unsupported version: {}",
            header.version
        )));
    }

    let mut offset = 28usize;

    // Parse feature table JSON
    let ft_json_end = offset + header.feature_table_json_byte_length as usize;
    let feature_table: serde_json::Value = if header.feature_table_json_byte_length > 0 {
        let json_str = std::str::from_utf8(&data[offset..ft_json_end])
            .map_err(|e| Tiles3dError::InvalidPnts(format!("Invalid UTF-8 in feature table: {}", e)))?;
        serde_json::from_str(json_str)?
    } else {
        serde_json::Value::Object(serde_json::Map::new())
    };
    offset = ft_json_end;

    // Feature table binary starts here
    let ft_bin_start = offset;
    let ft_bin = &data[ft_bin_start..ft_bin_start + header.feature_table_binary_byte_length as usize];
    offset += header.feature_table_binary_byte_length as usize;

    // Parse batch table JSON
    let bt_json_end = offset + header.batch_table_json_byte_length as usize;
    let batch_table: Option<serde_json::Value> = if header.batch_table_json_byte_length > 0 {
        let json_str = std::str::from_utf8(&data[offset..bt_json_end])
            .map_err(|e| Tiles3dError::InvalidPnts(format!("Invalid UTF-8 in batch table: {}", e)))?;
        Some(serde_json::from_str(json_str)?)
    } else {
        None
    };

    // Extract points length
    let points_length = feature_table
        .get("POINTS_LENGTH")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Tiles3dError::InvalidPnts("Missing POINTS_LENGTH".into()))? as u32;

    // Extract RTC_CENTER if present
    let rtc_center = feature_table
        .get("RTC_CENTER")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            if arr.len() == 3 {
                Some([
                    arr[0].as_f64()?,
                    arr[1].as_f64()?,
                    arr[2].as_f64()?,
                ])
            } else {
                None
            }
        });

    // Extract positions
    let positions = extract_positions(&feature_table, ft_bin, points_length)?;

    // Extract quantization parameters if present
    let quantized_volume_offset = feature_table
        .get("QUANTIZED_VOLUME_OFFSET")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            if arr.len() == 3 {
                Some([
                    arr[0].as_f64()? as f32,
                    arr[1].as_f64()? as f32,
                    arr[2].as_f64()? as f32,
                ])
            } else {
                None
            }
        });

    let quantized_volume_scale = feature_table
        .get("QUANTIZED_VOLUME_SCALE")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            if arr.len() == 3 {
                Some([
                    arr[0].as_f64()? as f32,
                    arr[1].as_f64()? as f32,
                    arr[2].as_f64()? as f32,
                ])
            } else {
                None
            }
        });

    // Extract colors (RGB or RGBA)
    let colors = extract_colors_rgb(&feature_table, ft_bin, points_length);
    let colors_rgba = extract_colors_rgba(&feature_table, ft_bin, points_length);

    // Extract normals
    let normals = extract_normals(&feature_table, ft_bin, points_length);

    // Extract batch IDs
    let batch_ids = extract_batch_ids(&feature_table, ft_bin, points_length);

    Ok(PntsPayload {
        header,
        points_length,
        positions,
        colors,
        colors_rgba,
        normals,
        batch_ids,
        rtc_center,
        quantized_volume_offset,
        quantized_volume_scale,
        feature_table,
        batch_table,
    })
}

/// Load and decode a PNTS file from path
pub fn load_pnts<P: AsRef<Path>>(path: P) -> Tiles3dResult<PntsPayload> {
    let data = std::fs::read(path)?;
    decode_pnts(&data)
}

fn extract_positions(ft: &serde_json::Value, bin: &[u8], count: u32) -> Tiles3dResult<Vec<f32>> {
    // Try POSITION first (float32 x 3)
    if let Some(pos_info) = ft.get("POSITION") {
        let byte_offset = pos_info
            .get("byteOffset")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let byte_count = count as usize * 3 * 4;
        if byte_offset + byte_count > bin.len() {
            return Err(Tiles3dError::InvalidPnts("POSITION buffer overrun".into()));
        }

        let mut positions = Vec::with_capacity(count as usize * 3);
        for i in 0..(count as usize * 3) {
            let idx = byte_offset + i * 4;
            let val = f32::from_le_bytes([bin[idx], bin[idx+1], bin[idx+2], bin[idx+3]]);
            positions.push(val);
        }
        return Ok(positions);
    }

    // Try POSITION_QUANTIZED (uint16 x 3, needs dequantization)
    if let Some(pos_info) = ft.get("POSITION_QUANTIZED") {
        let byte_offset = pos_info
            .get("byteOffset")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let byte_count = count as usize * 3 * 2;
        if byte_offset + byte_count > bin.len() {
            return Err(Tiles3dError::InvalidPnts("POSITION_QUANTIZED buffer overrun".into()));
        }

        let vol_offset = ft
            .get("QUANTIZED_VOLUME_OFFSET")
            .and_then(|v| v.as_array())
            .map(|arr| [
                arr.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            ])
            .unwrap_or([0.0, 0.0, 0.0]);

        let vol_scale = ft
            .get("QUANTIZED_VOLUME_SCALE")
            .and_then(|v| v.as_array())
            .map(|arr| [
                arr.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                arr.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                arr.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            ])
            .unwrap_or([1.0, 1.0, 1.0]);

        let mut positions = Vec::with_capacity(count as usize * 3);
        for i in 0..(count as usize) {
            for j in 0..3 {
                let idx = byte_offset + (i * 3 + j) * 2;
                let quantized = u16::from_le_bytes([bin[idx], bin[idx+1]]);
                let normalized = quantized as f32 / 65535.0;
                let pos = vol_offset[j] + normalized * vol_scale[j];
                positions.push(pos);
            }
        }
        return Ok(positions);
    }

    Err(Tiles3dError::InvalidPnts("No POSITION or POSITION_QUANTIZED found".into()))
}

fn extract_colors_rgb(ft: &serde_json::Value, bin: &[u8], count: u32) -> Option<Vec<u8>> {
    let color_info = ft.get("RGB")?;
    let byte_offset = color_info.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let byte_count = count as usize * 3;
    
    if byte_offset + byte_count > bin.len() {
        return None;
    }

    Some(bin[byte_offset..byte_offset + byte_count].to_vec())
}

fn extract_colors_rgba(ft: &serde_json::Value, bin: &[u8], count: u32) -> Option<Vec<u8>> {
    let color_info = ft.get("RGBA")?;
    let byte_offset = color_info.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let byte_count = count as usize * 4;
    
    if byte_offset + byte_count > bin.len() {
        return None;
    }

    Some(bin[byte_offset..byte_offset + byte_count].to_vec())
}

fn extract_normals(ft: &serde_json::Value, bin: &[u8], count: u32) -> Option<Vec<f32>> {
    // Try NORMAL (float32 x 3)
    if let Some(normal_info) = ft.get("NORMAL") {
        let byte_offset = normal_info.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let byte_count = count as usize * 3 * 4;
        
        if byte_offset + byte_count > bin.len() {
            return None;
        }

        let mut normals = Vec::with_capacity(count as usize * 3);
        for i in 0..(count as usize * 3) {
            let idx = byte_offset + i * 4;
            let val = f32::from_le_bytes([bin[idx], bin[idx+1], bin[idx+2], bin[idx+3]]);
            normals.push(val);
        }
        return Some(normals);
    }

    // Try NORMAL_OCT16P (oct-encoded uint8 x 2)
    if let Some(normal_info) = ft.get("NORMAL_OCT16P") {
        let byte_offset = normal_info.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let byte_count = count as usize * 2;
        
        if byte_offset + byte_count > bin.len() {
            return None;
        }

        let mut normals = Vec::with_capacity(count as usize * 3);
        for i in 0..(count as usize) {
            let idx = byte_offset + i * 2;
            let (nx, ny, nz) = decode_oct16p(bin[idx], bin[idx + 1]);
            normals.extend_from_slice(&[nx, ny, nz]);
        }
        return Some(normals);
    }

    None
}

fn extract_batch_ids(ft: &serde_json::Value, bin: &[u8], count: u32) -> Option<Vec<u32>> {
    let batch_info = ft.get("BATCH_ID")?;
    let byte_offset = batch_info.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let component_type = batch_info.get("componentType").and_then(|v| v.as_str()).unwrap_or("UNSIGNED_SHORT");
    
    match component_type {
        "UNSIGNED_BYTE" => {
            if byte_offset + count as usize > bin.len() {
                return None;
            }
            Some(bin[byte_offset..byte_offset + count as usize].iter().map(|&b| b as u32).collect())
        }
        "UNSIGNED_SHORT" => {
            let byte_count = count as usize * 2;
            if byte_offset + byte_count > bin.len() {
                return None;
            }
            let mut ids = Vec::with_capacity(count as usize);
            for i in 0..(count as usize) {
                let idx = byte_offset + i * 2;
                ids.push(u16::from_le_bytes([bin[idx], bin[idx+1]]) as u32);
            }
            Some(ids)
        }
        "UNSIGNED_INT" => {
            let byte_count = count as usize * 4;
            if byte_offset + byte_count > bin.len() {
                return None;
            }
            let mut ids = Vec::with_capacity(count as usize);
            for i in 0..(count as usize) {
                let idx = byte_offset + i * 4;
                ids.push(u32::from_le_bytes([bin[idx], bin[idx+1], bin[idx+2], bin[idx+3]]));
            }
            Some(ids)
        }
        _ => None,
    }
}

/// Decode oct16p encoded normal
fn decode_oct16p(x: u8, y: u8) -> (f32, f32, f32) {
    let fx = (x as f32 / 255.0) * 2.0 - 1.0;
    let fy = (y as f32 / 255.0) * 2.0 - 1.0;
    let fz = 1.0 - fx.abs() - fy.abs();
    
    let (fx, fy) = if fz < 0.0 {
        let sign_x = if fx >= 0.0 { 1.0 } else { -1.0 };
        let sign_y = if fy >= 0.0 { 1.0 } else { -1.0 };
        ((1.0 - fy.abs()) * sign_x, (1.0 - fx.abs()) * sign_y)
    } else {
        (fx, fy)
    };
    
    let len = (fx * fx + fy * fy + fz * fz).sqrt();
    (fx / len, fy / len, fz / len)
}
