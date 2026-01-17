//! B3DM (Batched 3D Model) payload parser

use super::error::{Tiles3dError, Tiles3dResult};
use bytemuck::{Pod, Zeroable};
use std::path::Path;

/// B3DM file header (28 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct B3dmHeader {
    /// Magic bytes "b3dm"
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

/// Decoded B3DM payload
#[derive(Debug)]
pub struct B3dmPayload {
    /// Header information
    pub header: B3dmHeader,
    /// Feature table JSON (parsed)
    pub feature_table: serde_json::Value,
    /// Batch table JSON (parsed, if present)
    pub batch_table: Option<serde_json::Value>,
    /// Vertex positions (3 floats per vertex)
    pub positions: Vec<f32>,
    /// Vertex normals (3 floats per vertex, optional)
    pub normals: Option<Vec<f32>>,
    /// Vertex colors (4 u8 per vertex, RGBA, optional)
    pub colors: Option<Vec<u8>>,
    /// Triangle indices
    pub indices: Vec<u32>,
    /// Batch IDs per vertex (optional)
    pub batch_ids: Option<Vec<u32>>,
}

impl B3dmPayload {
    /// Number of vertices
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    /// Number of triangles
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Decode a B3DM file from bytes
pub fn decode_b3dm(data: &[u8]) -> Tiles3dResult<B3dmPayload> {
    if data.len() < 28 {
        return Err(Tiles3dError::InvalidB3dm("File too small for header".into()));
    }

    let header: B3dmHeader = *bytemuck::from_bytes(&data[0..28]);

    if &header.magic != b"b3dm" {
        return Err(Tiles3dError::InvalidB3dm(format!(
            "Invalid magic: {:?}",
            header.magic
        )));
    }

    if header.version != 1 {
        return Err(Tiles3dError::InvalidB3dm(format!(
            "Unsupported version: {}",
            header.version
        )));
    }

    let mut offset = 28usize;

    // Parse feature table JSON
    let ft_json_end = offset + header.feature_table_json_byte_length as usize;
    let feature_table: serde_json::Value = if header.feature_table_json_byte_length > 0 {
        let json_str = std::str::from_utf8(&data[offset..ft_json_end])
            .map_err(|e| Tiles3dError::InvalidB3dm(format!("Invalid UTF-8 in feature table: {}", e)))?;
        serde_json::from_str(json_str)?
    } else {
        serde_json::Value::Object(serde_json::Map::new())
    };
    offset = ft_json_end;

    // Skip feature table binary
    offset += header.feature_table_binary_byte_length as usize;

    // Parse batch table JSON
    let bt_json_end = offset + header.batch_table_json_byte_length as usize;
    let batch_table: Option<serde_json::Value> = if header.batch_table_json_byte_length > 0 {
        let json_str = std::str::from_utf8(&data[offset..bt_json_end])
            .map_err(|e| Tiles3dError::InvalidB3dm(format!("Invalid UTF-8 in batch table: {}", e)))?;
        Some(serde_json::from_str(json_str)?)
    } else {
        None
    };
    offset = bt_json_end;

    // Skip batch table binary
    offset += header.batch_table_binary_byte_length as usize;

    // Remaining data is glTF (binary or embedded)
    let gltf_data = &data[offset..];
    
    // Parse glTF and extract geometry
    let (positions, normals, colors, indices, batch_ids) = parse_gltf_geometry(gltf_data)?;

    Ok(B3dmPayload {
        header,
        feature_table,
        batch_table,
        positions,
        normals,
        colors,
        indices,
        batch_ids,
    })
}

/// Load and decode a B3DM file from path
pub fn load_b3dm<P: AsRef<Path>>(path: P) -> Tiles3dResult<B3dmPayload> {
    let data = std::fs::read(path)?;
    decode_b3dm(&data)
}

/// Parse glTF geometry (simplified - handles common cases)
fn parse_gltf_geometry(data: &[u8]) -> Tiles3dResult<(Vec<f32>, Option<Vec<f32>>, Option<Vec<u8>>, Vec<u32>, Option<Vec<u32>>)> {
    if data.len() < 12 {
        return Err(Tiles3dError::InvalidGltf("glTF data too small".into()));
    }

    // Check for binary glTF (GLB)
    let is_glb = data.len() >= 4 && &data[0..4] == b"glTF";
    
    if is_glb {
        parse_glb_geometry(data)
    } else {
        // Embedded JSON glTF - try to parse as JSON
        Err(Tiles3dError::Unsupported("Embedded JSON glTF not yet supported, use GLB".into()))
    }
}

/// Parse GLB (binary glTF) geometry
fn parse_glb_geometry(data: &[u8]) -> Tiles3dResult<(Vec<f32>, Option<Vec<f32>>, Option<Vec<u8>>, Vec<u32>, Option<Vec<u32>>)> {
    if data.len() < 12 {
        return Err(Tiles3dError::InvalidGltf("GLB header too small".into()));
    }

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != 2 {
        return Err(Tiles3dError::InvalidGltf(format!("Unsupported glTF version: {}", version)));
    }

    let _total_length = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    
    let mut offset = 12usize;
    let mut json_chunk: Option<serde_json::Value> = None;
    let mut bin_chunk: Option<&[u8]> = None;

    // Parse chunks
    while offset + 8 <= data.len() {
        let chunk_length = u32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]) as usize;
        let chunk_type = u32::from_le_bytes([data[offset+4], data[offset+5], data[offset+6], data[offset+7]]);
        offset += 8;

        if offset + chunk_length > data.len() {
            break;
        }

        match chunk_type {
            0x4E4F534A => { // "JSON"
                let json_str = std::str::from_utf8(&data[offset..offset + chunk_length])
                    .map_err(|e| Tiles3dError::InvalidGltf(format!("Invalid UTF-8: {}", e)))?;
                json_chunk = Some(serde_json::from_str(json_str)?);
            }
            0x004E4942 => { // "BIN\0"
                bin_chunk = Some(&data[offset..offset + chunk_length]);
            }
            _ => {}
        }
        offset += chunk_length;
    }

    let json = json_chunk.ok_or_else(|| Tiles3dError::InvalidGltf("No JSON chunk".into()))?;
    let bin = bin_chunk.unwrap_or(&[]);

    extract_mesh_data(&json, bin)
}

/// Extract mesh data from glTF JSON and binary buffer
fn extract_mesh_data(json: &serde_json::Value, bin: &[u8]) -> Tiles3dResult<(Vec<f32>, Option<Vec<f32>>, Option<Vec<u8>>, Vec<u32>, Option<Vec<u32>>)> {
    let meshes = json.get("meshes").and_then(|m| m.as_array())
        .ok_or_else(|| Tiles3dError::InvalidGltf("No meshes".into()))?;
    
    if meshes.is_empty() {
        return Err(Tiles3dError::InvalidGltf("Empty meshes array".into()));
    }

    let accessors = json.get("accessors").and_then(|a| a.as_array());
    let buffer_views = json.get("bufferViews").and_then(|b| b.as_array());

    let mut all_positions = Vec::new();
    let mut all_normals: Option<Vec<f32>> = None;
    let mut all_colors: Option<Vec<u8>> = None;
    let mut all_indices = Vec::new();
    let mut all_batch_ids: Option<Vec<u32>> = None;

    for mesh in meshes {
        let primitives = mesh.get("primitives").and_then(|p| p.as_array())
            .ok_or_else(|| Tiles3dError::InvalidGltf("No primitives".into()))?;

        for prim in primitives {
            let attributes = prim.get("attributes").and_then(|a| a.as_object())
                .ok_or_else(|| Tiles3dError::InvalidGltf("No attributes".into()))?;

            let base_vertex = (all_positions.len() / 3) as u32;

            // POSITION (required)
            if let Some(pos_idx) = attributes.get("POSITION").and_then(|p| p.as_u64()) {
                let positions = read_accessor_f32(pos_idx as usize, accessors, buffer_views, bin)?;
                all_positions.extend(positions);
            }

            // NORMAL (optional)
            if let Some(norm_idx) = attributes.get("NORMAL").and_then(|n| n.as_u64()) {
                let normals = read_accessor_f32(norm_idx as usize, accessors, buffer_views, bin)?;
                all_normals.get_or_insert_with(Vec::new).extend(normals);
            }

            // COLOR_0 (optional)
            if let Some(color_idx) = attributes.get("COLOR_0").and_then(|c| c.as_u64()) {
                if let Ok(colors) = read_accessor_u8(color_idx as usize, accessors, buffer_views, bin) {
                    all_colors.get_or_insert_with(Vec::new).extend(colors);
                }
            }

            // _BATCHID (optional, 3D Tiles specific)
            if let Some(batch_idx) = attributes.get("_BATCHID").and_then(|b| b.as_u64()) {
                if let Ok(batch_ids) = read_accessor_u32(batch_idx as usize, accessors, buffer_views, bin) {
                    all_batch_ids.get_or_insert_with(Vec::new).extend(batch_ids);
                }
            }

            // Indices
            if let Some(idx_accessor) = prim.get("indices").and_then(|i| i.as_u64()) {
                let indices = read_accessor_indices(idx_accessor as usize, accessors, buffer_views, bin)?;
                all_indices.extend(indices.iter().map(|i| i + base_vertex));
            }
        }
    }

    if all_positions.is_empty() {
        return Err(Tiles3dError::InvalidGltf("No positions extracted".into()));
    }

    Ok((all_positions, all_normals, all_colors, all_indices, all_batch_ids))
}

fn read_accessor_f32(idx: usize, accessors: Option<&Vec<serde_json::Value>>, buffer_views: Option<&Vec<serde_json::Value>>, bin: &[u8]) -> Tiles3dResult<Vec<f32>> {
    let accessors = accessors.ok_or_else(|| Tiles3dError::InvalidGltf("No accessors".into()))?;
    let accessor = accessors.get(idx).ok_or_else(|| Tiles3dError::InvalidGltf("Accessor out of range".into()))?;
    
    let bv_idx = accessor.get("bufferView").and_then(|b| b.as_u64()).unwrap_or(0) as usize;
    let count = accessor.get("count").and_then(|c| c.as_u64()).unwrap_or(0) as usize;
    let acc_type = accessor.get("type").and_then(|t| t.as_str()).unwrap_or("SCALAR");
    let byte_offset = accessor.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
    
    let components = match acc_type {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        _ => 1,
    };

    let buffer_views = buffer_views.ok_or_else(|| Tiles3dError::InvalidGltf("No bufferViews".into()))?;
    let bv = buffer_views.get(bv_idx).ok_or_else(|| Tiles3dError::InvalidGltf("BufferView out of range".into()))?;
    
    let bv_offset = bv.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
    let start = bv_offset + byte_offset;
    let float_count = count * components;
    let end = start + float_count * 4;

    if end > bin.len() {
        return Err(Tiles3dError::InvalidGltf("Buffer overrun".into()));
    }

    let mut result = Vec::with_capacity(float_count);
    for i in 0..float_count {
        let idx = start + i * 4;
        let val = f32::from_le_bytes([bin[idx], bin[idx+1], bin[idx+2], bin[idx+3]]);
        result.push(val);
    }

    Ok(result)
}

fn read_accessor_u8(idx: usize, accessors: Option<&Vec<serde_json::Value>>, buffer_views: Option<&Vec<serde_json::Value>>, bin: &[u8]) -> Tiles3dResult<Vec<u8>> {
    let accessors = accessors.ok_or_else(|| Tiles3dError::InvalidGltf("No accessors".into()))?;
    let accessor = accessors.get(idx).ok_or_else(|| Tiles3dError::InvalidGltf("Accessor out of range".into()))?;
    
    let bv_idx = accessor.get("bufferView").and_then(|b| b.as_u64()).unwrap_or(0) as usize;
    let count = accessor.get("count").and_then(|c| c.as_u64()).unwrap_or(0) as usize;
    let acc_type = accessor.get("type").and_then(|t| t.as_str()).unwrap_or("SCALAR");
    let byte_offset = accessor.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
    
    let components = match acc_type {
        "VEC3" => 3,
        "VEC4" => 4,
        _ => 4,
    };

    let buffer_views = buffer_views.ok_or_else(|| Tiles3dError::InvalidGltf("No bufferViews".into()))?;
    let bv = buffer_views.get(bv_idx).ok_or_else(|| Tiles3dError::InvalidGltf("BufferView out of range".into()))?;
    
    let bv_offset = bv.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
    let start = bv_offset + byte_offset;
    let byte_count = count * components;

    if start + byte_count > bin.len() {
        return Err(Tiles3dError::InvalidGltf("Buffer overrun".into()));
    }

    Ok(bin[start..start + byte_count].to_vec())
}

fn read_accessor_u32(idx: usize, accessors: Option<&Vec<serde_json::Value>>, buffer_views: Option<&Vec<serde_json::Value>>, bin: &[u8]) -> Tiles3dResult<Vec<u32>> {
    let accessors = accessors.ok_or_else(|| Tiles3dError::InvalidGltf("No accessors".into()))?;
    let accessor = accessors.get(idx).ok_or_else(|| Tiles3dError::InvalidGltf("Accessor out of range".into()))?;
    
    let bv_idx = accessor.get("bufferView").and_then(|b| b.as_u64()).unwrap_or(0) as usize;
    let count = accessor.get("count").and_then(|c| c.as_u64()).unwrap_or(0) as usize;
    let byte_offset = accessor.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;

    let buffer_views = buffer_views.ok_or_else(|| Tiles3dError::InvalidGltf("No bufferViews".into()))?;
    let bv = buffer_views.get(bv_idx).ok_or_else(|| Tiles3dError::InvalidGltf("BufferView out of range".into()))?;
    
    let bv_offset = bv.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
    let component_type = accessor.get("componentType").and_then(|c| c.as_u64()).unwrap_or(5125);
    
    let start = bv_offset + byte_offset;
    
    match component_type {
        5121 => { // UNSIGNED_BYTE
            if start + count > bin.len() {
                return Err(Tiles3dError::InvalidGltf("Buffer overrun".into()));
            }
            Ok(bin[start..start + count].iter().map(|&b| b as u32).collect())
        }
        5123 => { // UNSIGNED_SHORT
            if start + count * 2 > bin.len() {
                return Err(Tiles3dError::InvalidGltf("Buffer overrun".into()));
            }
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let idx = start + i * 2;
                result.push(u16::from_le_bytes([bin[idx], bin[idx+1]]) as u32);
            }
            Ok(result)
        }
        5125 => { // UNSIGNED_INT
            if start + count * 4 > bin.len() {
                return Err(Tiles3dError::InvalidGltf("Buffer overrun".into()));
            }
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let idx = start + i * 4;
                result.push(u32::from_le_bytes([bin[idx], bin[idx+1], bin[idx+2], bin[idx+3]]));
            }
            Ok(result)
        }
        _ => Err(Tiles3dError::InvalidGltf(format!("Unsupported component type: {}", component_type))),
    }
}

fn read_accessor_indices(idx: usize, accessors: Option<&Vec<serde_json::Value>>, buffer_views: Option<&Vec<serde_json::Value>>, bin: &[u8]) -> Tiles3dResult<Vec<u32>> {
    read_accessor_u32(idx, accessors, buffer_views, bin)
}
