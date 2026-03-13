//! COPC chunk decoding: uncompressed point parsing and LAZ decompression.

use super::copc::{CopcHeader, PointData};
use super::error::{PointCloudError, PointCloudResult};

/// Decode a chunk of point data, dispatching to LAZ decompression if needed.
pub(crate) fn decode_chunk(
    data: &[u8],
    point_count: u32,
    header: &CopcHeader,
    laz_vlr_data: &Option<Vec<u8>>,
) -> PointCloudResult<PointData> {
    let record_len = header.point_record_length as usize;
    let expected = point_count as usize * record_len;

    if data.len() >= expected {
        return parse_uncompressed_points(data, point_count, header);
    }

    // Data is smaller than expected: must be LAZ compressed
    decompress_and_parse(data, point_count, header, laz_vlr_data)
}

/// Parse uncompressed (raw LAS) point records into `PointData`.
///
/// Shared parsing logic used by both the uncompressed and decompressed paths.
pub(crate) fn parse_uncompressed_points(
    data: &[u8],
    point_count: u32,
    header: &CopcHeader,
) -> PointCloudResult<PointData> {
    let record_len = header.point_record_length as usize;
    let expected = point_count as usize * record_len;
    if data.len() < expected {
        return Err(PointCloudError::InvalidCopc(format!(
            "Buffer too small: have {} bytes, need {} ({} points * {} record_len)",
            data.len(),
            expected,
            point_count,
            record_len,
        )));
    }

    let mut positions = Vec::with_capacity(point_count as usize * 3);
    let has_rgb = header.point_format == 2
        || header.point_format == 3
        || header.point_format == 5
        || header.point_format >= 7;
    let mut colors = if has_rgb {
        Some(Vec::with_capacity(point_count as usize * 3))
    } else {
        None
    };

    for i in 0..point_count as usize {
        let off = i * record_len;
        let x = i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let y = i32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let z = i32::from_le_bytes([data[off + 8], data[off + 9], data[off + 10], data[off + 11]]);

        positions.push((x as f64 * header.scale[0] + header.offset[0]) as f32);
        positions.push((y as f64 * header.scale[1] + header.offset[1]) as f32);
        positions.push((z as f64 * header.scale[2] + header.offset[2]) as f32);

        if let Some(ref mut cols) = colors {
            let rgb_off = off + 20;
            if rgb_off + 6 <= data.len() {
                cols.push((u16::from_le_bytes([data[rgb_off], data[rgb_off + 1]]) >> 8) as u8);
                cols.push((u16::from_le_bytes([data[rgb_off + 2], data[rgb_off + 3]]) >> 8) as u8);
                cols.push((u16::from_le_bytes([data[rgb_off + 4], data[rgb_off + 5]]) >> 8) as u8);
            }
        }
    }

    Ok(PointData {
        positions,
        colors,
        intensities: None,
    })
}

// ---------------------------------------------------------------------------
// LAZ decompression (feature-gated)
// ---------------------------------------------------------------------------

/// Dispatch to the feature-gated decompressor or return an explicit error.
fn decompress_and_parse(
    data: &[u8],
    point_count: u32,
    header: &CopcHeader,
    laz_vlr_data: &Option<Vec<u8>>,
) -> PointCloudResult<PointData> {
    #[cfg(feature = "copc_laz")]
    {
        decompress_laz_chunk(data, point_count, header, laz_vlr_data)
    }

    #[cfg(not(feature = "copc_laz"))]
    {
        let _ = (data, point_count, header, laz_vlr_data);
        Err(PointCloudError::InvalidLaz(
            "LAZ decompression requires the 'copc_laz' Cargo feature. \
             Rebuild with: maturin develop --release --features copc_laz"
                .into(),
        ))
    }
}

/// Decompress a LAZ chunk using the `laz` crate and parse the resulting points.
#[cfg(feature = "copc_laz")]
fn decompress_laz_chunk(
    compressed: &[u8],
    point_count: u32,
    header: &CopcHeader,
    laz_vlr_data: &Option<Vec<u8>>,
) -> PointCloudResult<PointData> {
    let vlr_bytes = laz_vlr_data.as_ref().ok_or_else(|| {
        PointCloudError::InvalidLaz("Data is LAZ-compressed but no laszip VLR found in file".into())
    })?;

    let laz_vlr = laz::LazVlr::read_from(std::io::Cursor::new(vlr_bytes))
        .map_err(|e| PointCloudError::InvalidLaz(format!("Failed to parse LAZ VLR: {}", e)))?;

    let record_len = header.point_record_length as usize;
    let decompressed_size = point_count as usize * record_len;
    let mut decompressed = vec![0u8; decompressed_size];

    let cursor = std::io::Cursor::new(compressed);
    let mut decompressor = laz::LasZipDecompressor::new(cursor, laz_vlr).map_err(|e| {
        PointCloudError::InvalidLaz(format!("Failed to create LAZ decompressor: {}", e))
    })?;

    decompressor
        .decompress_many(&mut decompressed)
        .map_err(|e| PointCloudError::InvalidLaz(format!("LAZ decompression failed: {}", e)))?;

    parse_uncompressed_points(&decompressed, point_count, header)
}
