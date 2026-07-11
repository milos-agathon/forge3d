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
    let mut intensities = Vec::with_capacity(point_count as usize);
    let mut classifications = Vec::with_capacity(point_count as usize);
    let has_rgb = header.point_format == 2
        || header.point_format == 3
        || header.point_format == 5
        || header.point_format >= 7;
    let rgb_offset = rgb_offset_for_point_format(header.point_format);
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

        positions.push(x as f64 * header.scale[0] + header.offset[0]);
        positions.push(y as f64 * header.scale[1] + header.offset[1]);
        positions.push(z as f64 * header.scale[2] + header.offset[2]);

        if off + 14 <= data.len() {
            intensities.push(u16::from_le_bytes([data[off + 12], data[off + 13]]));
        }
        if let Some(classification_off) =
            classification_offset_for_point_format(header.point_format)
        {
            if off + classification_off < data.len() {
                classifications.push(data[off + classification_off]);
            }
        }

        if let Some(ref mut cols) = colors {
            let rgb_off = off + rgb_offset.unwrap_or(20);
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
        intensities: (!intensities.is_empty()).then_some(intensities),
        classifications: (!classifications.is_empty()).then_some(classifications),
    })
}

fn classification_offset_for_point_format(point_format: u8) -> Option<usize> {
    match point_format {
        0..=5 => Some(15),
        6..=10 => Some(16),
        _ => None,
    }
}

fn rgb_offset_for_point_format(point_format: u8) -> Option<usize> {
    match point_format {
        2 => Some(20),
        3 | 5 => Some(28),
        7 | 8 | 10 => Some(30),
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn header(point_format: u8, record_len: u16) -> CopcHeader {
        CopcHeader {
            point_count: 1,
            point_format,
            point_record_length: record_len,
            scale: [0.01, 0.01, 0.01],
            offset: [100.0, 200.0, 300.0],
            min_bounds: [0.0; 3],
            max_bounds: [0.0; 3],
            num_vlrs: 0,
        }
    }

    fn write_xyz(record: &mut [u8], x: i32, y: i32, z: i32) {
        record[0..4].copy_from_slice(&x.to_le_bytes());
        record[4..8].copy_from_slice(&y.to_le_bytes());
        record[8..12].copy_from_slice(&z.to_le_bytes());
    }

    fn write_rgb16(record: &mut [u8], offset: usize, r: u16, g: u16, b: u16) {
        record[offset..offset + 2].copy_from_slice(&r.to_le_bytes());
        record[offset + 2..offset + 4].copy_from_slice(&g.to_le_bytes());
        record[offset + 4..offset + 6].copy_from_slice(&b.to_le_bytes());
    }

    #[test]
    fn parses_legacy_format3_attributes_and_rgb() {
        let mut record = vec![0u8; 34];
        write_xyz(&mut record, 100, 200, 300);
        record[12..14].copy_from_slice(&42u16.to_le_bytes());
        record[15] = 2;
        write_rgb16(&mut record, 28, 0xffff, 0x8000, 0x0000);

        let data = parse_uncompressed_points(&record, 1, &header(3, 34)).unwrap();

        assert_eq!(data.positions, vec![101.0, 202.0, 303.0]);
        assert_eq!(data.intensities, Some(vec![42]));
        assert_eq!(data.classifications, Some(vec![2]));
        assert_eq!(data.colors, Some(vec![255, 128, 0]));
    }

    #[test]
    fn parses_las14_format7_classification_and_rgb_offsets() {
        let mut record = vec![0u8; 36];
        write_xyz(&mut record, 1, 2, 3);
        record[12..14].copy_from_slice(&7u16.to_le_bytes());
        record[16] = 6;
        write_rgb16(&mut record, 30, 0x4000, 0x2000, 0xffff);

        let data = parse_uncompressed_points(&record, 1, &header(7, 36)).unwrap();

        assert_eq!(data.intensities, Some(vec![7]));
        assert_eq!(data.classifications, Some(vec![6]));
        assert_eq!(data.colors, Some(vec![64, 32, 255]));
    }
}
