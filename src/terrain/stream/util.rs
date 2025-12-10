use wgpu;

pub fn padded_bytes_per_row(unpadded: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    ((unpadded + align - 1) / align) * align
}

pub fn copy_rows_with_padding(
    src: &[u8],
    row_bytes: usize,
    padded_row_bytes: usize,
    rows: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; padded_row_bytes * rows];
    if row_bytes == padded_row_bytes {
        out.chunks_exact_mut(row_bytes)
            .zip(src.chunks_exact(row_bytes))
            .for_each(|(dst, src)| dst.copy_from_slice(src));
        return out;
    }
    for row in 0..rows {
        let src_offset = row * row_bytes;
        let dst_offset = row * padded_row_bytes;
        out[dst_offset..dst_offset + row_bytes]
            .copy_from_slice(&src[src_offset..src_offset + row_bytes]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_bytes_per_row_aligns() {
        let bpr = padded_bytes_per_row(12);
        assert_eq!(bpr, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        assert_eq!(padded_bytes_per_row(256), 256);
    }

    #[test]
    fn copy_rows_with_padding_preserves_data() {
        let rows = 3usize;
        let row_bytes = 6usize;
        let padded = 256usize;
        let mut src = Vec::with_capacity(row_bytes * rows);
        for row in 0..rows {
            for col in 0..row_bytes {
                src.push((row * row_bytes + col) as u8);
            }
        }
        let padded_bytes = copy_rows_with_padding(&src, row_bytes, padded, rows);
        assert_eq!(padded_bytes.len(), padded * rows);
        for row in 0..rows {
            let src_offset = row * row_bytes;
            let dst_offset = row * padded;
            assert_eq!(
                &padded_bytes[dst_offset..dst_offset + row_bytes],
                &src[src_offset..src_offset + row_bytes]
            );
        }
    }
}
