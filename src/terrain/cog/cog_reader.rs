//! P3.1-P3.2: COG HeightReader implementation.

use super::cache::CogTileCache;
use super::error::CogError;
use super::ifd_parser::{
    parse_cog_header, CogHeader, COMPRESSION_DEFLATE, COMPRESSION_DEFLATE_ALT,
    COMPRESSION_LZW, COMPRESSION_NONE, SAMPLE_FORMAT_FLOAT, SAMPLE_FORMAT_INT, SAMPLE_FORMAT_UINT,
};
use super::range_reader::RangeReader;
use crate::terrain::page_table::HeightReader;
use crate::terrain::tiling::TileBounds;
use glam::Vec2;
use std::sync::Arc;

/// COG-based height reader implementing the HeightReader trait.
pub struct CogHeightReader {
    reader: Arc<RangeReader>,
    header: CogHeader,
    cache: Arc<CogTileCache>,
    runtime: tokio::runtime::Handle,
}

impl CogHeightReader {
    /// Create a new COG height reader from a URL.
    pub async fn new(url: &str, cache_size_mb: u32) -> Result<Self, CogError> {
        let reader = if url.starts_with("file://") {
            let path = url.strip_prefix("file://").unwrap_or(url);
            RangeReader::new_local(path)?
        } else {
            RangeReader::new(url).await?
        };

        let reader = Arc::new(reader);
        let header = parse_cog_header(&reader).await?;
        let cache = Arc::new(CogTileCache::new(cache_size_mb));

        let runtime = tokio::runtime::Handle::current();

        Ok(Self {
            reader,
            header,
            cache,
            runtime,
        })
    }

    /// Create with an existing tokio runtime handle.
    pub async fn new_with_runtime(
        url: &str,
        cache_size_mb: u32,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, CogError> {
        let reader = if url.starts_with("file://") {
            let path = url.strip_prefix("file://").unwrap_or(url);
            RangeReader::new_local(path)?
        } else {
            RangeReader::new(url).await?
        };

        let reader = Arc::new(reader);
        let header = parse_cog_header(&reader).await?;
        let cache = Arc::new(CogTileCache::new(cache_size_mb));

        Ok(Self {
            reader,
            header,
            cache,
            runtime,
        })
    }

    /// Get geographic bounds (from first IFD).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        if let Some(ifd) = self.header.full_resolution() {
            (0.0, 0.0, ifd.width as f64, ifd.height as f64)
        } else {
            (0.0, 0.0, 1.0, 1.0)
        }
    }

    /// Get number of overview levels.
    pub fn overview_count(&self) -> usize {
        self.header.ifds.len()
    }

    /// Get the COG header for inspection.
    pub fn header(&self) -> &CogHeader {
        &self.header
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> super::cache::CogCacheStats {
        self.cache.stats()
    }

    /// Read a specific tile at given LOD.
    pub fn read_tile(&self, tile_x: u32, tile_y: u32, lod: u32) -> Result<Vec<f32>, CogError> {
        let ifd = self.header.select_ifd_for_lod(lod);

        let cache_key = (tile_x, tile_y, lod);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        let tile_idx = ifd.tile_index(tile_x, tile_y).ok_or(CogError::TileNotFound {
            x: tile_x,
            y: tile_y,
            lod,
        })?;

        if tile_idx >= ifd.tile_offsets.len() || tile_idx >= ifd.tile_byte_counts.len() {
            return Err(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            });
        }

        let offset = ifd.tile_offsets[tile_idx];
        let byte_count = ifd.tile_byte_counts[tile_idx];

        let reader = self.reader.clone();
        let compression = ifd.compression;
        let bits_per_sample = ifd.bits_per_sample;
        let sample_format = ifd.sample_format;
        let tile_width = ifd.tile_width;
        let tile_height = ifd.tile_height;

        let heights = self.runtime.block_on(async move {
            let compressed = reader.read_range(offset, byte_count).await?;
            let decompressed = decompress_tile(&compressed, compression)?;
            decode_heights(&decompressed, bits_per_sample, sample_format, tile_width, tile_height)
        })?;

        let tile_size = (tile_width * tile_height) as usize;
        let memory_bytes = tile_size * std::mem::size_of::<f32>();
        self.cache.insert(cache_key, heights.clone(), memory_bytes);

        Ok(heights)
    }

    /// Read tile async.
    pub async fn read_tile_async(
        &self,
        tile_x: u32,
        tile_y: u32,
        lod: u32,
    ) -> Result<Vec<f32>, CogError> {
        let ifd = self.header.select_ifd_for_lod(lod);

        let cache_key = (tile_x, tile_y, lod);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        let tile_idx = ifd.tile_index(tile_x, tile_y).ok_or(CogError::TileNotFound {
            x: tile_x,
            y: tile_y,
            lod,
        })?;

        if tile_idx >= ifd.tile_offsets.len() || tile_idx >= ifd.tile_byte_counts.len() {
            return Err(CogError::TileNotFound {
                x: tile_x,
                y: tile_y,
                lod,
            });
        }

        let offset = ifd.tile_offsets[tile_idx];
        let byte_count = ifd.tile_byte_counts[tile_idx];

        let compressed = self.reader.read_range(offset, byte_count).await?;
        let decompressed = decompress_tile(&compressed, ifd.compression)?;
        let heights = decode_heights(
            &decompressed,
            ifd.bits_per_sample,
            ifd.sample_format,
            ifd.tile_width,
            ifd.tile_height,
        )?;

        let tile_size = (ifd.tile_width * ifd.tile_height) as usize;
        let memory_bytes = tile_size * std::mem::size_of::<f32>();
        self.cache.insert(cache_key, heights.clone(), memory_bytes);

        Ok(heights)
    }
}

impl HeightReader for CogHeightReader {
    fn read(
        &self,
        _root_bounds: &TileBounds,
        _tile_size: Vec2,
        tile_id: crate::terrain::tiling::TileId,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        match self.read_tile(tile_id.x, tile_id.y, tile_id.lod) {
            Ok(heights) => {
                if heights.len() == (width * height) as usize {
                    heights
                } else {
                    resample_tile(&heights, width, height)
                }
            }
            Err(e) => {
                log::warn!("COG tile read failed: {:?}", e);
                vec![0.0f32; (width * height) as usize]
            }
        }
    }
}

fn decompress_tile(data: &[u8], compression: u16) -> Result<Vec<u8>, CogError> {
    match compression {
        COMPRESSION_NONE => Ok(data.to_vec()),
        COMPRESSION_DEFLATE | COMPRESSION_DEFLATE_ALT => {
            use flate2::read::ZlibDecoder;
            use std::io::Read;

            let mut decoder = ZlibDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| CogError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
        COMPRESSION_LZW => decompress_lzw(data),
        other => Err(CogError::UnsupportedCompression(other)),
    }
}

fn decompress_lzw(data: &[u8]) -> Result<Vec<u8>, CogError> {
    const CLEAR_CODE: u16 = 256;
    const EOI_CODE: u16 = 257;
    const FIRST_CODE: u16 = 258;

    let mut output = Vec::new();
    let mut table: Vec<Vec<u8>> = (0u16..256).map(|i| vec![i as u8]).collect();
    table.push(Vec::new()); // CLEAR_CODE placeholder
    table.push(Vec::new()); // EOI_CODE placeholder

    let mut bit_reader = LzwBitReader::new(data);
    let mut code_size = 9u8;
    let mut prev_code: Option<u16> = None;

    loop {
        let code = match bit_reader.read_bits(code_size) {
            Some(c) => c,
            None => break,
        };

        if code == EOI_CODE {
            break;
        }

        if code == CLEAR_CODE {
            table.truncate(258);
            code_size = 9;
            prev_code = None;
            continue;
        }

        let entry = if (code as usize) < table.len() {
            table[code as usize].clone()
        } else if code as usize == table.len() {
            if let Some(pc) = prev_code {
                let mut e = table[pc as usize].clone();
                e.push(e[0]);
                e
            } else {
                return Err(CogError::DecompressionError("LZW: invalid code sequence".into()));
            }
        } else {
            return Err(CogError::DecompressionError(format!(
                "LZW: code {} out of range (table size {})",
                code,
                table.len()
            )));
        };

        output.extend_from_slice(&entry);

        if let Some(pc) = prev_code {
            if table.len() < 4096 {
                let mut new_entry = table[pc as usize].clone();
                new_entry.push(entry[0]);
                table.push(new_entry);

                if table.len() == (1 << code_size) && code_size < 12 {
                    code_size += 1;
                }
            }
        }

        prev_code = Some(code);
    }

    Ok(output)
}

struct LzwBitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> LzwBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bits(&mut self, count: u8) -> Option<u16> {
        let mut result: u32 = 0;
        let mut bits_read = 0u8;

        while bits_read < count {
            if self.byte_pos >= self.data.len() {
                return None;
            }

            let bits_available = 8 - self.bit_pos;
            let bits_needed = count - bits_read;
            let bits_to_read = bits_available.min(bits_needed);

            let mask = ((1u16 << bits_to_read) - 1) as u8;
            let shift = 8 - self.bit_pos - bits_to_read;
            let bits = (self.data[self.byte_pos] >> shift) & mask;

            result = (result << bits_to_read) | (bits as u32);
            bits_read += bits_to_read;
            self.bit_pos += bits_to_read;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Some(result as u16)
    }
}

fn decode_heights(
    data: &[u8],
    bits_per_sample: u16,
    sample_format: u16,
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<f32>, CogError> {
    let pixel_count = (tile_width * tile_height) as usize;
    let mut heights = Vec::with_capacity(pixel_count);

    match (bits_per_sample, sample_format) {
        (32, SAMPLE_FORMAT_FLOAT) => {
            if data.len() < pixel_count * 4 {
                return Err(CogError::InvalidIfd(format!(
                    "Data too short: {} < {}",
                    data.len(),
                    pixel_count * 4
                )));
            }
            for i in 0..pixel_count {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().unwrap();
                heights.push(f32::from_le_bytes(bytes));
            }
        }
        (64, SAMPLE_FORMAT_FLOAT) => {
            if data.len() < pixel_count * 8 {
                return Err(CogError::InvalidIfd("Data too short for f64".into()));
            }
            for i in 0..pixel_count {
                let bytes: [u8; 8] = data[i * 8..(i + 1) * 8].try_into().unwrap();
                heights.push(f64::from_le_bytes(bytes) as f32);
            }
        }
        (16, SAMPLE_FORMAT_UINT) => {
            if data.len() < pixel_count * 2 {
                return Err(CogError::InvalidIfd("Data too short for u16".into()));
            }
            for i in 0..pixel_count {
                let bytes: [u8; 2] = data[i * 2..(i + 1) * 2].try_into().unwrap();
                let val = u16::from_le_bytes(bytes);
                heights.push(val as f32);
            }
        }
        (16, SAMPLE_FORMAT_INT) => {
            if data.len() < pixel_count * 2 {
                return Err(CogError::InvalidIfd("Data too short for i16".into()));
            }
            for i in 0..pixel_count {
                let bytes: [u8; 2] = data[i * 2..(i + 1) * 2].try_into().unwrap();
                let val = i16::from_le_bytes(bytes);
                heights.push(val as f32);
            }
        }
        (32, SAMPLE_FORMAT_INT) => {
            if data.len() < pixel_count * 4 {
                return Err(CogError::InvalidIfd("Data too short for i32".into()));
            }
            for i in 0..pixel_count {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into().unwrap();
                let val = i32::from_le_bytes(bytes);
                heights.push(val as f32);
            }
        }
        (8, _) => {
            for &byte in data.iter().take(pixel_count) {
                heights.push(byte as f32);
            }
        }
        _ => {
            return Err(CogError::UnsupportedSampleFormat {
                bits: bits_per_sample,
                format: sample_format,
            });
        }
    }

    while heights.len() < pixel_count {
        heights.push(0.0);
    }

    Ok(heights)
}

fn resample_tile(src: &[f32], dst_width: u32, dst_height: u32) -> Vec<f32> {
    let src_side = (src.len() as f32).sqrt() as u32;
    if src_side == 0 {
        return vec![0.0f32; (dst_width * dst_height) as usize];
    }

    let mut dst = Vec::with_capacity((dst_width * dst_height) as usize);
    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_x = (x as f32 / dst_width as f32 * src_side as f32) as u32;
            let src_y = (y as f32 / dst_height as f32 * src_side as f32) as u32;
            let src_idx = (src_y.min(src_side - 1) * src_side + src_x.min(src_side - 1)) as usize;
            dst.push(src.get(src_idx).copied().unwrap_or(0.0));
        }
    }
    dst
}
