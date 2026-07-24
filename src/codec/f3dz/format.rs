//! Byte-exact F3DZ v1 container primitives.
//!
//! All multibyte fields are little-endian. Floats are serialized by IEEE-754
//! bit pattern, never by text. The fixed header and every page-index record
//! are 64 bytes so readers can range-fetch the index without decoding pages.

use super::{F3dzError, F3dzResult};

pub const MAGIC: [u8; 4] = *b"F3DZ";
pub const PAGE_MAGIC: [u8; 4] = *b"F3PG";
pub const VERSION: u16 = 1;
pub const FIXED_HEADER_LEN: usize = 64;
pub const PAGE_INDEX_LEN: usize = 64;
pub const MAX_GPU_PAGE_SIZE: u16 = 64;

pub const FLAG_PROGRESSIVE: u32 = 1 << 0;
pub const FLAG_BASE_ONLY: u32 = 1 << 1;
pub const PAGE_FLAG_PROGRESSIVE: u8 = 1 << 0;
pub const PAGE_FLAG_BASE_ONLY: u8 = 1 << 1;
pub const PREDICTOR_LORENZO: u8 = 0;
pub const PREDICTOR_PLANE: u8 = 1;
pub const PREDICTOR_PREVIOUS_LOD: u8 = 2;
pub const PREDICTOR_ORDER_ZERO: u8 = 3;

/// Header predictor marker meaning that each index entry declares its own
/// selected predictor.
pub const PREDICTOR_ADAPTIVE: u8 = 0xff;

#[derive(Clone, Debug, PartialEq)]
pub struct ContainerHeader {
    pub flags: u32,
    pub width: u32,
    pub height: u32,
    pub tile_size: u16,
    pub predictor_id: u8,
    pub epsilon: f32,
    pub page_count: u32,
    pub height_datum: String,
    pub index_offset: u64,
    pub payload_offset: u64,
}

impl ContainerHeader {
    pub fn new(
        width: u32,
        height: u32,
        tile_size: u16,
        epsilon: f32,
        progressive: bool,
        height_datum: String,
    ) -> F3dzResult<Self> {
        if width == 0 || height == 0 {
            return Err(F3dzError::InvalidArgument(
                "grid dimensions must be non-zero".to_string(),
            ));
        }
        if tile_size == 0 || tile_size > MAX_GPU_PAGE_SIZE {
            return Err(F3dzError::InvalidArgument(format!(
                "tile_size must be in 1..={MAX_GPU_PAGE_SIZE}"
            )));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(F3dzError::InvalidArgument(
                "epsilon must be finite and greater than zero".to_string(),
            ));
        }
        let pages_x = width.div_ceil(u32::from(tile_size));
        let pages_y = height.div_ceil(u32::from(tile_size));
        let page_count = pages_x
            .checked_mul(pages_y)
            .ok_or_else(|| F3dzError::InvalidArgument("page count overflow".to_string()))?;
        let datum_len = u32::try_from(height_datum.len()).map_err(|_| {
            F3dzError::InvalidArgument("height datum exceeds u32 length".to_string())
        })?;
        let index_offset = (FIXED_HEADER_LEN as u64)
            .checked_add(u64::from(datum_len))
            .ok_or_else(|| F3dzError::InvalidArgument("index offset overflow".to_string()))?;
        let payload_offset = index_offset
            .checked_add(u64::from(page_count) * PAGE_INDEX_LEN as u64)
            .ok_or_else(|| F3dzError::InvalidArgument("payload offset overflow".to_string()))?;
        Ok(Self {
            flags: if progressive { FLAG_PROGRESSIVE } else { 0 },
            width,
            height,
            tile_size,
            predictor_id: PREDICTOR_ADAPTIVE,
            epsilon,
            page_count,
            height_datum,
            index_offset,
            payload_offset,
        })
    }

    pub fn progressive(&self) -> bool {
        self.flags & FLAG_PROGRESSIVE != 0
    }

    pub fn base_only(&self) -> bool {
        self.flags & FLAG_BASE_ONLY != 0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PageIndexEntry {
    pub page_x: u32,
    pub page_y: u32,
    pub width: u16,
    pub height: u16,
    pub predictor_id: u8,
    pub flags: u8,
    pub payload_offset: u64,
    pub payload_len: u32,
    pub base_layer_len: u32,
    pub crc32: u32,
    pub max_abs_err: f32,
    pub base_max_abs_err: f32,
    pub sample_count: u32,
    pub nan_count: u32,
}

impl PageIndexEntry {
    pub fn progressive(&self) -> bool {
        self.flags & PAGE_FLAG_PROGRESSIVE != 0
    }

    pub fn base_only(&self) -> bool {
        self.flags & PAGE_FLAG_BASE_ONLY != 0
    }
}

/// Serialize the header, datum bytes, and fixed-size page index. Page payloads
/// are appended by the encoder after this prefix.
pub fn write_prefix(header: &ContainerHeader, entries: &[PageIndexEntry]) -> F3dzResult<Vec<u8>> {
    validate_header(header)?;
    if entries.len() != header.page_count as usize {
        return Err(F3dzError::InvalidArgument(format!(
            "page index length {} does not match header page_count {}",
            entries.len(),
            header.page_count
        )));
    }
    let mut fixed = vec![0u8; FIXED_HEADER_LEN];
    fixed[0..4].copy_from_slice(&MAGIC);
    put_u16(&mut fixed, 4, VERSION);
    put_u16(&mut fixed, 6, FIXED_HEADER_LEN as u16);
    put_u32(&mut fixed, 8, header.flags);
    put_u32(&mut fixed, 12, header.width);
    put_u32(&mut fixed, 16, header.height);
    put_u16(&mut fixed, 20, header.tile_size);
    fixed[22] = header.predictor_id;
    put_u32(&mut fixed, 24, header.epsilon.to_bits());
    put_u32(&mut fixed, 28, header.page_count);
    put_u32(
        &mut fixed,
        32,
        u32::try_from(header.height_datum.len()).map_err(|_| {
            F3dzError::InvalidArgument("height datum exceeds u32 length".to_string())
        })?,
    );
    put_u64(&mut fixed, 36, header.index_offset);
    put_u64(&mut fixed, 44, header.payload_offset);

    let mut checksum_input = fixed.clone();
    checksum_input.extend_from_slice(header.height_datum.as_bytes());
    put_u32(&mut fixed, 52, crc32(&checksum_input));

    let capacity = usize::try_from(header.payload_offset)
        .map_err(|_| F3dzError::InvalidArgument("payload offset exceeds usize".to_string()))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&fixed);
    out.extend_from_slice(header.height_datum.as_bytes());
    for entry in entries {
        validate_entry(header, entry)?;
        let mut raw = [0u8; PAGE_INDEX_LEN];
        put_u32(&mut raw, 0, entry.page_x);
        put_u32(&mut raw, 4, entry.page_y);
        put_u16(&mut raw, 8, entry.width);
        put_u16(&mut raw, 10, entry.height);
        raw[12] = entry.predictor_id;
        raw[13] = entry.flags;
        put_u64(&mut raw, 16, entry.payload_offset);
        put_u32(&mut raw, 24, entry.payload_len);
        put_u32(&mut raw, 28, entry.base_layer_len);
        put_u32(&mut raw, 32, entry.crc32);
        put_u32(&mut raw, 36, entry.max_abs_err.to_bits());
        put_u32(&mut raw, 40, entry.base_max_abs_err.to_bits());
        put_u32(&mut raw, 44, entry.sample_count);
        put_u32(&mut raw, 48, entry.nan_count);
        out.extend_from_slice(&raw);
    }
    debug_assert_eq!(out.len(), capacity);
    Ok(out)
}

pub fn parse_prefix(data: &[u8]) -> F3dzResult<(ContainerHeader, Vec<PageIndexEntry>)> {
    require_len(data, FIXED_HEADER_LEN)?;
    if data[0..4] != MAGIC {
        return Err(F3dzError::InvalidHeader("magic must be F3DZ".to_string()));
    }
    let version = get_u16(data, 4);
    if version != VERSION {
        return Err(F3dzError::UnsupportedVersion(version));
    }
    if get_u16(data, 6) as usize != FIXED_HEADER_LEN {
        return Err(F3dzError::InvalidHeader(
            "fixed header length must be 64".to_string(),
        ));
    }
    if data[23] != 0 || data[56..64].iter().any(|&byte| byte != 0) {
        return Err(F3dzError::InvalidHeader(
            "reserved header bytes must be zero".to_string(),
        ));
    }
    let datum_len = get_u32(data, 32) as usize;
    let index_offset = get_u64(data, 36);
    let payload_offset = get_u64(data, 44);
    let datum_end = FIXED_HEADER_LEN
        .checked_add(datum_len)
        .ok_or_else(|| F3dzError::InvalidHeader("datum length overflow".to_string()))?;
    require_len(data, datum_end)?;
    if index_offset != datum_end as u64 {
        return Err(F3dzError::InvalidHeader(
            "index_offset does not follow height datum".to_string(),
        ));
    }
    let stored_crc = get_u32(data, 52);
    let mut checksum_input = data[..FIXED_HEADER_LEN].to_vec();
    put_u32(&mut checksum_input, 52, 0);
    checksum_input.extend_from_slice(&data[FIXED_HEADER_LEN..datum_end]);
    let actual_crc = crc32(&checksum_input);
    if stored_crc != actual_crc {
        return Err(F3dzError::InvalidHeader(format!(
            "header CRC mismatch: stored=0x{stored_crc:08x} actual=0x{actual_crc:08x}"
        )));
    }
    let height_datum = std::str::from_utf8(&data[FIXED_HEADER_LEN..datum_end])
        .map_err(|_| F3dzError::InvalidHeader("height datum is not UTF-8".to_string()))?
        .to_string();
    let header = ContainerHeader {
        flags: get_u32(data, 8),
        width: get_u32(data, 12),
        height: get_u32(data, 16),
        tile_size: get_u16(data, 20),
        predictor_id: data[22],
        epsilon: f32::from_bits(get_u32(data, 24)),
        page_count: get_u32(data, 28),
        height_datum,
        index_offset,
        payload_offset,
    };
    validate_header(&header)?;
    let expected_payload = header
        .index_offset
        .checked_add(u64::from(header.page_count) * PAGE_INDEX_LEN as u64)
        .ok_or_else(|| F3dzError::InvalidHeader("page index overflow".to_string()))?;
    if header.payload_offset != expected_payload {
        return Err(F3dzError::InvalidHeader(
            "payload_offset does not follow page index".to_string(),
        ));
    }
    let payload_start = usize::try_from(header.payload_offset)
        .map_err(|_| F3dzError::InvalidHeader("payload offset exceeds usize".to_string()))?;
    require_len(data, payload_start)?;
    let mut entries = Vec::with_capacity(header.page_count as usize);
    let mut offset = datum_end;
    for page in 0..header.page_count as usize {
        let end = offset + PAGE_INDEX_LEN;
        require_len(data, end)?;
        let raw = &data[offset..end];
        if raw[14..16].iter().any(|&byte| byte != 0) || raw[52..64].iter().any(|&byte| byte != 0) {
            return Err(F3dzError::CorruptPage {
                page,
                reason: "reserved page-index bytes must be zero".to_string(),
            });
        }
        let entry = PageIndexEntry {
            page_x: get_u32(raw, 0),
            page_y: get_u32(raw, 4),
            width: get_u16(raw, 8),
            height: get_u16(raw, 10),
            predictor_id: raw[12],
            flags: raw[13],
            payload_offset: get_u64(raw, 16),
            payload_len: get_u32(raw, 24),
            base_layer_len: get_u32(raw, 28),
            crc32: get_u32(raw, 32),
            max_abs_err: f32::from_bits(get_u32(raw, 36)),
            base_max_abs_err: f32::from_bits(get_u32(raw, 40)),
            sample_count: get_u32(raw, 44),
            nan_count: get_u32(raw, 48),
        };
        validate_entry(&header, &entry).map_err(|error| F3dzError::CorruptPage {
            page,
            reason: error.to_string(),
        })?;
        entries.push(entry);
        offset = end;
    }
    Ok((header, entries))
}

fn validate_header(header: &ContainerHeader) -> F3dzResult<()> {
    if header.width == 0 || header.height == 0 {
        return Err(F3dzError::InvalidHeader(
            "grid dimensions must be non-zero".to_string(),
        ));
    }
    if header.tile_size == 0 || header.tile_size > MAX_GPU_PAGE_SIZE {
        return Err(F3dzError::InvalidHeader(format!(
            "tile_size must be in 1..={MAX_GPU_PAGE_SIZE}"
        )));
    }
    if !header.epsilon.is_finite() || header.epsilon <= 0.0 {
        return Err(F3dzError::InvalidHeader(
            "epsilon must be finite and greater than zero".to_string(),
        ));
    }
    let expected_count = header
        .width
        .div_ceil(u32::from(header.tile_size))
        .checked_mul(header.height.div_ceil(u32::from(header.tile_size)))
        .ok_or_else(|| F3dzError::InvalidHeader("page count overflow".to_string()))?;
    if header.page_count != expected_count {
        return Err(F3dzError::InvalidHeader(format!(
            "page_count {} does not match grid-derived count {expected_count}",
            header.page_count
        )));
    }
    if header.flags & !(FLAG_PROGRESSIVE | FLAG_BASE_ONLY) != 0 {
        return Err(F3dzError::InvalidHeader(
            "unknown container flag bits are set".to_string(),
        ));
    }
    if header.base_only() && !header.progressive() {
        return Err(F3dzError::InvalidHeader(
            "base-only flag requires progressive flag".to_string(),
        ));
    }
    Ok(())
}

fn validate_entry(header: &ContainerHeader, entry: &PageIndexEntry) -> F3dzResult<()> {
    if entry.width == 0
        || entry.height == 0
        || entry.width > header.tile_size
        || entry.height > header.tile_size
    {
        return Err(F3dzError::InvalidArgument(
            "page dimensions exceed tile_size or are zero".to_string(),
        ));
    }
    let pages_x = header.width.div_ceil(u32::from(header.tile_size));
    let pages_y = header.height.div_ceil(u32::from(header.tile_size));
    if entry.page_x >= pages_x || entry.page_y >= pages_y {
        return Err(F3dzError::InvalidArgument(
            "page coordinates are outside the container grid".to_string(),
        ));
    }
    let expected_width = (header.width - entry.page_x * u32::from(header.tile_size))
        .min(u32::from(header.tile_size));
    let expected_height = (header.height - entry.page_y * u32::from(header.tile_size))
        .min(u32::from(header.tile_size));
    if u32::from(entry.width) != expected_width || u32::from(entry.height) != expected_height {
        return Err(F3dzError::InvalidArgument(
            "page dimensions do not match their grid-edge extent".to_string(),
        ));
    }
    if entry.predictor_id > PREDICTOR_ORDER_ZERO {
        return Err(F3dzError::InvalidArgument(
            "unknown page predictor id".to_string(),
        ));
    }
    let sample_count = u32::from(entry.width) * u32::from(entry.height);
    if entry.sample_count != sample_count || entry.nan_count > sample_count {
        return Err(F3dzError::InvalidArgument(
            "page sample/nodata counts are inconsistent".to_string(),
        ));
    }
    if entry.payload_offset < header.payload_offset {
        return Err(F3dzError::InvalidArgument(
            "page payload precedes payload region".to_string(),
        ));
    }
    if entry.payload_len == 0
        || entry.base_layer_len == 0
        || entry.base_layer_len > entry.payload_len
    {
        return Err(F3dzError::InvalidArgument(
            "page payload lengths are inconsistent".to_string(),
        ));
    }
    entry
        .payload_offset
        .checked_add(u64::from(entry.payload_len))
        .ok_or_else(|| F3dzError::InvalidArgument("page payload range overflow".to_string()))?;
    if !entry.max_abs_err.is_finite()
        || entry.max_abs_err < 0.0
        || !entry.base_max_abs_err.is_finite()
        || entry.base_max_abs_err < 0.0
    {
        return Err(F3dzError::InvalidArgument(
            "page error fields must be finite and non-negative".to_string(),
        ));
    }
    if entry.flags & !(PAGE_FLAG_PROGRESSIVE | PAGE_FLAG_BASE_ONLY) != 0 {
        return Err(F3dzError::InvalidArgument(
            "unknown page flag bits are set".to_string(),
        ));
    }
    if entry.progressive() != header.progressive() || entry.base_only() != header.base_only() {
        return Err(F3dzError::InvalidArgument(
            "page quality flags disagree with the container header".to_string(),
        ));
    }
    if !entry.progressive() && entry.base_layer_len != entry.payload_len {
        return Err(F3dzError::InvalidArgument(
            "non-progressive page must contain exactly one layer".to_string(),
        ));
    }
    if entry.max_abs_err > header.epsilon {
        return Err(F3dzError::InvalidArgument(
            "page max_abs_err exceeds the declared epsilon".to_string(),
        ));
    }
    let base_bound = if entry.progressive() {
        header.epsilon * 4.0
    } else {
        header.epsilon
    };
    if entry.base_max_abs_err > base_bound {
        return Err(F3dzError::InvalidArgument(
            "page base_max_abs_err exceeds its declared bound".to_string(),
        ));
    }
    Ok(())
}

fn require_len(data: &[u8], needed: usize) -> F3dzResult<()> {
    if data.len() < needed {
        Err(F3dzError::Truncated {
            needed,
            available: data.len(),
        })
    } else {
        Ok(())
    }
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn get_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn get_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn get_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

fn put_u16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(data: &mut [u8], offset: usize, value: u64) {
    data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(offset: u64) -> PageIndexEntry {
        PageIndexEntry {
            page_x: 0,
            page_y: 0,
            width: 32,
            height: 16,
            predictor_id: 1,
            flags: PAGE_FLAG_PROGRESSIVE,
            payload_offset: offset,
            payload_len: 100,
            base_layer_len: 40,
            crc32: 0x1234_5678,
            max_abs_err: 0.25,
            base_max_abs_err: 0.9,
            sample_count: 512,
            nan_count: 3,
        }
    }

    #[test]
    fn prefix_is_byte_exact_and_round_trips() {
        let header = ContainerHeader::new(32, 16, 32, 0.25, true, "EGM96".to_string()).unwrap();
        let raw = write_prefix(&header, &[entry(header.payload_offset)]).unwrap();
        assert_eq!(raw.len(), header.payload_offset as usize);
        assert_eq!(&raw[0..4], b"F3DZ");
        assert_eq!(u16::from_le_bytes([raw[4], raw[5]]), VERSION);
        let (decoded, entries) = parse_prefix(&raw).unwrap();
        assert_eq!(decoded, header);
        assert_eq!(entries, vec![entry(header.payload_offset)]);
    }

    #[test]
    fn corrupt_header_crc_fails_closed() {
        let header = ContainerHeader::new(32, 16, 32, 0.25, true, "EGM96".to_string()).unwrap();
        let mut raw = write_prefix(&header, &[entry(header.payload_offset)]).unwrap();
        raw[24] ^= 0x01;
        assert!(matches!(
            parse_prefix(&raw),
            Err(F3dzError::InvalidHeader(message)) if message.contains("CRC mismatch")
        ));
    }

    #[test]
    fn crc32_matches_standard_check_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }
}
