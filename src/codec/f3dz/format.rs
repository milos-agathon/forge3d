//! Byte-exact F3DZ v1 container primitives.
//!
//! All multibyte fields are little-endian. Floats are serialized by IEEE-754
//! bit pattern, never by text. The fixed header and every page-index record
//! are 64 bytes so readers can range-fetch the index without decoding pages.

use super::predict::PlaneModel;
use super::{F3dzError, F3dzResult};

pub const MAGIC: [u8; 4] = *b"F3DZ";
pub const PAGE_MAGIC: [u8; 4] = *b"F3PG";
pub const VERSION: u16 = 1;
pub const FIXED_HEADER_LEN: usize = 64;
pub const PAGE_INDEX_LEN: usize = 64;
pub(crate) const PAGE_HEADER_LEN: usize = 56;
pub const MAX_GPU_PAGE_SIZE: u16 = 64;

pub const FLAG_PROGRESSIVE: u32 = 1 << 0;
pub const FLAG_BASE_ONLY: u32 = 1 << 1;
pub const PAGE_FLAG_PROGRESSIVE: u8 = 1 << 0;
pub const PAGE_FLAG_BASE_ONLY: u8 = 1 << 1;
/// The base layer supports the low-barrier GPU path: a proved separable prefix
/// scan for Lorenzo or independent reconstruction for plane/order-zero. The
/// GPU still validates every sample, so a forged flag fails closed.
pub const PAGE_FLAG_GPU_FAST: u8 = 1 << 2;
/// Both entropy layers contain exactly one canonical single-byte residual token
/// per sample, allowing workgroup-parallel token parsing.
pub const PAGE_FLAG_GPU_DIRECT: u8 = 1 << 3;
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

    pub fn gpu_fast(&self) -> bool {
        self.flags & PAGE_FLAG_GPU_FAST != 0
    }

    pub fn gpu_direct(&self) -> bool {
        self.flags & PAGE_FLAG_GPU_DIRECT != 0
    }
}

/// Structural facts checked before either execution adapter decodes symbols.
/// Entropy layers borrow the original bytes; validation never entropy-decodes.
pub(crate) struct ValidatedPage<'a> {
    pub entry: PageIndexEntry,
    pub enhancement_predictor: u8,
    pub plane: PlaneModel,
    pub fine_step: f32,
    pub base_step: f32,
    pub base_scale: u32,
    pub enhancement_scale: u32,
    pub base_layer: &'a [u8],
    pub enhancement_layer: &'a [u8],
}

pub(crate) struct ValidatedStream<'a> {
    pub header: ContainerHeader,
    pub pages: Vec<ValidatedPage<'a>>,
}

pub(crate) fn validate_stream(data: &[u8]) -> F3dzResult<ValidatedStream<'_>> {
    let (header, entries) = parse_prefix(data)?;
    let pages = validate_pages(data, &header, entries)?;
    Ok(ValidatedStream { header, pages })
}

/// Keep prefix parsing separate for callers that compare the demanded epsilon
/// before validating payloads, and for range readers that only have the index.
pub(crate) fn validate_pages<'a>(
    data: &'a [u8],
    header: &ContainerHeader,
    entries: Vec<PageIndexEntry>,
) -> F3dzResult<Vec<ValidatedPage<'a>>> {
    let mut pages = Vec::with_capacity(entries.len());
    let mut expected_payload_offset = header.payload_offset;
    let pages_x = header.width.div_ceil(u32::from(header.tile_size));
    for (page_index, entry) in entries.into_iter().enumerate() {
        let corrupt = |reason| F3dzError::CorruptPage {
            page: page_index,
            reason,
        };
        if entry.page_x != page_index as u32 % pages_x
            || entry.page_y != page_index as u32 / pages_x
        {
            return Err(corrupt(
                "page index entries must appear once in canonical row-major order".to_string(),
            ));
        }
        if entry.payload_offset != expected_payload_offset {
            return Err(corrupt(format!(
                "payload offset is not canonical/contiguous: expected={expected_payload_offset} actual={}",
                entry.payload_offset
            )));
        }
        let start = usize::try_from(entry.payload_offset)
            .map_err(|_| corrupt("payload offset exceeds usize".to_string()))?;
        let end = start
            .checked_add(entry.payload_len as usize)
            .ok_or_else(|| corrupt("payload range overflow".to_string()))?;
        require_len(data, end)?;
        let payload = &data[start..end];
        let actual_crc = crc32(payload);
        if actual_crc != entry.crc32 {
            return Err(corrupt(format!(
                "CRC mismatch: stored=0x{:08x} actual=0x{actual_crc:08x}",
                entry.crc32
            )));
        }
        expected_payload_offset = entry
            .payload_offset
            .checked_add(u64::from(entry.payload_len))
            .ok_or_else(|| corrupt("payload offset overflow".to_string()))?;
        let page =
            validate_page(header, entry, payload).map_err(|error| corrupt(error.to_string()))?;
        pages.push(page);
    }
    if expected_payload_offset != data.len() as u64 {
        return Err(F3dzError::InvalidHeader(format!(
            "container has trailing or missing payload bytes: indexed_end={expected_payload_offset} actual={}",
            data.len()
        )));
    }
    Ok(pages)
}

fn validate_page<'a>(
    header: &ContainerHeader,
    entry: PageIndexEntry,
    payload: &'a [u8],
) -> F3dzResult<ValidatedPage<'a>> {
    require_len(payload, PAGE_HEADER_LEN)?;
    if payload[0..4] != PAGE_MAGIC {
        return Err(F3dzError::InvalidArgument(
            "page payload magic must be F3PG".to_string(),
        ));
    }
    if get_u16(payload, 4) != VERSION {
        return Err(F3dzError::InvalidArgument(
            "page payload version mismatch".to_string(),
        ));
    }
    if get_u16(payload, 6) != u16::from(entry.flags)
        || get_u16(payload, 10) != 0
        || get_u32(payload, 52) != 0
    {
        return Err(F3dzError::InvalidArgument(
            "page payload flags/reserved fields disagree with its container".to_string(),
        ));
    }
    let predictor_id = payload[8];
    let enhancement_predictor = payload[9];
    if !matches!(
        predictor_id,
        PREDICTOR_LORENZO | PREDICTOR_PLANE | PREDICTOR_ORDER_ZERO
    ) || (header.progressive() && enhancement_predictor != PREDICTOR_PREVIOUS_LOD)
        || (!header.progressive() && enhancement_predictor != predictor_id)
    {
        return Err(F3dzError::InvalidArgument(
            "invalid base/enhancement predictor ids".to_string(),
        ));
    }
    if predictor_id != entry.predictor_id {
        return Err(F3dzError::InvalidArgument(
            "payload predictor disagrees with page index".to_string(),
        ));
    }
    if get_u32(payload, 32) != entry.sample_count {
        return Err(F3dzError::InvalidArgument(
            "page token count disagrees with page index".to_string(),
        ));
    }
    let fine_step = f32::from_bits(get_u32(payload, 36));
    let base_step = f32::from_bits(get_u32(payload, 40));
    if !fine_step.is_finite() || fine_step <= 0.0 || !base_step.is_finite() || base_step <= 0.0 {
        return Err(F3dzError::InvalidArgument(
            "invalid page quantization steps".to_string(),
        ));
    }
    let expected_fine_step = header.epsilon * 2.0;
    let expected_base_step = if header.progressive() {
        header.epsilon * 8.0
    } else {
        expected_fine_step
    };
    if fine_step.to_bits() != expected_fine_step.to_bits()
        || base_step.to_bits() != expected_base_step.to_bits()
    {
        return Err(F3dzError::InvalidArgument(format!(
            "page quantization steps disagree with container epsilon: fine=0x{:08x}/0x{:08x} base=0x{:08x}/0x{:08x}",
            fine_step.to_bits(), expected_fine_step.to_bits(), base_step.to_bits(), expected_base_step.to_bits()
        )));
    }
    let base_scale = get_u32(payload, 44);
    let enhancement_scale = get_u32(payload, 48);
    if base_scale == 0
        || base_scale > i32::MAX as u32
        || enhancement_scale == 0
        || enhancement_scale > i32::MAX as u32
    {
        return Err(F3dzError::InvalidArgument(
            "page residual scales must be in 1..=i32::MAX".to_string(),
        ));
    }
    let base_layer_len = get_u32(payload, 24) as usize;
    let enhancement_layer_len = get_u32(payload, 28) as usize;
    let base_end = PAGE_HEADER_LEN
        .checked_add(base_layer_len)
        .ok_or_else(|| F3dzError::InvalidArgument("base page layer overflow".to_string()))?;
    let payload_end = base_end
        .checked_add(enhancement_layer_len)
        .ok_or_else(|| F3dzError::InvalidArgument("enhancement page layer overflow".to_string()))?;
    if base_end != entry.base_layer_len as usize || payload_end != payload.len() {
        return Err(F3dzError::InvalidArgument(
            "page layer lengths disagree with the index/payload".to_string(),
        ));
    }
    if (header.base_only() || !header.progressive()) && enhancement_layer_len != 0 {
        return Err(F3dzError::InvalidArgument(format!(
            "page quality mode forbids enhancement bytes: {enhancement_layer_len}"
        )));
    }
    if header.progressive() && !header.base_only() && enhancement_layer_len == 0 {
        return Err(F3dzError::InvalidArgument(
            "refined progressive page is missing its enhancement layer".to_string(),
        ));
    }
    let plane = PlaneModel {
        x_slope: get_u32(payload, 12) as i32,
        y_slope: get_u32(payload, 16) as i32,
        intercept: get_u32(payload, 20) as i32,
    };
    if predictor_id != PREDICTOR_PLANE && plane != PlaneModel::default() {
        return Err(F3dzError::InvalidArgument(
            "non-plane predictor carries non-zero plane coefficients".to_string(),
        ));
    }
    Ok(ValidatedPage {
        entry,
        enhancement_predictor,
        plane,
        fine_step,
        base_step,
        base_scale,
        enhancement_scale,
        base_layer: &payload[PAGE_HEADER_LEN..base_end],
        enhancement_layer: &payload[base_end..payload_end],
    })
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
    if entry.flags
        & !(PAGE_FLAG_PROGRESSIVE | PAGE_FLAG_BASE_ONLY | PAGE_FLAG_GPU_FAST | PAGE_FLAG_GPU_DIRECT)
        != 0
    {
        return Err(F3dzError::InvalidArgument(
            "unknown page flag bits are set".to_string(),
        ));
    }
    if entry.progressive() != header.progressive() || entry.base_only() != header.base_only() {
        return Err(F3dzError::InvalidArgument(
            "page quality flags disagree with the container header".to_string(),
        ));
    }
    if entry.gpu_fast() && entry.predictor_id == PREDICTOR_LORENZO && entry.nan_count != 0 {
        return Err(F3dzError::InvalidArgument(
            "fast-path Lorenzo page cannot contain nodata".to_string(),
        ));
    }
    if entry.gpu_direct() && !entry.gpu_fast() {
        return Err(F3dzError::InvalidArgument(
            "direct-token GPU page must also declare the low-barrier path".to_string(),
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

    fn stream(progressive: bool) -> Vec<u8> {
        let mut options = super::super::EncodeOptions::new(0.125);
        options.tile_size = 4;
        options.progressive = progressive;
        options.force_order_zero = true;
        options.height_datum = "EGM96".to_string();
        let source: Vec<_> = (0..35).map(|index| index as f32 * 0.3).collect();
        super::super::encode_dem(&source, 7, 5, &options).unwrap()
    }

    fn assert_shared_rejection(data: &[u8], reason: &str) {
        let shared = match validate_stream(data) {
            Err(error) => error,
            Ok(_) => panic!("shared validator accepted malformed stream: {reason}"),
        };
        assert!(
            shared.to_string().contains(reason),
            "{shared}: expected {reason}"
        );
        assert_eq!(super::super::decode_dem(data, None).unwrap_err(), shared);
        assert_eq!(
            super::super::gpu::validate_stream(data).unwrap_err(),
            shared
        );
    }

    fn edit_payload(data: &[u8], page: usize, offset: usize, bytes: &[u8]) -> Vec<u8> {
        let (header, entries) = parse_prefix(data).unwrap();
        let entry = &entries[page];
        let start = entry.payload_offset as usize;
        let end = start + entry.payload_len as usize;
        let mut edited = data.to_vec();
        edited[start + offset..start + offset + bytes.len()].copy_from_slice(bytes);
        let checksum = crc32(&edited[start..end]);
        put_u32(
            &mut edited,
            header.index_offset as usize + page * PAGE_INDEX_LEN + 32,
            checksum,
        );
        edited
    }

    #[test]
    fn structural_page_errors_agree_before_cpu_or_gpu_execution() {
        let refined = stream(true);
        let base = super::super::base_only_stream(&refined).unwrap();
        let single = stream(false);
        for data in [&refined, &base, &single] {
            let (_, entries) = parse_prefix(data).unwrap();
            let cases = [
                (0, b"BAD!".to_vec(), "magic"),
                (4, 2u16.to_le_bytes().to_vec(), "version"),
                (
                    6,
                    (u16::from(entries[0].flags) ^ 0x10).to_le_bytes().to_vec(),
                    "flags/reserved",
                ),
                (10, 1u16.to_le_bytes().to_vec(), "flags/reserved"),
                (52, 1u32.to_le_bytes().to_vec(), "flags/reserved"),
                (8, vec![PREDICTOR_PREVIOUS_LOD], "predictor ids"),
                (8, vec![0xff], "predictor ids"),
                (
                    8,
                    vec![
                        PREDICTOR_LORENZO,
                        if data == &single {
                            PREDICTOR_LORENZO
                        } else {
                            PREDICTOR_PREVIOUS_LOD
                        },
                    ],
                    "predictor disagrees",
                ),
                (9, vec![0xff], "predictor ids"),
                (12, 1i32.to_le_bytes().to_vec(), "plane coefficients"),
                (24, 0u32.to_le_bytes().to_vec(), "layer lengths"),
                (28, u32::MAX.to_le_bytes().to_vec(), "layer"),
                (32, 0u32.to_le_bytes().to_vec(), "token count"),
                (
                    36,
                    0.0f32.to_bits().to_le_bytes().to_vec(),
                    "quantization steps",
                ),
                (
                    36,
                    (-0.25f32).to_bits().to_le_bytes().to_vec(),
                    "quantization steps",
                ),
                (
                    36,
                    f32::NAN.to_bits().to_le_bytes().to_vec(),
                    "quantization steps",
                ),
                (
                    40,
                    f32::INFINITY.to_bits().to_le_bytes().to_vec(),
                    "quantization steps",
                ),
                (
                    36,
                    (0.25f32.to_bits() + 1).to_le_bytes().to_vec(),
                    "container epsilon",
                ),
                (
                    40,
                    (if data == &single { 0.25f32 } else { 1.0f32 })
                        .to_bits()
                        .wrapping_add(1)
                        .to_le_bytes()
                        .to_vec(),
                    "container epsilon",
                ),
                (44, 0u32.to_le_bytes().to_vec(), "residual scales"),
                (44, u32::MAX.to_le_bytes().to_vec(), "residual scales"),
                (48, 0u32.to_le_bytes().to_vec(), "residual scales"),
                (48, u32::MAX.to_le_bytes().to_vec(), "residual scales"),
            ];
            for (offset, bytes, reason) in cases {
                assert_shared_rejection(&edit_payload(data, 0, offset, &bytes), reason);
            }
        }
    }

    #[test]
    fn malformed_prefix_index_and_payload_ranges_agree() {
        let data = stream(true);
        let (header, entries) = parse_prefix(&data).unwrap();
        let index = header.index_offset as usize;
        let payload = header.payload_offset as usize;
        let cases = [
            (8, 0x80u32.to_le_bytes().to_vec(), "unknown container flag"),
            (
                index + 13,
                vec![entries[0].flags | 0x80],
                "unknown page flag",
            ),
            (
                index + 13,
                vec![entries[0].flags ^ PAGE_FLAG_PROGRESSIVE],
                "quality flags",
            ),
            (
                index + 13,
                vec![PAGE_FLAG_PROGRESSIVE | PAGE_FLAG_GPU_DIRECT],
                "low-barrier path",
            ),
            (index + 14, vec![1], "reserved page-index"),
            (index + 52, vec![1], "reserved page-index"),
            (index + 12, vec![0xff], "unknown page predictor"),
            (
                index + 44,
                0u32.to_le_bytes().to_vec(),
                "sample/nodata counts",
            ),
            (
                index + 16,
                (entries[0].payload_offset + 1).to_le_bytes().to_vec(),
                "canonical/contiguous",
            ),
            (
                index + PAGE_INDEX_LEN + 16,
                entries[0].payload_offset.to_le_bytes().to_vec(),
                "canonical/contiguous",
            ),
            (
                index + PAGE_INDEX_LEN + 16,
                (entries[1].payload_offset + 1).to_le_bytes().to_vec(),
                "canonical/contiguous",
            ),
            (
                index + PAGE_INDEX_LEN,
                0u32.to_le_bytes().to_vec(),
                "grid-edge extent",
            ),
            (payload, vec![data[payload] ^ 1], "CRC mismatch"),
        ];
        for (offset, bytes, reason) in cases {
            let mut edited = data.clone();
            edited[offset..offset + bytes.len()].copy_from_slice(&bytes);
            if offset < FIXED_HEADER_LEN {
                put_u32(&mut edited, 52, 0);
                let checksum = crc32(&edited[..index]);
                put_u32(&mut edited, 52, checksum);
            }
            assert_shared_rejection(&edited, reason);
        }
        let mut edited = data.clone();
        edited[24] ^= 1;
        assert_shared_rejection(&edited, "header CRC mismatch");
        assert_shared_rejection(&data[..data.len() - 1], "truncated_stream");
        assert_shared_rejection(&data[..FIXED_HEADER_LEN - 1], "truncated_stream");
        let mut edited = data.clone();
        edited.push(0);
        assert_shared_rejection(&edited, "trailing or missing payload");
        let mut edited = data.clone();
        edited[index..index + 2 * PAGE_INDEX_LEN].rotate_left(PAGE_INDEX_LEN);
        assert_shared_rejection(&edited, "canonical row-major");
        // Duplicate coordinates on equal-sized pages reach the coverage check.
        let mut duplicate = super::super::encode_dem(
            &vec![0.0; 128 * 64],
            128,
            64,
            &super::super::EncodeOptions::new(0.125),
        )
        .unwrap();
        put_u32(&mut duplicate, FIXED_HEADER_LEN + PAGE_INDEX_LEN, 0);
        assert_shared_rejection(&duplicate, "canonical row-major");
    }

    #[test]
    fn quality_modes_reject_missing_or_forbidden_enhancement_layers() {
        let refined = stream(true);
        let (header, entries) = parse_prefix(&refined).unwrap();
        let entry = &entries[0];
        let start = entry.payload_offset as usize;
        let mut missing = refined.clone();
        let base_len = entry.payload_len - PAGE_HEADER_LEN as u32;
        put_u32(&mut missing, start + 24, base_len);
        put_u32(&mut missing, start + 28, 0);
        put_u32(
            &mut missing,
            header.index_offset as usize + 28,
            entry.payload_len,
        );
        let checksum = crc32(&missing[start..start + entry.payload_len as usize]);
        put_u32(&mut missing, header.index_offset as usize + 32, checksum);
        assert_shared_rejection(&missing, "missing its enhancement");

        for data in [
            stream(false),
            super::super::base_only_stream(&refined).unwrap(),
        ] {
            let (header, entries) = parse_prefix(&data).unwrap();
            let entry = &entries[0];
            let start = entry.payload_offset as usize;
            let mut forbidden = data.clone();
            // Move one stored byte from the base layer to the enhancement layer.
            put_u32(
                &mut forbidden,
                start + 24,
                entry.payload_len - PAGE_HEADER_LEN as u32 - 1,
            );
            put_u32(&mut forbidden, start + 28, 1);
            put_u32(
                &mut forbidden,
                header.index_offset as usize + 28,
                entry.payload_len - 1,
            );
            let checksum = crc32(&forbidden[start..start + entry.payload_len as usize]);
            put_u32(&mut forbidden, header.index_offset as usize + 32, checksum);
            assert_shared_rejection(
                &forbidden,
                if header.progressive() {
                    "forbids enhancement"
                } else {
                    "exactly one layer"
                },
            );
        }
    }

    #[test]
    fn all_structural_pages_are_checked_before_entropy_decoding() {
        let data = stream(true);
        // Structurally valid but invalid entropy framing on the first page.
        let mut edited = edit_payload(&data, 0, PAGE_HEADER_LEN, &0u32.to_le_bytes());
        assert!(validate_stream(&edited).is_ok());
        assert!(super::super::decode_dem(&edited, None).is_err());
        assert!(super::super::gpu::validate_stream(&edited).is_err());
        // Both paths must now report the later structural error first.
        let (_, entries) = parse_prefix(&data).unwrap();
        edited[entries[2].payload_offset as usize] ^= 1;
        assert_shared_rejection(&edited, "CRC mismatch");
        assert!(matches!(
            super::super::decode_dem(&edited, None),
            Err(F3dzError::CorruptPage { page: 2, .. })
        ));
    }
}
