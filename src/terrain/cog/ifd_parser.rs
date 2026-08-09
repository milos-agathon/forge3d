//! P3.2: TIFF IFD (Image File Directory) parser for COG overview detection.

use super::error::CogError;
use super::range_reader::RangeReader;

/// TIFF tag constants.
const TAG_IMAGE_WIDTH: u16 = 256;
const TAG_IMAGE_LENGTH: u16 = 257;
const TAG_BITS_PER_SAMPLE: u16 = 258;
const TAG_COMPRESSION: u16 = 259;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_PREDICTOR: u16 = 317;
const TAG_ROWS_PER_STRIP: u16 = 278;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_SAMPLE_FORMAT: u16 = 339;
const TAG_TILE_WIDTH: u16 = 322;
const TAG_TILE_LENGTH: u16 = 323;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_MODEL_TRANSFORMATION: u16 = 34264;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;

/// Compression constants.
pub const COMPRESSION_NONE: u16 = 1;
pub const COMPRESSION_LZW: u16 = 5;
pub const COMPRESSION_DEFLATE: u16 = 8;
pub const COMPRESSION_DEFLATE_ALT: u16 = 32946;
/// forge3d private TIFF compression tag for an embedded F3DZ v1 height tile.
/// Values 65000..=65535 are reserved for private TIFF use.
pub const COMPRESSION_F3DZ: u16 = 65003;

/// TIFF predictor constants.
pub const TIFF_PREDICTOR_NONE: u16 = 1;
pub const TIFF_PREDICTOR_HORIZONTAL: u16 = 2;

/// Sample format constants.
pub const SAMPLE_FORMAT_UINT: u16 = 1;
pub const SAMPLE_FORMAT_INT: u16 = 2;
pub const SAMPLE_FORMAT_FLOAT: u16 = 3;

/// Parsed IFD entry representing one image/overview level.
#[derive(Debug, Clone)]
pub struct IfdEntry {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub bits_per_sample: u16,
    pub sample_format: u16,
    pub compression: u16,
    pub predictor: u16,
    pub tile_offsets: Vec<u64>,
    pub tile_byte_counts: Vec<u64>,
    pub overview_level: u32,
    pub tiles_across: u32,
    pub tiles_down: u32,
    pub geo_reference: Option<GeoReference>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeoReference {
    pub model_pixel_scale: Option<[f64; 3]>,
    pub model_tiepoint: Option<[f64; 6]>,
    pub model_transformation: Option<[f64; 16]>,
    pub epsg: u32,
}

impl GeoReference {
    pub fn pixel_to_model(&self, x: f64, y: f64) -> Result<(f64, f64), CogError> {
        let result = if let Some(matrix) = self.model_transformation {
            (
                matrix[0] * x + matrix[1] * y + matrix[3],
                matrix[4] * x + matrix[5] * y + matrix[7],
            )
        } else if let (Some(scale), Some(tie)) = (self.model_pixel_scale, self.model_tiepoint) {
            (
                tie[3] + (x - tie[0]) * scale[0],
                tie[4] - (y - tie[1]) * scale[1],
            )
        } else {
            return Err(CogError::InvalidIfd(
                "GeoTIFF requires ModelTransformation or ModelPixelScale+ModelTiepoint"
                    .to_string(),
            ));
        };
        if result.0.is_finite() && result.1.is_finite() {
            Ok(result)
        } else {
            Err(CogError::InvalidIfd(
                "GeoTIFF transform produced non-finite coordinates".to_string(),
            ))
        }
    }

    pub fn model_to_pixel(&self, model_x: f64, model_y: f64) -> Result<(f64, f64), CogError> {
        let result = if let Some(matrix) = self.model_transformation {
            let a = matrix[0];
            let b = matrix[1];
            let c = matrix[3];
            let d = matrix[4];
            let e = matrix[5];
            let f = matrix[7];
            let det = a * e - b * d;
            if !det.is_finite() || det.abs() <= f64::EPSILON {
                return Err(CogError::InvalidIfd(
                    "GeoTIFF ModelTransformation is not invertible".to_string(),
                ));
            }
            let dx = model_x - c;
            let dy = model_y - f;
            ((e * dx - b * dy) / det, (-d * dx + a * dy) / det)
        } else if let (Some(scale), Some(tie)) = (self.model_pixel_scale, self.model_tiepoint) {
            if scale[0] == 0.0 || scale[1] == 0.0 {
                return Err(CogError::InvalidIfd(
                    "GeoTIFF ModelPixelScale is not invertible".to_string(),
                ));
            }
            (
                tie[0] + (model_x - tie[3]) / scale[0],
                tie[1] - (model_y - tie[4]) / scale[1],
            )
        } else {
            return Err(CogError::InvalidIfd(
                "GeoTIFF requires ModelTransformation or ModelPixelScale+ModelTiepoint"
                    .to_string(),
            ));
        };
        if result.0.is_finite() && result.1.is_finite() {
            Ok(result)
        } else {
            Err(CogError::InvalidIfd(
                "GeoTIFF inverse transform produced non-finite pixels".to_string(),
            ))
        }
    }
}

impl IfdEntry {
    /// Get total number of tiles in this IFD.
    pub fn tile_count(&self) -> usize {
        self.tile_offsets.len()
    }

    /// Get bytes per sample based on bits_per_sample.
    pub fn bytes_per_sample(&self) -> usize {
        (self.bits_per_sample as usize + 7) / 8
    }

    /// Check if this IFD uses float samples.
    pub fn is_float(&self) -> bool {
        self.sample_format == SAMPLE_FORMAT_FLOAT
    }

    /// Get tile index for given tile coordinates.
    pub fn tile_index(&self, tile_x: u32, tile_y: u32) -> Option<usize> {
        if tile_x >= self.tiles_across || tile_y >= self.tiles_down {
            return None;
        }
        Some((tile_y * self.tiles_across + tile_x) as usize)
    }
}

/// COG header information.
#[derive(Debug, Clone)]
pub struct CogHeader {
    pub is_big_endian: bool,
    pub is_bigtiff: bool,
    pub ifds: Vec<IfdEntry>,
}

impl CogHeader {
    pub fn geo_reference(&self) -> Result<&GeoReference, CogError> {
        self.ifds
            .iter()
            .find_map(|ifd| ifd.geo_reference.as_ref())
            .ok_or_else(|| {
                CogError::InvalidIfd(
                    "COG lacks required GeoTIFF transform/CRS metadata".to_string(),
                )
            })
    }
    /// Get the full-resolution IFD (first one).
    pub fn full_resolution(&self) -> Option<&IfdEntry> {
        self.ifds.first()
    }

    /// Get IFD for a specific overview level.
    pub fn overview(&self, level: u32) -> Option<&IfdEntry> {
        self.ifds.iter().find(|ifd| ifd.overview_level == level)
    }

    /// Select best IFD for requested LOD.
    pub fn select_ifd_for_lod(&self, lod: u32) -> Result<&IfdEntry, CogError> {
        self.ifds
            .iter()
            .filter(|ifd| ifd.overview_level <= lod)
            .max_by_key(|ifd| ifd.width)
            .or_else(|| self.ifds.last())
            .ok_or_else(|| CogError::InvalidIfd("COG contains no image file directories".into()))
    }
}

/// Parse COG header and all IFDs.
pub async fn parse_cog_header(reader: &RangeReader) -> Result<CogHeader, CogError> {
    let header_bytes = reader.read_range(0, 16).await?;

    let (is_big_endian, is_bigtiff) = parse_tiff_header(&header_bytes)?;

    let first_ifd_offset = first_ifd_offset(&header_bytes, is_big_endian, is_bigtiff)?;

    let mut ifds = Vec::new();
    let mut ifd_offset = first_ifd_offset;
    let mut overview_level = 0u32;

    while ifd_offset != 0 {
        let (ifd, next_offset) = parse_ifd(
            reader,
            ifd_offset,
            is_big_endian,
            is_bigtiff,
            overview_level,
        )
        .await?;
        ifds.push(ifd);
        ifd_offset = next_offset;
        overview_level += 1;

        if ifds.len() > 20 {
            break;
        }
    }

    if ifds.is_empty() {
        return Err(CogError::InvalidIfd(
            "COG contains no image file directories".into(),
        ));
    }

    Ok(CogHeader {
        is_big_endian,
        is_bigtiff,
        ifds,
    })
}

fn first_ifd_offset(
    header_bytes: &[u8],
    is_big_endian: bool,
    is_bigtiff: bool,
) -> Result<u64, CogError> {
    if is_bigtiff {
        if header_bytes.len() < 16 {
            return Err(CogError::InvalidTiffHeader(
                "BigTIFF header too short for first IFD offset".into(),
            ));
        }
        Ok(if is_big_endian {
            u64::from_be_bytes([
                header_bytes[8],
                header_bytes[9],
                header_bytes[10],
                header_bytes[11],
                header_bytes[12],
                header_bytes[13],
                header_bytes[14],
                header_bytes[15],
            ])
        } else {
            u64::from_le_bytes([
                header_bytes[8],
                header_bytes[9],
                header_bytes[10],
                header_bytes[11],
                header_bytes[12],
                header_bytes[13],
                header_bytes[14],
                header_bytes[15],
            ])
        })
    } else {
        let offset = if is_big_endian {
            u32::from_be_bytes([
                header_bytes[4],
                header_bytes[5],
                header_bytes[6],
                header_bytes[7],
            ])
        } else {
            u32::from_le_bytes([
                header_bytes[4],
                header_bytes[5],
                header_bytes[6],
                header_bytes[7],
            ])
        };
        Ok(offset as u64)
    }
}

fn parse_tiff_header(header: &[u8]) -> Result<(bool, bool), CogError> {
    if header.len() < 8 {
        return Err(CogError::InvalidTiffHeader("Header too short".into()));
    }

    let is_big_endian = match &header[0..2] {
        b"II" => false,
        b"MM" => true,
        _ => return Err(CogError::InvalidTiffHeader("Invalid byte order".into())),
    };

    let magic = if is_big_endian {
        u16::from_be_bytes([header[2], header[3]])
    } else {
        u16::from_le_bytes([header[2], header[3]])
    };

    let is_bigtiff = match magic {
        42 => false,
        43 => true,
        _ => {
            return Err(CogError::InvalidTiffHeader(format!(
                "Invalid magic number: {}",
                magic
            )))
        }
    };

    Ok((is_big_endian, is_bigtiff))
}

async fn parse_ifd(
    reader: &RangeReader,
    offset: u64,
    big_endian: bool,
    bigtiff: bool,
    overview_level: u32,
) -> Result<(IfdEntry, u64), CogError> {
    let entry_size: u64 = if bigtiff { 20 } else { 12 };
    let count_size: u64 = if bigtiff { 8 } else { 2 };
    let next_size: u64 = if bigtiff { 8 } else { 4 };

    let count_bytes = reader.read_range(offset, count_size).await?;
    let entry_count = if bigtiff {
        read_u64(&count_bytes, 0, big_endian)
    } else {
        read_u16(&count_bytes, 0, big_endian) as u64
    };

    let entries_size = entry_count
        .checked_mul(entry_size)
        .ok_or_else(|| CogError::InvalidIfd("IFD entry table size overflow".into()))?;
    let entries_and_next = entries_size
        .checked_add(next_size)
        .ok_or_else(|| CogError::InvalidIfd("IFD read size overflow".into()))?;
    let entries_offset = offset
        .checked_add(count_size)
        .ok_or_else(|| CogError::InvalidIfd("IFD offset overflow".into()))?;
    let ifd_data = reader.read_range(entries_offset, entries_and_next).await?;

    let mut width = 0u32;
    let mut height = 0u32;
    let mut tile_width = 256u32;
    let mut tile_height = 256u32;
    let mut bits_per_sample = 8u16;
    let mut sample_format = SAMPLE_FORMAT_UINT;
    let mut compression = COMPRESSION_NONE;
    let mut predictor = TIFF_PREDICTOR_NONE;
    let mut tile_offsets_info: Option<(u64, u64, u16)> = None;
    let mut tile_byte_counts_info: Option<(u64, u64, u16)> = None;
    let mut strip_offsets_info: Option<(u64, u64, u16)> = None;
    let mut strip_byte_counts_info: Option<(u64, u64, u16)> = None;
    let mut rows_per_strip = 0u32;
    let mut pixel_scale_info: Option<(u64, u64, u16)> = None;
    let mut tiepoint_info: Option<(u64, u64, u16)> = None;
    let mut transformation_info: Option<(u64, u64, u16)> = None;
    let mut geo_keys_info: Option<(u64, u64, u16)> = None;

    for i in 0..entry_count {
        let entry_offset = (i * entry_size) as usize;
        let tag = read_u16(&ifd_data, entry_offset, big_endian);
        let field_type = read_u16(&ifd_data, entry_offset + 2, big_endian);
        let count = if bigtiff {
            read_u64(&ifd_data, entry_offset + 4, big_endian)
        } else {
            read_u32(&ifd_data, entry_offset + 4, big_endian) as u64
        };

        let value_offset = if bigtiff { 12 } else { 8 };
        let value = read_tag_value(
            &ifd_data,
            entry_offset + value_offset,
            field_type,
            big_endian,
        );
        let external_offset = if bigtiff {
            read_u64(&ifd_data, entry_offset + value_offset, big_endian)
        } else {
            u64::from(read_u32(
                &ifd_data,
                entry_offset + value_offset,
                big_endian,
            ))
        };

        match tag {
            TAG_IMAGE_WIDTH => width = value as u32,
            TAG_IMAGE_LENGTH => height = value as u32,
            TAG_BITS_PER_SAMPLE => bits_per_sample = value as u16,
            TAG_COMPRESSION => compression = value as u16,
            TAG_STRIP_OFFSETS => {
                let inline_capacity = if bigtiff { 8 } else { 4 };
                let data_offset = if type_size(field_type) * count as usize > inline_capacity {
                    external_offset
                } else {
                    offset + count_size + (i * entry_size) + value_offset as u64
                };
                strip_offsets_info = Some((data_offset, count, field_type));
            }
            TAG_PREDICTOR => predictor = value as u16,
            TAG_ROWS_PER_STRIP => rows_per_strip = value as u32,
            TAG_STRIP_BYTE_COUNTS => {
                let inline_capacity = if bigtiff { 8 } else { 4 };
                let data_offset = if type_size(field_type) * count as usize > inline_capacity {
                    external_offset
                } else {
                    offset + count_size + (i * entry_size) + value_offset as u64
                };
                strip_byte_counts_info = Some((data_offset, count, field_type));
            }
            TAG_SAMPLE_FORMAT => sample_format = value as u16,
            TAG_TILE_WIDTH => tile_width = value as u32,
            TAG_TILE_LENGTH => tile_height = value as u32,
            TAG_TILE_OFFSETS => {
                let inline_capacity = if bigtiff { 8 } else { 4 };
                let data_offset = if type_size(field_type) * count as usize > inline_capacity {
                    external_offset
                } else {
                    offset + count_size + (i * entry_size) + value_offset as u64
                };
                tile_offsets_info = Some((data_offset, count, field_type));
            }
            TAG_TILE_BYTE_COUNTS => {
                let inline_capacity = if bigtiff { 8 } else { 4 };
                let data_offset = if type_size(field_type) * count as usize > inline_capacity {
                    external_offset
                } else {
                    offset + count_size + (i * entry_size) + value_offset as u64
                };
                tile_byte_counts_info = Some((data_offset, count, field_type));
            }
            TAG_MODEL_PIXEL_SCALE | TAG_MODEL_TIEPOINT | TAG_MODEL_TRANSFORMATION
            | TAG_GEO_KEY_DIRECTORY => {
                let inline_capacity = if bigtiff { 8 } else { 4 };
                let data_offset = if type_size(field_type) * count as usize > inline_capacity {
                    external_offset
                } else {
                    offset + count_size + (i * entry_size) + value_offset as u64
                };
                let info = Some((data_offset, count, field_type));
                match tag {
                    TAG_MODEL_PIXEL_SCALE => pixel_scale_info = info,
                    TAG_MODEL_TIEPOINT => tiepoint_info = info,
                    TAG_MODEL_TRANSFORMATION => transformation_info = info,
                    TAG_GEO_KEY_DIRECTORY => geo_keys_info = info,
                    _ => unreachable!(),
                }
            }
            _ => {}
        }
    }

    let model_pixel_scale = if let Some((off, count, field_type)) = pixel_scale_info {
        let values = read_f64_array(reader, off, count as usize, field_type, big_endian).await?;
        (values.len() >= 3).then(|| [values[0], values[1], values[2]])
    } else {
        None
    };
    let model_tiepoint = if let Some((off, count, field_type)) = tiepoint_info {
        let values = read_f64_array(reader, off, count as usize, field_type, big_endian).await?;
        (values.len() >= 6).then(|| [values[0], values[1], values[2], values[3], values[4], values[5]])
    } else {
        None
    };
    let model_transformation = if let Some((off, count, field_type)) = transformation_info {
        let values = read_f64_array(reader, off, count as usize, field_type, big_endian).await?;
        (values.len() >= 16).then(|| {
            let mut matrix = [0.0; 16];
            matrix.copy_from_slice(&values[..16]);
            matrix
        })
    } else {
        None
    };
    let epsg = if let Some((off, count, field_type)) = geo_keys_info {
        let keys = read_u16_array(reader, off, count as usize, field_type, big_endian).await?;
        parse_epsg_from_geo_keys(&keys)
    } else {
        None
    };
    let geo_reference = epsg.map(|epsg| GeoReference {
        model_pixel_scale,
        model_tiepoint,
        model_transformation,
        epsg,
    });

    let mut tile_offsets = if let Some((off, count, field_type)) = tile_offsets_info {
        read_offset_array(reader, off, count as usize, field_type, big_endian).await?
    } else {
        Vec::new()
    };

    let mut tile_byte_counts = if let Some((off, count, field_type)) = tile_byte_counts_info {
        read_offset_array(reader, off, count as usize, field_type, big_endian).await?
    } else {
        Vec::new()
    };

    if tile_offsets.is_empty() {
        if let Some((off, count, field_type)) = strip_offsets_info {
            tile_offsets =
                read_offset_array(reader, off, count as usize, field_type, big_endian).await?;
            tile_width = width;
            tile_height = rows_per_strip.max(1).min(height);
        }
    }
    if tile_byte_counts.is_empty() {
        if let Some((off, count, field_type)) = strip_byte_counts_info {
            tile_byte_counts =
                read_offset_array(reader, off, count as usize, field_type, big_endian).await?;
        }
    }

    if tile_width == 0 || tile_height == 0 {
        return Err(CogError::InvalidIfd(
            "tile width and height must be non-zero".into(),
        ));
    }
    let tiles_across = width
        .checked_add(tile_width - 1)
        .ok_or_else(|| CogError::InvalidIfd("tile column count overflow".into()))?
        / tile_width;
    let tiles_down = height
        .checked_add(tile_height - 1)
        .ok_or_else(|| CogError::InvalidIfd("tile row count overflow".into()))?
        / tile_height;

    let next_ifd_offset = if bigtiff {
        read_u64(&ifd_data, entries_size as usize, big_endian)
    } else {
        read_u32(&ifd_data, entries_size as usize, big_endian) as u64
    };

    Ok((
        IfdEntry {
            width,
            height,
            tile_width,
            tile_height,
            bits_per_sample,
            sample_format,
            compression,
            predictor,
            tile_offsets,
            tile_byte_counts,
            overview_level,
            tiles_across,
            tiles_down,
            geo_reference,
        },
        next_ifd_offset,
    ))
}

async fn read_f64_array(
    reader: &RangeReader,
    offset: u64,
    count: usize,
    field_type: u16,
    big_endian: bool,
) -> Result<Vec<f64>, CogError> {
    if field_type != 12 {
        return Err(CogError::InvalidIfd(format!(
            "GeoTIFF floating tag uses unsupported TIFF field type {field_type}"
        )));
    }
    let bytes = reader.read_range(offset, (count as u64).saturating_mul(8)).await?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            if big_endian {
                f64::from_be_bytes(chunk.try_into().unwrap())
            } else {
                f64::from_le_bytes(chunk.try_into().unwrap())
            }
        })
        .collect())
}

async fn read_u16_array(
    reader: &RangeReader,
    offset: u64,
    count: usize,
    field_type: u16,
    big_endian: bool,
) -> Result<Vec<u16>, CogError> {
    if field_type != 3 {
        return Err(CogError::InvalidIfd(format!(
            "GeoKeyDirectory uses unsupported TIFF field type {field_type}"
        )));
    }
    let bytes = reader.read_range(offset, (count as u64).saturating_mul(2)).await?;
    Ok((0..count)
        .map(|index| read_u16(&bytes, index * 2, big_endian))
        .collect())
}

fn parse_epsg_from_geo_keys(keys: &[u16]) -> Option<u32> {
    let key_count = usize::from(*keys.get(3)?);
    let mut geographic = None;
    let mut projected = None;
    for entry in keys.get(4..)?.chunks_exact(4).take(key_count) {
        if entry[1] != 0 || entry[2] != 1 {
            continue;
        }
        match entry[0] {
            2048 => geographic = Some(u32::from(entry[3])),
            3072 => projected = Some(u32::from(entry[3])),
            _ => {}
        }
    }
    projected.or(geographic).filter(|code| *code != 0 && *code != 32767)
}

async fn read_offset_array(
    reader: &RangeReader,
    offset: u64,
    count: usize,
    field_type: u16,
    big_endian: bool,
) -> Result<Vec<u64>, CogError> {
    let item_size = type_size(field_type);
    let byte_count = count
        .checked_mul(item_size)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| CogError::InvalidIfd("IFD offset array size overflow".into()))?;
    let bytes = reader.read_range(offset, byte_count).await?;
    let mut offsets = Vec::with_capacity(count);

    for i in 0..count {
        let base = i * item_size;
        let val = match field_type {
            3 if bytes.len() >= base + 2 => read_u16(&bytes, base, big_endian) as u64,
            4 if bytes.len() >= base + 4 => read_u32(&bytes, base, big_endian) as u64,
            16 if bytes.len() >= base + 8 => read_u64(&bytes, base, big_endian),
            _ => 0,
        };
        offsets.push(val);
    }

    Ok(offsets)
}

fn read_u16(data: &[u8], offset: usize, big_endian: bool) -> u16 {
    if offset + 2 > data.len() {
        return 0;
    }
    if big_endian {
        u16::from_be_bytes([data[offset], data[offset + 1]])
    } else {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    }
}

fn read_u32(data: &[u8], offset: usize, big_endian: bool) -> u32 {
    if offset + 4 > data.len() {
        return 0;
    }
    if big_endian {
        u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    } else {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }
}

fn read_u64(data: &[u8], offset: usize, big_endian: bool) -> u64 {
    if offset + 8 > data.len() {
        return 0;
    }
    if big_endian {
        u64::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ])
    } else {
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
}

fn read_tag_value(data: &[u8], offset: usize, field_type: u16, big_endian: bool) -> u64 {
    match field_type {
        1 | 2 => data.get(offset).copied().unwrap_or(0) as u64,
        3 => read_u16(data, offset, big_endian) as u64,
        4 => read_u32(data, offset, big_endian) as u64,
        16 => read_u64(data, offset, big_endian),
        _ => read_u32(data, offset, big_endian) as u64,
    }
}

fn type_size(field_type: u16) -> usize {
    match field_type {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 | 16 | 17 | 18 => 8,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_full_file_array_fixture(bigtiff: bool, external: bool) -> CogHeader {
        let (ifd_offset, count_size, entry_size, value_offset, inline_capacity) = if bigtiff {
            (16usize, 8usize, 20usize, 12usize, 8usize)
        } else {
            (8usize, 2usize, 12usize, 8usize, 4usize)
        };
        let value_count = match (bigtiff, external) {
            (false, false) => 2usize,
            (false, true) => 3usize,
            (true, false) => 4usize,
            (true, true) => 5usize,
        };
        let field_type = 3u16;
        let element_size = 2usize;
        assert_eq!(value_count * element_size > inline_capacity, external);
        let entries_offset = ifd_offset + count_size;
        let next_offset = entries_offset + 2 * entry_size;
        let external_offset = 0x1_0000usize;
        let mut bytes = vec![0u8; external_offset + 2 * value_count * element_size + 16];
        bytes[0..2].copy_from_slice(b"II");
        if bigtiff {
            bytes[2..4].copy_from_slice(&43u16.to_le_bytes());
            bytes[4..6].copy_from_slice(&8u16.to_le_bytes());
            bytes[6..8].copy_from_slice(&0u16.to_le_bytes());
            bytes[8..16].copy_from_slice(&(ifd_offset as u64).to_le_bytes());
            bytes[ifd_offset..ifd_offset + 8].copy_from_slice(&2u64.to_le_bytes());
        } else {
            bytes[2..4].copy_from_slice(&42u16.to_le_bytes());
            bytes[4..8].copy_from_slice(&(ifd_offset as u32).to_le_bytes());
            bytes[ifd_offset..ifd_offset + 2].copy_from_slice(&2u16.to_le_bytes());
        }
        for (entry_index, (tag, values)) in [
            (TAG_TILE_OFFSETS, vec![40u64, 50, 60, 70, 80]),
            (TAG_TILE_BYTE_COUNTS, vec![4u64, 5, 6, 7, 8]),
        ]
        .into_iter()
        .enumerate()
        {
            let start = entries_offset + entry_index * entry_size;
            bytes[start..start + 2].copy_from_slice(&tag.to_le_bytes());
            bytes[start + 2..start + 4].copy_from_slice(&field_type.to_le_bytes());
            if bigtiff {
                bytes[start + 4..start + 12]
                    .copy_from_slice(&(value_count as u64).to_le_bytes());
            } else {
                bytes[start + 4..start + 8]
                    .copy_from_slice(&(value_count as u32).to_le_bytes());
            }
            let data_start = if external {
                let array_offset = external_offset + entry_index * value_count * element_size;
                if bigtiff {
                    bytes[start + value_offset..start + value_offset + 8]
                        .copy_from_slice(&(array_offset as u64).to_le_bytes());
                } else {
                    bytes[start + value_offset..start + value_offset + 4]
                        .copy_from_slice(&(array_offset as u32).to_le_bytes());
                }
                array_offset
            } else {
                start + value_offset
            };
            for (index, value) in values.into_iter().take(value_count).enumerate() {
                let pos = data_start + index * element_size;
                bytes[pos..pos + 2].copy_from_slice(&(value as u16).to_le_bytes());
            }
        }
        // A valid zero next-IFD pointer completes the full TIFF directory.
        let next_size = if bigtiff { 8 } else { 4 };
        bytes[next_offset..next_offset + next_size].fill(0);
        let path = std::env::temp_dir().join(format!(
            "forge3d-full-tiff-array-{}-{}-{}.tif",
            std::process::id(),
            u8::from(bigtiff),
            u8::from(external)
        ));
        std::fs::write(&path, bytes).unwrap();
        let reader = RangeReader::new_local(path.to_str().unwrap()).unwrap();
        let parsed = pollster::block_on(parse_cog_header(&reader)).unwrap();
        let _ = std::fs::remove_file(path);
        parsed
    }

    #[test]
    fn short_bigtiff_header_returns_typed_error() {
        let header = b"II+\0\x08\0\0\0";
        let (big_endian, bigtiff) = parse_tiff_header(header).expect("valid BigTIFF prefix");
        let error = first_ifd_offset(header, big_endian, bigtiff).unwrap_err();
        assert!(matches!(
            error,
            CogError::InvalidTiffHeader(message) if message.contains("too short")
        ));
    }

    #[test]
    fn selecting_from_empty_header_returns_typed_error() {
        let header = CogHeader {
            is_big_endian: false,
            is_bigtiff: false,
            ifds: Vec::new(),
        };
        let error = header.select_ifd_for_lod(0).unwrap_err();
        assert!(matches!(
            error,
            CogError::InvalidIfd(message) if message.contains("no image file directories")
        ));
    }

    #[test]
    fn geokeys_and_pixel_transform_are_validated() {
        let keys = [
            1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326,
        ];
        assert_eq!(parse_epsg_from_geo_keys(&keys), Some(4326));
        let georef = GeoReference {
            model_pixel_scale: Some([0.25, 0.5, 0.0]),
            model_tiepoint: Some([0.0, 0.0, 0.0, -123.0, 48.0, 0.0]),
            model_transformation: None,
            epsg: 4326,
        };
        assert_eq!(georef.pixel_to_model(4.0, 6.0).unwrap(), (-122.0, 45.0));
    }

    #[test]
    fn classic_full_file_arrays_classify_inline_and_external_by_byte_size() {
        let inline_header = parse_full_file_array_fixture(false, false);
        assert!(!inline_header.is_bigtiff);
        assert_eq!(inline_header.ifds.len(), 1);
        let inline = &inline_header.ifds[0];
        assert_eq!(inline.tile_offsets, vec![40, 50]);
        assert_eq!(inline.tile_byte_counts, vec![4, 5]);
        let external_header = parse_full_file_array_fixture(false, true);
        assert!(!external_header.is_bigtiff);
        assert_eq!(external_header.ifds.len(), 1);
        let external = &external_header.ifds[0];
        assert_eq!(external.tile_offsets, vec![40, 50, 60]);
        assert_eq!(external.tile_byte_counts, vec![4, 5, 6]);
    }

    #[test]
    fn bigtiff_full_file_arrays_classify_inline_and_external_by_byte_size() {
        let inline_header = parse_full_file_array_fixture(true, false);
        assert!(inline_header.is_bigtiff);
        assert_eq!(inline_header.ifds.len(), 1);
        let inline = &inline_header.ifds[0];
        assert_eq!(inline.tile_offsets, vec![40, 50, 60, 70]);
        assert_eq!(inline.tile_byte_counts, vec![4, 5, 6, 7]);
        let external_header = parse_full_file_array_fixture(true, true);
        assert!(external_header.is_bigtiff);
        assert_eq!(external_header.ifds.len(), 1);
        let external = &external_header.ifds[0];
        assert_eq!(external.tile_offsets, vec![40, 50, 60, 70, 80]);
        assert_eq!(external.tile_byte_counts, vec![4, 5, 6, 7, 8]);
    }
}
