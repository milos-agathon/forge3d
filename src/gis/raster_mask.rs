//! Explicit GeoTIFF validity, independent of sample values and GDAL_NODATA.
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read, Seek, SeekFrom};
use std::path::Path;

use flate2::read::ZlibDecoder;
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::Tag;

use super::error::{GisError, GisResult};
use super::raster_values::deinterleave;

/// Return true-valid samples in band-first order. A single plane applies to all bands.
pub(crate) fn read_explicit_mask<R: Read + Seek>(
    path: &Path,
    decoder: &mut Decoder<R>,
    bands: usize,
) -> GisResult<Option<Vec<bool>>> {
    let shape = decoder.dimensions()?;
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".msk");
    let sidecar = Path::new(&sidecar);
    match File::open(sidecar) {
        Ok(file) => {
            let mut mask = Decoder::new(BufReader::new(file))?;
            let samples = mask
                .find_tag_unsigned(Tag::SamplesPerPixel)?
                .unwrap_or(1usize);
            if mask.dimensions()? != shape || (samples != 1 && samples != bands) {
                return Err(GisError::InvalidRaster(
                    "external mask shape differs from raster".into(),
                ));
            }
            let mut valid = if samples > 1
                && mask
                    .find_tag_unsigned(Tag::PlanarConfiguration)?
                    .unwrap_or(1u16)
                    == 2
            {
                // TIFF 0.9 read_image returns only the first plane for band-interleaved data.
                if super::raster_tags::dtype_per_band(&mut mask, samples as u16)?
                    .iter()
                    .any(|dtype| dtype != "uint8")
                {
                    return Err(GisError::InvalidRaster(
                        "expected a byte-valued GDAL mask".into(),
                    ));
                }
                read_mask_blocks(sidecar, &mut mask, shape, samples, false)?
            } else {
                let DecodingResult::U8(values) = mask.read_image()? else {
                    return Err(GisError::InvalidRaster(
                        "expected a byte-valued GDAL mask".into(),
                    ));
                };
                if (shape.0 as usize)
                    .checked_mul(shape.1 as usize)
                    .and_then(|n| n.checked_mul(samples))
                    != Some(values.len())
                {
                    return Err(GisError::InvalidRaster(
                        "external mask sample count differs from raster".into(),
                    ));
                }
                deinterleave(values, samples)
                    .into_iter()
                    .map(|value| value != 0)
                    .collect()
            };
            if samples > 1 {
                if let Some(metadata) = mask.find_tag(Tag::Unknown(42112))? {
                    // GDAL leaves unassigned mask planes zero-filled. Only bands named
                    // by INTERNAL_MASK_FLAGS_n have an explicit mask in this sidecar.
                    let xml = metadata.into_string()?;
                    let mut assigned = vec![false; samples];
                    for attr in xml.split(['"', '\'']) {
                        if let Some(band) = attr
                            .strip_prefix("INTERNAL_MASK_FLAGS_")
                            .and_then(|suffix| suffix.parse::<usize>().ok())
                            .and_then(|band| band.checked_sub(1))
                            .and_then(|index| assigned.get_mut(index))
                        {
                            *band = true;
                        }
                    }
                    if assigned.iter().any(|&present| present) {
                        let pixels = (shape.0 as usize) * (shape.1 as usize);
                        for (plane, present) in valid.chunks_exact_mut(pixels).zip(assigned) {
                            if !present {
                                plane.fill(true);
                            }
                        }
                    }
                }
            }
            return Ok(Some(valid));
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }

    while decoder.more_images() {
        decoder.next_image()?;
        let flags = decoder
            .find_tag_unsigned(Tag::NewSubfileType)?
            .unwrap_or(0u32);
        // FILETYPE_MASK (4), excluding FILETYPE_REDUCEDIMAGE (1) overview masks.
        if flags & 5 == 4 {
            if decoder.dimensions()? != shape {
                return Err(GisError::InvalidRaster(
                    "internal mask shape differs from raster".into(),
                ));
            }
            if decoder.get_tag_u32(Tag::PhotometricInterpretation)? != 4
                || decoder
                    .find_tag_unsigned(Tag::BitsPerSample)?
                    .unwrap_or(1u16)
                    != 1
                || decoder
                    .find_tag_unsigned(Tag::SamplesPerPixel)?
                    .unwrap_or(1u16)
                    != 1
            {
                return Err(GisError::UnsupportedDriver(
                    "unsupported internal TIFF mask encoding".into(),
                ));
            }
            return read_mask_blocks(path, decoder, shape, 1, true).map(Some);
        }
    }
    Ok(None)
}

// TIFF 0.9 exposes the mask IFD and its strip/tile tables, but read_image
// cannot decode PhotometricInterpretation=TransparencyMask. GDAL stores
// internal masks as row-aligned bits, with full-width padding on edge tiles.
fn read_mask_blocks<R: Read + Seek>(
    path: &Path,
    decoder: &mut Decoder<R>,
    (width, height): (u32, u32),
    planes: usize,
    bit_packed: bool,
) -> GisResult<Vec<bool>> {
    let compression = decoder.find_tag_unsigned(Tag::Compression)?.unwrap_or(1u16);
    let order = decoder.find_tag_unsigned(Tag::FillOrder)?.unwrap_or(1u16);
    if decoder.find_tag_unsigned(Tag::Predictor)?.unwrap_or(1u16) != 1
        || (bit_packed && !matches!(order, 1 | 2))
        || !matches!(compression, 1 | 8 | 32946)
    {
        return Err(GisError::UnsupportedDriver(
            "unsupported internal TIFF mask encoding".into(),
        ));
    }
    let tiled = decoder.get_chunk_type() == ChunkType::Tile;
    let (offset_tag, count_tag) = if tiled {
        (Tag::TileOffsets, Tag::TileByteCounts)
    } else {
        (Tag::StripOffsets, Tag::StripByteCounts)
    };
    let offsets = decoder.get_tag_u64_vec(offset_tag)?;
    let counts = decoder.get_tag_u64_vec(count_tag)?;
    let (chunk_width, chunk_height) = decoder.chunk_dimensions();
    if chunk_width == 0 || chunk_height == 0 {
        return Err(GisError::InvalidRaster(
            "empty mask block dimensions".into(),
        ));
    }
    let across = width.div_ceil(chunk_width);
    let blocks_per_plane = u64::from(across) * u64::from(height.div_ceil(chunk_height));
    if blocks_per_plane == 0
        || offsets.len() != counts.len()
        || Some(offsets.len() as u64) != blocks_per_plane.checked_mul(planes as u64)
    {
        return Err(GisError::InvalidRaster("invalid mask block table".into()));
    }
    let mut valid =
        vec![
            false;
            (width as usize)
                .checked_mul(height as usize)
                .and_then(|pixels| pixels.checked_mul(planes))
                .ok_or_else(|| GisError::InvalidRaster("mask dimensions overflow".into()))?
        ];
    let mut file = BufReader::new(File::open(path)?);
    let mut row = Vec::new();
    let stride = if bit_packed {
        chunk_width.div_ceil(8)
    } else {
        chunk_width
    } as usize;
    if stride > tiff::decoder::Limits::default().intermediate_buffer_size {
        return Err(GisError::InvalidRaster(
            "mask row exceeds TIFF decoder memory limit".into(),
        ));
    }
    row.resize(stride, 0);
    for (i, (offset, count)) in offsets.into_iter().zip(counts).enumerate() {
        if offset == 0 && count == 0 {
            continue; // GDAL sparse block, all bits clear
        }
        file.seek(SeekFrom::Start(offset))?;
        let input = (&mut file).take(count);
        let mut input: Box<dyn Read> = if compression == 1 {
            Box::new(input)
        } else {
            Box::new(ZlibDecoder::new(input))
        };
        let block = i as u64 % blocks_per_plane;
        let plane = (i as u64 / blocks_per_plane) as usize;
        let x0 = (block % u64::from(across)) as u32 * chunk_width;
        let y0 = (block / u64::from(across)) as u32 * chunk_height;
        for dy in 0..chunk_height.min(height - y0) {
            input.read_exact(&mut row)?;
            let start =
                (plane * height as usize + (y0 + dy) as usize) * width as usize + x0 as usize;
            for x in 0..chunk_width.min(width - x0) as usize {
                valid[start + x] = if bit_packed {
                    let bit = if order == 1 { 7 - x % 8 } else { x % 8 };
                    row[x / 8] & (1 << bit) != 0
                } else {
                    row[x] != 0
                };
            }
        }
    }
    Ok(valid)
}
