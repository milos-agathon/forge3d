//! Windowed raster decoding: decode ONLY the strips or tiles overlapping a
//! pixel window, over any `Read + Seek` source. This is what lets a remote
//! range reader transfer just the bytes for the requested window instead of
//! downloading the whole object.

use std::path::PathBuf;

use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::Tag;

use crate::gis::affine::PixelWindow;
use crate::gis::error::{GisError, GisResult};
use crate::gis::raster_info::{
    raster_info_from_decoder, read_result_info, validate_read_window, RasterReadResult,
};
use crate::gis::raster_values::{copy_window, deinterleave, f64_to_raster_data};
use crate::gis::raster_write::{RasterArray, RasterData};
use crate::gis::types::{RasterDType, RasterInfo};

/// Windowed read that decodes ONLY the strips or tiles overlapping `window`.
/// Supports chunky-planar (interleaved) striped AND tiled TIFFs across every
/// dtype the decoder produces; planar-separate inputs return
/// `BackendUnavailable` so the caller can fall back to a full read-and-slice.
pub(crate) fn read_window_from_decoder<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    path_label: PathBuf,
    window: PixelWindow,
) -> GisResult<RasterReadResult> {
    let info = raster_info_from_decoder(decoder, path_label)?;

    // Validate BEFORE the layout checks and any pixel fetch: an out-of-bounds
    // window is a request error (InvalidArgument) that must propagate. If a
    // layout check ran first, its BackendUnavailable would send the caller to
    // the full-object download fallback before the invalid window ever raised.
    let window = validate_read_window(window, info.width, info.height)?;

    // Chunky (interleaved) planar config only. Planar-separate stores each band
    // in its own strips/tiles; assembling that here would duplicate the full
    // decoder, so we defer to the full fetch-and-slice fallback with an explicit
    // reason.
    let planar = decoder
        .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)?
        .unwrap_or(1);
    if planar != 1 {
        return Err(GisError::BackendUnavailable(
            "unsupported_layout: windowed range read supports chunky-planar TIFFs only".to_string(),
        ));
    }

    // The tiff decoder's per-chunk `read_chunk` only decodes photometric/sample
    // combinations its `colortype()` accepts: grayscale (1 band), RGB/RGBA (3/4
    // bands), CMYK (4 bands), YCbCr (3 bands). Other combinations (e.g. a 2-band
    // grayscale) can only be assembled by the full read-and-slice path, so we
    // declare them an unsupported layout upfront and fall back — never fetching
    // chunk payloads we cannot decode.
    if !chunk_decodable_layout(decoder, info.band_count)? {
        return Err(GisError::BackendUnavailable(format!(
            "unsupported_layout: windowed range read cannot assemble a {}-band chunk of this \
             photometric type",
            info.band_count
        )));
    }

    let array = match decoder.get_chunk_type() {
        ChunkType::Strip => read_striped_window_array(decoder, &info, window)?,
        ChunkType::Tile => read_tiled_window_array(decoder, &info, window)?,
    };

    let selected_bands: Vec<u16> = (1..=info.band_count).collect();
    let window_transform = if info.transform.is_some() {
        Some(crate::gis::affine::window_transform(&info, window)?)
    } else {
        None
    };
    let result_info = read_result_info(
        &info,
        &array,
        &selected_bands,
        Some(window),
        window_transform,
        info.nodata_per_band.clone(),
    );
    let warnings = result_info.warnings.clone();
    Ok(RasterReadResult {
        array,
        info: result_info,
        bands: selected_bands,
        window: Some(window),
        window_transform,
        mask: None,
        nodata_per_band: info.nodata_per_band.clone(),
        warnings,
    })
}

/// Decode the strips overlapping `window` and crop it out. Reads only the
/// covering strip range; every strip is full image width.
fn read_striped_window_array<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    info: &RasterInfo,
    window: PixelWindow,
) -> GisResult<RasterArray> {
    let bands = info.band_count as usize;
    let width = info.width as usize;
    let (_chunk_w, rows_per_strip) = decoder.chunk_dimensions();
    let rows_per_strip = rows_per_strip as usize;
    if rows_per_strip == 0 {
        return Err(GisError::InvalidRaster(
            "invalid_raster: TIFF reports zero rows per strip".to_string(),
        ));
    }

    let row_start = window.row_off as usize;
    let row_end = row_start + window.height as usize; // exclusive
    let first_strip = row_start / rows_per_strip;
    let last_strip = (row_end - 1) / rows_per_strip;
    let covered_start = first_strip * rows_per_strip;

    let mut chunks: Vec<DecodingResult> = Vec::with_capacity(last_strip - first_strip + 1);
    let mut covered_rows = 0usize;
    for strip in first_strip..=last_strip {
        let chunk = decoder.read_chunk(strip as u32).map_err(|err| {
            GisError::InvalidRaster(format!("invalid_raster: strip {strip}: {err}"))
        })?;
        let (_data_w, data_height) = decoder.chunk_data_dimensions(strip as u32);
        covered_rows += data_height as usize;
        chunks.push(chunk);
    }

    // Assemble the overlapping strips into a partial full-width image, then slice
    // the requested window out of it with the shared window copier.
    let partial = strips_to_array(chunks, bands, covered_rows, width)?;
    let local_window = PixelWindow {
        col_off: window.col_off,
        row_off: window.row_off - covered_start as i64,
        width: window.width,
        height: window.height,
    };
    copy_window(&partial, &info.nodata_per_band, local_window)
}

/// Decode the tiles overlapping `window` and crop it out. Reads only the tiles
/// intersecting the window (never a non-intersecting tile payload), assembles
/// them into the bounding box of those tiles, then slices out the window.
fn read_tiled_window_array<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    info: &RasterInfo,
    window: PixelWindow,
) -> GisResult<RasterArray> {
    let bands = info.band_count as usize;
    let width = info.width as usize;
    let height = info.height as usize;
    let (tile_w, tile_h) = decoder.chunk_dimensions();
    let (tile_w, tile_h) = (tile_w as usize, tile_h as usize);
    if tile_w == 0 || tile_h == 0 {
        return Err(GisError::InvalidRaster(
            "invalid_raster: TIFF reports zero tile size".to_string(),
        ));
    }
    let tiles_across = (width - 1) / tile_w + 1;

    // Window bounds in pixels (validated in-range by the caller).
    let win_col0 = window.col_off as usize;
    let win_row0 = window.row_off as usize;
    let win_col1 = win_col0 + window.width as usize; // exclusive
    let win_row1 = win_row0 + window.height as usize; // exclusive

    // Inclusive tile-grid range covering the window.
    let tile_col0 = win_col0 / tile_w;
    let tile_col1 = (win_col1 - 1) / tile_w;
    let tile_row0 = win_row0 / tile_h;
    let tile_row1 = (win_row1 - 1) / tile_h;

    // Bounding box of the covering tiles (clamped to the image extent). The
    // window is always a subset of this box, so the crop below never reads
    // outside the assembled buffer.
    let covered_col0 = tile_col0 * tile_w;
    let covered_row0 = tile_row0 * tile_h;
    let covered_col1 = ((tile_col1 + 1) * tile_w).min(width);
    let covered_row1 = ((tile_row1 + 1) * tile_h).min(height);
    let covered_w = covered_col1 - covered_col0;
    let covered_h = covered_row1 - covered_row0;

    // Scatter each intersecting tile into a band-first f64 scratch buffer. Every
    // pixel of the covered box belongs to exactly one tile in the rectangular
    // tile range, so the buffer is fully written. Round-tripping through f64
    // mirrors the striped path (copy_window does the same) and is lossless for
    // every dtype this backend supports.
    let mut partial = vec![0.0f64; bands * covered_h * covered_w];
    let mut dtype: Option<RasterDType> = None;
    for tile_row in tile_row0..=tile_row1 {
        for tile_col in tile_col0..=tile_col1 {
            let chunk_index = (tile_row * tiles_across + tile_col) as u32;
            let (valid_w, valid_h) = decoder.chunk_data_dimensions(chunk_index);
            let (valid_w, valid_h) = (valid_w as usize, valid_h as usize);
            let chunk = decoder.read_chunk(chunk_index).map_err(|err| {
                GisError::InvalidRaster(format!("invalid_raster: tile {chunk_index}: {err}"))
            })?;
            let (chunk_dtype, values) = chunk_to_f64(chunk)?;
            match dtype {
                None => dtype = Some(chunk_dtype),
                Some(existing) if existing != chunk_dtype => {
                    return Err(GisError::InvalidRaster(
                        "invalid_raster: tiles report mixed dtypes".to_string(),
                    ));
                }
                Some(_) => {}
            }
            // Tile pixels are chunky-interleaved: (row, col, band) lives at
            // (row * valid_w + col) * bands + band.
            let dst_col0 = tile_col * tile_w - covered_col0;
            let dst_row0 = tile_row * tile_h - covered_row0;
            for band in 0..bands {
                for r in 0..valid_h {
                    for c in 0..valid_w {
                        let src = (r * valid_w + c) * bands + band;
                        let dst = band * covered_w * covered_h
                            + (dst_row0 + r) * covered_w
                            + (dst_col0 + c);
                        partial[dst] = values[src];
                    }
                }
            }
        }
    }
    let dtype = dtype.ok_or_else(|| {
        GisError::InvalidRaster("invalid_raster: no tiles read for window".to_string())
    })?;
    let partial = f64_to_raster_data(dtype, partial, bands, covered_h, covered_w)?;
    let local_window = PixelWindow {
        col_off: window.col_off - covered_col0 as i64,
        row_off: window.row_off - covered_row0 as i64,
        width: window.width,
        height: window.height,
    };
    copy_window(&partial, &info.nodata_per_band, local_window)
}

/// Whether the tiff decoder's `read_chunk` can decode this photometric/band
/// combination (mirrors its `colortype()` acceptance): grayscale needs 1 band,
/// RGB 3 or 4, CMYK 4, YCbCr 3. Anything else must use the full read-and-slice
/// path. `raster_info_from_decoder` has already restricted bit depth to the
/// supported set, so only the photometric/band pairing needs checking here.
fn chunk_decodable_layout<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    band_count: u16,
) -> GisResult<bool> {
    // PhotometricInterpretation: 0=WhiteIsZero, 1=BlackIsZero, 2=RGB, 5=CMYK,
    // 6=YCbCr. Absent defaults to BlackIsZero per the decoder.
    let photometric = decoder
        .find_tag_unsigned::<u16>(Tag::PhotometricInterpretation)?
        .unwrap_or(1);
    Ok(match photometric {
        0 | 1 => band_count == 1,
        2 => band_count == 3 || band_count == 4,
        5 => band_count == 4,
        6 => band_count == 3,
        _ => false,
    })
}

/// Convert a decoded chunk (strip or tile) into interleaved f64 samples plus its
/// dtype, rejecting any variant outside this backend's supported set.
fn chunk_to_f64(chunk: DecodingResult) -> GisResult<(RasterDType, Vec<f64>)> {
    Ok(match chunk {
        DecodingResult::U8(v) => (
            RasterDType::UInt8,
            v.into_iter().map(|x| x as f64).collect(),
        ),
        DecodingResult::U16(v) => (
            RasterDType::UInt16,
            v.into_iter().map(|x| x as f64).collect(),
        ),
        DecodingResult::I16(v) => (
            RasterDType::Int16,
            v.into_iter().map(|x| x as f64).collect(),
        ),
        DecodingResult::U32(v) => (
            RasterDType::UInt32,
            v.into_iter().map(|x| x as f64).collect(),
        ),
        DecodingResult::I32(v) => (
            RasterDType::Int32,
            v.into_iter().map(|x| x as f64).collect(),
        ),
        DecodingResult::F32(v) => (
            RasterDType::Float32,
            v.into_iter().map(|x| x as f64).collect(),
        ),
        DecodingResult::F64(v) => (RasterDType::Float64, v),
        _ => {
            return Err(GisError::UnsupportedDType(
                "unsupported_dtype: unsupported TIFF dtype for windowed read".to_string(),
            ))
        }
    })
}

/// Concatenate same-dtype strip chunks (chunky/interleaved) into a band-first
/// partial `RasterArray` of shape `(bands, rows, width)`.
fn strips_to_array(
    chunks: Vec<DecodingResult>,
    bands: usize,
    rows: usize,
    width: usize,
) -> GisResult<RasterArray> {
    let shape = [bands, rows, width];
    let mut iter = chunks.into_iter();
    let first = iter
        .next()
        .ok_or_else(|| GisError::InvalidRaster("invalid_raster: no strips read".to_string()))?;
    macro_rules! collect {
        ($variant:ident, $data:expr) => {{
            let mut all = $data;
            for chunk in iter {
                match chunk {
                    DecodingResult::$variant(v) => all.extend(v),
                    _ => {
                        return Err(GisError::InvalidRaster(
                            "invalid_raster: strips report mixed dtypes".to_string(),
                        ))
                    }
                }
            }
            RasterArray::new(RasterData::$variant(deinterleave(all, bands)), &shape)
        }};
    }
    match first {
        DecodingResult::U8(data) => collect!(U8, data),
        DecodingResult::U16(data) => collect!(U16, data),
        DecodingResult::I16(data) => collect!(I16, data),
        DecodingResult::U32(data) => collect!(U32, data),
        DecodingResult::I32(data) => collect!(I32, data),
        DecodingResult::F32(data) => collect!(F32, data),
        DecodingResult::F64(data) => collect!(F64, data),
        _ => Err(GisError::UnsupportedDType(
            "unsupported_dtype: unsupported TIFF dtype for windowed read".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gis::raster_write::{write_raster, CreationOptions, WriteRasterOptions};
    use std::fs::File;
    use std::io::BufReader;

    /// An out-of-bounds window must surface as the caller's request error
    /// (`InvalidArgument`) even when the file's layout is one the windowed
    /// decoder cannot assemble (e.g. 2-band grayscale). If the layout check
    /// ran first, the `BackendUnavailable` would send read_cog to the
    /// full-object download fallback before the invalid window ever raised.
    #[test]
    fn invalid_window_raises_before_unsupported_layout_fallback() {
        let path = std::env::temp_dir().join(format!(
            "forge3d_window_order_{}_{}.tif",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before unix epoch")
                .as_nanos()
        ));
        let array = RasterArray::new(
            crate::gis::raster_write::RasterData::U16((0..(2 * 8 * 8) as u16).collect()),
            &[2, 8, 8],
        )
        .expect("test raster shape");
        write_raster(
            &path,
            array,
            WriteRasterOptions {
                crs: None,
                transform: None,
                nodata: vec![None, None],
                driver: "GTiff".to_string(),
                overwrite: true,
                creation_options: CreationOptions::default(),
                creation_options_explicit: false,
                like_info: None,
                height_system: crate::gis::types::HEIGHT_SYSTEM_UNSPECIFIED.to_string(),
            },
        )
        .expect("write 2-band raster");

        let mut decoder =
            Decoder::new(BufReader::new(File::open(&path).expect("open"))).expect("decoder");
        let result = read_window_from_decoder(
            &mut decoder,
            path.clone(),
            PixelWindow {
                col_off: 100,
                row_off: 100,
                width: 4,
                height: 4,
            },
        );
        let _ = std::fs::remove_file(&path);
        match result {
            Err(GisError::InvalidArgument(_)) => {}
            other => panic!("expected InvalidArgument for out-of-bounds window, got {other:?}"),
        }
    }
}
