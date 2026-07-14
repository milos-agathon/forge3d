//! Pixel-value utilities shared by the raster readers, warp, rasterize, and
//! thematic paths: band-first f64 round-trips, window copies, validity masks,
//! and chunky-to-planar deinterleaving.

use crate::gis::affine::PixelWindow;
use crate::gis::error::GisResult;
use crate::gis::raster_write::{RasterArray, RasterData};
use crate::gis::types::RasterDType;

pub(crate) fn copy_window(
    source: &RasterArray,
    nodata: &[Option<f64>],
    window: PixelWindow,
) -> GisResult<RasterArray> {
    let source_values = raster_to_f64(source);
    let out_width = window.width as usize;
    let out_height = window.height as usize;
    let mut out = vec![0.0; source.bands * out_width * out_height];
    for band in 0..source.bands {
        let fill = nodata.get(band).copied().flatten().unwrap_or(0.0);
        for out_row in 0..out_height {
            for out_col in 0..out_width {
                let src_col = window.col_off + out_col as i64;
                let src_row = window.row_off + out_row as i64;
                let out_index = band * out_width * out_height + out_row * out_width + out_col;
                out[out_index] = if src_col >= 0
                    && src_row >= 0
                    && src_col < source.width as i64
                    && src_row < source.height as i64
                {
                    source_values[band * source.width * source.height
                        + src_row as usize * source.width
                        + src_col as usize]
                } else {
                    fill
                };
            }
        }
    }
    f64_to_raster_data(source.dtype(), out, source.bands, out_height, out_width)
}

pub(crate) fn valid_mask(
    array: &RasterArray,
    nodata: &[Option<f64>],
    explicit_mask: Option<&[bool]>,
) -> Vec<bool> {
    let values = raster_to_f64(array);
    let pixels = array.height * array.width;
    let mut mask = Vec::with_capacity(values.len());
    for band in 0..array.bands {
        let nodata = nodata.get(band).copied().flatten();
        for pixel in 0..pixels {
            let index = band * pixels + pixel;
            let value = values[index];
            let explicit_valid = explicit_mask
                .and_then(|mask| mask.get(index))
                .copied()
                .unwrap_or(true);
            mask.push(explicit_valid && !value.is_nan() && !nodata_matches(value, nodata));
        }
    }
    mask
}

pub(crate) fn raster_to_f64(array: &RasterArray) -> Vec<f64> {
    match &array.data {
        RasterData::U8(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::I16(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::U16(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::I32(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::U32(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::F32(values) => values.iter().map(|&value| value as f64).collect(),
        RasterData::F64(values) => values.clone(),
    }
}

pub(crate) fn f64_to_raster_data(
    dtype: RasterDType,
    values: Vec<f64>,
    bands: usize,
    height: usize,
    width: usize,
) -> GisResult<RasterArray> {
    let shape = [bands, height, width];
    let data = match dtype {
        RasterDType::UInt8 => RasterData::U8(
            values
                .iter()
                .map(|&value| value.round().clamp(0.0, u8::MAX as f64) as u8)
                .collect(),
        ),
        RasterDType::Int16 => RasterData::I16(
            values
                .iter()
                .map(|&value| value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
                .collect(),
        ),
        RasterDType::UInt16 => RasterData::U16(
            values
                .iter()
                .map(|&value| value.round().clamp(0.0, u16::MAX as f64) as u16)
                .collect(),
        ),
        RasterDType::Int32 => RasterData::I32(
            values
                .iter()
                .map(|&value| value.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
                .collect(),
        ),
        RasterDType::UInt32 => RasterData::U32(
            values
                .iter()
                .map(|&value| value.round().clamp(0.0, u32::MAX as f64) as u32)
                .collect(),
        ),
        RasterDType::Float32 => RasterData::F32(values.iter().map(|&value| value as f32).collect()),
        RasterDType::Float64 => RasterData::F64(values),
    };
    RasterArray::new(data, &shape)
}

fn nodata_matches(value: f64, nodata: Option<f64>) -> bool {
    nodata.is_some_and(|nodata| value == nodata || (value.is_nan() && nodata.is_nan()))
}

pub(crate) fn deinterleave<T: Copy>(data: Vec<T>, bands: usize) -> Vec<T> {
    if bands <= 1 {
        return data;
    }
    let pixels = data.len() / bands;
    let mut out = Vec::with_capacity(data.len());
    for band in 0..bands {
        for pixel in 0..pixels {
            out.push(data[pixel * bands + band]);
        }
    }
    out
}
