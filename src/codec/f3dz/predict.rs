//! Deterministic lattice quantization and DEM predictors.

use super::format::{
    PREDICTOR_LORENZO, PREDICTOR_ORDER_ZERO, PREDICTOR_PLANE, PREDICTOR_PREVIOUS_LOD,
};
use super::{F3dzError, F3dzResult};

pub const NAN_TOKEN: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlaneModel {
    pub x_slope: i32,
    pub y_slope: i32,
    pub intercept: i32,
}

pub fn quantize_source(values: &[f32], step: f32, bound: f32) -> F3dzResult<Vec<Option<i32>>> {
    if !step.is_finite() || step <= 0.0 || !bound.is_finite() || bound < 0.0 {
        return Err(F3dzError::InvalidArgument(
            "quantization step/bound must be finite and positive".to_string(),
        ));
    }
    let mut quantized = Vec::with_capacity(values.len());
    for (index, &value) in values.iter().enumerate() {
        if value.is_nan() {
            quantized.push(None);
            continue;
        }
        if !value.is_finite() {
            return Err(F3dzError::InvalidArgument(format!(
                "infinite height at sample {index}"
            )));
        }
        // Both operands are binary32 dyadic rationals. Compute their ratio and
        // round to nearest, ties-to-even with integer arithmetic so the stream
        // is independent of host floating-point division behavior.
        let code = round_f32_ratio_ties_even(value, step).ok_or_else(|| {
            F3dzError::InvalidArgument(format!(
                "height at sample {index} is outside the f3dz v1 i32 lattice"
            ))
        })?;
        let reconstructed = (code as f32) * step;
        let error = (reconstructed - value).abs();
        if !reconstructed.is_finite() || error > bound {
            return Err(F3dzError::InvalidArgument(format!(
                "height at sample {index} cannot be represented within the declared bound: error={error} bound={bound}"
            )));
        }
        quantized.push(Some(code));
    }
    Ok(quantized)
}

pub fn fit_least_squares_plane(
    quantized: &[Option<i32>],
    width: usize,
    height: usize,
) -> PlaneModel {
    if width == 0 || height == 0 || quantized.len() != width * height {
        return PlaneModel::default();
    }
    let mut n = 0i128;
    let mut sx = 0i128;
    let mut sy = 0i128;
    let mut sxx = 0i128;
    let mut syy = 0i128;
    let mut sxy = 0i128;
    let mut sz = 0i128;
    let mut sxz = 0i128;
    let mut syz = 0i128;
    for y in 0..height {
        for x in 0..width {
            let Some(z) = quantized[y * width + x] else {
                continue;
            };
            let x = x as i128;
            let y = y as i128;
            let z = i128::from(z);
            n += 1;
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            sz += z;
            sxz += x * z;
            syz += y * z;
        }
    }
    if n == 0 {
        return PlaneModel::default();
    }
    let matrix = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
    let rhs = [sxz, syz, sz];
    let determinant = det3(matrix);
    if determinant == 0 {
        return PlaneModel {
            intercept: round_ratio_i128(sz, n)
                .and_then(|value| i32::try_from(value).ok())
                .unwrap_or_default(),
            ..PlaneModel::default()
        };
    }
    let x_num = det3([
        [rhs[0], matrix[0][1], matrix[0][2]],
        [rhs[1], matrix[1][1], matrix[1][2]],
        [rhs[2], matrix[2][1], matrix[2][2]],
    ]);
    let y_num = det3([
        [matrix[0][0], rhs[0], matrix[0][2]],
        [matrix[1][0], rhs[1], matrix[1][2]],
        [matrix[2][0], rhs[2], matrix[2][2]],
    ]);
    let c_num = det3([
        [matrix[0][0], matrix[0][1], rhs[0]],
        [matrix[1][0], matrix[1][1], rhs[1]],
        [matrix[2][0], matrix[2][1], rhs[2]],
    ]);
    PlaneModel {
        x_slope: coefficient_i32(x_num, determinant),
        y_slope: coefficient_i32(y_num, determinant),
        intercept: coefficient_i32(c_num, determinant),
    }
}

pub fn encode_residual_tokens(
    quantized: &[Option<i32>],
    width: usize,
    predictor_id: u8,
    plane: PlaneModel,
    previous: Option<&[Option<i32>]>,
) -> F3dzResult<Vec<u32>> {
    if width == 0 || !quantized.len().is_multiple_of(width) {
        return Err(F3dzError::InvalidArgument(
            "invalid predictor grid dimensions".to_string(),
        ));
    }
    if predictor_id == PREDICTOR_PREVIOUS_LOD
        && previous.map(|values| values.len()) != Some(quantized.len())
    {
        return Err(F3dzError::InvalidArgument(
            "previous-LOD predictor requires a same-sized reconstructed base".to_string(),
        ));
    }
    let mut reconstructed = vec![None; quantized.len()];
    let mut tokens = Vec::with_capacity(quantized.len());
    for index in 0..quantized.len() {
        let Some(value) = quantized[index] else {
            tokens.push(NAN_TOKEN);
            reconstructed[index] = None;
            continue;
        };
        // Load-bearing feedback invariant: prediction consults only values
        // already reconstructed by the decoder model (`reconstructed`), never
        // the higher-precision source heights. Thus residual quantization error
        // cannot feed forward and compound beyond epsilon.
        let predicted = predict(predictor_id, &reconstructed, width, index, plane, previous)?;
        let residual = i64::from(value) - i64::from(predicted);
        let residual = i32::try_from(residual).map_err(|_| {
            F3dzError::InvalidArgument(format!("predictor residual overflow at sample {index}"))
        })?;
        let token = zigzag(residual);
        if token == NAN_TOKEN {
            return Err(F3dzError::InvalidArgument(format!(
                "predictor residual at sample {index} collides with the NaN escape"
            )));
        }
        tokens.push(token);
        reconstructed[index] = Some(value);
    }
    Ok(tokens)
}

pub fn decode_residual_tokens(
    tokens: &[u32],
    width: usize,
    predictor_id: u8,
    plane: PlaneModel,
    previous: Option<&[Option<i32>]>,
) -> F3dzResult<Vec<Option<i32>>> {
    if width == 0 || !tokens.len().is_multiple_of(width) {
        return Err(F3dzError::InvalidArgument(
            "invalid predictor grid dimensions".to_string(),
        ));
    }
    if predictor_id == PREDICTOR_PREVIOUS_LOD
        && previous.map(|values| values.len()) != Some(tokens.len())
    {
        return Err(F3dzError::InvalidArgument(
            "previous-LOD predictor requires a same-sized reconstructed base".to_string(),
        ));
    }
    let mut reconstructed = vec![None; tokens.len()];
    for (index, &token) in tokens.iter().enumerate() {
        if token == NAN_TOKEN {
            continue;
        }
        let predicted = predict(predictor_id, &reconstructed, width, index, plane, previous)?;
        let value = i64::from(predicted) + i64::from(unzigzag(token));
        reconstructed[index] = Some(i32::try_from(value).map_err(|_| {
            F3dzError::InvalidArgument(format!(
                "reconstructed lattice value overflow at sample {index}"
            ))
        })?);
    }
    Ok(reconstructed)
}

pub fn reconstruct_values(quantized: &[Option<i32>], step: f32) -> Vec<f32> {
    quantized
        .iter()
        .map(|value| match value {
            Some(value) => (*value as f32) * step,
            None => f32::from_bits(0x7fc0_0000),
        })
        .collect()
}

fn predict(
    predictor_id: u8,
    reconstructed: &[Option<i32>],
    width: usize,
    index: usize,
    plane: PlaneModel,
    previous: Option<&[Option<i32>]>,
) -> F3dzResult<i32> {
    match predictor_id {
        PREDICTOR_LORENZO => Ok(lorenzo(reconstructed, width, index)),
        PREDICTOR_PLANE => {
            let x = (index % width) as i64;
            let y = (index / width) as i64;
            let prediction = i64::from(plane.x_slope) * x
                + i64::from(plane.y_slope) * y
                + i64::from(plane.intercept);
            i32::try_from(prediction).map_err(|_| {
                F3dzError::InvalidArgument(format!(
                    "least-squares plane overflows at sample {index}"
                ))
            })
        }
        PREDICTOR_PREVIOUS_LOD => Ok(previous
            .and_then(|values| values[index])
            .and_then(|value| value.checked_mul(4))
            .unwrap_or(0)),
        PREDICTOR_ORDER_ZERO => Ok(0),
        other => Err(F3dzError::InvalidArgument(format!(
            "unknown predictor id {other}"
        ))),
    }
}

fn lorenzo(reconstructed: &[Option<i32>], width: usize, index: usize) -> i32 {
    let x = index % width;
    let y = index / width;
    let left = (x > 0).then(|| reconstructed[index - 1]).flatten();
    let up = (y > 0).then(|| reconstructed[index - width]).flatten();
    let upper_left = (x > 0 && y > 0)
        .then(|| reconstructed[index - width - 1])
        .flatten();
    match (left, up, upper_left) {
        (Some(left), Some(up), Some(upper_left)) => {
            let prediction = i64::from(left) + i64::from(up) - i64::from(upper_left);
            i32::try_from(prediction).unwrap_or(0)
        }
        (Some(left), _, _) => left,
        (_, Some(up), _) => up,
        _ => 0,
    }
}

fn zigzag(value: i32) -> u32 {
    ((value << 1) ^ (value >> 31)) as u32
}

fn unzigzag(value: u32) -> i32 {
    ((value >> 1) as i32) ^ (-((value & 1) as i32))
}

fn det3(matrix: [[i128; 3]; 3]) -> i128 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn coefficient_i32(numerator: i128, denominator: i128) -> i32 {
    round_ratio_i128(numerator, denominator)
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or_default()
}

fn round_ratio_i128(numerator: i128, denominator: i128) -> Option<i128> {
    if denominator == 0 {
        return None;
    }
    let negative = (numerator < 0) ^ (denominator < 0);
    let numerator = numerator.unsigned_abs();
    let denominator = denominator.unsigned_abs();
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let doubled = remainder.checked_mul(2)?;
    let rounded = if doubled > denominator || (doubled == denominator && quotient & 1 == 1) {
        quotient.checked_add(1)?
    } else {
        quotient
    };
    let rounded = i128::try_from(rounded).ok()?;
    Some(if negative { -rounded } else { rounded })
}

fn round_f32_ratio_ties_even(value: f32, step: f32) -> Option<i32> {
    let (value_negative, value_mantissa, value_exponent) = f32_dyadic(value)?;
    let (_, step_mantissa, step_exponent) = f32_dyadic(step)?;
    if step_mantissa == 0 {
        return None;
    }
    if value_mantissa == 0 {
        return Some(0);
    }
    let exponent_delta = value_exponent - step_exponent;
    let (numerator, denominator) = if exponent_delta >= 0 {
        let shift = u32::try_from(exponent_delta).ok()?;
        (
            u128::from(value_mantissa).checked_shl(shift)?,
            u128::from(step_mantissa),
        )
    } else {
        let shift = u32::try_from(-exponent_delta).ok()?;
        if shift >= 128 {
            return Some(0);
        }
        (
            u128::from(value_mantissa),
            u128::from(step_mantissa).checked_shl(shift)?,
        )
    };
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let doubled = remainder.checked_mul(2)?;
    let rounded = if doubled > denominator || (doubled == denominator && quotient & 1 == 1) {
        quotient.checked_add(1)?
    } else {
        quotient
    };
    if value_negative {
        let magnitude = i64::try_from(rounded).ok()?;
        i32::try_from(-magnitude).ok()
    } else {
        i32::try_from(rounded).ok()
    }
}

/// Return `(negative, mantissa, exponent)` such that
/// `value == (-1)^negative * mantissa * 2^exponent`.
fn f32_dyadic(value: f32) -> Option<(bool, u32, i32)> {
    if !value.is_finite() {
        return None;
    }
    let bits = value.to_bits();
    let negative = bits >> 31 != 0;
    let exponent_bits = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ffff;
    if exponent_bits == 0 {
        Some((negative, fraction, -149))
    } else {
        Some((negative, (1 << 23) | fraction, exponent_bits - 127 - 23))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ratio_rounds_ties_to_even() {
        assert_eq!(round_f32_ratio_ties_even(0.25, 0.5), Some(0));
        assert_eq!(round_f32_ratio_ties_even(0.75, 0.5), Some(2));
        assert_eq!(round_f32_ratio_ties_even(-0.75, 0.5), Some(-2));
        assert_eq!(round_f32_ratio_ties_even(-0.25, 0.5), Some(0));
    }

    #[test]
    fn encoder_feedback_uses_reconstructed_lattice() {
        let source = [0.24, 0.76, 1.24, 1.76, 0.26, 0.74, 1.26, 1.74];
        let quantized = quantize_source(&source, 0.5, 0.25).unwrap();
        let tokens = encode_residual_tokens(
            &quantized,
            4,
            PREDICTOR_LORENZO,
            PlaneModel::default(),
            None,
        )
        .unwrap();
        let decoded =
            decode_residual_tokens(&tokens, 4, PREDICTOR_LORENZO, PlaneModel::default(), None)
                .unwrap();
        assert_eq!(decoded, quantized);
        let reconstructed = reconstruct_values(&decoded, 0.5);
        assert!(reconstructed
            .iter()
            .zip(source)
            .all(|(decoded, source)| (*decoded - source).abs() <= 0.25));
    }

    #[test]
    fn nan_uses_explicit_escape_and_does_not_poison_neighbors() {
        let quantized = vec![Some(10), None, Some(12), Some(13)];
        let tokens = encode_residual_tokens(
            &quantized,
            2,
            PREDICTOR_LORENZO,
            PlaneModel::default(),
            None,
        )
        .unwrap();
        assert_eq!(tokens[1], NAN_TOKEN);
        let decoded =
            decode_residual_tokens(&tokens, 2, PREDICTOR_LORENZO, PlaneModel::default(), None)
                .unwrap();
        assert_eq!(decoded, quantized);
        assert!(reconstruct_values(&decoded, 1.0)[1].is_nan());
    }

    #[test]
    fn integer_least_squares_plane_is_exact_for_a_plane() {
        let values: Vec<Option<i32>> = (0..5)
            .flat_map(|y| (0..7).map(move |x| Some(3 * x - 2 * y + 11)))
            .collect();
        assert_eq!(
            fit_least_squares_plane(&values, 7, 5),
            PlaneModel {
                x_slope: 3,
                y_slope: -2,
                intercept: 11
            }
        );
    }
}
