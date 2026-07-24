//! Deterministic lattice quantization and DEM predictors.

use super::format::{
    PREDICTOR_LORENZO, PREDICTOR_ORDER_ZERO, PREDICTOR_PLANE, PREDICTOR_PREVIOUS_LOD,
};
use super::{F3dzError, F3dzResult};

pub const NAN_TOKEN: u32 = u32::MAX;
pub const RAW_TOKEN: u32 = u32::MAX - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizedSample {
    Code(i32),
    Raw(u32),
    Nan(u32),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlaneModel {
    pub x_slope: i32,
    pub y_slope: i32,
    pub intercept: i32,
}

pub fn quantize_source(values: &[f32], step: f32, bound: f32) -> F3dzResult<Vec<QuantizedSample>> {
    if !step.is_finite() || step <= 0.0 || !bound.is_finite() || bound < 0.0 {
        return Err(F3dzError::InvalidArgument(
            "quantization step/bound must be finite and positive".to_string(),
        ));
    }
    let mut quantized = Vec::with_capacity(values.len());
    for (index, &value) in values.iter().enumerate() {
        if value.is_nan() {
            quantized.push(QuantizedSample::Nan(value.to_bits()));
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
        let nearest = round_f32_ratio_ties_even(value, step).ok_or_else(|| {
            F3dzError::InvalidArgument(format!(
                "height at sample {index} is outside the f3dz v1 i32 lattice"
            ))
        })?;
        // Binary32 multiplication can move the mathematically nearest lattice
        // point by one output ULP. Evaluate that code and its adjacent lattice
        // points against the exact f32 source values, then deterministically
        // prefer an even code on an exact tie.
        let code = [
            nearest.checked_sub(1),
            Some(nearest),
            nearest.checked_add(1),
        ]
        .into_iter()
        .flatten()
        .min_by(|left, right| {
            let left_value = (*left as f32) * step;
            let right_value = (*right as f32) * step;
            let left_error = (f64::from(left_value) - f64::from(value)).abs();
            let right_error = (f64::from(right_value) - f64::from(value)).abs();
            left_error
                .total_cmp(&right_error)
                .then_with(|| (left & 1).cmp(&(right & 1)))
                .then_with(|| left.unsigned_abs().cmp(&right.unsigned_abs()))
        })
        .expect("the nearest lattice code is always present");
        let reconstructed = (code as f32) * step;
        let error = (reconstructed - value).abs();
        // The GPU decoder reserves i32::MIN as the "no causal value" marker in
        // workgroup memory. Preserve such a perfectly finite source through
        // the RAW escape instead of making the marker ambiguous.
        if code == i32::MIN || !reconstructed.is_finite() || error > bound {
            quantized.push(QuantizedSample::Raw(value.to_bits()));
            continue;
        }
        quantized.push(QuantizedSample::Code(code));
    }
    Ok(quantized)
}

pub fn fit_least_squares_plane(
    quantized: &[QuantizedSample],
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
            let QuantizedSample::Code(z) = quantized[y * width + x] else {
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
    quantized: &[QuantizedSample],
    width: usize,
    predictor_id: u8,
    plane: PlaneModel,
    previous: Option<&[QuantizedSample]>,
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
        let value = match quantized[index] {
            QuantizedSample::Nan(_) => {
                tokens.push(NAN_TOKEN);
                reconstructed[index] = None;
                continue;
            }
            QuantizedSample::Raw(bits) => {
                tokens.push(RAW_TOKEN);
                tokens.push(bits);
                reconstructed[index] = None;
                continue;
            }
            QuantizedSample::Code(value) => value,
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
        if token >= RAW_TOKEN {
            return Err(F3dzError::InvalidArgument(format!(
                "predictor residual at sample {index} collides with an escape token"
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
    previous: Option<&[QuantizedSample]>,
) -> F3dzResult<Vec<QuantizedSample>> {
    if width == 0 {
        return Err(F3dzError::InvalidArgument(
            "invalid predictor grid dimensions".to_string(),
        ));
    }
    let mut reconstructed: Vec<Option<i32>> = Vec::with_capacity(tokens.len());
    let mut output = Vec::with_capacity(tokens.len());
    let mut cursor = 0usize;
    while cursor < tokens.len() {
        let token = tokens[cursor];
        cursor += 1;
        let index = output.len();
        if token == NAN_TOKEN {
            let bits = f32::NAN.to_bits();
            reconstructed.push(None);
            output.push(QuantizedSample::Nan(bits));
            continue;
        }
        if token == RAW_TOKEN {
            let bits = *tokens.get(cursor).ok_or_else(|| {
                F3dzError::InvalidArgument(format!(
                    "escape token at sample {index} is missing its binary32 payload"
                ))
            })?;
            cursor += 1;
            let value = f32::from_bits(bits);
            if !value.is_finite() {
                return Err(F3dzError::InvalidArgument(format!(
                    "raw escape at sample {index} carries non-finite bits"
                )));
            }
            reconstructed.push(None);
            output.push(QuantizedSample::Raw(bits));
            continue;
        }
        let predicted = predict(predictor_id, &reconstructed, width, index, plane, previous)?;
        let value = i64::from(predicted) + i64::from(unzigzag(token));
        let value = i32::try_from(value).map_err(|_| {
            F3dzError::InvalidArgument(format!(
                "reconstructed lattice value overflow at sample {index}"
            ))
        })?;
        reconstructed.push(Some(value));
        output.push(QuantizedSample::Code(value));
    }
    if !output.len().is_multiple_of(width) {
        return Err(F3dzError::InvalidArgument(
            "decoded predictor grid is not row-aligned".to_string(),
        ));
    }
    if predictor_id == PREDICTOR_PREVIOUS_LOD
        && previous.map(|values| values.len()) != Some(output.len())
    {
        return Err(F3dzError::InvalidArgument(
            "previous-LOD predictor requires a same-sized reconstructed base".to_string(),
        ));
    }
    Ok(output)
}

pub fn reconstruct_values(quantized: &[QuantizedSample], step: f32) -> Vec<f32> {
    quantized
        .iter()
        .map(|value| match value {
            QuantizedSample::Code(value) => (*value as f32) * step,
            QuantizedSample::Raw(bits) | QuantizedSample::Nan(bits) => f32::from_bits(*bits),
        })
        .collect()
}

/// Pack predictor tokens into the byte stream consumed by rANS.
///
/// Each logical sample begins with a canonical unsigned LEB128 value:
/// `0` is canonical NaN, `1` is RAW followed by four little-endian f32 bits,
/// `2` is a run followed by `(sample_code, run_length)`, and an ordinary
/// zig-zag residual `r` is encoded as `r + 3`. Run sample code `0` means NaN
/// and `r + 1` means residual `r`. Runs of four or more non-RAW samples make
/// the spatial coherence available to an otherwise static entropy model.
pub(crate) fn pack_residual_tokens(tokens: &[u32]) -> F3dzResult<Vec<u8>> {
    let mut samples = Vec::with_capacity(tokens.len());
    let mut cursor = 0usize;
    while cursor < tokens.len() {
        let token = tokens[cursor];
        cursor += 1;
        match token {
            NAN_TOKEN => samples.push((NAN_TOKEN, None)),
            RAW_TOKEN => {
                let bits = *tokens.get(cursor).ok_or_else(|| {
                    F3dzError::InvalidArgument(
                        "RAW token is missing its binary32 payload".to_string(),
                    )
                })?;
                cursor += 1;
                if !f32::from_bits(bits).is_finite() {
                    return Err(F3dzError::InvalidArgument(
                        "RAW token carries non-finite binary32 bits".to_string(),
                    ));
                }
                samples.push((RAW_TOKEN, Some(bits)));
            }
            residual => samples.push((residual, None)),
        }
    }
    let mut packed = Vec::with_capacity(samples.len());
    let mut sample = 0usize;
    while sample < samples.len() {
        let (token, bits) = samples[sample];
        let mut run_length = 1usize;
        if bits.is_none() {
            while sample + run_length < samples.len()
                && samples[sample + run_length] == (token, None)
            {
                run_length += 1;
            }
        }
        if run_length >= 4 {
            write_uleb128(2, &mut packed);
            write_uleb128(if token == NAN_TOKEN { 0 } else { token + 1 }, &mut packed);
            write_uleb128(run_length as u32, &mut packed);
            sample += run_length;
            continue;
        }
        match token {
            NAN_TOKEN => write_uleb128(0, &mut packed),
            RAW_TOKEN => {
                write_uleb128(1, &mut packed);
                packed
                    .extend_from_slice(&bits.expect("RAW samples always carry bits").to_le_bytes());
            }
            residual => write_uleb128(residual + 3, &mut packed),
        }
        sample += 1;
    }
    Ok(packed)
}

/// Return the canonical token byte stream when every logical sample is a
/// single ordinary residual whose unsigned LEB128 representation is one byte.
/// The stream remains valid for the general CPU decoder; the page flag merely
/// proves that the GPU may map byte `i` directly to sample `i`.
pub(crate) fn pack_direct_residual_tokens(tokens: &[u32]) -> Option<Vec<u8>> {
    let mut packed = Vec::with_capacity(tokens.len());
    for &token in tokens {
        if token >= RAW_TOKEN {
            return None;
        }
        let mapped = token.checked_add(3)?;
        if mapped > 0x7f {
            return None;
        }
        packed.push(mapped as u8);
    }
    Some(packed)
}

pub(crate) fn unpack_residual_tokens(packed: &[u8], sample_count: usize) -> F3dzResult<Vec<u32>> {
    let mut tokens = Vec::with_capacity(sample_count);
    let mut cursor = 0usize;
    let mut sample = 0usize;
    while sample < sample_count {
        let mapped = read_uleb128(packed, &mut cursor).map_err(|reason| {
            F3dzError::InvalidArgument(format!("sample {sample} token: {reason}"))
        })?;
        match mapped {
            0 => {
                tokens.push(NAN_TOKEN);
                sample += 1;
            }
            1 => {
                let end = cursor.checked_add(4).ok_or_else(|| {
                    F3dzError::InvalidArgument("RAW payload range overflow".to_string())
                })?;
                let bytes = packed.get(cursor..end).ok_or_else(|| {
                    F3dzError::InvalidArgument(format!(
                        "sample {sample} RAW token is missing its binary32 payload"
                    ))
                })?;
                let bits = u32::from_le_bytes(bytes.try_into().unwrap());
                if !f32::from_bits(bits).is_finite() {
                    return Err(F3dzError::InvalidArgument(format!(
                        "sample {sample} RAW token carries non-finite bits"
                    )));
                }
                tokens.push(RAW_TOKEN);
                tokens.push(bits);
                cursor = end;
                sample += 1;
            }
            2 => {
                let run_code = read_uleb128(packed, &mut cursor).map_err(|reason| {
                    F3dzError::InvalidArgument(format!("sample {sample} run code: {reason}"))
                })?;
                let run_length = read_uleb128(packed, &mut cursor).map_err(|reason| {
                    F3dzError::InvalidArgument(format!("sample {sample} run length: {reason}"))
                })? as usize;
                if run_length < 4 || run_length > sample_count - sample {
                    return Err(F3dzError::InvalidArgument(format!(
                        "sample {sample} run length {run_length} is invalid"
                    )));
                }
                if run_code >= RAW_TOKEN {
                    return Err(F3dzError::InvalidArgument(format!(
                        "sample {sample} run code {run_code} is invalid"
                    )));
                }
                let token = if run_code == 0 {
                    NAN_TOKEN
                } else {
                    run_code - 1
                };
                tokens.extend(std::iter::repeat_n(token, run_length));
                sample += run_length;
            }
            value => {
                tokens.push(value - 3);
                sample += 1;
            }
        }
    }
    if cursor != packed.len() {
        return Err(F3dzError::InvalidArgument(format!(
            "token stream has {} trailing bytes",
            packed.len() - cursor
        )));
    }
    Ok(tokens)
}

/// Divide all finite residuals by their exact per-layer GCD before entropy
/// coding. The scale is stored in the page header and restored before
/// prediction, so this changes no reconstructed lattice value. This matters
/// for fine tolerances: integer-metre DEMs quantized at 0.1 m otherwise carry
/// a redundant factor of ten in every residual.
pub(crate) fn normalize_residual_tokens(tokens: &[u32]) -> F3dzResult<(u32, Vec<u32>)> {
    let mut scale = 0u32;
    let mut cursor = 0usize;
    while cursor < tokens.len() {
        let token = tokens[cursor];
        cursor += 1;
        match token {
            NAN_TOKEN => {}
            RAW_TOKEN => {
                if cursor >= tokens.len() {
                    return Err(F3dzError::InvalidArgument(
                        "RAW token is missing its binary32 payload".to_string(),
                    ));
                }
                cursor += 1;
            }
            residual => {
                scale = gcd(scale, unzigzag(residual).unsigned_abs());
            }
        }
    }
    let scale = scale.max(1);
    let mut normalized = Vec::with_capacity(tokens.len());
    cursor = 0;
    while cursor < tokens.len() {
        let token = tokens[cursor];
        cursor += 1;
        match token {
            NAN_TOKEN => normalized.push(NAN_TOKEN),
            RAW_TOKEN => {
                normalized.push(RAW_TOKEN);
                normalized.push(tokens[cursor]);
                cursor += 1;
            }
            residual => normalized.push(zigzag(unzigzag(residual) / scale as i32)),
        }
    }
    Ok((scale, normalized))
}

pub(crate) fn denormalize_residual_tokens(tokens: &[u32], scale: u32) -> F3dzResult<Vec<u32>> {
    if scale == 0 || scale > i32::MAX as u32 {
        return Err(F3dzError::InvalidArgument(format!(
            "invalid residual scale {scale}"
        )));
    }
    let mut restored = Vec::with_capacity(tokens.len());
    let mut cursor = 0usize;
    while cursor < tokens.len() {
        let token = tokens[cursor];
        cursor += 1;
        match token {
            NAN_TOKEN => restored.push(NAN_TOKEN),
            RAW_TOKEN => {
                let bits = *tokens.get(cursor).ok_or_else(|| {
                    F3dzError::InvalidArgument(
                        "RAW token is missing its binary32 payload".to_string(),
                    )
                })?;
                restored.push(RAW_TOKEN);
                restored.push(bits);
                cursor += 1;
            }
            residual => {
                let value = i64::from(unzigzag(residual)) * i64::from(scale);
                let value = i32::try_from(value).map_err(|_| {
                    F3dzError::InvalidArgument(
                        "scaled predictor residual overflows i32".to_string(),
                    )
                })?;
                restored.push(zigzag(value));
            }
        }
    }
    Ok(restored)
}

/// Prove that a Lorenzo page is exactly reconstructible by a horizontal
/// prefix scan followed by a vertical prefix scan.
///
/// This is an integer identity for a complete Lorenzo lattice. We explicitly
/// reject escape samples and any intermediate i32 overflow, then compare the
/// result with the canonical causal decoder. The encoder records the result in
/// the page flags so the GPU can select a low-barrier pipeline without
/// inspecting or CPU-decoding the entropy stream.
pub(crate) fn lorenzo_prefix_safe(
    residual_tokens: &[u32],
    width: usize,
    expected: &[QuantizedSample],
) -> bool {
    if width == 0
        || expected.is_empty()
        || !expected.len().is_multiple_of(width)
        || residual_tokens.len() != expected.len()
    {
        return false;
    }
    let mut prefix = Vec::with_capacity(residual_tokens.len());
    for &token in residual_tokens {
        if token >= RAW_TOKEN {
            return false;
        }
        prefix.push(unzigzag(token));
    }
    let height = expected.len() / width;
    for y in 0..height {
        for x in 1..width {
            let index = y * width + x;
            let Some(value) = prefix[index].checked_add(prefix[index - 1]) else {
                return false;
            };
            prefix[index] = value;
        }
    }
    for x in 0..width {
        for y in 1..height {
            let index = y * width + x;
            let Some(value) = prefix[index].checked_add(prefix[index - width]) else {
                return false;
            };
            prefix[index] = value;
        }
    }
    prefix.iter().zip(expected).all(
        |(&actual, expected)| matches!(expected, QuantizedSample::Code(value) if *value == actual),
    )
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn write_uleb128(mut value: u32, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn read_uleb128(data: &[u8], cursor: &mut usize) -> Result<u32, &'static str> {
    let mut value = 0u32;
    for byte_index in 0..5 {
        let byte = *data.get(*cursor).ok_or("truncated unsigned LEB128")?;
        *cursor += 1;
        if byte_index == 4 && byte > 0x0f {
            return Err("unsigned LEB128 overflows u32");
        }
        value |= u32::from(byte & 0x7f) << (byte_index * 7);
        if byte & 0x80 == 0 {
            if byte_index > 0 && byte == 0 {
                return Err("unsigned LEB128 is not minimally encoded");
            }
            return Ok(value);
        }
    }
    Err("unsigned LEB128 exceeds five bytes")
}

fn predict(
    predictor_id: u8,
    reconstructed: &[Option<i32>],
    width: usize,
    index: usize,
    plane: PlaneModel,
    previous: Option<&[QuantizedSample]>,
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
        PREDICTOR_PREVIOUS_LOD => {
            match previous
                .and_then(|values| values.get(index))
                .and_then(|value| match value {
                    QuantizedSample::Code(value) => Some(*value),
                    QuantizedSample::Raw(_) | QuantizedSample::Nan(_) => None,
                }) {
                Some(value) => value.checked_mul(4).ok_or_else(|| {
                    F3dzError::InvalidArgument(format!(
                        "previous-LOD predictor overflows at sample {index}"
                    ))
                }),
                None => Ok(0),
            }
        }
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
        let nan_bits = f32::NAN.to_bits();
        let quantized = vec![
            QuantizedSample::Code(10),
            QuantizedSample::Nan(nan_bits),
            QuantizedSample::Code(12),
            QuantizedSample::Code(13),
        ];
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
        assert!(matches!(decoded[1], QuantizedSample::Nan(_)));
        assert_eq!(decoded[0], quantized[0]);
        assert_eq!(decoded[2..], quantized[2..]);
        assert!(reconstruct_values(&decoded, 1.0)[1].is_nan());
    }

    #[test]
    fn lorenzo_prefix_proof_matches_causal_decode_and_rejects_unsafe_inputs() {
        let quantized: Vec<QuantizedSample> = (0..6)
            .flat_map(|y| {
                (0..8).map(move |x| QuantizedSample::Code(17 + x * 3 - y * 2 + (x * y) % 5))
            })
            .collect();
        let residuals = encode_residual_tokens(
            &quantized,
            8,
            PREDICTOR_LORENZO,
            PlaneModel::default(),
            None,
        )
        .unwrap();
        let decoded = decode_residual_tokens(
            &residuals,
            8,
            PREDICTOR_LORENZO,
            PlaneModel::default(),
            None,
        )
        .unwrap();
        assert!(lorenzo_prefix_safe(&residuals, 8, &decoded));

        let mut escaped = residuals.clone();
        escaped[5] = NAN_TOKEN;
        assert!(!lorenzo_prefix_safe(&escaped, 8, &decoded));

        let overflow = [zigzag(1_500_000_000), zigzag(1_000_000_000)];
        let expected = [QuantizedSample::Code(0), QuantizedSample::Code(0)];
        assert!(!lorenzo_prefix_safe(&overflow, 2, &expected));
    }

    #[test]
    fn packed_tokens_are_canonical_and_round_trip_escapes() {
        let tokens = [
            0,
            1,
            127,
            128,
            16_384,
            NAN_TOKEN,
            RAW_TOKEN,
            12.25f32.to_bits(),
        ];
        let packed = pack_residual_tokens(&tokens).unwrap();
        let decoded = unpack_residual_tokens(&packed, 7).unwrap();
        assert_eq!(decoded, tokens);
        let mut overlong = packed;
        overlong[0..1].copy_from_slice(&[0x82]);
        overlong.insert(1, 0);
        assert!(unpack_residual_tokens(&overlong, 7).is_err());
    }

    #[test]
    fn direct_tokens_accept_only_one_byte_ordinary_residuals() {
        let tokens = [0, 1, 7, 124];
        let direct = pack_direct_residual_tokens(&tokens).unwrap();
        assert_eq!(direct, [3, 4, 10, 127]);
        assert_eq!(
            unpack_residual_tokens(&direct, tokens.len()).unwrap(),
            tokens
        );
        assert!(pack_direct_residual_tokens(&[125]).is_none());
        assert!(pack_direct_residual_tokens(&[NAN_TOKEN]).is_none());
        assert!(pack_direct_residual_tokens(&[RAW_TOKEN, 12.25f32.to_bits()]).is_none());
    }

    #[test]
    fn packed_tokens_run_length_encode_residuals_and_nodata() {
        let tokens = [
            0, 0, 0, 0, NAN_TOKEN, NAN_TOKEN, NAN_TOKEN, NAN_TOKEN, NAN_TOKEN, 9,
        ];
        let packed = pack_residual_tokens(&tokens).unwrap();
        assert!(packed.len() < tokens.len());
        assert_eq!(unpack_residual_tokens(&packed, 10).unwrap(), tokens);
    }

    #[test]
    fn residual_gcd_normalization_is_exact_across_escapes() {
        let tokens = [
            zigzag(-30),
            zigzag(0),
            NAN_TOKEN,
            zigzag(90),
            RAW_TOKEN,
            12.25f32.to_bits(),
        ];
        let (scale, normalized) = normalize_residual_tokens(&tokens).unwrap();
        assert_eq!(scale, 30);
        assert_eq!(
            denormalize_residual_tokens(&normalized, scale).unwrap(),
            tokens
        );
    }

    #[test]
    fn finite_escape_preserves_bound_when_binary32_lattice_rounding_cannot() {
        let source = [1206.0, 1214.0, 1222.0, 1230.0];
        let quantized = quantize_source(&source, 0.8, 0.4).unwrap();
        assert!(quantized
            .iter()
            .any(|value| matches!(value, QuantizedSample::Raw(_))));
        let tokens = encode_residual_tokens(
            &quantized,
            4,
            PREDICTOR_LORENZO,
            PlaneModel::default(),
            None,
        )
        .unwrap();
        assert!(tokens.contains(&RAW_TOKEN));
        let decoded =
            decode_residual_tokens(&tokens, 4, PREDICTOR_LORENZO, PlaneModel::default(), None)
                .unwrap();
        assert_eq!(reconstruct_values(&decoded, 0.8), source);
    }

    #[test]
    fn integer_least_squares_plane_is_exact_for_a_plane() {
        let values: Vec<QuantizedSample> = (0..5)
            .flat_map(|y| (0..7).map(move |x| QuantizedSample::Code(3 * x - 2 * y + 11)))
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
