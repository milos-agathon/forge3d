use super::TextError;
use std::sync::Arc;
use ttf_parser::{Face, NormalizedCoordinate, Tag, VariationAxis};

pub(crate) struct PreparedVariations {
    pub face_bytes: Arc<[u8]>,
    pub coordinates: Vec<([u8; 4], f32)>,
}

fn normalize(axis: VariationAxis, value: f32) -> NormalizedCoordinate {
    let value = value.clamp(axis.min_value, axis.max_value);
    let normalized = if value == axis.def_value {
        0.0
    } else if value < axis.def_value {
        (value - axis.def_value) / (axis.def_value - axis.min_value)
    } else {
        (value - axis.def_value) / (axis.max_value - axis.def_value)
    };
    NormalizedCoordinate::from(normalized)
}

fn denormalize(axis: VariationAxis, coordinate: NormalizedCoordinate) -> f32 {
    let normalized = f32::from(coordinate.get()) / 16_384.0;
    if normalized < 0.0 {
        axis.def_value + normalized * (axis.def_value - axis.min_value)
    } else {
        axis.def_value + normalized * (axis.max_value - axis.def_value)
    }
}

fn face_offset(bytes: &[u8], face_index: u32) -> Result<usize, TextError> {
    if bytes.get(0..4) == Some(b"ttcf") {
        let count = u32::from_be_bytes(
            bytes
                .get(8..12)
                .ok_or(TextError::MalformedFvar)?
                .try_into()
                .map_err(|_| TextError::MalformedFvar)?,
        );
        if face_index >= count {
            return Err(TextError::MalformedFvar);
        }
        let offset = 12usize
            .checked_add(face_index as usize * 4)
            .ok_or(TextError::MalformedFvar)?;
        Ok(u32::from_be_bytes(
            bytes
                .get(offset..offset + 4)
                .ok_or(TextError::MalformedFvar)?
                .try_into()
                .map_err(|_| TextError::MalformedFvar)?,
        ) as usize)
    } else if face_index == 0 {
        Ok(0)
    } else {
        Err(TextError::MalformedFvar)
    }
}

fn without_avar(bytes: &Arc<[u8]>, face_index: u32) -> Result<Arc<[u8]>, TextError> {
    let base = face_offset(bytes, face_index)?;
    let count = u16::from_be_bytes(
        bytes
            .get(base + 4..base + 6)
            .ok_or(TextError::MalformedFvar)?
            .try_into()
            .map_err(|_| TextError::MalformedFvar)?,
    ) as usize;
    let mut copy = bytes.to_vec();
    for index in 0..count {
        let record = base + 12 + index * 16;
        if copy.get(record..record + 4) == Some(b"avar") {
            copy[record..record + 4].copy_from_slice(b"AVAR");
            return Ok(copy.into());
        }
    }
    Ok(Arc::clone(bytes))
}

pub(crate) fn prepare(
    face: &Face<'_>,
    bytes: &Arc<[u8]>,
    face_index: u32,
    fixed: &[([u8; 4], i32)],
) -> Result<PreparedVariations, TextError> {
    if fixed.is_empty() {
        return Ok(PreparedVariations {
            face_bytes: Arc::clone(bytes),
            coordinates: Vec::new(),
        });
    }
    let axes: Vec<_> = face.variation_axes().into_iter().collect();
    if axes.len() != fixed.len() {
        return Err(TextError::MalformedFvar);
    }
    let mut normalized = Vec::with_capacity(axes.len());
    for axis in &axes {
        let tag = axis.tag.0.to_be_bytes();
        let value = fixed
            .iter()
            .find_map(|(candidate, value)| (*candidate == tag).then_some(*value))
            .ok_or(TextError::MalformedFvar)?;
        normalized.push(normalize(*axis, value as f32 / 65_536.0));
    }
    if let Some(avar) = face.tables().avar {
        avar.map_coordinates(&mut normalized)
            .ok_or(TextError::MalformedFvar)?;
    }
    let coordinates = axes
        .into_iter()
        .zip(normalized)
        .map(|(axis, coordinate)| (axis.tag.0.to_be_bytes(), denormalize(axis, coordinate)))
        .collect();
    Ok(PreparedVariations {
        face_bytes: without_avar(bytes, face_index)?,
        coordinates,
    })
}

pub(crate) fn apply(face: &mut Face<'_>, coordinates: &[([u8; 4], f32)]) -> Result<(), TextError> {
    for (tag, value) in coordinates {
        face.set_variation(Tag::from_bytes(tag), *value)
            .ok_or(TextError::MalformedFvar)?;
    }
    Ok(())
}
