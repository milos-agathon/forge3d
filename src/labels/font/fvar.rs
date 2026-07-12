use super::TextError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NamedInstance {
    pub name_id: u16,
    pub coordinates: Vec<([u8; 4], i32)>,
}

fn u16_at(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_be_bytes(
        data.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn i32_at(data: &[u8], offset: usize) -> Option<i32> {
    Some(i32::from_be_bytes(
        data.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

pub(crate) fn parse_named_instances(data: &[u8]) -> Result<Vec<NamedInstance>, TextError> {
    if data.get(0..4) != Some(&0x0001_0000u32.to_be_bytes()) {
        return Err(TextError::MalformedFvar);
    }
    let axes_offset = usize::from(u16_at(data, 4).ok_or(TextError::MalformedFvar)?);
    let axis_count = usize::from(u16_at(data, 8).ok_or(TextError::MalformedFvar)?);
    let axis_size = usize::from(u16_at(data, 10).ok_or(TextError::MalformedFvar)?);
    let instance_count = usize::from(u16_at(data, 12).ok_or(TextError::MalformedFvar)?);
    let instance_size = usize::from(u16_at(data, 14).ok_or(TextError::MalformedFvar)?);
    if axis_size < 20 || instance_size < 4 + axis_count * 4 {
        return Err(TextError::MalformedFvar);
    }

    let axes_end = axes_offset
        .checked_add(
            axis_count
                .checked_mul(axis_size)
                .ok_or(TextError::MalformedFvar)?,
        )
        .ok_or(TextError::MalformedFvar)?;
    let instances_end = axes_end
        .checked_add(
            instance_count
                .checked_mul(instance_size)
                .ok_or(TextError::MalformedFvar)?,
        )
        .ok_or(TextError::MalformedFvar)?;
    if axes_offset < 16 || axes_end > data.len() || instances_end > data.len() {
        return Err(TextError::MalformedFvar);
    }

    let mut tags = Vec::with_capacity(axis_count);
    for axis in 0..axis_count {
        let offset = axes_offset
            .checked_add(
                axis.checked_mul(axis_size)
                    .ok_or(TextError::MalformedFvar)?,
            )
            .ok_or(TextError::MalformedFvar)?;
        tags.push(
            data.get(offset..offset + 4)
                .ok_or(TextError::MalformedFvar)?
                .try_into()
                .map_err(|_| TextError::MalformedFvar)?,
        );
    }

    let instances_offset = axes_end;
    let mut instances = Vec::with_capacity(instance_count);
    for index in 0..instance_count {
        let offset = instances_offset
            .checked_add(
                index
                    .checked_mul(instance_size)
                    .ok_or(TextError::MalformedFvar)?,
            )
            .ok_or(TextError::MalformedFvar)?;
        let name_id = u16_at(data, offset).ok_or(TextError::MalformedFvar)?;
        let mut coordinates = Vec::with_capacity(axis_count);
        for (axis, tag) in tags.iter().enumerate() {
            coordinates.push((
                *tag,
                i32_at(data, offset + 4 + axis * 4).ok_or(TextError::MalformedFvar)?,
            ));
        }
        instances.push(NamedInstance {
            name_id,
            coordinates,
        });
    }
    Ok(instances)
}
