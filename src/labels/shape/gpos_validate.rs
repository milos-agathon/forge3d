use super::super::{ot::Reader, TextError};
use ttf_parser::Tag;

pub(super) fn validate_lookup_type(
    data: &[u8],
    index: u16,
    script: Tag,
) -> Result<usize, TextError> {
    let reader = Reader::new(data);
    let list = usize::from(reader.u16(8)?);
    let count = usize::from(reader.u16(list)?);
    reader.slice_at(list + 2, count * 2)?;
    if usize::from(index) >= count {
        return Err(TextError::MalformedOpenType("GPOS LookupList"));
    }
    let lookup = list
        .checked_add(usize::from(reader.u16(list + 2 + usize::from(index) * 2)?))
        .ok_or(TextError::MalformedOpenType("GPOS LookupList"))?;
    let lookup_type = reader.u16(lookup)?;
    if !matches!(lookup_type, 1 | 2 | 4 | 5 | 6 | 9) {
        return Err(TextError::UnsupportedLookup {
            table: "GPOS",
            lookup_type,
            script,
        });
    }
    let subtable_count = usize::from(reader.u16(lookup + 4)?);
    reader.slice_at(lookup + 6, subtable_count * 2)?;
    if reader.u16(lookup + 2)? & 0x0010 != 0 {
        reader.u16(lookup + 6 + subtable_count * 2)?;
    }
    for subtable in 0..subtable_count {
        let mut target = lookup
            .checked_add(usize::from(reader.u16(lookup + 6 + subtable * 2)?))
            .ok_or(TextError::MalformedOpenType("GPOS subtable"))?;
        reader.slice_at(target, 2)?;
        let mut effective_type = lookup_type;
        if lookup_type == 9 {
            let extension = target;
            if reader.u16(extension)? != 1 {
                return Err(TextError::MalformedOpenType("GPOS extension"));
            }
            let nested = reader.u16(extension + 2)?;
            effective_type = nested;
            if !matches!(nested, 1 | 2 | 4 | 5 | 6) {
                return Err(TextError::UnsupportedLookup {
                    table: "GPOS",
                    lookup_type: nested,
                    script,
                });
            }
            target = extension
                .checked_add(reader.u32(extension + 4)? as usize)
                .ok_or(TextError::MalformedOpenType("GPOS extension"))?;
            reader.slice_at(target, 2)?;
        }
        validate_subtable(reader, effective_type, target)?;
    }
    Ok(subtable_count)
}

fn validate_subtable(reader: Reader<'_>, kind: u16, base: usize) -> Result<(), TextError> {
    match kind {
        1 => validate_single(reader, base),
        2 => validate_pair(reader, base),
        4 | 6 => validate_mark_matrix(reader, base),
        5 => validate_mark_ligature(reader, base),
        _ => Ok(()),
    }
}

fn validate_pair(reader: Reader<'_>, base: usize) -> Result<(), TextError> {
    let first_format = reader.u16(base + 4)?;
    let second_format = reader.u16(base + 6)?;
    let record_size = 2 + value_size(first_format) + value_size(second_format);
    match reader.u16(base)? {
        1 => {
            let count = usize::from(reader.u16(base + 8)?);
            reader.slice_at(base + 10, count * 2)?;
            for index in 0..count {
                let set = add_offset(reader, base, base + 10 + index * 2)?;
                let records = usize::from(reader.u16(set)?);
                reader.slice_at(
                    set + 2,
                    records.checked_mul(record_size).ok_or(malformed())?,
                )?;
                for record in 0..records {
                    let at = set + 2 + record * record_size + 2;
                    validate_value(reader, base, at, first_format)?;
                    validate_value(reader, base, at + value_size(first_format), second_format)?;
                }
            }
        }
        2 => {
            reader.slice_at(base, 16)?;
            let first_count = usize::from(reader.u16(base + 12)?);
            let second_count = usize::from(reader.u16(base + 14)?);
            let records = first_count.checked_mul(second_count).ok_or(malformed())?;
            let size = value_size(first_format) + value_size(second_format);
            reader.slice_at(base + 16, records.checked_mul(size).ok_or(malformed())?)?;
            for record in 0..records {
                let at = base + 16 + record * size;
                validate_value(reader, base, at, first_format)?;
                validate_value(reader, base, at + value_size(first_format), second_format)?;
            }
        }
        _ => return Err(malformed()),
    }
    Ok(())
}

fn validate_single(reader: Reader<'_>, base: usize) -> Result<(), TextError> {
    let format = reader.u16(base)?;
    let value_format = reader.u16(base + 4)?;
    match format {
        1 => {
            validate_value(reader, base, base + 6, value_format)?;
        }
        2 => {
            let count = usize::from(reader.u16(base + 6)?);
            let size = value_size(value_format);
            reader.slice_at(base + 8, count.checked_mul(size).ok_or(malformed())?)?;
            for index in 0..count {
                validate_value(reader, base, base + 8 + index * size, value_format)?;
            }
        }
        _ => return Err(malformed()),
    }
    Ok(())
}

fn value_size(format: u16) -> usize {
    usize::try_from(format.count_ones()).unwrap_or(0) * 2
}

fn validate_value(
    reader: Reader<'_>,
    base: usize,
    record: usize,
    format: u16,
) -> Result<(), TextError> {
    reader.slice_at(record, value_size(format))?;
    let mut cursor = record + usize::try_from((format & 0x000f).count_ones()).unwrap_or(0) * 2;
    for bit in [0x0010, 0x0020, 0x0040, 0x0080] {
        if format & bit != 0 {
            validate_device(reader, base, reader.u16(cursor)?)?;
            cursor += 2;
        }
    }
    Ok(())
}

fn validate_mark_matrix(reader: Reader<'_>, base: usize) -> Result<(), TextError> {
    if reader.u16(base)? != 1 {
        return Err(malformed());
    }
    let classes = usize::from(reader.u16(base + 6)?);
    validate_mark_array(reader, add_offset(reader, base, base + 8)?)?;
    validate_anchor_matrix(reader, add_offset(reader, base, base + 10)?, classes)
}

fn validate_mark_ligature(reader: Reader<'_>, base: usize) -> Result<(), TextError> {
    if reader.u16(base)? != 1 {
        return Err(malformed());
    }
    let classes = usize::from(reader.u16(base + 6)?);
    validate_mark_array(reader, add_offset(reader, base, base + 8)?)?;
    let array = add_offset(reader, base, base + 10)?;
    let count = usize::from(reader.u16(array)?);
    reader.slice_at(array + 2, count * 2)?;
    for index in 0..count {
        let attach = add_offset(reader, array, array + 2 + index * 2)?;
        let components = usize::from(reader.u16(attach)?);
        validate_anchor_offsets(reader, attach, attach + 2, components * classes)?;
    }
    Ok(())
}

fn validate_mark_array(reader: Reader<'_>, array: usize) -> Result<(), TextError> {
    let count = usize::from(reader.u16(array)?);
    reader.slice_at(array + 2, count * 4)?;
    for index in 0..count {
        let record = array + 2 + index * 4;
        validate_anchor(reader, array, reader.u16(record + 2)?)?;
    }
    Ok(())
}

fn validate_anchor_matrix(
    reader: Reader<'_>,
    matrix: usize,
    classes: usize,
) -> Result<(), TextError> {
    let rows = usize::from(reader.u16(matrix)?);
    validate_anchor_offsets(reader, matrix, matrix + 2, rows * classes)
}

fn validate_anchor_offsets(
    reader: Reader<'_>,
    base: usize,
    offsets: usize,
    count: usize,
) -> Result<(), TextError> {
    reader.slice_at(offsets, count * 2)?;
    for index in 0..count {
        validate_anchor(reader, base, reader.u16(offsets + index * 2)?)?;
    }
    Ok(())
}

fn validate_anchor(reader: Reader<'_>, base: usize, offset: u16) -> Result<(), TextError> {
    if offset == 0 {
        return Ok(());
    }
    let anchor = base.checked_add(usize::from(offset)).ok_or(malformed())?;
    match reader.u16(anchor)? {
        1 => reader.slice_at(anchor, 6).map(|_| ()),
        2 => reader.slice_at(anchor, 8).map(|_| ()),
        3 => {
            reader.slice_at(anchor, 10)?;
            validate_device(reader, anchor, reader.u16(anchor + 6)?)?;
            validate_device(reader, anchor, reader.u16(anchor + 8)?)
        }
        _ => Err(malformed()),
    }
}

fn validate_device(reader: Reader<'_>, base: usize, offset: u16) -> Result<(), TextError> {
    if offset == 0 {
        return Ok(());
    }
    let device = base.checked_add(usize::from(offset)).ok_or(malformed())?;
    reader.slice_at(device, 6)?;
    match reader.u16(device + 4)? {
        0x8000 => Ok(()),
        format @ 1..=3 => {
            let start = reader.u16(device)?;
            let end = reader.u16(device + 2)?;
            let span = end.checked_sub(start).ok_or(malformed())?;
            let delta_count = 1 + usize::from(span);
            let per_word = 1usize << (4 - usize::from(format));
            let words = delta_count.div_ceil(per_word);
            reader.slice_at(device + 6, words * 2).map(|_| ())
        }
        _ => Err(malformed()),
    }
}

fn add_offset(reader: Reader<'_>, base: usize, at: usize) -> Result<usize, TextError> {
    base.checked_add(usize::from(reader.u16(at)?))
        .ok_or(malformed())
}

fn malformed() -> TextError {
    TextError::MalformedOpenType("GPOS subtable")
}
