use super::super::{ot::Reader, TextError};
use ttf_parser::Tag;

pub(super) fn validate_lookup_type(data: &[u8], index: u16, script: Tag) -> Result<(), TextError> {
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
        let target = lookup
            .checked_add(usize::from(reader.u16(lookup + 6 + subtable * 2)?))
            .ok_or(TextError::MalformedOpenType("GPOS subtable"))?;
        reader.slice_at(target, 2)?;
        if lookup_type == 9 {
            let extension = target;
            if reader.u16(extension)? != 1 {
                return Err(TextError::MalformedOpenType("GPOS extension"));
            }
            let nested = reader.u16(extension + 2)?;
            if !matches!(nested, 1 | 2 | 4 | 5 | 6) {
                return Err(TextError::UnsupportedLookup {
                    table: "GPOS",
                    lookup_type: nested,
                    script,
                });
            }
            let target = extension
                .checked_add(reader.u32(extension + 4)? as usize)
                .ok_or(TextError::MalformedOpenType("GPOS extension"))?;
            reader.slice_at(target, 2)?;
        }
    }
    Ok(())
}
