use super::{PositioningGlyph, TextError};
use ttf_parser::gpos::ValueRecord;

pub(super) fn apply_value(
    glyph: &mut PositioningGlyph,
    value: ValueRecord<'_>,
) -> Result<(), TextError> {
    glyph.x_offset = glyph
        .x_offset
        .checked_add(i32::from(value.x_placement))
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    glyph.y_offset = glyph
        .y_offset
        .checked_add(i32::from(value.y_placement))
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    glyph.x_advance = glyph
        .x_advance
        .checked_add(i32::from(value.x_advance))
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    glyph.y_advance = glyph
        .y_advance
        .checked_add(i32::from(value.y_advance))
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    Ok(())
}
