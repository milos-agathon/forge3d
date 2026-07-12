use super::PositioningGlyph;
use ttf_parser::Face;

pub fn apply_legacy_kern(
    face: &Face<'_>,
    buffer: &mut [PositioningGlyph],
) -> Result<(), super::TextError> {
    if face.tables().gpos.is_some() {
        return Ok(());
    }
    let Some(kern) = face.tables().kern else {
        return Ok(());
    };
    for subtable in kern.subtables {
        if !subtable.horizontal || subtable.variable || subtable.has_cross_stream {
            continue;
        }
        for index in 0..buffer.len().saturating_sub(1) {
            if let Some(value) = subtable.glyphs_kerning(buffer[index].id, buffer[index + 1].id) {
                buffer[index].x_advance = buffer[index]
                    .x_advance
                    .checked_add(i32::from(value))
                    .ok_or(super::TextError::MalformedOpenType(
                        "kern positioning overflow",
                    ))?;
            }
        }
    }
    Ok(())
}
