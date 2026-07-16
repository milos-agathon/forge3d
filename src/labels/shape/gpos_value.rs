use super::{PositioningGlyph, TextError};
use ttf_parser::gpos::{Device, ValueRecord};
use ttf_parser::Face;

pub(super) fn device_delta(face: &Face<'_>, device: Option<Device<'_>>) -> Result<i32, TextError> {
    let Some(device) = device else { return Ok(0) };
    match device {
        Device::Hinting(_) => Ok(0),
        Device::Variation(variation) => face
            .tables()
            .gdef
            .and_then(|gdef| {
                gdef.glyph_variation_delta(
                    variation.outer_index,
                    variation.inner_index,
                    face.variation_coordinates(),
                )
            })
            .map(|delta| delta.round() as i32)
            .ok_or(TextError::MalformedOpenType("GPOS variation device")),
    }
}

pub(super) fn apply_value(
    face: &Face<'_>,
    glyph: &mut PositioningGlyph,
    value: ValueRecord<'_>,
) -> Result<(), TextError> {
    let x_placement = i32::from(value.x_placement)
        .checked_add(device_delta(face, value.x_placement_device)?)
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    let y_placement = i32::from(value.y_placement)
        .checked_add(device_delta(face, value.y_placement_device)?)
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    let x_advance = i32::from(value.x_advance)
        .checked_add(device_delta(face, value.x_advance_device)?)
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    let y_advance = i32::from(value.y_advance)
        .checked_add(device_delta(face, value.y_advance_device)?)
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    glyph.x_offset = glyph
        .x_offset
        .checked_add(x_placement)
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    glyph.y_offset = glyph
        .y_offset
        .checked_add(y_placement)
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    glyph.x_advance = glyph
        .x_advance
        .checked_add(x_advance)
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    glyph.y_advance = glyph
        .y_advance
        .checked_add(y_advance)
        .ok_or(TextError::MalformedOpenType("GPOS positioning overflow"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{apply_value, PositioningGlyph};
    use ttf_parser::gpos::{Device, ValueRecord, VariationDevice};
    use ttf_parser::{Face, GlyphId};

    #[test]
    fn variation_device_delta_is_applied() {
        let mut head = [0u8; 54];
        head[18..20].copy_from_slice(&1000u16.to_be_bytes());
        let mut hhea = [0u8; 36];
        hhea[34..36].copy_from_slice(&1u16.to_be_bytes());
        let maxp = [0, 0, 0x50, 0, 0, 100];
        let gdef = [
            0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 1, 0, 0, 0, 12, 0, 1, 0, 0,
            0, 16, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 7,
        ];
        let face = Face::from_raw_tables(ttf_parser::RawFaceTables {
            head: &head,
            hhea: &hhea,
            maxp: &maxp,
            gdef: Some(&gdef),
            ..ttf_parser::RawFaceTables::default()
        })
        .unwrap();
        let mut glyph = PositioningGlyph::new(GlyphId(10), 0);
        apply_value(
            &face,
            &mut glyph,
            ValueRecord {
                x_advance_device: Some(Device::Variation(VariationDevice {
                    outer_index: 0,
                    inner_index: 0,
                })),
                ..ValueRecord::default()
            },
        )
        .unwrap();
        assert_eq!(glyph.x_advance, 7);
    }
}
