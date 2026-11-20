use super::gsub::GlyphFilter;
use super::TextError;
use ttf_parser::gdef::GlyphClass;
use ttf_parser::gpos::{PairAdjustment, PositioningSubtable, SingleAdjustment};
use ttf_parser::{Face, Tag};

#[path = "gpos_attach.rs"]
mod attach;
#[path = "gpos_buffer.rs"]
mod buffer;
#[path = "gpos_kern.rs"]
mod kern;
#[path = "gpos_validate.rs"]
mod validate;
#[path = "gpos_value.rs"]
mod value;
use attach::attach;
pub use buffer::PositioningGlyph;
pub use kern::apply_legacy_kern;
use validate::validate_lookup_type;
use value::apply_value_with_mode;

#[derive(Clone, Copy)]
pub(super) struct PositionMode {
    units_per_em: Option<u16>,
    backward: bool,
}

impl PositionMode {
    const DESIGN_UNITS: Self = Self {
        units_per_em: None,
        backward: false,
    };

    fn run(units_per_em: u16, backward: bool) -> Self {
        Self {
            units_per_em: Some(units_per_em),
            backward,
        }
    }

    fn scale(self, value: i32) -> Result<i32, TextError> {
        match self.units_per_em {
            Some(0) => Err(TextError::MalformedOpenType("head unitsPerEm")),
            Some(units_per_em) => {
                let overflow = || TextError::MalformedOpenType("GPOS positioning overflow");
                let numerator = i64::from(value).checked_mul(64).ok_or_else(overflow)?;
                let denominator = i64::from(units_per_em);
                let quotient = numerator / denominator;
                let remainder = numerator % denominator;
                let adjustment = if remainder.abs() * 2 >= denominator {
                    numerator.signum()
                } else {
                    0
                };
                let rounded = quotient.checked_add(adjustment).ok_or_else(overflow)?;
                i32::try_from(rounded).map_err(|_| overflow())
            }
            None => Ok(value),
        }
    }
}

pub fn apply_gpos(
    face: &Face<'_>,
    buffer: &mut [PositioningGlyph],
    selection: &[u16],
    script: Tag,
) -> Result<(), TextError> {
    let data = face.raw_face().table(Tag::from_bytes(b"GPOS"));
    apply_gpos_with_data(face, data, buffer, selection, script)
}

pub(super) fn apply_gpos_for_run(
    face: &Face<'_>,
    buffer: &mut [PositioningGlyph],
    selection: &[u16],
    script: Tag,
    units_per_em: u16,
    backward: bool,
) -> Result<(), TextError> {
    let data = face.raw_face().table(Tag::from_bytes(b"GPOS"));
    apply_gpos_with_mode(
        face,
        data,
        buffer,
        selection,
        script,
        PositionMode::run(units_per_em, backward),
    )
}

fn apply_gpos_with_data(
    face: &Face<'_>,
    data: Option<&[u8]>,
    buffer: &mut [PositioningGlyph],
    selection: &[u16],
    script: Tag,
) -> Result<(), TextError> {
    apply_gpos_with_mode(
        face,
        data,
        buffer,
        selection,
        script,
        PositionMode::DESIGN_UNITS,
    )
}

fn apply_gpos_with_mode(
    face: &Face<'_>,
    data: Option<&[u8]>,
    buffer: &mut [PositioningGlyph],
    selection: &[u16],
    script: Tag,
    mode: PositionMode,
) -> Result<(), TextError> {
    let table = face
        .tables()
        .gpos
        .ok_or(TextError::MalformedOpenType("GPOS"))?;
    for &index in selection {
        let declared_subtables = data
            .map(|data| validate_lookup_type(data, index, script))
            .transpose()?;
        let lookup = table
            .lookups
            .get(index)
            .ok_or(TextError::MalformedOpenType("GPOS LookupList"))?;
        let subtables: Vec<_> = lookup
            .subtables
            .into_iter::<PositioningSubtable<'_>>()
            .collect();
        if declared_subtables.is_some_and(|count| count != subtables.len()) {
            return Err(TextError::MalformedOpenType("GPOS subtable"));
        }
        if subtables.is_empty() {
            return Err(TextError::UnsupportedLookup {
                table: "GPOS",
                lookup_type: 0,
                script,
            });
        }
        let filter = GlyphFilter::new(face, lookup);
        let mut claimed = vec![false; buffer.len()];
        for subtable in subtables {
            apply_subtable(face, subtable, buffer, &mut claimed, filter, script, mode)?;
        }
    }
    if mode.backward {
        finish_backward_attachments(buffer)?;
    }
    Ok(())
}

fn apply_subtable(
    face: &Face<'_>,
    subtable: PositioningSubtable<'_>,
    buffer: &mut [PositioningGlyph],
    claimed: &mut [bool],
    filter: GlyphFilter<'_>,
    script: Tag,
    mode: PositionMode,
) -> Result<(), TextError> {
    match subtable {
        PositioningSubtable::Single(single) => {
            for (index, glyph) in buffer.iter_mut().enumerate() {
                if claimed[index] || filter.ignored_id(glyph.id) {
                    continue;
                }
                let value = match single {
                    SingleAdjustment::Format1 { coverage, value } => {
                        coverage.contains(glyph.id).then_some(value)
                    }
                    SingleAdjustment::Format2 { coverage, values } => {
                        coverage.get(glyph.id).and_then(|index| values.get(index))
                    }
                };
                if let Some(value) = value {
                    apply_value_with_mode(face, glyph, value, mode)?;
                    claimed[index] = true;
                }
            }
        }
        PositioningSubtable::Pair(pair) => {
            for index in 0..buffer.len().saturating_sub(1) {
                if claimed[index] || filter.ignored_id(buffer[index].id) {
                    continue;
                }
                let Some(next) =
                    (index + 1..buffer.len()).find(|&next| !filter.ignored_id(buffer[next].id))
                else {
                    continue;
                };
                let values = match pair {
                    PairAdjustment::Format1 { coverage, sets } => coverage
                        .get(buffer[index].id)
                        .and_then(|coverage| sets.get(coverage))
                        .and_then(|set| set.get(buffer[next].id)),
                    PairAdjustment::Format2 {
                        coverage,
                        classes,
                        matrix,
                    } => coverage
                        .contains(buffer[index].id)
                        .then(|| {
                            matrix.get((
                                classes.0.get(buffer[index].id),
                                classes.1.get(buffer[next].id),
                            ))
                        })
                        .flatten(),
                };
                if let Some((first, second)) = values {
                    apply_value_with_mode(face, &mut buffer[index], first, mode)?;
                    apply_value_with_mode(face, &mut buffer[next], second, mode)?;
                    claimed[index] = true;
                }
            }
        }
        PositioningSubtable::MarkToBase(mark) => {
            for index in 1..buffer.len() {
                if claimed[index] || filter.ignored_id(buffer[index].id) {
                    continue;
                }
                let Some(mark_index) = mark.mark_coverage.get(buffer[index].id) else {
                    continue;
                };
                let Some(parent) = (0..index).rev().find(|&parent| {
                    !matches!(
                        face.tables()
                            .gdef
                            .and_then(|gdef| gdef.glyph_class(buffer[parent].id)),
                        Some(GlyphClass::Mark)
                    )
                }) else {
                    continue;
                };
                if !mark.base_coverage.contains(buffer[parent].id) {
                    continue;
                }
                let Some(base_index) = mark.base_coverage.get(buffer[parent].id) else {
                    continue;
                };
                let Some((class, mark_anchor)) = mark.marks.get(mark_index) else {
                    continue;
                };
                let Some(parent_anchor) = mark.anchors.get(base_index, class) else {
                    continue;
                };
                attach(
                    face,
                    buffer,
                    index,
                    parent,
                    mark_anchor,
                    parent_anchor,
                    None,
                    mode,
                )?;
                claimed[index] = true;
            }
        }
        PositioningSubtable::MarkToLigature(mark) => {
            for index in 1..buffer.len() {
                if claimed[index] || filter.ignored_id(buffer[index].id) {
                    continue;
                }
                let Some(mark_index) = mark.mark_coverage.get(buffer[index].id) else {
                    continue;
                };
                let Some(parent) = (0..index)
                    .rev()
                    .find(|&parent| !filter.ignored_id(buffer[parent].id))
                else {
                    continue;
                };
                if !mark.ligature_coverage.contains(buffer[parent].id) {
                    continue;
                }
                let Some(ligature_index) = mark.ligature_coverage.get(buffer[parent].id) else {
                    continue;
                };
                let Some(matrix) = mark.ligature_array.get(ligature_index) else {
                    continue;
                };
                let component = buffer[index]
                    .ligature_component
                    .or_else(|| {
                        buffer[parent]
                            .component_clusters
                            .iter()
                            .position(|&cluster| cluster == buffer[index].cluster)
                            .and_then(|component| u16::try_from(component).ok())
                    })
                    .unwrap_or(0)
                    .min(matrix.rows.saturating_sub(1));
                let Some((class, mark_anchor)) = mark.marks.get(mark_index) else {
                    continue;
                };
                let Some(parent_anchor) = matrix.get(component, class) else {
                    continue;
                };
                attach(
                    face,
                    buffer,
                    index,
                    parent,
                    mark_anchor,
                    parent_anchor,
                    Some(component),
                    mode,
                )?;
                claimed[index] = true;
            }
        }
        PositioningSubtable::MarkToMark(mark) => {
            for index in 1..buffer.len() {
                if claimed[index] || filter.ignored_id(buffer[index].id) {
                    continue;
                }
                let Some(mark_index) = mark.mark1_coverage.get(buffer[index].id) else {
                    continue;
                };
                let Some(parent) = (0..index)
                    .rev()
                    .find(|&parent| !filter.ignored_id(buffer[parent].id))
                else {
                    continue;
                };
                if !mark.mark2_coverage.contains(buffer[parent].id) {
                    continue;
                }
                let Some(parent_index) = mark.mark2_coverage.get(buffer[parent].id) else {
                    continue;
                };
                let Some((class, mark_anchor)) = mark.marks.get(mark_index) else {
                    continue;
                };
                let Some(parent_anchor) = mark.mark2_matrix.get(parent_index, class) else {
                    continue;
                };
                attach(
                    face,
                    buffer,
                    index,
                    parent,
                    mark_anchor,
                    parent_anchor,
                    None,
                    mode,
                )?;
                claimed[index] = true;
            }
        }
        _ => {
            return Err(TextError::UnsupportedLookup {
                table: "GPOS",
                lookup_type: match subtable {
                    PositioningSubtable::Cursive(_) => 3,
                    PositioningSubtable::Context(_) => 7,
                    PositioningSubtable::ChainContext(_) => 8,
                    _ => 0,
                },
                script,
            });
        }
    }
    Ok(())
}

fn finish_backward_attachments(buffer: &mut [PositioningGlyph]) -> Result<(), TextError> {
    fn apply_parent_offset(buffer: &mut [PositioningGlyph], index: usize) -> Result<(), TextError> {
        let Some(parent) = buffer[index].attached_to else {
            return Ok(());
        };
        let overflow = || TextError::MalformedOpenType("GPOS positioning overflow");
        let (x_advance, y_advance, subtract) = if parent < index {
            let advances =
                buffer[parent + 1..=index]
                    .iter()
                    .try_fold((0i32, 0i32), |(x, y), glyph| {
                        Ok::<_, TextError>((
                            x.checked_add(glyph.x_advance).ok_or_else(overflow)?,
                            y.checked_add(glyph.y_advance).ok_or_else(overflow)?,
                        ))
                    })?;
            (advances.0, advances.1, false)
        } else {
            let advances =
                buffer[index + 1..=parent]
                    .iter()
                    .try_fold((0i32, 0i32), |(x, y), glyph| {
                        Ok::<_, TextError>((
                            x.checked_add(glyph.x_advance).ok_or_else(overflow)?,
                            y.checked_add(glyph.y_advance).ok_or_else(overflow)?,
                        ))
                    })?;
            (advances.0, advances.1, true)
        };
        let parent_x = buffer[parent].x_offset;
        let parent_y = buffer[parent].y_offset;
        buffer[index].x_offset = buffer[index]
            .x_offset
            .checked_add(parent_x)
            .and_then(|value| {
                if subtract {
                    value.checked_sub(x_advance)
                } else {
                    value.checked_add(x_advance)
                }
            })
            .ok_or_else(overflow)?;
        buffer[index].y_offset = buffer[index]
            .y_offset
            .checked_add(parent_y)
            .and_then(|value| {
                if subtract {
                    value.checked_sub(y_advance)
                } else {
                    value.checked_add(y_advance)
                }
            })
            .ok_or_else(overflow)?;
        Ok(())
    }

    const VISITING: u8 = 1;
    const RESOLVED: u8 = 2;
    let mut state = vec![0u8; buffer.len()];
    for start in 0..buffer.len() {
        if state[start] == RESOLVED {
            continue;
        }
        let mut path = Vec::new();
        let mut index = start;
        loop {
            match state[index] {
                RESOLVED => break,
                VISITING => {
                    return Err(TextError::MalformedOpenType("GPOS attachment cycle"));
                }
                _ => {}
            }
            state[index] = VISITING;
            path.push(index);
            let Some(parent) = buffer[index].attached_to else {
                break;
            };
            if parent >= buffer.len() {
                return Err(TextError::MalformedOpenType("GPOS attachment parent"));
            }
            index = parent;
        }
        for index in path.into_iter().rev() {
            apply_parent_offset(buffer, index)?;
            state[index] = RESOLVED;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "gpos_edge_tests.rs"]
mod gpos_edge_tests;
#[cfg(test)]
#[path = "gpos_tests.rs"]
mod gpos_tests;

#[cfg(test)]
mod backward_attachment_tests {
    use super::{finish_backward_attachments, PositioningGlyph, TextError};
    use ttf_parser::GlyphId;

    fn glyph(advance: i32, offset: i32, parent: Option<usize>) -> PositioningGlyph {
        let mut glyph = PositioningGlyph::new(GlyphId(1), 0);
        glyph.x_advance = advance;
        glyph.x_offset = offset;
        glyph.attached_to = parent;
        glyph
    }

    #[test]
    fn backward_marks_resolve_parent_offsets_without_base_advance() {
        let mut glyphs = vec![
            glyph(35, 2, None),
            glyph(0, 5, Some(0)),
            glyph(0, -3, Some(1)),
        ];
        finish_backward_attachments(&mut glyphs).unwrap();
        assert_eq!(
            glyphs.iter().map(|g| g.x_offset).collect::<Vec<_>>(),
            vec![2, 7, 4]
        );
    }

    #[test]
    fn backward_attachment_rejects_invalid_parent_cycle_and_overflow() {
        let mut invalid = vec![glyph(0, 0, Some(1))];
        assert!(matches!(
            finish_backward_attachments(&mut invalid),
            Err(TextError::MalformedOpenType("GPOS attachment parent"))
        ));
        let mut cyclic = vec![glyph(0, 0, Some(1)), glyph(0, 0, Some(0))];
        assert!(matches!(
            finish_backward_attachments(&mut cyclic),
            Err(TextError::MalformedOpenType("GPOS attachment cycle"))
        ));
        let mut overflow = vec![glyph(0, i32::MAX, None), glyph(0, 1, Some(0))];
        assert!(matches!(
            finish_backward_attachments(&mut overflow),
            Err(TextError::MalformedOpenType("GPOS positioning overflow"))
        ));
    }

    #[test]
    fn backward_attachment_chain_does_not_use_the_call_stack() {
        // Exercise the full OpenType glyph-index space as a text-run fixture;
        // this is not an input cap.
        let length = usize::from(u16::MAX) + 1;
        let mut glyphs = (0..length)
            .map(|index| glyph(0, 1, (index + 1 < length).then_some(index + 1)))
            .collect::<Vec<_>>();
        glyphs[length - 1].x_offset = 4;
        finish_backward_attachments(&mut glyphs).unwrap();
        assert_eq!(glyphs[0].x_offset, i32::from(u16::MAX) + 4);
        assert_eq!(glyphs[length - 1].x_offset, 4);
    }

    #[test]
    fn scaled_positioning_rejects_both_signed_overflow_directions() {
        let mode = super::PositionMode::run(1, true);
        for value in [i32::MIN, i32::MAX] {
            assert!(matches!(
                mode.scale(value),
                Err(TextError::MalformedOpenType("GPOS positioning overflow"))
            ));
        }
        assert_eq!(super::PositionMode::run(1000, true).scale(-40).unwrap(), -3);
        assert!(matches!(
            super::PositionMode::run(0, true).scale(1),
            Err(TextError::MalformedOpenType("head unitsPerEm"))
        ));
    }
}
