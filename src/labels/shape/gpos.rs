use super::gsub::GlyphFilter;
use super::TextError;
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
use value::apply_value;

pub fn apply_gpos(
    face: &Face<'_>,
    buffer: &mut [PositioningGlyph],
    selection: &[u16],
    script: Tag,
) -> Result<(), TextError> {
    let data = face.raw_face().table(Tag::from_bytes(b"GPOS"));
    apply_gpos_with_data(face, data, buffer, selection, script)
}

fn apply_gpos_with_data(
    face: &Face<'_>,
    data: Option<&[u8]>,
    buffer: &mut [PositioningGlyph],
    selection: &[u16],
    script: Tag,
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
            apply_subtable(face, subtable, buffer, &mut claimed, filter, script)?;
        }
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
                    apply_value(face, glyph, value)?;
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
                    apply_value(face, &mut buffer[index], first)?;
                    apply_value(face, &mut buffer[next], second)?;
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
                let Some(parent) = (0..index)
                    .rev()
                    .find(|&parent| !filter.ignored_id(buffer[parent].id))
                else {
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

#[cfg(test)]
#[path = "gpos_edge_tests.rs"]
mod gpos_edge_tests;
#[cfg(test)]
#[path = "gpos_tests.rs"]
mod gpos_tests;
