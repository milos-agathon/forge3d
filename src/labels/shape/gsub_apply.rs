use super::{context, Glyph, GlyphFilter};
use crate::labels::shape::TextError;
use ttf_parser::gsub::SubstitutionSubtable;
use ttf_parser::{Face, GlyphId, Tag};

pub(super) fn apply_subtable(
    face: &Face<'_>,
    data: Option<&[u8]>,
    table: ttf_parser::opentype_layout::LayoutTable<'_>,
    filter: GlyphFilter<'_>,
    subtable: SubstitutionSubtable<'_>,
    buffer: &mut Vec<Glyph>,
    position: usize,
    script: Tag,
    depth: u8,
) -> Result<Option<usize>, TextError> {
    let Some(glyph) = buffer.get(position).cloned() else {
        return Ok(None);
    };
    match subtable {
        SubstitutionSubtable::Single(single) => {
            let Some(index) = single.coverage().get(glyph.id) else {
                return Ok(None);
            };
            buffer[position].id = match single {
                ttf_parser::gsub::SingleSubstitution::Format1 { delta, .. } => {
                    GlyphId(glyph.id.0.wrapping_add_signed(delta))
                }
                ttf_parser::gsub::SingleSubstitution::Format2 { substitutes, .. } => substitutes
                    .get(index)
                    .ok_or(TextError::MalformedOpenType("GSUB single"))?,
            };
            Ok(Some(1))
        }
        SubstitutionSubtable::Multiple(multiple) => {
            let Some(index) = multiple.coverage.get(glyph.id) else {
                return Ok(None);
            };
            let sequence = multiple
                .sequences
                .get(index)
                .ok_or(TextError::MalformedOpenType("GSUB multiple"))?;
            let replacements: Vec<_> = (0..sequence.substitutes.len())
                .map(|index| {
                    sequence
                        .substitutes
                        .get(index)
                        .map(|id| {
                            let mut replacement = Glyph::new(id, glyph.cluster);
                            replacement.features.clone_from(&glyph.features);
                            replacement
                        })
                        .ok_or(TextError::MalformedOpenType("GSUB multiple"))
                })
                .collect::<Result<_, _>>()?;
            let count = replacements.len();
            buffer.splice(position..=position, replacements);
            Ok(Some(count))
        }
        SubstitutionSubtable::Alternate(alternate) => {
            let Some(index) = alternate.coverage.get(glyph.id) else {
                return Ok(None);
            };
            buffer[position].id = alternate
                .alternate_sets
                .get(index)
                .and_then(|set| set.alternates.get(0))
                .ok_or(TextError::MalformedOpenType("GSUB alternate"))?;
            Ok(Some(1))
        }
        SubstitutionSubtable::Ligature(ligature) => {
            let Some(index) = ligature.coverage.get(glyph.id) else {
                return Ok(None);
            };
            let set = ligature
                .ligature_sets
                .get(index)
                .ok_or(TextError::MalformedOpenType("GSUB ligature"))?;
            for candidate_index in 0..set.len() {
                let Some(candidate) = set.get(candidate_index) else {
                    continue;
                };
                let mut positions = vec![position];
                let mut cursor = position;
                for component in 0..candidate.components.len() {
                    let Some(next) = filter.next(buffer, cursor + 1) else {
                        positions.clear();
                        break;
                    };
                    if candidate.components.get(component) != Some(buffer[next].id) {
                        positions.clear();
                        break;
                    }
                    positions.push(next);
                    cursor = next;
                }
                if !positions.is_empty() {
                    let cluster = positions
                        .iter()
                        .map(|&index| buffer[index].cluster)
                        .min()
                        .unwrap_or(glyph.cluster);
                    let mut replacement = Glyph::new(candidate.glyph, cluster);
                    replacement.features.clone_from(&glyph.features);
                    buffer[position] = replacement;
                    for &index in positions[1..].iter().rev() {
                        buffer.remove(index);
                    }
                    return Ok(Some(1));
                }
            }
            Ok(None)
        }
        SubstitutionSubtable::ChainContext(context) => context::apply_chain_context(
            face, data, table, filter, context, buffer, position, script, depth,
        ),
        SubstitutionSubtable::Context(_) | SubstitutionSubtable::ReverseChainSingle(_) => {
            Err(TextError::UnsupportedLookup {
                table: "GSUB",
                lookup_type: if matches!(subtable, SubstitutionSubtable::Context(_)) {
                    5
                } else {
                    8
                },
                script,
            })
        }
    }
}
