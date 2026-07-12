use super::ot::Reader;
use super::TextError;
use ttf_parser::{gsub::SubstitutionSubtable, Face, GlyphId, Tag};

#[path = "gsub_context.rs"]
mod context;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glyph {
    pub id: GlyphId,
    pub cluster: u32,
}

impl Glyph {
    pub fn new(id: GlyphId, cluster: u32) -> Self {
        Self { id, cluster }
    }
}

pub fn apply_gsub(
    face: &Face<'_>,
    buffer: &mut Vec<Glyph>,
    selection: &[u16],
    script: Tag,
) -> Result<(), TextError> {
    if let Some(data) = face.raw_face().table(Tag::from_bytes(b"GSUB")) {
        validate_lookup_types(data, selection, script)?;
    }
    let table = face
        .tables()
        .gsub
        .ok_or(TextError::MalformedOpenType("GSUB"))?;
    for &index in selection {
        let lookup = table
            .lookups
            .get(index)
            .ok_or(TextError::MalformedOpenType("GSUB LookupList"))?;
        let mut position = 0;
        while position < buffer.len() {
            let changed = apply_lookup_at(table, lookup, buffer, position, script, 0)?;
            position += changed.max(1);
        }
    }
    Ok(())
}

fn validate_lookup_types(data: &[u8], selection: &[u16], script: Tag) -> Result<(), TextError> {
    let reader = Reader::new(data);
    let list = usize::from(reader.u16(8)?);
    let count = usize::from(reader.u16(list)?);
    reader.slice_at(list + 2, count * 2)?;
    for &index in selection {
        if usize::from(index) >= count {
            return Err(TextError::MalformedOpenType("GSUB LookupList"));
        }
        let lookup = list + usize::from(reader.u16(list + 2 + usize::from(index) * 2)?);
        let lookup_type = reader.u16(lookup)?;
        if !matches!(lookup_type, 1 | 2 | 3 | 4 | 6 | 7) {
            return Err(TextError::UnsupportedLookup {
                table: "GSUB",
                lookup_type,
                script,
            });
        }
    }
    Ok(())
}

pub(super) fn apply_lookup_at(
    table: ttf_parser::opentype_layout::LayoutTable<'_>,
    lookup: ttf_parser::opentype_layout::Lookup<'_>,
    buffer: &mut Vec<Glyph>,
    position: usize,
    script: Tag,
    depth: u8,
) -> Result<usize, TextError> {
    if depth == 16 {
        return Err(TextError::MalformedOpenType("GSUB recursion"));
    }
    for subtable in lookup.subtables.into_iter::<SubstitutionSubtable<'_>>() {
        if let Some(count) = apply_subtable(table, subtable, buffer, position, script, depth)? {
            return Ok(count);
        }
    }
    Ok(0)
}

fn apply_subtable(
    table: ttf_parser::opentype_layout::LayoutTable<'_>,
    subtable: SubstitutionSubtable<'_>,
    buffer: &mut Vec<Glyph>,
    position: usize,
    script: Tag,
    depth: u8,
) -> Result<Option<usize>, TextError> {
    let Some(glyph) = buffer.get(position).copied() else {
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
                        .map(|id| Glyph::new(id, glyph.cluster))
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
                let count = usize::from(candidate.components.len()) + 1;
                if position + count > buffer.len() {
                    continue;
                }
                let matches = (0..candidate.components.len()).all(|component| {
                    candidate.components.get(component)
                        == buffer
                            .get(position + usize::from(component) + 1)
                            .map(|glyph| glyph.id)
                });
                if matches {
                    let cluster = buffer[position..position + count]
                        .iter()
                        .map(|glyph| glyph.cluster)
                        .min()
                        .unwrap_or(glyph.cluster);
                    buffer.splice(
                        position..position + count,
                        [Glyph::new(candidate.glyph, cluster)],
                    );
                    return Ok(Some(1));
                }
            }
            Ok(None)
        }
        SubstitutionSubtable::ChainContext(context) => {
            context::apply_chain_context(table, context, buffer, position, script, depth)
        }
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

#[cfg(test)]
fn apply_gsub_data(
    data: &[u8],
    buffer: &mut Vec<Glyph>,
    selection: &[u16],
    script: Tag,
) -> Result<(), TextError> {
    validate_lookup_types(data, selection, script)?;
    let mut head = [0u8; 54];
    head[18..20].copy_from_slice(&1000u16.to_be_bytes());
    let mut hhea = [0u8; 36];
    hhea[34..36].copy_from_slice(&1u16.to_be_bytes());
    let maxp = [0, 0, 0x50, 0, 0, 100];
    let face = Face::from_raw_tables(ttf_parser::RawFaceTables {
        head: &head,
        hhea: &hhea,
        maxp: &maxp,
        gsub: Some(data),
        ..ttf_parser::RawFaceTables::default()
    })
    .map_err(|_| TextError::MalformedOpenType("test face"))?;
    apply_gsub(&face, buffer, selection, script)
}

#[cfg(test)]
#[path = "gsub_tests.rs"]
mod tests;
