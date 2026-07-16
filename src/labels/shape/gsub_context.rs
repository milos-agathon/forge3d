use super::{apply_lookup_at, Glyph, GlyphFilter};
use crate::labels::shape::TextError;
use ttf_parser::opentype_layout::ChainedContextLookup;
use ttf_parser::{Face, Tag};

fn backward_matches(
    filter: GlyphFilter<'_>,
    buffer: &[Glyph],
    before: usize,
    count: u16,
    mut matches: impl FnMut(u16, &Glyph) -> bool,
) -> bool {
    let mut cursor = before;
    for index in 0..count {
        let Some(at) = filter.previous(buffer, cursor) else {
            return false;
        };
        if !matches(index, &buffer[at]) {
            return false;
        }
        cursor = at;
    }
    true
}

fn forward_positions(
    filter: GlyphFilter<'_>,
    buffer: &[Glyph],
    after: usize,
    count: u16,
    mut matches: impl FnMut(u16, &Glyph) -> bool,
) -> Option<Vec<usize>> {
    let mut positions = Vec::with_capacity(usize::from(count));
    let mut cursor = after;
    for index in 0..count {
        let at = filter.next(buffer, cursor + 1)?;
        if !matches(index, &buffer[at]) {
            return None;
        }
        positions.push(at);
        cursor = at;
    }
    Some(positions)
}

pub(super) fn apply_chain_context(
    face: &Face<'_>,
    data: Option<&[u8]>,
    table: ttf_parser::opentype_layout::LayoutTable<'_>,
    filter: GlyphFilter<'_>,
    context: ChainedContextLookup<'_>,
    buffer: &mut Vec<Glyph>,
    position: usize,
    script: Tag,
    depth: u8,
) -> Result<Option<usize>, TextError> {
    let (records, input_positions) = match context {
        ChainedContextLookup::Format1 { coverage, sets } => {
            let Some(set) = coverage
                .get(buffer[position].id)
                .and_then(|index| sets.get(index))
            else {
                return Ok(None);
            };
            let mut matched = None;
            for index in 0..set.len() {
                let Some(rule) = set.get(index) else { continue };
                if !backward_matches(
                    filter,
                    buffer,
                    position,
                    rule.backtrack.len(),
                    |i, glyph| rule.backtrack.get(i) == Some(glyph.id.0),
                ) {
                    continue;
                }
                let Some(tail) =
                    forward_positions(filter, buffer, position, rule.input.len(), |i, glyph| {
                        rule.input.get(i) == Some(glyph.id.0)
                    })
                else {
                    continue;
                };
                let end = tail.last().copied().unwrap_or(position);
                if forward_positions(filter, buffer, end, rule.lookahead.len(), |i, glyph| {
                    rule.lookahead.get(i) == Some(glyph.id.0)
                })
                .is_some()
                {
                    matched = Some((rule.lookups, tail));
                    break;
                }
            }
            let Some((records, tail)) = matched else {
                return Ok(None);
            };
            let mut positions = vec![position];
            positions.extend(tail);
            (records, positions)
        }
        ChainedContextLookup::Format2 {
            coverage,
            backtrack_classes,
            input_classes,
            lookahead_classes,
            sets,
        } => {
            if !coverage.contains(buffer[position].id) {
                return Ok(None);
            }
            let Some(set) = sets.get(input_classes.get(buffer[position].id)) else {
                return Ok(None);
            };
            let mut matched = None;
            for index in 0..set.len() {
                let Some(rule) = set.get(index) else { continue };
                if !backward_matches(
                    filter,
                    buffer,
                    position,
                    rule.backtrack.len(),
                    |i, glyph| rule.backtrack.get(i) == Some(backtrack_classes.get(glyph.id)),
                ) {
                    continue;
                }
                let Some(tail) =
                    forward_positions(filter, buffer, position, rule.input.len(), |i, glyph| {
                        rule.input.get(i) == Some(input_classes.get(glyph.id))
                    })
                else {
                    continue;
                };
                let end = tail.last().copied().unwrap_or(position);
                if forward_positions(filter, buffer, end, rule.lookahead.len(), |i, glyph| {
                    rule.lookahead.get(i) == Some(lookahead_classes.get(glyph.id))
                })
                .is_some()
                {
                    matched = Some((rule.lookups, tail));
                    break;
                }
            }
            let Some((records, tail)) = matched else {
                return Ok(None);
            };
            let mut positions = vec![position];
            positions.extend(tail);
            (records, positions)
        }
        ChainedContextLookup::Format3 {
            coverage,
            backtrack_coverages,
            input_coverages,
            lookahead_coverages,
            lookups,
        } => {
            if !coverage.contains(buffer[position].id)
                || !backward_matches(
                    filter,
                    buffer,
                    position,
                    backtrack_coverages.len(),
                    |i, glyph| {
                        backtrack_coverages
                            .get(i)
                            .is_some_and(|coverage| coverage.contains(glyph.id))
                    },
                )
            {
                return Ok(None);
            }
            let Some(tail) = forward_positions(
                filter,
                buffer,
                position,
                input_coverages.len(),
                |i, glyph| {
                    input_coverages
                        .get(i)
                        .is_some_and(|coverage| coverage.contains(glyph.id))
                },
            ) else {
                return Ok(None);
            };
            let end = tail.last().copied().unwrap_or(position);
            if forward_positions(
                filter,
                buffer,
                end,
                lookahead_coverages.len(),
                |i, glyph| {
                    lookahead_coverages
                        .get(i)
                        .is_some_and(|coverage| coverage.contains(glyph.id))
                },
            )
            .is_none()
            {
                return Ok(None);
            }
            let mut positions = vec![position];
            positions.extend(tail);
            (lookups, positions)
        }
    };

    let records: Vec<_> = (0..records.len())
        .filter_map(|index| records.get(index))
        .collect();
    let mut positions: Vec<Option<usize>> = input_positions.into_iter().map(Some).collect();
    let origins: Vec<u64> = positions
        .iter()
        .map(|position| buffer[position.expect("matched input")].origin)
        .collect();
    for record in records {
        let sequence_index = usize::from(record.sequence_index);
        if sequence_index >= positions.len() {
            return Err(TextError::MalformedOpenType(
                "GSUB contextual sequence index",
            ));
        }
        let Some(target) = positions[sequence_index] else {
            continue;
        };
        let nested = table
            .lookups
            .get(record.lookup_list_index)
            .ok_or(TextError::MalformedOpenType("GSUB contextual lookup"))?;
        apply_lookup_at(
            face,
            data,
            table,
            record.lookup_list_index,
            nested,
            buffer,
            target,
            script,
            depth + 1,
        )?;
        for (slot, origin) in positions.iter_mut().zip(&origins) {
            *slot = buffer.iter().position(|glyph| glyph.origin == *origin);
        }
    }
    Ok(Some(1))
}
