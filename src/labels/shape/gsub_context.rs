use super::{apply_lookup_at, Glyph};
use crate::labels::shape::TextError;
use ttf_parser::opentype_layout::ChainedContextLookup;
use ttf_parser::Tag;

pub(super) fn apply_chain_context(
    table: ttf_parser::opentype_layout::LayoutTable<'_>,
    context: ChainedContextLookup<'_>,
    buffer: &mut Vec<Glyph>,
    position: usize,
    script: Tag,
    depth: u8,
) -> Result<Option<usize>, TextError> {
    let (records, input_len) = match context {
        ChainedContextLookup::Format1 { coverage, sets } => {
            let Some(set_index) = coverage.get(buffer[position].id) else {
                return Ok(None);
            };
            let Some(set) = sets.get(set_index) else {
                return Ok(None);
            };
            let mut matched = None;
            for index in 0..set.len() {
                let Some(rule) = set.get(index) else { continue };
                let backtrack = (0..rule.backtrack.len()).all(|index| {
                    position
                        .checked_sub(usize::from(index) + 1)
                        .and_then(|at| buffer.get(at))
                        .is_some_and(|glyph| rule.backtrack.get(index) == Some(glyph.id.0))
                });
                let input = (0..rule.input.len()).all(|index| {
                    buffer
                        .get(position + usize::from(index) + 1)
                        .is_some_and(|glyph| rule.input.get(index) == Some(glyph.id.0))
                });
                let start = position + usize::from(rule.input.len()) + 1;
                let lookahead = (0..rule.lookahead.len()).all(|index| {
                    buffer
                        .get(start + usize::from(index))
                        .is_some_and(|glyph| rule.lookahead.get(index) == Some(glyph.id.0))
                });
                if backtrack && input && lookahead {
                    matched = Some((rule.lookups, usize::from(rule.input.len()) + 1));
                    break;
                }
            }
            let Some((records, input_len)) = matched else {
                return Ok(None);
            };
            (records, input_len)
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
                let backtrack = (0..rule.backtrack.len()).all(|index| {
                    position
                        .checked_sub(usize::from(index) + 1)
                        .and_then(|at| buffer.get(at))
                        .is_some_and(|glyph| {
                            rule.backtrack.get(index) == Some(backtrack_classes.get(glyph.id))
                        })
                });
                let input = (0..rule.input.len()).all(|index| {
                    buffer
                        .get(position + usize::from(index) + 1)
                        .is_some_and(|glyph| {
                            rule.input.get(index) == Some(input_classes.get(glyph.id))
                        })
                });
                let start = position + usize::from(rule.input.len()) + 1;
                let lookahead = (0..rule.lookahead.len()).all(|index| {
                    buffer.get(start + usize::from(index)).is_some_and(|glyph| {
                        rule.lookahead.get(index) == Some(lookahead_classes.get(glyph.id))
                    })
                });
                if backtrack && input && lookahead {
                    matched = Some((rule.lookups, usize::from(rule.input.len()) + 1));
                    break;
                }
            }
            let Some((records, input_len)) = matched else {
                return Ok(None);
            };
            (records, input_len)
        }
        ChainedContextLookup::Format3 {
            coverage,
            backtrack_coverages,
            input_coverages,
            lookahead_coverages,
            lookups,
        } => {
            if !coverage.contains(buffer[position].id)
                || (0..backtrack_coverages.len()).any(|index| {
                    position
                        .checked_sub(usize::from(index) + 1)
                        .and_then(|at| buffer.get(at))
                        .is_none_or(|glyph| {
                            !backtrack_coverages
                                .get(index)
                                .is_some_and(|coverage| coverage.contains(glyph.id))
                        })
                })
                || (0..input_coverages.len()).any(|index| {
                    buffer
                        .get(position + usize::from(index) + 1)
                        .is_none_or(|glyph| {
                            !input_coverages
                                .get(index)
                                .is_some_and(|coverage| coverage.contains(glyph.id))
                        })
                })
            {
                return Ok(None);
            }
            let start = position + usize::from(input_coverages.len()) + 1;
            if (0..lookahead_coverages.len()).any(|index| {
                buffer.get(start + usize::from(index)).is_none_or(|glyph| {
                    !lookahead_coverages
                        .get(index)
                        .is_some_and(|coverage| coverage.contains(glyph.id))
                })
            }) {
                return Ok(None);
            }
            (lookups, usize::from(input_coverages.len()) + 1)
        }
    };
    let records: Vec<_> = (0..records.len())
        .filter_map(|index| records.get(index))
        .collect();
    let mut positions: Vec<Option<usize>> =
        (0..input_len).map(|index| Some(position + index)).collect();
    for record in records {
        let Some(Some(target)) = positions.get(usize::from(record.sequence_index)).copied() else {
            continue;
        };
        let nested = table
            .lookups
            .get(record.lookup_list_index)
            .ok_or(TextError::MalformedOpenType("GSUB contextual lookup"))?;
        let old_len = buffer.len();
        apply_lookup_at(table, nested, buffer, target, script, depth + 1)?;
        let delta = buffer.len() as isize - old_len as isize;
        for slot in positions
            .iter_mut()
            .skip(usize::from(record.sequence_index) + 1)
        {
            let Some(at) = *slot else { continue };
            if delta < 0 && at <= target + (-delta) as usize {
                *slot = None;
            } else {
                *slot = Some((at as isize + delta) as usize);
            }
        }
    }
    Ok(Some(1))
}
