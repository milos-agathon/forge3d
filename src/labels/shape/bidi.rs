use crate::labels::unicode::{bidi_class, mirrored, BidiClass};
use std::fmt;
use std::ops::Range;

#[path = "bidi_brackets.rs"]
mod brackets;
#[path = "bidi_explicit.rs"]
mod explicit;
#[path = "bidi_resolve.rs"]
mod resolve;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BidiParagraph {
    pub text: String,
    pub paragraph_level: u8,
    pub levels: Vec<u8>,
    pub classes: Vec<BidiClass>,
    pub mirrored: Vec<Option<char>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BidiError {
    InvalidParagraphLevel(u8),
    InvalidLineRange(Range<usize>),
    MalformedParagraph,
}

impl fmt::Display for BidiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParagraphLevel(level) => {
                write!(formatter, "invalid bidi paragraph level {level}")
            }
            Self::InvalidLineRange(range) => write!(
                formatter,
                "invalid bidi line range {}..{}",
                range.start, range.end
            ),
            Self::MalformedParagraph => write!(formatter, "inconsistent bidi paragraph fields"),
        }
    }
}

impl std::error::Error for BidiError {}

pub fn resolve_bidi(text: &str, paragraph_level: Option<u8>) -> Result<BidiParagraph, BidiError> {
    if paragraph_level.is_some_and(|level| level > 1) {
        return Err(BidiError::InvalidParagraphLevel(paragraph_level.unwrap()));
    }
    let characters: Vec<_> = text.chars().collect();
    let classes: Vec<_> = characters.iter().copied().map(bidi_class).collect();
    let paragraph_level = paragraph_level.unwrap_or_else(|| explicit::paragraph_level(&classes));
    let mut units = explicit::resolve_explicit(&classes, paragraph_level);
    resolve::resolve_sequences(&characters, &mut units, paragraph_level);
    let levels: Vec<_> = units.iter().map(|unit| unit.level).collect();
    let mirrored = characters
        .iter()
        .copied()
        .zip(&levels)
        .map(|(character, level)| (level & 1 == 1).then(|| mirrored(character)).flatten())
        .collect();
    Ok(BidiParagraph {
        text: text.to_owned(),
        paragraph_level,
        levels,
        classes,
        mirrored,
    })
}

pub fn visual_order(
    paragraph: &BidiParagraph,
    line_ranges: &[Range<usize>],
) -> Result<Vec<usize>, BidiError> {
    let length = paragraph.levels.len();
    if paragraph.classes.len() != length
        || paragraph.mirrored.len() != length
        || paragraph.text.chars().count() != length
        || paragraph.paragraph_level > 1
    {
        return Err(BidiError::MalformedParagraph);
    }
    let mut order = Vec::with_capacity(paragraph.levels.len());
    for line in line_ranges {
        if line.start > line.end || line.end > paragraph.levels.len() {
            return Err(BidiError::InvalidLineRange(line.clone()));
        }
        let levels = line_levels(paragraph, line.clone());
        let mut line_order: Vec<_> = (line.start..line.end)
            .filter(|&index| !explicit::removed_by_x9(paragraph.classes[index]))
            .collect();
        reorder_l2(&levels, line.start, &mut line_order);
        // L3 is applied when positioned glyph attachments are traversed; it does
        // not alter the conformance character-index order produced by L2.
        order.extend(line_order);
    }
    Ok(order)
}

pub(super) fn line_levels(paragraph: &BidiParagraph, line: Range<usize>) -> Vec<u8> {
    use BidiClass::{Bn, Fsi, Lre, Lri, Lro, Pdf, Pdi, Rle, Rli, Rlo, Ws, B, S};
    let mut levels = paragraph.levels[line.clone()].to_vec();
    let classes = &paragraph.classes[line];
    let mut reset_from = Some(0usize);
    let mut reset_to = None;
    let mut previous = paragraph.paragraph_level;
    for (index, class) in classes.iter().copied().enumerate() {
        match class {
            B | S => {
                reset_to = Some(index + 1);
                reset_from.get_or_insert(index);
            }
            Ws | Fsi | Lri | Rli | Pdi => {
                reset_from.get_or_insert(index);
            }
            Rle | Lre | Rlo | Lro | Pdf | Bn => {
                reset_from.get_or_insert(index);
                levels[index] = previous;
            }
            _ => reset_from = None,
        }
        if let (Some(from), Some(to)) = (reset_from, reset_to) {
            levels[from..to].fill(paragraph.paragraph_level);
            reset_from = None;
            reset_to = None;
        }
        previous = levels[index];
    }
    if let Some(from) = reset_from {
        levels[from..].fill(paragraph.paragraph_level);
    }
    levels
}

fn reorder_l2(levels: &[u8], line_start: usize, order: &mut [usize]) {
    let Some(maximum) = levels.iter().copied().max() else {
        return;
    };
    let Some(lowest_odd) = levels.iter().copied().filter(|level| level & 1 == 1).min() else {
        return;
    };
    for threshold in (lowest_odd..=maximum).rev() {
        let mut start = 0;
        while start < order.len() {
            if levels[order[start] - line_start] < threshold {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < order.len() && levels[order[end] - line_start] >= threshold {
                end += 1;
            }
            order[start..end].reverse();
            start = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_bidi, visual_order, BidiError};

    #[test]
    fn visual_order_is_deferred_until_lines_are_known() {
        let paragraph = resolve_bidi("abc אבגדה", Some(0)).unwrap();
        let whole = visual_order(&paragraph, std::slice::from_ref(&(0..9))).unwrap();
        let wrapped = visual_order(&paragraph, &[0..6, 6..9]).unwrap();
        assert_ne!(whole, wrapped);
    }

    #[test]
    fn odd_levels_request_mirroring() {
        let paragraph = resolve_bidi("(א)", Some(1)).unwrap();
        assert_eq!(paragraph.mirrored, vec![Some(')'), None, Some('(')]);
    }

    #[test]
    fn invalid_inputs_are_structured_errors() {
        assert_eq!(
            resolve_bidi("a", Some(2)),
            Err(BidiError::InvalidParagraphLevel(2))
        );
        let paragraph = resolve_bidi("a", Some(0)).unwrap();
        assert_eq!(
            visual_order(&paragraph, std::slice::from_ref(&(0..2))),
            Err(BidiError::InvalidLineRange(0..2))
        );
        let mut malformed = paragraph;
        malformed.classes.clear();
        assert_eq!(
            visual_order(&malformed, std::slice::from_ref(&(0..1))),
            Err(BidiError::MalformedParagraph)
        );
    }
}

#[cfg(test)]
#[path = "bidi_conformance_tests.rs"]
mod conformance;
#[cfg(test)]
#[path = "bidi_rule_tests.rs"]
mod rules;
