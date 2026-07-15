use crate::labels::shape::bidi::{resolve_bidi, visual_order, BidiError};
use crate::labels::shape::{ShapedGlyph, ShapedText, TextError};
use lyon_path::{math::point, Event, Path};
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use ttf_parser::GlyphId;

#[derive(Debug)]
pub enum PositionedError {
    Bidi(BidiError),
    Text(TextError),
    InvalidLineRange(Range<usize>),
}

impl fmt::Display for PositionedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bidi(error) => error.fmt(formatter),
            Self::Text(error) => error.fmt(formatter),
            Self::InvalidLineRange(range) => {
                write!(
                    formatter,
                    "invalid text line range {}..{}",
                    range.start, range.end
                )
            }
        }
    }
}

impl std::error::Error for PositionedError {}

impl From<BidiError> for PositionedError {
    fn from(value: BidiError) -> Self {
        Self::Bidi(value)
    }
}

impl From<crate::labels::font::TextError> for PositionedError {
    fn from(value: crate::labels::font::TextError) -> Self {
        Self::Text(TextError::Font(value))
    }
}

#[derive(Clone, Debug)]
pub struct PositionedOutline {
    pub glyph_id: u16,
    pub font_index: usize,
    pub cluster: u32,
    pub line_index: usize,
    pub path: Path,
}

fn transformed(path: &Path, scale: f32, x: f32, y: f32) -> Path {
    let transform = |p: lyon_path::math::Point| point(x + p.x * scale, y + p.y * scale);
    let mut builder = Path::builder();
    for event in path.iter() {
        match event {
            Event::Begin { at } => {
                builder.begin(transform(at));
            }
            Event::Line { to, .. } => {
                builder.line_to(transform(to));
            }
            Event::Quadratic { ctrl, to, .. } => {
                builder.quadratic_bezier_to(transform(ctrl), transform(to));
            }
            Event::Cubic {
                ctrl1, ctrl2, to, ..
            } => {
                builder.cubic_bezier_to(transform(ctrl1), transform(ctrl2), transform(to));
            }
            Event::End { close, .. } => {
                builder.end(close);
            }
        }
    }
    builder.build()
}

fn glyphs_by_character(shaped: &ShapedText) -> Vec<Vec<&ShapedGlyph>> {
    let byte_to_character: HashMap<usize, usize> = shaped
        .text
        .char_indices()
        .enumerate()
        .map(|(character, (byte, _))| (byte, character))
        .collect();
    let mut glyphs = vec![Vec::new(); shaped.text.chars().count()];
    for glyph in shaped.runs.iter().flat_map(|run| &run.glyphs) {
        if let Some(&character) = byte_to_character.get(&(glyph.cluster as usize)) {
            glyphs[character].push(glyph);
        }
    }
    glyphs
}

pub fn positioned_outlines(
    shaped: &ShapedText,
    line_ranges: &[Range<usize>],
) -> Result<Vec<PositionedOutline>, PositionedError> {
    let character_count = shaped.text.chars().count();
    if let Some(range) = line_ranges
        .iter()
        .find(|range| range.start > range.end || range.end > character_count)
    {
        return Err(PositionedError::InvalidLineRange(range.clone()));
    }
    let mut paragraph = resolve_bidi(&shaped.text, None)?;
    paragraph.levels.clone_from(&shaped.levels);
    let visual = visual_order(&paragraph, line_ranges)?;
    let glyphs = glyphs_by_character(shaped);
    let line_for_character: Vec<_> = (0..character_count)
        .map(|character| {
            line_ranges
                .iter()
                .position(|range| range.contains(&character))
        })
        .collect();
    let mut cursors = vec![0.0f32; line_ranges.len()];
    let mut output = Vec::new();

    for character in visual {
        let Some(line_index) = line_for_character[character] else {
            continue;
        };
        for (glyph_position, glyph) in glyphs[character].iter().enumerate() {
            let mirrored = if glyph_position == 0 {
                paragraph.mirrored[character]
                    .map(|character| shaped.fonts.glyph_for(character))
                    .transpose()?
            } else {
                None
            };
            let font_index = mirrored.map_or(glyph.font_index, |value| value.font_index);
            let glyph_id = mirrored.map_or(glyph.glyph_id, |value| value.glyph_id.0);
            let metrics = shaped.fonts.metrics(font_index)?;
            let scale = shaped.size / f32::from(metrics.units_per_em);
            let x = cursors[line_index] + glyph.x_offset as f32 * shaped.size / 64.0;
            let y =
                line_index as f32 * shaped.size * 1.2 - glyph.y_offset as f32 * shaped.size / 64.0;
            let advance = glyph.x_advance as f32 * shaped.size / 64.0;
            let outline = match shaped.fonts.outline(font_index, GlyphId(glyph_id)) {
                Ok(outline) => outline,
                Err(crate::labels::font::TextError::MissingOutline { .. }) => {
                    cursors[line_index] += advance;
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            output.push(PositionedOutline {
                glyph_id,
                font_index,
                cluster: glyph.cluster,
                line_index,
                path: transformed(&outline, scale, x, y),
            });
            cursors[line_index] += advance;
        }
    }
    Ok(output)
}

fn coordinate(value: f32, precision: usize) -> String {
    let mut result = format!("{value:.precision$}");
    if result.contains('.') {
        result = result
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned();
    }
    if result == "-0" {
        result = "0".to_owned();
    }
    result
}

pub fn svg_path_data(outlines: &[PositionedOutline], precision: u8) -> String {
    let precision = precision as usize;
    let mut output = String::new();
    for outline in outlines {
        for event in outline.path.iter() {
            match event {
                Event::Begin { at } => output.push_str(&format!(
                    "M{} {}",
                    coordinate(at.x, precision),
                    coordinate(at.y, precision)
                )),
                Event::Line { to, .. } => output.push_str(&format!(
                    "L{} {}",
                    coordinate(to.x, precision),
                    coordinate(to.y, precision)
                )),
                Event::Quadratic { ctrl, to, .. } => output.push_str(&format!(
                    "Q{} {} {} {}",
                    coordinate(ctrl.x, precision),
                    coordinate(ctrl.y, precision),
                    coordinate(to.x, precision),
                    coordinate(to.y, precision)
                )),
                Event::Cubic {
                    ctrl1, ctrl2, to, ..
                } => output.push_str(&format!(
                    "C{} {} {} {} {} {}",
                    coordinate(ctrl1.x, precision),
                    coordinate(ctrl1.y, precision),
                    coordinate(ctrl2.x, precision),
                    coordinate(ctrl2.y, precision),
                    coordinate(to.x, precision),
                    coordinate(to.y, precision)
                )),
                Event::End { close: true, .. } => output.push('Z'),
                Event::End { close: false, .. } => {}
            }
        }
    }
    output
}

pub fn outline_bounds(outlines: &[PositionedOutline]) -> Option<[f32; 4]> {
    let mut bounds = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let mut saw_point = false;
    let mut include = |point: lyon_path::math::Point| {
        saw_point = true;
        bounds[0] = bounds[0].min(point.x);
        bounds[1] = bounds[1].min(point.y);
        bounds[2] = bounds[2].max(point.x);
        bounds[3] = bounds[3].max(point.y);
    };
    for outline in outlines {
        for event in outline.path.iter() {
            match event {
                Event::Begin { at } | Event::Line { to: at, .. } => include(at),
                Event::Quadratic { ctrl, to, .. } => {
                    include(ctrl);
                    include(to);
                }
                Event::Cubic {
                    ctrl1, ctrl2, to, ..
                } => {
                    include(ctrl1);
                    include(ctrl2);
                    include(to);
                }
                Event::End { .. } => {}
            }
        }
    }
    saw_point.then_some(bounds)
}

#[cfg(test)]
mod tests {
    use super::{positioned_outlines, svg_path_data};
    use crate::labels::font::{FontCollection, FontRequest};
    use crate::labels::shape;
    use std::sync::Arc;

    #[test]
    fn supplied_line_partition_changes_positioned_outline_geometry() {
        let fonts = Arc::new(
            FontCollection::load(&[FontRequest::from_bytes(
                "NotoSansLatin-subset.ttf",
                include_bytes!("../../assets/fonts/NotoSansLatin-subset.ttf").to_vec(),
            )])
            .unwrap(),
        );
        let shaped = shape::shape("AB", fonts, 16.0, None, None, &[]).unwrap();
        let one_line_range = 0..2;
        let one_line = positioned_outlines(&shaped, std::slice::from_ref(&one_line_range)).unwrap();
        let two_lines = positioned_outlines(&shaped, &[0..1, 1..2]).unwrap();

        assert_ne!(svg_path_data(&one_line, 4), svg_path_data(&two_lines, 4));
        assert_eq!(two_lines[0].line_index, 0);
        assert_eq!(two_lines[1].line_index, 1);
    }

    #[test]
    fn odd_level_punctuation_uses_uax9_l4_mirrored_glyphs() {
        let fonts = Arc::new(
            FontCollection::load(&[
                FontRequest::from_bytes(
                    "NotoSansLatin-subset.ttf",
                    include_bytes!("../../assets/fonts/NotoSansLatin-subset.ttf").to_vec(),
                ),
                FontRequest::from_bytes(
                    "NotoSansHebrew-subset.ttf",
                    include_bytes!("../../assets/fonts/NotoSansHebrew-subset.ttf").to_vec(),
                ),
            ])
            .unwrap(),
        );
        let shaped = shape::shape("(בד)", fonts.clone(), 16.0, None, None, &[]).unwrap();
        let line = 0..4;
        let outlines = positioned_outlines(&shaped, std::slice::from_ref(&line)).unwrap();
        let open = fonts.glyph_for('(').unwrap().glyph_id.0;
        let close = fonts.glyph_for(')').unwrap().glyph_id.0;

        assert_eq!(outlines.first().unwrap().glyph_id, open);
        assert_eq!(outlines.last().unwrap().glyph_id, close);
    }
}
