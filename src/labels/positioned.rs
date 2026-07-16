use crate::labels::shape::bidi::{resolve_bidi, visual_order_groups, BidiError};
use crate::labels::shape::{ShapedGlyph, ShapedText, TextError};
use crate::labels::unicode::{bidi_class, line_break_class, BidiClass, LineBreakClass};
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

/// A visually ordered shaped glyph with its exact font identity and device-space origin.
///
/// Unlike [`PositionedOutline`], this record is retained for whitespace and other glyphs
/// without outlines so consumers can preserve advances from the shaped stream.
#[derive(Clone, Debug)]
pub struct PositionedGlyph {
    pub glyph_id: u16,
    pub font_index: usize,
    pub cluster: u32,
    pub line_index: usize,
    pub origin: [f32; 2],
    pub advance: [f32; 2],
    pub path: Option<Path>,
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

struct GlyphGroup<'a> {
    logical_character: usize,
    glyphs: Vec<&'a ShapedGlyph>,
}

fn attachment_root(glyphs: &[ShapedGlyph], index: usize) -> usize {
    let mut current = index;
    for _ in 0..glyphs.len() {
        let Some(parent) = glyphs[current].attached_to else {
            return current;
        };
        if parent >= glyphs.len() || parent == current {
            return index;
        }
        current = parent;
    }
    index
}

fn is_logical_mark(character: char) -> bool {
    matches!(bidi_class(character), BidiClass::Nsm)
        || matches!(line_break_class(character), LineBreakClass::Cm)
}

/// Build base-plus-attached-mark groups in logical glyph order.
///
/// UAX #9 L3 requires a combining mark to remain after its base even when L2 reverses
/// the source characters. The GPOS attachment graph refines explicit mark placement,
/// but unattached Unicode marks still group with their preceding logical base.
fn glyph_groups(shaped: &ShapedText) -> Vec<GlyphGroup<'_>> {
    let byte_to_character: HashMap<usize, usize> = shaped
        .text
        .char_indices()
        .enumerate()
        .map(|(character, (byte, _))| (byte, character))
        .collect();
    let characters = shaped.text.chars().collect::<Vec<_>>();
    let mut groups = Vec::new();
    for run in &shaped.runs {
        let mut roots = Vec::new();
        for index in 0..run.glyphs.len() {
            let root = attachment_root(&run.glyphs, index);
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        for root in roots {
            let root_glyph = &run.glyphs[root];
            let Some(&character) = byte_to_character.get(&(root_glyph.cluster as usize)) else {
                continue;
            };
            let mut glyphs = vec![root_glyph];
            glyphs.extend(
                run.glyphs
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        *index != root && attachment_root(&run.glyphs, *index) == root
                    })
                    .map(|(_, glyph)| glyph),
            );
            if is_logical_mark(characters[character]) {
                if let Some(base) = groups.iter_mut().rfind(|group: &&mut GlyphGroup<'_>| {
                    !is_logical_mark(characters[group.logical_character])
                }) {
                    base.glyphs.extend(glyphs);
                    continue;
                }
            }
            groups.push(GlyphGroup {
                logical_character: character,
                glyphs,
            });
        }
    }
    groups
}

pub fn positioned_glyphs(
    shaped: &ShapedText,
    line_ranges: &[Range<usize>],
) -> Result<Vec<PositionedGlyph>, PositionedError> {
    let character_count = shaped.text.chars().count();
    if let Some(range) = line_ranges
        .iter()
        .find(|range| range.start > range.end || range.end > character_count)
    {
        return Err(PositionedError::InvalidLineRange(range.clone()));
    }
    let mut paragraph = resolve_bidi(&shaped.text, None)?;
    paragraph.levels.clone_from(&shaped.levels);
    // UAX #9 L3: construct attachment groups before applying L2 so the base
    // and all attached marks are one indivisible visual-reordering unit.
    let groups = glyph_groups(shaped);
    let logical_characters = groups
        .iter()
        .map(|group| group.logical_character)
        .collect::<Vec<_>>();
    let visual_groups = visual_order_groups(&paragraph, line_ranges, &logical_characters)?;
    let line_for_character: Vec<_> = (0..character_count)
        .map(|character| {
            line_ranges
                .iter()
                .position(|range| range.contains(&character))
        })
        .collect();
    let mut cursors = vec![[0.0f32, 0.0f32]; line_ranges.len()];
    let mut output = Vec::new();

    for group_index in visual_groups {
        let group = &groups[group_index];
        let character = group.logical_character;
        let Some(line_index) = line_for_character[character] else {
            continue;
        };
        let mut mirror_pending = true;
        for glyph in &group.glyphs {
            let mirrored = if mirror_pending {
                paragraph.mirrored[character]
                    .map(|character| shaped.fonts.glyph_for(character))
                    .transpose()?
            } else {
                None
            };
            mirror_pending = false;
            let font_index = mirrored.map_or(glyph.font_index, |value| value.font_index);
            let glyph_id = mirrored.map_or(glyph.glyph_id, |value| value.glyph_id.0);
            let metrics = shaped.fonts.metrics(font_index)?;
            let scale = shaped.size / f32::from(metrics.units_per_em);
            let line_origin_y = line_index as f32 * shaped.size * 1.2;
            let x = cursors[line_index][0] + glyph.x_offset as f32 * shaped.size / 64.0;
            let y =
                line_origin_y + cursors[line_index][1] - glyph.y_offset as f32 * shaped.size / 64.0;
            let advance = [
                glyph.x_advance as f32 * shaped.size / 64.0,
                glyph.y_advance as f32 * shaped.size / 64.0,
            ];
            let path = match shaped.fonts.outline(font_index, GlyphId(glyph_id)) {
                Ok(outline) => Some(transformed(&outline, scale, x, y)),
                Err(crate::labels::font::TextError::MissingOutline { .. }) => None,
                Err(error) => return Err(error.into()),
            };
            output.push(PositionedGlyph {
                glyph_id,
                font_index,
                cluster: glyph.cluster,
                line_index,
                origin: [x, y],
                advance,
                path,
            });
            cursors[line_index][0] += advance[0];
            cursors[line_index][1] += advance[1];
        }
    }
    Ok(output)
}

pub fn positioned_outlines(
    shaped: &ShapedText,
    line_ranges: &[Range<usize>],
) -> Result<Vec<PositionedOutline>, PositionedError> {
    Ok(positioned_glyphs(shaped, line_ranges)?
        .into_iter()
        .filter_map(|glyph| {
            glyph.path.map(|path| PositionedOutline {
                glyph_id: glyph.glyph_id,
                font_index: glyph.font_index,
                cluster: glyph.cluster,
                line_index: glyph.line_index,
                path,
            })
        })
        .collect())
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
    use super::{positioned_glyphs, positioned_outlines, svg_path_data};
    use crate::labels::font::{FontCollection, FontRequest};
    use crate::labels::shape::{self, Direction, ShapedGlyph, ShapedRun, ShapedText};
    use std::sync::Arc;
    use ttf_parser::Tag;

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

    #[test]
    fn rtl_attached_mark_stays_after_its_base_per_uax9_l3() {
        let fonts = Arc::new(
            FontCollection::load(&[FontRequest::from_bytes(
                "NotoSansHebrew-subset.ttf",
                include_bytes!("../../assets/fonts/NotoSansHebrew-subset.ttf").to_vec(),
            )])
            .unwrap(),
        );
        let base = fonts.glyph_for('ב').unwrap().glyph_id.0;
        let first_mark = fonts.glyph_for('י').unwrap().glyph_id.0;
        let second_mark = fonts.glyph_for('ו').unwrap().glyph_id.0;
        let trailing = fonts.glyph_for('ד').unwrap().glyph_id.0;
        let shaped = ShapedText {
            text: "בְֱד".to_owned(),
            runs: vec![ShapedRun {
                text_range: 0..8,
                glyphs: vec![
                    ShapedGlyph {
                        glyph_id: base,
                        font_index: 0,
                        cluster: 0,
                        x_advance: 64,
                        y_advance: 0,
                        x_offset: 0,
                        y_offset: 0,
                        attached_to: None,
                    },
                    ShapedGlyph {
                        glyph_id: first_mark,
                        font_index: 0,
                        cluster: 2,
                        x_advance: 0,
                        y_advance: 0,
                        x_offset: -48,
                        y_offset: 24,
                        attached_to: Some(0),
                    },
                    ShapedGlyph {
                        glyph_id: second_mark,
                        font_index: 0,
                        cluster: 4,
                        x_advance: 0,
                        y_advance: 0,
                        x_offset: -32,
                        y_offset: 32,
                        attached_to: Some(0),
                    },
                    ShapedGlyph {
                        glyph_id: trailing,
                        font_index: 0,
                        cluster: 6,
                        x_advance: 64,
                        y_advance: 0,
                        x_offset: 0,
                        y_offset: 0,
                        attached_to: None,
                    },
                ],
                bidi_levels: vec![1, 1, 1, 1],
                direction: Direction::RightToLeft,
                script: Tag::from_bytes(b"hebr"),
                language: None,
            }],
            levels: vec![1, 1, 1, 1],
            legal_breaks: vec![0, 4],
            face_descriptors: fonts.descriptors(),
            fonts,
            size: 16.0,
        };

        let line = 0..4;
        let positioned = positioned_glyphs(&shaped, std::slice::from_ref(&line)).unwrap();
        let clusters: Vec<_> = positioned.iter().map(|glyph| glyph.cluster).collect();
        let glyph_ids: Vec<_> = positioned.iter().map(|glyph| glyph.glyph_id).collect();
        let origins: Vec<_> = positioned.iter().map(|glyph| glyph.origin).collect();
        let advances: Vec<_> = positioned.iter().map(|glyph| glyph.advance).collect();

        // L2 reverses the character groups. L3 keeps both marks after the base,
        // in their original attachment order, while retaining exact GPOS data.
        assert_eq!(clusters, vec![6, 0, 2, 4]);
        assert_eq!(glyph_ids, vec![trailing, base, first_mark, second_mark]);
        assert_eq!(
            origins,
            vec![[0.0, 0.0], [16.0, 0.0], [20.0, -6.0], [24.0, -8.0]]
        );
        assert_eq!(
            advances,
            vec![[16.0, 0.0], [16.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        );
    }

    #[test]
    fn rtl_unattached_combining_mark_stays_after_logical_base_per_uax9_l3() {
        let fonts = Arc::new(
            FontCollection::load(&[FontRequest::from_bytes(
                "NotoSansArabic-subset.ttf",
                include_bytes!("../../assets/fonts/NotoSansArabic-subset.ttf").to_vec(),
            )])
            .unwrap(),
        );
        let base = fonts.glyph_for('ب').unwrap().glyph_id.0;
        let trailing = fonts.glyph_for('ت').unwrap().glyph_id.0;
        let shaped = ShapedText {
            text: "ب\u{064E}ت".to_owned(),
            runs: vec![ShapedRun {
                text_range: 0..6,
                glyphs: vec![
                    ShapedGlyph {
                        glyph_id: base,
                        font_index: 0,
                        cluster: 0,
                        x_advance: 64,
                        y_advance: 0,
                        x_offset: 0,
                        y_offset: 0,
                        attached_to: None,
                    },
                    ShapedGlyph {
                        glyph_id: base,
                        font_index: 0,
                        cluster: 2,
                        x_advance: 0,
                        y_advance: 0,
                        x_offset: 0,
                        y_offset: 0,
                        attached_to: None,
                    },
                    ShapedGlyph {
                        glyph_id: trailing,
                        font_index: 0,
                        cluster: 4,
                        x_advance: 64,
                        y_advance: 0,
                        x_offset: 0,
                        y_offset: 0,
                        attached_to: None,
                    },
                ],
                bidi_levels: vec![1, 1, 1],
                direction: Direction::RightToLeft,
                script: Tag::from_bytes(b"arab"),
                language: None,
            }],
            levels: vec![1, 1, 1],
            legal_breaks: vec![0, 3],
            face_descriptors: fonts.descriptors(),
            fonts,
            size: 16.0,
        };

        let line = 0..3;
        let positioned = positioned_glyphs(&shaped, std::slice::from_ref(&line)).unwrap();
        let clusters: Vec<_> = positioned.iter().map(|glyph| glyph.cluster).collect();

        assert_eq!(clusters, vec![4, 0, 2]);
    }

    #[test]
    fn positioned_cursor_applies_x_and_y_advance_once() {
        let fonts = Arc::new(
            FontCollection::load(&[FontRequest::from_bytes(
                "NotoSansLatin-subset.ttf",
                include_bytes!("../../assets/fonts/NotoSansLatin-subset.ttf").to_vec(),
            )])
            .unwrap(),
        );
        let a = fonts.glyph_for('A').unwrap().glyph_id.0;
        let b = fonts.glyph_for('B').unwrap().glyph_id.0;
        let shaped = ShapedText {
            text: "AB".to_owned(),
            runs: vec![ShapedRun {
                text_range: 0..2,
                glyphs: vec![
                    ShapedGlyph {
                        glyph_id: a,
                        font_index: 0,
                        cluster: 0,
                        x_advance: 64,
                        y_advance: 32,
                        x_offset: 16,
                        y_offset: 8,
                        attached_to: None,
                    },
                    ShapedGlyph {
                        glyph_id: b,
                        font_index: 0,
                        cluster: 1,
                        x_advance: 64,
                        y_advance: 0,
                        x_offset: 0,
                        y_offset: 0,
                        attached_to: None,
                    },
                ],
                bidi_levels: vec![0, 0],
                direction: Direction::LeftToRight,
                script: Tag::from_bytes(b"latn"),
                language: None,
            }],
            levels: vec![0, 0],
            legal_breaks: vec![0, 2],
            face_descriptors: fonts.descriptors(),
            fonts,
            size: 16.0,
        };

        let ranges: Vec<std::ops::Range<usize>> = std::iter::once(0..2).collect();
        let positioned = positioned_glyphs(&shaped, &ranges).unwrap();

        assert_eq!(positioned[0].origin, [4.0, -2.0]);
        assert_eq!(positioned[0].advance, [16.0, 8.0]);
        assert_eq!(positioned[1].origin, [16.0, 8.0]);
    }
}
