use crate::labels::font::{FaceDescriptor, FontCollection};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use ttf_parser::Tag;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    LeftToRight,
    RightToLeft,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapedGlyph {
    pub glyph_id: u16,
    pub font_index: usize,
    pub cluster: u32,
    pub x_advance: i32,
    pub y_advance: i32,
    pub x_offset: i32,
    pub y_offset: i32,
    pub attached_to: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapedRun {
    pub text_range: Range<usize>,
    pub glyphs: Vec<ShapedGlyph>,
    pub bidi_levels: Vec<u8>,
    pub direction: Direction,
    pub script: Tag,
    pub language: Option<Tag>,
}

#[derive(Clone)]
pub struct ShapedText {
    pub text: String,
    pub runs: Vec<ShapedRun>,
    pub levels: Vec<u8>,
    pub legal_breaks: Vec<usize>,
    pub fonts: Arc<FontCollection>,
    pub face_descriptors: Vec<FaceDescriptor>,
    pub size: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextError {
    OutOfBounds {
        offset: usize,
        length: usize,
    },
    MalformedOpenType(&'static str),
    UnsupportedLookup {
        table: &'static str,
        lookup_type: u16,
        script: Tag,
    },
    InvalidSize,
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { offset, length } => {
                write!(
                    f,
                    "OpenType read out of bounds at {offset} for {length} bytes"
                )
            }
            Self::MalformedOpenType(table) => write!(f, "malformed OpenType {table}"),
            Self::UnsupportedLookup {
                table,
                lookup_type,
                script,
            } => write!(
                f,
                "unsupported {table} lookup type {lookup_type} for script {script}"
            ),
            Self::InvalidSize => write!(f, "text size must be finite and positive"),
        }
    }
}

impl std::error::Error for TextError {}

#[cfg(test)]
mod tests {
    use super::{Direction, ShapedRun};
    use ttf_parser::Tag;

    #[test]
    fn shaped_run_preserves_distinct_resolved_levels() {
        let run = ShapedRun {
            text_range: 0..2,
            glyphs: Vec::new(),
            bidi_levels: vec![0, 2],
            direction: Direction::LeftToRight,
            script: Tag::from_bytes(b"latn"),
            language: None,
        };
        assert_eq!(run.bidi_levels, vec![0, 2]);
    }
}
