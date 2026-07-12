use super::super::{gsub::Glyph, ShapedGlyph, TextError};
use crate::labels::font::to_q26_6;
use ttf_parser::GlyphId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositioningGlyph {
    pub id: GlyphId,
    pub cluster: u32,
    pub x_advance: i32,
    pub y_advance: i32,
    pub x_offset: i32,
    pub y_offset: i32,
    pub attached_to: Option<usize>,
    pub component_clusters: Vec<u32>,
    pub ligature_component: Option<u16>,
}

impl PositioningGlyph {
    pub fn new(id: GlyphId, cluster: u32) -> Self {
        Self {
            id,
            cluster,
            x_advance: 0,
            y_advance: 0,
            x_offset: 0,
            y_offset: 0,
            attached_to: None,
            component_clusters: vec![cluster],
            ligature_component: None,
        }
    }

    pub fn normalize(self, font_index: usize, units_per_em: u16) -> Result<ShapedGlyph, TextError> {
        if units_per_em == 0 {
            return Err(TextError::MalformedOpenType("head unitsPerEm"));
        }
        Ok(ShapedGlyph {
            glyph_id: self.id.0,
            font_index,
            cluster: self.cluster,
            x_advance: to_q26_6(self.x_advance, units_per_em),
            y_advance: to_q26_6(self.y_advance, units_per_em),
            x_offset: to_q26_6(self.x_offset, units_per_em),
            y_offset: to_q26_6(self.y_offset, units_per_em),
            attached_to: self.attached_to,
        })
    }
}

impl From<&Glyph> for PositioningGlyph {
    fn from(glyph: &Glyph) -> Self {
        let mut positioned = Self::new(glyph.id, glyph.cluster);
        positioned
            .component_clusters
            .clone_from(&glyph.component_clusters);
        positioned
    }
}
