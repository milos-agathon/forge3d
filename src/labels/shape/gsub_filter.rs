use super::Glyph;
use ttf_parser::gdef::GlyphClass;
use ttf_parser::Face;

#[derive(Clone, Copy)]
pub(crate) struct GlyphFilter<'a> {
    gdef: Option<ttf_parser::gdef::Table<'a>>,
    flags: ttf_parser::opentype_layout::LookupFlags,
    mark_set: Option<u16>,
}

impl<'a> GlyphFilter<'a> {
    pub(super) fn new(face: &Face<'a>, lookup: ttf_parser::opentype_layout::Lookup<'a>) -> Self {
        Self {
            gdef: face.tables().gdef,
            flags: lookup.flags,
            mark_set: lookup.mark_filtering_set,
        }
    }

    pub(super) fn ignored(self, glyph: &Glyph) -> bool {
        let Some(gdef) = self.gdef else { return false };
        match gdef.glyph_class(glyph.id) {
            Some(GlyphClass::Base) => self.flags.ignore_base_glyphs(),
            Some(GlyphClass::Ligature) => self.flags.ignore_ligatures(),
            Some(GlyphClass::Mark) => {
                if self.flags.ignore_marks() {
                    return true;
                }
                if self.flags.use_mark_filtering_set() {
                    return !gdef.is_mark_glyph(glyph.id, self.mark_set);
                }
                let attachment_type = self.flags.0 >> 8;
                attachment_type != 0
                    && gdef.glyph_mark_attachment_class(glyph.id) != attachment_type
            }
            _ => false,
        }
    }

    pub(super) fn next(self, buffer: &[Glyph], from: usize) -> Option<usize> {
        (from..buffer.len()).find(|&index| !self.ignored(&buffer[index]))
    }

    pub(super) fn previous(self, buffer: &[Glyph], before: usize) -> Option<usize> {
        (0..before)
            .rev()
            .find(|&index| !self.ignored(&buffer[index]))
    }
}
