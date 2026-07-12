mod generated;

pub use generated::*;

#[cfg(test)]
mod tests {
    use super::{
        bidi_class, joining_type, line_break_class, mirrored, script, BidiClass, JoiningType,
        LineBreakClass, Script,
    };

    #[test]
    fn generated_tables_cover_required_scripts() {
        assert_eq!(script('A'), Script::Latin);
        assert_eq!(script('ب'), Script::Arabic);
        assert_eq!(script('ק'), Script::Hebrew);
        assert_eq!(script('क'), Script::Devanagari);
        assert_eq!(script('漢'), Script::Han);
        assert_eq!(script('\u{0301}'), Script::Inherited);
    }

    #[test]
    fn joining_data_distinguishes_letters_and_marks() {
        assert_eq!(joining_type('ب'), JoiningType::DualJoining);
        assert_eq!(joining_type('ا'), JoiningType::RightJoining);
        assert_eq!(joining_type('\u{064E}'), JoiningType::Transparent);
        assert_eq!(joining_type('A'), JoiningType::NonJoining);
    }

    #[test]
    fn bidi_classes_include_isolates_and_required_scripts() {
        assert_eq!(bidi_class('\u{2066}'), BidiClass::Lri);
        assert_eq!(bidi_class('\u{2067}'), BidiClass::Rli);
        assert_eq!(bidi_class('\u{2068}'), BidiClass::Fsi);
        assert_eq!(bidi_class('\u{2069}'), BidiClass::Pdi);
        assert_eq!(bidi_class('ק'), BidiClass::R);
        assert_eq!(bidi_class('ب'), BidiClass::Al);
    }

    #[test]
    fn mirroring_pairs_are_bidirectional() {
        assert_eq!(mirrored('('), Some(')'));
        assert_eq!(mirrored(')'), Some('('));
        assert_eq!(mirrored('<'), Some('>'));
        assert_eq!(mirrored('A'), None);
    }

    #[test]
    fn line_break_data_covers_text_and_cjk() {
        assert_eq!(line_break_class('\n'), LineBreakClass::Lf);
        assert_eq!(line_break_class(' '), LineBreakClass::Sp);
        assert_eq!(line_break_class('\u{0301}'), LineBreakClass::Cm);
        assert_eq!(line_break_class('漢'), LineBreakClass::Id);
        assert_eq!(line_break_class('\u{2010}'), LineBreakClass::Hh);
    }
}
