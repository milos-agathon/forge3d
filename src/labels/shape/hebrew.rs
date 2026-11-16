use super::gsub::Glyph;
use std::collections::BTreeMap;

fn modified_combining_class(ch: char) -> Option<u16> {
    Some(match ch {
        '\u{0591}' | '\u{0596}' | '\u{059B}' | '\u{05A2}'..='\u{05A7}' | '\u{05AA}' => 220,
        '\u{059A}' | '\u{05AD}' => 222,
        '\u{05AE}' => 228,
        '\u{0592}'..='\u{0595}'
        | '\u{0597}'..='\u{0599}'
        | '\u{059C}'..='\u{05A1}'
        | '\u{05A8}'..='\u{05A9}'
        | '\u{05AB}'..='\u{05AC}'
        | '\u{05AF}'
        | '\u{05C4}' => 230,
        '\u{05B0}' => 22,
        '\u{05B1}' => 15,
        '\u{05B2}' => 16,
        '\u{05B3}' => 17,
        '\u{05B4}' => 23,
        '\u{05B5}' => 18,
        '\u{05B6}' => 19,
        '\u{05B7}' => 20,
        '\u{05B8}' | '\u{05C7}' => 21,
        '\u{05B9}' | '\u{05BA}' => 14,
        '\u{05BB}' => 24,
        '\u{05BC}' => 12,
        '\u{05BD}' => 25,
        '\u{05BF}' => 13,
        '\u{05C1}' => 10,
        '\u{05C2}' => 11,
        '\u{05C5}' => 220,
        '\u{FB1E}' => 26,
        _ => return None,
    })
}

fn glyph_combining_class(source: &BTreeMap<usize, char>, glyph: &Glyph) -> Option<u16> {
    source
        .get(&(glyph.cluster as usize))
        .copied()
        .and_then(modified_combining_class)
}

fn apply_hebrew_mark_exception(source: &BTreeMap<usize, char>, glyphs: &mut [Glyph]) {
    for index in 2..glyphs.len() {
        let first = glyph_combining_class(source, &glyphs[index - 2]);
        let second = glyph_combining_class(source, &glyphs[index - 1]);
        let third = glyph_combining_class(source, &glyphs[index]);
        if matches!(first, Some(20 | 21))
            && matches!(second, Some(22 | 23))
            && matches!(third, Some(25 | 220))
        {
            glyphs.swap(index - 1, index);
            break;
        }
    }
}

pub(super) fn reorder_marks(text: &str, glyphs: &mut [Glyph]) {
    let source = text.char_indices().collect::<BTreeMap<_, _>>();
    let mut start = 0;
    while start < glyphs.len() {
        let has_starter = glyph_combining_class(&source, &glyphs[start]).is_none();
        let marks_start = if has_starter { start + 1 } else { start };
        let mut end = marks_start;
        while end < glyphs.len() && glyph_combining_class(&source, &glyphs[end]).is_some() {
            end += 1;
        }
        if end > marks_start {
            let marks = &mut glyphs[marks_start..end];
            marks.sort_by_key(|glyph| glyph_combining_class(&source, glyph).unwrap_or(u16::MAX));
            apply_hebrew_mark_exception(&source, marks);
        }
        if has_starter {
            let cluster = glyphs[start].cluster;
            for mark in &mut glyphs[marks_start..end] {
                mark.cluster = cluster;
            }
        }
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::reorder_marks;
    use crate::labels::shape::gsub::Glyph;
    use ttf_parser::GlyphId;

    fn source_glyphs(text: &str) -> Vec<Glyph> {
        text.char_indices()
            .map(|(cluster, ch)| Glyph::new(GlyphId(ch as u16), cluster as u32))
            .collect()
    }

    #[test]
    fn hebrew_marks_use_modified_classes_and_base_clusters() {
        let text = "\u{05E9}\u{05B8}\u{05BC}\u{05C1}\u{05DC}";
        let mut glyphs = source_glyphs(text);
        reorder_marks(text, &mut glyphs);
        assert_eq!(
            glyphs.iter().map(|g| g.id.0).collect::<Vec<_>>(),
            vec![0x05E9, 0x05C1, 0x05BC, 0x05B8, 0x05DC]
        );
        assert_eq!(
            glyphs.iter().map(|g| g.cluster).collect::<Vec<_>>(),
            vec![0, 0, 0, 0, 8]
        );
    }

    #[test]
    fn equal_class_marks_keep_source_order_and_separate_bases() {
        let text = "\u{05D1}\u{05B8}\u{05C7}\u{05D1}\u{05B0}";
        let mut glyphs = source_glyphs(text);
        reorder_marks(text, &mut glyphs);
        assert_eq!(
            glyphs.iter().map(|g| g.id.0).collect::<Vec<_>>(),
            vec![0x05D1, 0x05B8, 0x05C7, 0x05D1, 0x05B0]
        );
        assert_eq!(
            glyphs.iter().map(|g| g.cluster).collect::<Vec<_>>(),
            vec![0, 0, 0, 6, 6]
        );
    }

    #[test]
    fn leading_marks_are_sorted_without_inventing_a_starter() {
        let text = "\u{05B0}\u{05BC}\u{05D1}";
        let mut glyphs = source_glyphs(text);
        reorder_marks(text, &mut glyphs);
        assert_eq!(
            glyphs.iter().map(|g| g.id.0).collect::<Vec<_>>(),
            vec![0x05BC, 0x05B0, 0x05D1]
        );
        assert_eq!(
            glyphs.iter().map(|g| g.cluster).collect::<Vec<_>>(),
            vec![2, 0, 4]
        );
    }

    #[test]
    fn qamats_sheva_meteg_uses_hebrew_post_sort_swap() {
        let text = "\u{05D1}\u{05B8}\u{05B0}\u{05BD}";
        let mut glyphs = source_glyphs(text);
        reorder_marks(text, &mut glyphs);
        assert_eq!(
            glyphs.iter().map(|g| g.id.0).collect::<Vec<_>>(),
            vec![0x05D1, 0x05B8, 0x05BD, 0x05B0]
        );
        assert_eq!(
            glyphs.iter().map(|g| g.cluster).collect::<Vec<_>>(),
            vec![0, 0, 0, 0]
        );
    }
}
