#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusteredChar {
    pub ch: char,
    pub cluster: u32,
    pub features: Vec<ttf_parser::Tag>,
}

fn is_consonant(ch: char) -> bool {
    matches!(ch, '\u{0915}'..='\u{0939}' | '\u{0958}'..='\u{095F}' | '\u{0978}'..='\u{097F}')
}

fn is_postbase_matra(ch: char) -> bool {
    matches!(ch, '\u{093E}' | '\u{0940}' | '\u{0947}'..='\u{094C}' | '\u{094E}'..='\u{094F}')
}

fn syllable_ranges(text: &str) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut seen_consonant = false;
    let mut previous = None;
    for (offset, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if start < offset {
                ranges.push((start as u32, offset as u32));
            }
            start = offset + ch.len_utf8();
            seen_consonant = false;
            previous = None;
            continue;
        }
        if is_consonant(ch) {
            if seen_consonant && previous != Some('\u{094D}') {
                ranges.push((start as u32, offset as u32));
                start = offset;
            }
            seen_consonant = true;
        }
        previous = Some(ch);
    }
    if start < text.len() {
        ranges.push((start as u32, text.len() as u32));
    }
    ranges
}

pub fn reorder_devanagari(text: &str) -> Vec<ClusteredChar> {
    let mut output: Vec<_> = text
        .char_indices()
        .map(|(cluster, ch)| ClusteredChar {
            ch,
            cluster: cluster as u32,
            features: Vec::new(),
        })
        .collect();
    let mut index = 0;
    while index < output.len() {
        if output[index].ch != '\u{093F}' || index == 0 {
            index += 1;
            continue;
        }
        let mut start = index - 1;
        if !is_consonant(output[start].ch) {
            index += 1;
            continue;
        }
        while start >= 2 && output[start - 1].ch == '\u{094D}' && is_consonant(output[start - 2].ch)
        {
            start -= 2;
        }
        let matra = output.remove(index);
        output.insert(start, matra);
        index += 1;
    }
    output
}

pub fn devanagari_features(text: &str) -> Vec<ClusteredChar> {
    let mut chars: Vec<_> = text
        .char_indices()
        .map(|(cluster, ch)| ClusteredChar {
            ch,
            cluster: cluster as u32,
            features: Vec::new(),
        })
        .collect();
    let tag = |bytes| ttf_parser::Tag::from_bytes(bytes);
    let syllables = syllable_ranges(text);
    for index in 0..chars.len().saturating_sub(1) {
        let range = syllables
            .iter()
            .copied()
            .find(|(start, end)| (*start..*end).contains(&chars[index].cluster));
        let same_syllable =
            range.is_some_and(|(start, end)| (start..end).contains(&chars[index + 1].cluster));
        if !same_syllable {
            continue;
        }
        if chars[index].ch == 'र'
            && chars[index + 1].ch == '\u{094D}'
            && range.is_some_and(|(start, _)| chars[index].cluster == start)
        {
            chars[index].features.push(tag(b"rphf"));
            chars[index + 1].features.push(tag(b"rphf"));
        } else if is_consonant(chars[index].ch) && chars[index + 1].ch == '\u{094D}' {
            chars[index].features.push(tag(b"half"));
            chars[index + 1].features.push(tag(b"half"));
        }
        if chars[index].ch == '\u{094D}' && chars[index + 1].ch == 'र' {
            chars[index].features.push(tag(b"pref"));
            chars[index + 1].features.push(tag(b"pref"));
        }
    }
    chars
}

pub fn apply_feature_masks(text: &str, glyphs: &mut [crate::labels::shape::gsub::Glyph]) {
    for item in devanagari_features(text) {
        for glyph in glyphs
            .iter_mut()
            .filter(|glyph| glyph.cluster == item.cluster)
        {
            for &feature in &item.features {
                glyph.enable_feature(feature);
            }
            if item.ch == '\u{093F}' {
                glyph.enable_feature(ttf_parser::Tag::from_bytes(b"imtr"));
            }
        }
    }
}

pub fn finish_reordering(text: &str, glyphs: &mut Vec<crate::labels::shape::gsub::Glyph>) {
    let pref = ttf_parser::Tag::from_bytes(b"pref");
    let reph = ttf_parser::Tag::from_bytes(b"rphf");
    let imatra = ttf_parser::Tag::from_bytes(b"imtr");
    for (start, end) in syllable_ranges(text).into_iter().rev() {
        let Some(first) = glyphs
            .iter()
            .position(|glyph| (start..end).contains(&glyph.cluster))
        else {
            continue;
        };
        let last = glyphs
            .iter()
            .rposition(|glyph| (start..end).contains(&glyph.cluster))
            .unwrap_or(first);
        if let Some(index) = (first..=last)
            .find(|&index| glyphs[index].has_feature(pref) || glyphs[index].has_feature(imatra))
        {
            let glyph = glyphs.remove(index);
            glyphs.insert(first, glyph);
        }
        if let Some(index) = (first..glyphs.len()).find(|&index| {
            (start..end).contains(&glyphs[index].cluster) && glyphs[index].has_feature(reph)
        }) {
            let glyph = glyphs.remove(index);
            let insert = glyphs
                .iter()
                .enumerate()
                .skip(first)
                .find(|(_, item)| {
                    (start..end).contains(&item.cluster)
                        && text
                            .get(item.cluster as usize..)
                            .and_then(|tail| tail.chars().next())
                            .is_some_and(is_postbase_matra)
                })
                .map(|(index, _)| index)
                .or_else(|| {
                    glyphs
                        .iter()
                        .rposition(|item| (start..end).contains(&item.cluster))
                        .map(|index| index + 1)
                })
                .unwrap_or(first);
            glyphs.insert(insert, glyph);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{devanagari_features, reorder_devanagari};

    fn reordered(text: &str) -> String {
        reorder_devanagari(text)
            .iter()
            .map(|item| item.ch)
            .collect()
    }

    #[test]
    fn prebase_i_matra_moves_before_consonant_cluster() {
        assert_eq!(reordered("कि"), "िक");
    }

    #[test]
    fn prebase_i_matra_moves_before_conjunct() {
        assert_eq!(reordered("क्षि"), "िक्ष");
    }

    #[test]
    fn conjunct_patterns_keep_halant_sequences_intact() {
        assert_eq!(reordered("स्त्र"), "स्त्र");
        assert_eq!(reordered("ज्ञ"), "ज्ञ");
        assert_eq!(reordered("श्र"), "श्र");
    }

    #[test]
    fn matra_moves_within_its_syllable_and_keeps_utf8_cluster() {
        let output = reorder_devanagari("रामकि");
        assert_eq!(
            output.iter().map(|item| item.ch).collect::<String>(),
            "रामिक"
        );
        let matra = output.iter().find(|item| item.ch == 'ि').unwrap();
        assert_eq!(matra.cluster, "रामक".len() as u32);
    }

    #[test]
    fn conjunct_roles_enable_reph_half_and_pref_features() {
        let tag = |bytes| ttf_parser::Tag::from_bytes(bytes);
        let reph = devanagari_features("र्क");
        assert!(reph[0].features.contains(&tag(b"rphf")));
        let half = devanagari_features("क्त");
        assert!(half[0].features.contains(&tag(b"half")));
        let pref = devanagari_features("क्र");
        assert!(pref[1].features.contains(&tag(b"pref")));
        assert!(pref[2].features.contains(&tag(b"pref")));
    }

    #[test]
    fn final_reordering_does_not_cross_unspaced_syllables() {
        let text = "रामकि";
        let mut glyphs: Vec<_> = text
            .char_indices()
            .enumerate()
            .map(|(index, (cluster, _))| {
                crate::labels::shape::gsub::Glyph::new(
                    ttf_parser::GlyphId(index as u16 + 1),
                    cluster as u32,
                )
            })
            .collect();
        super::apply_feature_masks(text, &mut glyphs);
        super::finish_reordering(text, &mut glyphs);
        assert_eq!(
            glyphs.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
            vec![1, 2, 3, 5, 4]
        );
    }

    #[test]
    fn reph_precedes_postbase_matra() {
        let text = "र्का";
        let mut reph = crate::labels::shape::gsub::Glyph::new(ttf_parser::GlyphId(90), 0);
        reph.enable_feature(ttf_parser::Tag::from_bytes(b"rphf"));
        let mut glyphs = vec![
            reph,
            crate::labels::shape::gsub::Glyph::new(ttf_parser::GlyphId(12), "र्".len() as u32),
            crate::labels::shape::gsub::Glyph::new(ttf_parser::GlyphId(13), "र्क".len() as u32),
        ];
        super::finish_reordering(text, &mut glyphs);
        assert_eq!(
            glyphs.iter().map(|glyph| glyph.id.0).collect::<Vec<_>>(),
            vec![12, 90, 13]
        );
    }
}
