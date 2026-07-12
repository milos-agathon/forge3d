#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClusteredChar {
    pub ch: char,
    pub cluster: u32,
}

fn is_consonant(ch: char) -> bool {
    matches!(ch, '\u{0915}'..='\u{0939}' | '\u{0958}'..='\u{095F}' | '\u{0978}'..='\u{097F}')
}

pub fn reorder_devanagari(text: &str) -> Vec<ClusteredChar> {
    let mut output: Vec<_> = text
        .char_indices()
        .map(|(cluster, ch)| ClusteredChar {
            ch,
            cluster: cluster as u32,
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

#[cfg(test)]
mod tests {
    use super::reorder_devanagari;

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
}
