use super::resolve_bidi;

fn levels(text: &str, paragraph: u8) -> Vec<u8> {
    resolve_bidi(text, Some(paragraph)).unwrap().levels
}

#[test]
fn p2_p3_skip_isolated_strong_text() {
    assert_eq!(
        resolve_bidi("\u{2067}א\u{2069}a", None)
            .unwrap()
            .paragraph_level,
        0
    );
}

#[test]
fn x1_x8_embeddings_overrides_and_isolates() {
    let resolved = levels("a\u{202b}b\u{202c}c", 0);
    assert_eq!([resolved[0], resolved[2], resolved[4]], [0, 2, 0]);
    assert_eq!(levels("a\u{2067}ב\u{2069}c", 0), vec![0, 0, 1, 0, 0]);
}

#[test]
fn w1_nsm_inherits_previous_type() {
    assert_eq!(levels("א\u{0301}", 1), vec![1, 1]);
}

#[test]
fn w2_en_after_al_becomes_an() {
    assert_eq!(levels("ا1", 1), vec![1, 2]);
}

#[test]
fn w3_al_becomes_r() {
    assert_eq!(levels("ا", 1), vec![1]);
}

#[test]
fn w4_separator_between_numbers_becomes_number() {
    assert_eq!(levels("1+2", 1), vec![2, 2, 2]);
}

#[test]
fn w5_terminator_run_adjacent_to_number_becomes_number() {
    assert_eq!(levels("$1", 1), vec![2, 2]);
}

#[test]
fn w6_remaining_separators_and_terminators_become_neutral() {
    assert_eq!(levels("+", 0), vec![0]);
}

#[test]
fn w7_en_after_l_becomes_l() {
    assert_eq!(levels("a1", 0), vec![0, 0]);
}

#[test]
fn n0_brackets_follow_enclosed_strong_type() {
    assert_eq!(levels("(a)", 1), vec![1, 2, 1]);
}

#[test]
fn n1_neutral_between_matching_strong_types_matches_them() {
    assert_eq!(levels("א א", 1), vec![1, 1, 1]);
}

#[test]
fn n2_other_neutral_uses_embedding_direction() {
    assert_eq!(levels("a א", 0), vec![0, 0, 1]);
}

#[test]
fn i1_even_embedding_raises_r_and_numbers() {
    assert_eq!(levels("א1", 0), vec![1, 2]);
}

#[test]
fn i2_odd_embedding_raises_l_and_numbers() {
    assert_eq!(levels("a1", 1), vec![2, 2]);
}
