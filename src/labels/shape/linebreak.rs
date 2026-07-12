use crate::labels::unicode::{bidi_class, line_break_class, BidiClass, LineBreakClass as C};

use super::linebreak_emoji::is_unassigned_extended_pictographic;
use super::linebreak_rules::{
    aksara, closing_quote, east_asian_parenthesis, east_asian_width, effective_class, hangul,
    numeric, opening_quote,
};

pub fn line_breaks(text: &str) -> Vec<usize> {
    let chars: Vec<_> = text.char_indices().collect();
    if chars.is_empty() {
        return vec![0];
    }
    let classes: Vec<_> = chars
        .iter()
        .map(|&(_, character)| resolve_class(character))
        .collect();
    let mut breaks = Vec::new();
    let mut regional_indicators = usize::from(classes[0] == C::Ri);
    for index in 1..chars.len() {
        if allowed(&chars, &classes, index, regional_indicators) {
            breaks.push(chars[index].0);
        }
        regional_indicators = if classes[index] == C::Ri {
            regional_indicators + 1
        } else if matches!(classes[index], C::Cm | C::Zwj) {
            regional_indicators
        } else {
            0
        };
    }
    breaks.push(text.len());
    breaks
}

fn resolve_class(character: char) -> C {
    match line_break_class(character) {
        C::Ai | C::Sg | C::Xx => C::Al,
        C::Sa if bidi_class(character) == BidiClass::Nsm || sa_spacing_mark(character) => C::Cm,
        C::Sa => C::Al,
        C::Cj => C::Ns,
        class => class,
    }
}

fn sa_spacing_mark(character: char) -> bool {
    matches!(
        character as u32,
        0x102b..=0x103e
            | 0x1056..=0x1059
            | 0x105e..=0x1060
            | 0x1062..=0x1064
            | 0x1067..=0x106d
            | 0x1071..=0x1074
            | 0x1082..=0x108d
            | 0x108f
            | 0x109a..=0x109d
            | 0x17b6
            | 0x17be..=0x17c5
            | 0x17c7..=0x17c8
    )
}

fn allowed(
    chars: &[(usize, char)],
    classes: &[C],
    index: usize,
    regional_indicators: usize,
) -> bool {
    let mut previous = classes[index - 1];
    let next = classes[index];
    if previous == C::Cr && next == C::Lf {
        return false;
    }
    if matches!(previous, C::Bk | C::Cr | C::Lf | C::Nl) {
        return true;
    }
    if matches!(next, C::Bk | C::Cr | C::Lf | C::Nl | C::Sp | C::Zw) {
        return false;
    }
    if previous == C::Zw {
        return true;
    }
    if previous == C::Zwj {
        return false;
    }
    if matches!(next, C::Cm | C::Zwj) && !matches!(previous, C::Sp | C::Zw) {
        return false;
    }
    if previous == C::Cm {
        previous = effective_class(classes, index - 1);
    }
    if previous == C::Sp && classes[..index].iter().rfind(|class| **class != C::Sp) == Some(&C::Zw)
    {
        return true;
    }
    if previous == C::Wj || next == C::Wj || previous == C::Gl {
        return false;
    }
    if next == C::Gl && !matches!(previous, C::Sp | C::Ba | C::Hh | C::Hy) {
        return false;
    }
    if matches!(next, C::Cl | C::Cp | C::Ex | C::Sy) {
        return false;
    }
    if next == C::Is {
        return previous == C::Sp && classes.get(index + 1) == Some(&C::Nu);
    }
    let before_spaces_position = classes[..index].iter().rposition(|class| *class != C::Sp);
    let before_spaces = before_spaces_position.map(|position| {
        if position == index - 1 {
            previous
        } else {
            effective_class(classes, position)
        }
    });
    let quote_position = before_spaces_position.map(|position| {
        if matches!(classes[position], C::Cm | C::Zwj) {
            classes[..position]
                .iter()
                .rposition(|class| !matches!(class, C::Cm | C::Zwj))
                .unwrap_or(position)
        } else {
            position
        }
    });
    let after_initial_quote = quote_position.is_some_and(|quote| {
        classes[quote] == C::Qu
            && opening_quote(chars[quote].1)
            && (quote == 0
                || matches!(
                    classes[quote - 1],
                    C::Bk | C::Cr | C::Lf | C::Nl | C::Op | C::Qu | C::Gl | C::Sp | C::Zw
                ))
    });
    if before_spaces == Some(C::Op)
        || after_initial_quote
        || (matches!(before_spaces, Some(C::Cl | C::Cp)) && next == C::Ns)
        || (before_spaces == Some(C::B2) && next == C::B2)
    {
        return false;
    }
    let quote_can_follow_space = previous == C::Sp
        && (opening_quote(chars[index].1)
            || matches!(chars[index].1, '"' | '\'')
            || closing_quote(chars[index].1)
                && classes.get(index + 1).is_some_and(|class| {
                    !matches!(
                        class,
                        C::Sp
                            | C::Gl
                            | C::Wj
                            | C::Cl
                            | C::Qu
                            | C::Cp
                            | C::Ex
                            | C::Is
                            | C::Sy
                            | C::Bk
                            | C::Cr
                            | C::Lf
                            | C::Nl
                            | C::Zw
                    )
                }));
    let quote_between_east_asian = next == C::Qu
        && opening_quote(chars[index].1)
        && chars.get(index + 1).is_some_and(|next_character| {
            east_asian_width(chars[index - 1].1) && east_asian_width(next_character.1)
        });
    if next == C::Qu && !(quote_can_follow_space || quote_between_east_asian)
        || previous == C::Qu
            && !(closing_quote(chars[index - 1].1)
                && matches!(next, C::Id | C::Eb | C::Em)
                && index >= 2
                && east_asian_width(chars[index - 2].1))
    {
        return false;
    }
    if previous == C::Sp {
        return true;
    }
    if previous == C::Cb || next == C::Cb {
        return true;
    }
    if matches!(next, C::Ba | C::Hh | C::Hy | C::Ns) || previous == C::Bb {
        return false;
    }
    if matches!(previous, C::Hh | C::Hy)
        && matches!(next, C::Al | C::Hl)
        && (classes[..index]
            .iter()
            .rposition(|class| !matches!(class, C::Cm | C::Zwj))
            .is_some_and(|hyphen| {
                hyphen == 0
                    || matches!(
                        effective_class(classes, hyphen - 1),
                        C::Bk | C::Cr | C::Lf | C::Nl | C::Sp | C::Zw | C::Cb | C::Gl
                    )
            }))
    {
        return false;
    }
    if index >= 2
        && classes[index - 2] == C::Hl
        && matches!(previous, C::Hh | C::Hy)
        && next != C::Hl
    {
        return false;
    }
    if previous == C::Sy && next == C::Hl || next == C::In {
        return false;
    }
    if matches!(previous, C::Al | C::Hl) && next == C::Nu
        || previous == C::Nu && matches!(next, C::Al | C::Hl)
    {
        return false;
    }
    if previous == C::Pr && matches!(next, C::Id | C::Eb | C::Em)
        || matches!(previous, C::Id | C::Eb | C::Em) && next == C::Po
    {
        return false;
    }
    if matches!(previous, C::Pr | C::Po) && matches!(next, C::Al | C::Hl)
        || matches!(previous, C::Al | C::Hl) && matches!(next, C::Pr | C::Po)
    {
        return false;
    }
    if numeric(classes, index) {
        return false;
    }
    if hangul(previous, next)
        || matches!(previous, C::Jl | C::Jv | C::Jt | C::H2 | C::H3)
            && matches!(next, C::In | C::Po)
        || previous == C::Pr && matches!(next, C::Jl | C::Jv | C::Jt | C::H2 | C::H3)
    {
        return false;
    }
    if matches!(previous, C::Al | C::Hl) && matches!(next, C::Al | C::Hl)
        || previous == C::Is && matches!(next, C::Al | C::Hl)
        || matches!(previous, C::Al | C::Hl | C::Nu)
            && next == C::Op
            && !east_asian_parenthesis(chars[index].1)
        || previous == C::Cp
            && !east_asian_parenthesis(chars[index - 1].1)
            && matches!(next, C::Al | C::Hl | C::Nu)
    {
        return false;
    }
    let previous_base = classes[..index]
        .iter()
        .rposition(|class| !matches!(class, C::Cm | C::Zwj))
        .unwrap_or(index - 1);
    let previous_is_aksara = previous == C::Ak || chars[previous_base].1 == '\u{25cc}';
    let next_is_aksara = aksara(classes[index], chars[index].1);
    if previous == C::Ap && (next_is_aksara || next == C::As)
        || (previous_is_aksara || previous == C::As) && matches!(next, C::Vf | C::Vi)
        || previous == C::Vi
            && next_is_aksara
            && classes[..previous_base]
                .iter()
                .rposition(|class| !matches!(class, C::Cm | C::Zwj))
                .is_some_and(|base| {
                    aksara(effective_class(classes, base), chars[base].1)
                        || effective_class(classes, base) == C::As
                })
        || (previous_is_aksara || previous == C::As)
            && (next_is_aksara || next == C::As)
            && classes.get(index + 1) == Some(&C::Vf)
    {
        return false;
    }
    if previous == C::Ri && next == C::Ri && regional_indicators % 2 == 1 {
        return false;
    }
    (previous != C::Eb && !is_unassigned_extended_pictographic(chars[previous_base].1))
        || next != C::Em
}

#[cfg(test)]
mod tests {
    use super::line_breaks;

    #[test]
    fn mandatory_breaks_and_combining_marks() {
        assert_eq!(line_breaks("a\r\nb"), vec![3, 4]);
        assert_eq!(line_breaks("a\u{0301}b"), vec![4]);
    }

    #[test]
    fn spaces_cjk_and_regional_indicators_break() {
        assert_eq!(line_breaks("a b"), vec![2, 3]);
        assert_eq!(line_breaks("漢字"), vec![3, 6]);
        assert_eq!(line_breaks("🇳🇱🇧🇪"), vec![8, 16]);
    }
}

#[cfg(test)]
#[path = "linebreak_conformance_tests.rs"]
mod conformance;
