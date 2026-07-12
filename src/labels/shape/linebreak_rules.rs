use crate::labels::unicode::LineBreakClass as C;

pub(super) fn aksara(class: C, character: char) -> bool {
    class == C::Ak || character == '\u{25cc}'
}

pub(super) fn east_asian_parenthesis(character: char) -> bool {
    matches!(
        character as u32,
        0x2329..=0x232a | 0x3008..=0x3011 | 0x3014..=0x301b | 0xfe59..=0xfe5e | 0xff08 | 0xff09
    )
}

pub(super) fn east_asian_width(character: char) -> bool {
    matches!(
        character as u32,
        0x1100..=0x115f
            | 0x2329..=0x232a
            | 0x2e80..=0xa4cf
            | 0xac00..=0xd7a3
            | 0xf900..=0xfaff
            | 0xfe10..=0xfe19
            | 0xfe30..=0xfe6f
            | 0xff01..=0xff60
            | 0xffe0..=0xffe6
            | 0x1f300..=0x1faff
            | 0x20000..=0x3fffd
    )
}

pub(super) fn effective_class(classes: &[C], position: usize) -> C {
    if !matches!(classes[position], C::Cm | C::Zwj) {
        return classes[position];
    }
    classes[..position]
        .iter()
        .copied()
        .rfind(|class| !matches!(class, C::Cm | C::Zwj))
        .filter(|class| !matches!(class, C::Bk | C::Cr | C::Lf | C::Nl | C::Sp | C::Zw))
        .unwrap_or(C::Al)
}

pub(super) fn opening_quote(character: char) -> bool {
    matches!(
        character,
        '\u{00ab}' | '\u{2018}' | '\u{201b}' | '\u{201c}' | '\u{201f}' | '\u{2039}'
    )
}

pub(super) fn closing_quote(character: char) -> bool {
    matches!(character, '\u{00bb}' | '\u{2019}' | '\u{201d}' | '\u{203a}')
}

pub(super) fn numeric(classes: &[C], index: usize) -> bool {
    let previous = effective_class(classes, index - 1);
    let next = classes[index];
    if matches!(previous, C::Pr | C::Po | C::Hy | C::Is) && next == C::Nu {
        return true;
    }
    if matches!(previous, C::Pr | C::Po)
        && next == C::Op
        && (classes.get(index + 1) == Some(&C::Nu)
            || classes.get(index + 1) == Some(&C::Is) && classes.get(index + 2) == Some(&C::Nu))
    {
        return true;
    }
    if matches!(next, C::Nu | C::Po | C::Pr) {
        let start = classes[..index]
            .iter()
            .rposition(|class| !matches!(class, C::Sy | C::Is | C::Cm | C::Zwj));
        if start.is_some_and(|position| effective_class(classes, position) == C::Nu) {
            return true;
        }
    }
    if matches!(previous, C::Cl | C::Cp) && matches!(next, C::Po | C::Pr) {
        let start = classes[..index - 1]
            .iter()
            .rposition(|class| !matches!(class, C::Sy | C::Is | C::Cm | C::Zwj));
        return start.is_some_and(|position| effective_class(classes, position) == C::Nu);
    }
    false
}

pub(super) fn hangul(previous: C, next: C) -> bool {
    previous == C::Jl && matches!(next, C::Jl | C::Jv | C::H2 | C::H3)
        || matches!(previous, C::Jv | C::H2) && matches!(next, C::Jv | C::Jt)
        || matches!(previous, C::Jt | C::H3) && next == C::Jt
}
