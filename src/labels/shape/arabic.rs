use crate::labels::unicode::{joining_type, JoiningType};
use ttf_parser::Tag;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Form {
    None,
    Isolated,
    Initial,
    Medial,
    Final,
}

impl Form {
    pub fn feature(self) -> Option<Tag> {
        let tag = match self {
            Self::None => return None,
            Self::Isolated => b"isol",
            Self::Initial => b"init",
            Self::Medial => b"medi",
            Self::Final => b"fina",
        };
        Some(Tag::from_bytes(tag))
    }
}

pub fn required_features() -> [Tag; 2] {
    [Tag::from_bytes(b"ccmp"), Tag::from_bytes(b"rlig")]
}

fn joins_previous(kind: JoiningType) -> bool {
    matches!(
        kind,
        JoiningType::RightJoining | JoiningType::DualJoining | JoiningType::JoinCausing
    )
}

fn joins_next(kind: JoiningType) -> bool {
    matches!(
        kind,
        JoiningType::LeftJoining | JoiningType::DualJoining | JoiningType::JoinCausing
    )
}

pub fn arabic_features(text: &str) -> Vec<Form> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .iter()
        .enumerate()
        .map(|(index, &ch)| {
            let kind = joining_type(ch);
            if kind == JoiningType::Transparent {
                return Form::None;
            }
            let previous = chars[..index]
                .iter()
                .rev()
                .map(|ch| joining_type(*ch))
                .find(|kind| *kind != JoiningType::Transparent);
            let next = chars[index + 1..]
                .iter()
                .map(|ch| joining_type(*ch))
                .find(|kind| *kind != JoiningType::Transparent);
            let joins_before = previous.is_some_and(joins_next) && joins_previous(kind);
            let joins_after = joins_next(kind) && next.is_some_and(joins_previous);
            match (joins_before, joins_after) {
                (true, true) => Form::Medial,
                (true, false) => Form::Final,
                (false, true) => Form::Initial,
                (false, false) => Form::Isolated,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{arabic_features, required_features, Form};

    #[test]
    fn joining_skips_transparent_marks() {
        assert_eq!(
            arabic_features("ب\u{064E}ب"),
            vec![Form::Initial, Form::None, Form::Final]
        );
    }

    #[test]
    fn right_joining_letter_ends_the_connection() {
        assert_eq!(
            arabic_features("باب"),
            vec![Form::Initial, Form::Final, Form::Isolated]
        );
    }

    #[test]
    fn left_joining_letter_connects_to_following_neighbor() {
        assert_eq!(
            arabic_features("\u{A872}\u{0640}"),
            vec![Form::Initial, Form::Final]
        );
    }

    #[test]
    fn lam_alef_enables_required_ligature_forms() {
        assert_eq!(arabic_features("لا"), vec![Form::Initial, Form::Final]);
        assert!(required_features().contains(&ttf_parser::Tag::from_bytes(b"rlig")));
    }
}
