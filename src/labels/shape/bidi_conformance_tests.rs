use super::{line_levels, resolve_bidi, visual_order};
use std::fs;
use std::path::PathBuf;

const REQUIRED: usize = 2_000;

#[test]
fn bidi_conformance_corpus() {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/unicode");
    let character_count = check_character_tests(&data.join("BidiCharacterTest.txt"));
    let bidi_count = check_class_tests(&data.join("BidiTest.txt"));
    assert!(character_count >= REQUIRED);
    assert!(bidi_count >= REQUIRED);
    println!("BidiCharacterTest.txt={character_count} BidiTest.txt={bidi_count} failures=0");
}

fn check_character_tests(path: &std::path::Path) -> usize {
    let input = fs::read_to_string(path).unwrap();
    let mut count = 0;
    for line in input
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
    {
        let fields: Vec<_> = line.split(';').collect();
        let text: String = fields[0]
            .split_whitespace()
            .map(|value| char::from_u32(u32::from_str_radix(value, 16).unwrap()).unwrap())
            .collect();
        let requested = match fields[1].trim() {
            "0" => Some(0),
            "1" => Some(1),
            "2" => None,
            _ => unreachable!(),
        };
        check_case(
            &text,
            requested,
            Some(fields[2].trim().parse().unwrap()),
            fields[3],
            fields[4],
            line,
        );
        count += 1;
    }
    count
}

fn check_class_tests(path: &std::path::Path) -> usize {
    let input = fs::read_to_string(path).unwrap();
    let mut levels = "";
    let mut reorder = "";
    let mut count = 0;
    for line in input.lines() {
        if let Some(value) = line.strip_prefix("@Levels:") {
            levels = value;
        } else if let Some(value) = line.strip_prefix("@Reorder:") {
            reorder = value;
        } else if !line.starts_with('#') && line.contains(';') {
            let fields: Vec<_> = line.split(';').collect();
            let text: String = fields[0].split_whitespace().map(class_character).collect();
            let bitset = u8::from_str_radix(fields[1].trim(), 16).unwrap();
            for (bit, requested) in [(1, None), (2, Some(0)), (4, Some(1))] {
                if bitset & bit != 0 {
                    check_case(&text, requested, requested, levels, reorder, line);
                    count += 1;
                }
            }
        }
    }
    count
}

fn check_case(
    text: &str,
    requested: Option<u8>,
    expected_paragraph: Option<u8>,
    expected_levels: &str,
    expected_order: &str,
    source: &str,
) {
    let paragraph = resolve_bidi(text, requested).unwrap();
    if let Some(expected) = expected_paragraph {
        assert_eq!(paragraph.paragraph_level, expected, "{source}");
    }
    let resolved = line_levels(&paragraph, 0..paragraph.levels.len());
    for (actual, expected) in resolved.iter().zip(expected_levels.split_whitespace()) {
        if expected != "x" {
            assert_eq!(*actual, expected.parse::<u8>().unwrap(), "{source}");
        }
    }
    let order = visual_order(
        &paragraph,
        std::slice::from_ref(&(0..paragraph.levels.len())),
    )
    .unwrap();
    let expected: Vec<_> = expected_order
        .split_whitespace()
        .map(|value| value.parse::<usize>().unwrap())
        .collect();
    assert_eq!(order, expected, "{source}");
}

fn class_character(class: &str) -> char {
    match class {
        "L" => 'a',
        "R" => 'א',
        "AL" => 'ا',
        "EN" => '1',
        "ES" => '+',
        "ET" => '$',
        "AN" => '١',
        "CS" => ',',
        "NSM" => '\u{0300}',
        "BN" => '\u{00ad}',
        "B" => '\u{2029}',
        "S" => '\t',
        "WS" => ' ',
        "ON" => '!',
        "LRE" => '\u{202a}',
        "LRO" => '\u{202d}',
        "RLE" => '\u{202b}',
        "RLO" => '\u{202e}',
        "PDF" => '\u{202c}',
        "LRI" => '\u{2066}',
        "RLI" => '\u{2067}',
        "FSI" => '\u{2068}',
        "PDI" => '\u{2069}',
        _ => panic!("{class}"),
    }
}
