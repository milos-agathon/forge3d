use super::line_breaks;
use std::fs;
use std::path::PathBuf;

#[test]
fn unicode_line_break_conformance() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/unicode/LineBreakTest.txt");
    let input = fs::read_to_string(path).unwrap();
    let mut count = 0usize;
    for source in input.lines().filter(|line| line.starts_with(['×', '÷'])) {
        // The public API uses UAX #14's default AI -> AL resolution, so exclude only
        // the alternative AI -> ID fixture variant.
        if source.contains("(AI_EastAsian)") {
            continue;
        }
        let tokens: Vec<_> = source
            .split('#')
            .next()
            .unwrap()
            .split_whitespace()
            .collect();
        let mut text = String::new();
        let mut expected = Vec::new();
        for pair in tokens[1..].chunks_exact(2) {
            let character = char::from_u32(u32::from_str_radix(pair[0], 16).unwrap()).unwrap();
            text.push(character);
            if pair[1] == "÷" {
                expected.push(text.len());
            }
        }
        assert_eq!(line_breaks(&text), expected, "{source}");
        count += 1;
    }
    assert!(count >= 10_000);
    println!("LineBreakTest.txt={count} failures=0");
}
