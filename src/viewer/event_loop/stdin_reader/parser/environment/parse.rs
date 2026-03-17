pub(super) fn parse_f32(line: &str) -> Option<f32> {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<f32>().ok())
}

pub(super) fn parse_u32(line: &str) -> Option<u32> {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u32>().ok())
}
