//! Strict parser for the absolute M/L/Q/C/Z grammar emitted by svg_path_data.
//! This reconsumes exported coordinates, rather than reusing the source font.
use lyon_path::{math::point, Path};

pub fn parse(data: &str) -> Result<Path, String> {
    let mut builder = Path::builder();
    let mut offset = 0usize;
    let mut open = false;
    let bytes = data.as_bytes();
    while offset < bytes.len() {
        while offset < bytes.len() && (bytes[offset].is_ascii_whitespace() || bytes[offset] == b',')
        {
            offset += 1;
        }
        if offset == bytes.len() {
            break;
        }
        let command = bytes[offset];
        offset += 1;
        let count = match command {
            b'M' | b'L' => 2,
            b'Q' => 4,
            b'C' => 6,
            b'Z' => 0,
            _ => {
                return Err(format!(
                    "unsupported outline path command at byte {}",
                    offset - 1
                ))
            }
        };
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            while offset < bytes.len()
                && (bytes[offset].is_ascii_whitespace() || bytes[offset] == b',')
            {
                offset += 1;
            }
            let start = offset;
            if offset < bytes.len() && matches!(bytes[offset], b'+' | b'-') {
                offset += 1;
            }
            while offset < bytes.len() && (bytes[offset].is_ascii_digit() || bytes[offset] == b'.')
            {
                offset += 1;
            }
            if offset < bytes.len() && matches!(bytes[offset], b'e' | b'E') {
                offset += 1;
                if offset < bytes.len() && matches!(bytes[offset], b'+' | b'-') {
                    offset += 1;
                }
                while offset < bytes.len() && bytes[offset].is_ascii_digit() {
                    offset += 1;
                }
            }
            let value = data
                .get(start..offset)
                .and_then(|number| number.parse::<f32>().ok())
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("invalid outline coordinate at byte {start}"))?;
            values.push(value);
        }
        if command != b'M' && !open {
            return Err("outline contour must start with M".to_owned());
        }
        match command {
            b'M' => {
                if open {
                    builder.end(false);
                }
                builder.begin(point(values[0], values[1]));
                open = true;
            }
            b'L' => {
                builder.line_to(point(values[0], values[1]));
            }
            b'Q' => {
                builder
                    .quadratic_bezier_to(point(values[0], values[1]), point(values[2], values[3]));
            }
            b'C' => {
                builder.cubic_bezier_to(
                    point(values[0], values[1]),
                    point(values[2], values[3]),
                    point(values[4], values[5]),
                );
            }
            b'Z' => {
                builder.close();
                open = false;
            }
            _ => unreachable!(),
        }
    }
    if open {
        builder.end(false);
    }
    Ok(builder.build())
}
