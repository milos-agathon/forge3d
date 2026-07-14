//! Strict `Content-Range` header parsing for validating HTTP 206 responses.

/// Parse an HTTP `Content-Range: bytes START-END/TOTAL` value into
/// `(start, end, total)`, enforcing the RFC 9110 grammar
/// `range-unit SP first-pos "-" last-pos "/" complete-length` — exactly one
/// space after the unit, digits only. The `bytes` unit name is matched
/// case-insensitively per RFC 9110 §14.1. Returns `None` for any other form —
/// a missing/duplicated space, an unsatisfied range (`bytes */TOTAL`), an
/// unknown total (`.../*`), a `multipart/byteranges` response, or anything
/// otherwise unparseable — all of which callers must reject.
pub(crate) fn parse_content_range(value: &str) -> Option<(u64, u64, u64)> {
    let (unit, rest) = value.trim().split_once(' ')?;
    if !unit.eq_ignore_ascii_case("bytes") {
        return None;
    }
    let (range_part, total_part) = rest.split_once('/')?;
    let (start_str, end_str) = range_part.split_once('-')?;
    Some((
        parse_decimal_u64(start_str)?,
        parse_decimal_u64(end_str)?,
        parse_decimal_u64(total_part)?,
    ))
}

/// Strict decimal parse: ASCII digits only (no sign, whitespace, or empty
/// string, all of which `str::parse::<u64>` would otherwise tolerate or which
/// would loosen the Content-Range grammar).
fn parse_decimal_u64(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_content_range_accepts_well_formed_values() {
        assert_eq!(parse_content_range("bytes 0-99/1000"), Some((0, 99, 1000)));
        assert_eq!(
            parse_content_range("bytes 500-999/1000"),
            Some((500, 999, 1000))
        );
        // RFC 9110 §14.1: range-unit names are case-insensitive, so a valid
        // origin may reply `Bytes`/`BYTES` and must not trigger a full-object
        // fallback.
        assert_eq!(parse_content_range("Bytes 0-99/1000"), Some((0, 99, 1000)));
        assert_eq!(
            parse_content_range("BYTES 500-999/1000"),
            Some((500, 999, 1000))
        );
    }

    #[test]
    fn parse_content_range_rejects_malformed_or_unknown_forms() {
        // Missing "bytes" unit, unsatisfied range, unknown sizes, missing total,
        // wrong unit, empty, and grammar violations (missing/doubled space after
        // the unit, signed or padded numbers) — all must be rejected so callers
        // never trust them.
        for value in [
            "0-99/1000",
            "bytes */1000",
            "bytes 0-99/*",
            "bytes 0-99",
            "items 0-99/1000",
            "bytes 0/1000",
            "",
            "bytes0-99/1000",
            "bytes  0-99/1000",
            "bytes +0-99/1000",
            "bytes 0- 99/1000",
            "bytes 0-99/ 1000",
            "bytes 0-99/+1000",
        ] {
            assert_eq!(parse_content_range(value), None, "value: {value:?}");
        }
    }
}
