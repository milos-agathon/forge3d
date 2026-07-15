#!/usr/bin/env python3
"""Generate the small UAX #9/#14 tables owned by the shaping module."""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path


SOURCE_HASHES = {
    "BidiBrackets.txt": "dadbaf38a0d0246e5b805bf8725cb81b7c621f93d030595635f5ba2c2f179428",
    "emoji-data.txt": "2cb2bb9455cda83e8481541ecf5b6dfda66a3bb89efa3fa7c5297eccf607b72b",
}


def verify_sources(bidi_brackets: Path, emoji_data: Path) -> None:
    for path in (bidi_brackets, emoji_data):
        expected = SOURCE_HASHES[path.name]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
            )


def rust_char(value: int) -> str:
    character = chr(value)
    if character in "()[]{}":
        return repr(character)
    return f"'\\u{{{value:04x}}}'"


def generate_brackets(source: Path) -> str:
    data = source.read_bytes()
    openings = []
    for line in data.decode("utf-8").splitlines():
        fields = line.split("#", 1)[0].strip().split(";")
        if len(fields) == 3 and fields[2].strip() == "o":
            opening = int(fields[0].strip(), 16)
            closing = int(fields[1].strip(), 16)
            skeleton = ord(unicodedata.normalize("NFC", chr(opening)))
            openings.append((opening, closing, skeleton))
    digest = hashlib.sha256(data).hexdigest()
    entries = "\n".join(
        f"    ({rust_char(opening)}, {rust_char(closing)}, {rust_char(skeleton)}),"
        for opening, closing, skeleton in openings
    )
    return f"""// Generated from Unicode 17.0.0 BidiBrackets.txt, SHA-256
// {digest}. Each entry is
// (opening, closing, normalized opening skeleton).
const PAIRS: &[(char, char, char)] = &[
{entries}
];

pub(super) fn bracket(character: char) -> Option<(char, bool)> {{
    PAIRS.iter().find_map(|&(open, close, skeleton)| {{
        (character == open || character == close).then_some((skeleton, character == open))
    }})
}}
"""


def generate_emoji(source: Path) -> str:
    data = source.read_bytes()
    ranges = []
    for line in data.decode("utf-8").splitlines():
        if "Extended_Pictographic" not in line or "<reserved-" not in line:
            continue
        match = re.match(r"^([0-9A-F]+)(?:\.\.([0-9A-F]+))?", line)
        if not match:
            raise ValueError(f"malformed emoji-data line: {line}")
        start = int(match.group(1), 16)
        end = int(match.group(2) or match.group(1), 16)
        ranges.append((start, end))
    digest = hashlib.sha256(data).hexdigest()
    patterns = []
    for start, end in ranges:
        patterns.append(f"0x{start:x}" if start == end else f"0x{start:x}..=0x{end:x}")
    joined = "\n            | ".join(patterns)
    return f"""// Unicode 17.0 emoji-data.txt Extended_Pictographic ranges whose code points are
// reserved (General_Category=Cn). SHA-256:
// {digest}

pub(super) fn is_unassigned_extended_pictographic(character: char) -> bool {{
    matches!(
        character as u32,
        {joined}
    )
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bidi-brackets", type=Path, required=True)
    parser.add_argument("--emoji-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    verify_sources(arguments.bidi_brackets, arguments.emoji_data)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "bidi_brackets.rs").write_text(
        generate_brackets(arguments.bidi_brackets), encoding="utf-8", newline="\n"
    )
    (arguments.output_dir / "linebreak_emoji.rs").write_text(
        generate_emoji(arguments.emoji_data), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
