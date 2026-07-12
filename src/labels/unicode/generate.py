#!/usr/bin/env python3
"""Generate compact Rust Unicode property tables from a pinned local UCD."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

MAX_CODEPOINT = 0x10FFFF
MISSING = re.compile(r"^#\s*@missing:\s*([^;]+);\s*([^#\s]+)")
HERE = Path(__file__).resolve().parent

BIDI_ALIASES = {
    "Arabic_Letter": "AL",
    "Arabic_Number": "AN",
    "Boundary_Neutral": "BN",
    "Common_Separator": "CS",
    "European_Number": "EN",
    "European_Separator": "ES",
    "European_Terminator": "ET",
    "First_Strong_Isolate": "FSI",
    "Left_To_Right": "L",
    "Left_To_Right_Embedding": "LRE",
    "Left_To_Right_Isolate": "LRI",
    "Left_To_Right_Override": "LRO",
    "Nonspacing_Mark": "NSM",
    "Other_Neutral": "ON",
    "Paragraph_Separator": "B",
    "Pop_Directional_Format": "PDF",
    "Pop_Directional_Isolate": "PDI",
    "Right_To_Left": "R",
    "Right_To_Left_Embedding": "RLE",
    "Right_To_Left_Isolate": "RLI",
    "Right_To_Left_Override": "RLO",
    "Segment_Separator": "S",
    "White_Space": "WS",
}

JOINING_NAMES = {
    "U": "NonJoining",
    "R": "RightJoining",
    "L": "LeftJoining",
    "D": "DualJoining",
    "C": "JoinCausing",
    "T": "Transparent",
}
JOINING_ALIASES = {
    **JOINING_NAMES,
    "Non_Joining": "NonJoining",
    "Right_Joining": "RightJoining",
    "Left_Joining": "LeftJoining",
    "Dual_Joining": "DualJoining",
    "Join_Causing": "JoinCausing",
    "Transparent": "Transparent",
}


def codepoint_range(text: str) -> tuple[int, int]:
    parts = text.strip().split("..")
    start = int(parts[0], 16)
    return start, int(parts[-1], 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(args: argparse.Namespace) -> None:
    recorded_version = (HERE / "UCD_VERSION").read_text(encoding="ascii").strip()
    if args.version != recorded_version:
        raise ValueError(
            f"Unicode version mismatch: expected {recorded_version}, got {args.version}"
        )
    expected = {}
    for line in (HERE / "SOURCES.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    inputs = {
        "Scripts.txt": args.scripts,
        "ArabicShaping.txt": args.arabic_shaping,
        "extracted/DerivedJoiningType.txt": args.joining_type,
        "extracted/DerivedBidiClass.txt": args.bidi_class,
        "BidiMirroring.txt": args.bidi_mirroring,
        "LineBreak.txt": args.line_break,
    }
    if inputs.keys() != expected.keys():
        raise ValueError("SOURCES.sha256 does not name the required Unicode inputs")
    for name, path in inputs.items():
        actual = sha256(path)
        if actual != expected[name]:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected[name]}, got {actual}"
            )


def property_ranges(
    path: Path,
    field: int,
    default: str,
    aliases: dict[str, str] | None = None,
) -> list[tuple[int, int, str]]:
    aliases = aliases or {}
    values = [aliases.get(default, default)] * (MAX_CODEPOINT + 1)
    explicit: list[tuple[int, int, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        missing = MISSING.match(raw)
        if missing:
            start, end = codepoint_range(missing.group(1))
            value = aliases.get(missing.group(2), missing.group(2))
            values[start : end + 1] = [value] * (end - start + 1)
            continue
        data = raw.split("#", 1)[0].strip()
        if not data:
            continue
        fields = [part.strip() for part in data.split(";")]
        start, end = codepoint_range(fields[0])
        explicit.append((start, end, aliases.get(fields[field], fields[field])))
    for start, end, value in explicit:
        values[start : end + 1] = [value] * (end - start + 1)

    ranges: list[tuple[int, int, str]] = []
    start = 0
    current = values[0]
    for codepoint in range(1, MAX_CODEPOINT + 1):
        if values[codepoint] != current:
            ranges.append((start, codepoint - 1, current))
            start = codepoint
            current = values[codepoint]
    ranges.append((start, MAX_CODEPOINT, current))
    return ranges


def rust_variant(value: str) -> str:
    if len(value) <= 3 and value.isupper():
        return value[0] + value[1:].lower()
    return "".join(part[:1].upper() + part[1:] for part in value.split("_"))


def emit_enum(name: str, values: set[str]) -> list[str]:
    return [
        "#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]",
        f"pub enum {name} {{",
        *(f"    {rust_variant(value)}," for value in sorted(values)),
        "}",
        "",
    ]


def emit_ranges(name: str, enum_name: str, ranges: list[tuple[int, int, str]]) -> list[str]:
    lines = ["#[rustfmt::skip]", f"const {name}: &[Range<{enum_name}>] = &["]
    lines.extend(
        f"    Range {{ start: 0x{start:X}, end: 0x{end:X}, value: {enum_name}::{rust_variant(value)} }},"
        for start, end, value in ranges
    )
    lines.extend(["];", ""])
    return lines


def mirror_pairs(path: Path) -> list[tuple[int, int]]:
    pairs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        data = raw.split("#", 1)[0].strip()
        if data:
            left, right = (part.strip() for part in data.split(";")[:2])
            pairs.append((int(left, 16), int(right, 16)))
    return sorted(pairs)


def value_at(ranges: list[tuple[int, int, str]], codepoint: int) -> str:
    low, high = 0, len(ranges)
    while low < high:
        middle = (low + high) // 2
        if ranges[middle][0] <= codepoint:
            low = middle + 1
        else:
            high = middle
    start, end, value = ranges[low - 1]
    assert start <= codepoint <= end
    return value


def validate_arabic_shaping(path: Path, joining: list[tuple[int, int, str]]) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        data = raw.split("#", 1)[0].strip()
        if not data:
            continue
        fields = [part.strip() for part in data.split(";")]
        start, end = codepoint_range(fields[0])
        expected = JOINING_ALIASES[fields[2]]
        for codepoint in range(start, end + 1):
            if value_at(joining, codepoint) != expected:
                raise ValueError(f"joining type mismatch at U+{codepoint:04X}")


def generate(args: argparse.Namespace) -> str:
    verify_sources(args)
    scripts = property_ranges(args.scripts, 1, "Unknown")
    joining = property_ranges(args.joining_type, 1, "Non_Joining", JOINING_ALIASES)
    validate_arabic_shaping(args.arabic_shaping, joining)
    bidi = property_ranges(args.bidi_class, 1, "L", BIDI_ALIASES)
    line_break = property_ranges(args.line_break, 1, "XX")
    mirrors = mirror_pairs(args.bidi_mirroring)

    lines = [
        "// @generated by src/labels/unicode/generate.py; do not edit.",
        f"// Unicode {args.version}",
        "",
    ]
    for enum_name, ranges in (
        ("Script", scripts),
        ("JoiningType", joining),
        ("BidiClass", bidi),
        ("LineBreakClass", line_break),
    ):
        lines.extend(emit_enum(enum_name, {value for _, _, value in ranges}))
    lines.extend(
        [
            "#[derive(Clone, Copy)]",
            "struct Range<T> {",
            "    start: u32,",
            "    end: u32,",
            "    value: T,",
            "}",
            "",
            "fn lookup<T: Copy>(ranges: &[Range<T>], codepoint: u32) -> T {",
            "    let index = ranges.partition_point(|range| range.start <= codepoint) - 1;",
            "    let range = ranges[index];",
            "    debug_assert!(codepoint <= range.end);",
            "    range.value",
            "}",
            "",
        ]
    )
    lines.extend(emit_ranges("SCRIPT_RANGES", "Script", scripts))
    lines.extend(emit_ranges("JOINING_RANGES", "JoiningType", joining))
    lines.extend(emit_ranges("BIDI_RANGES", "BidiClass", bidi))
    lines.extend(emit_ranges("LINE_BREAK_RANGES", "LineBreakClass", line_break))
    lines.extend(["#[rustfmt::skip]", "const MIRRORING: &[(u32, u32)] = &["])
    lines.extend(f"    (0x{left:X}, 0x{right:X})," for left, right in mirrors)
    lines.extend(
        [
            "];",
            "",
            "pub fn script(value: char) -> Script {",
            "    lookup(SCRIPT_RANGES, value as u32)",
            "}",
            "pub fn joining_type(value: char) -> JoiningType {",
            "    lookup(JOINING_RANGES, value as u32)",
            "}",
            "pub fn bidi_class(value: char) -> BidiClass {",
            "    lookup(BIDI_RANGES, value as u32)",
            "}",
            "pub fn line_break_class(value: char) -> LineBreakClass {",
            "    lookup(LINE_BREAK_RANGES, value as u32)",
            "}",
            "pub fn mirrored(value: char) -> Option<char> {",
            "    let codepoint = value as u32;",
            "    MIRRORING",
            "        .binary_search_by_key(&codepoint, |pair| pair.0)",
            "        .ok()",
            "        .and_then(|index| char::from_u32(MIRRORING[index].1))",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--arabic-shaping", type=Path, required=True)
    parser.add_argument("--joining-type", type=Path, required=True)
    parser.add_argument("--bidi-class", type=Path, required=True)
    parser.add_argument("--bidi-mirroring", type=Path, required=True)
    parser.add_argument("--line-break", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = generate(arguments)
    arguments.output.write_text(output, encoding="utf-8", newline="\n")
    print(hashlib.sha256(output.encode()).hexdigest())
