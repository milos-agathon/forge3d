#!/usr/bin/env python3
"""Generate the Rust/WGSL DUPLA arithmetic mirrors from one checked source."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUST_PATH = ROOT / "src" / "core" / "dd.rs"
WGSL_PATH = ROOT / "src" / "shaders" / "includes" / "dd.wgsl"
BEGIN = "// BEGIN GENERATED DD MIRROR"
END = "// END GENERATED DD MIRROR"

LOCKSTEP_FUNCTIONS = (
    "two_sum",
    "quick_two_sum",
    "split_scaled",
    "two_prod_split",
    "two_prod_fma",
    "dd_renorm",
    "dd_add",
    "dd_sub",
    "dd_mul",
    "dd_reciprocal_refine",
    "dd_div",
    "inverse_sqrt_residual",
    "dd_inverse_sqrt_refine",
    "dd_sqrt",
    "dd_dot3",
    "dd_length3",
    "dd_sub_vec3",
)

RUST_TEMPLATE_PATH = ROOT / "scripts" / "dd.rs.in"
WGSL_TEMPLATE_PATH = ROOT / "scripts" / "dd.wgsl.in"


def generated_wgsl() -> str:
    return WGSL_TEMPLATE_PATH.read_text(encoding="utf-8")


def generated_rust(current: str) -> str:
    before, found, rest = current.partition(BEGIN)
    if not found:
        raise SystemExit(f"missing {BEGIN} in {RUST_PATH}")
    _, found, after = rest.partition(END)
    if not found:
        raise SystemExit(f"missing {END} in {RUST_PATH}")
    template = RUST_TEMPLATE_PATH.read_text(encoding="utf-8").rstrip("\n")
    return before + template + after


def rustfmt(source: str) -> str:
    result = subprocess.run(
        ["rustfmt", "--edition", "2021"],
        input=source,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr)
    return result.stdout


def function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{name}\s*\(", source)
    if match is None:
        raise SystemExit(f"missing lockstep function {name}")
    start = source.index("{", match.end())
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise SystemExit(f"unterminated lockstep function {name}")


def normalized_body(body: str) -> str:
    normalized = re.sub(r"//.*", "", body)
    normalized = re.sub(r"debug_assert!\([^;]*;", "", normalized, flags=re.DOTALL)
    normalized = normalized.replace("dd_barrier", "barrier").replace("det_barrier", "barrier")
    normalized = normalized.replace("DD::ZERO", "DD(0.0, 0.0)")
    normalized = re.sub(r"(\w+)\.abs\(\)", r"abs(\1)", normalized)
    normalized = normalized.replace("f32::abs", "abs")
    normalized = re.sub(r"(\w+)\.mul_add\(", r"fma(\1, ", normalized)
    normalized = re.sub(r"f32::from_bits\(", "from_bits(", normalized)
    normalized = re.sub(r"(\w+)\.to_bits\(\)", r"to_bits(\1)", normalized)
    normalized = re.sub(r"bitcast\s*<\s*f32\s*>", "from_bits", normalized)
    normalized = re.sub(r"bitcast\s*<\s*u32\s*>", "to_bits", normalized)
    normalized = normalized.replace("_", "")
    normalized = re.sub(r"\b(0x[0-9a-fA-F]+)(?:u32|u)\b", r"\1", normalized)
    return normalized


def operation_trace(body: str) -> list[str]:
    """Reduce syntax to ordered calls/operators shared by Rust and WGSL."""
    normalized = normalized_body(body)
    normalized = re.sub(r"\.abs\(\)", " abs()", normalized)
    ignored = {"if", "return", "DD", "DDVec3", "abs", "from_bits", "to_bits", "bitcast"}
    tokens: list[tuple[int, str]] = []
    for match in re.finditer(r"([A-Za-z_]\w*(?:::\w+)?)(?:<[^>]+>)?\s*\(", normalized):
        name = match.group(1).split("::")[-1]
        if name not in ignored:
            tokens.append((match.start(), name))
    for match in re.finditer(r"==|!=|<=|>=|&&|\|\||[+*/<>-]", normalized):
        tokens.append((match.start(), match.group(0)))
    return [token for _, token in sorted(tokens)]


def structural_trace(body: str) -> list[str]:
    """Preserve operands and statement order while erasing language syntax."""
    normalized = normalized_body(body)
    normalized = re.sub(r"\b(?:hi|lo|x|y|z)\s*:", "", normalized)
    tokens = re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||[+*/<>-]", normalized)
    ignored = {"let", "mut", "var", "return", "u", "u32"}
    return [token.rstrip("u") for token in tokens if token not in ignored]


def validate_lockstep(rust: str, wgsl: str) -> None:
    mismatches = []
    for name in LOCKSTEP_FUNCTIONS:
        rust_trace = operation_trace(function_body(rust, name))
        wgsl_trace = operation_trace(function_body(wgsl, name))
        if rust_trace != wgsl_trace:
            mismatches.append(f"{name}: Rust {rust_trace!r} != WGSL {wgsl_trace!r}")
        rust_structure = structural_trace(function_body(rust, name))
        wgsl_structure = structural_trace(function_body(wgsl, name))
        if rust_structure != wgsl_structure:
            mismatches.append(
                f"{name}: Rust structure {rust_structure!r} != WGSL {wgsl_structure!r}"
            )
    if mismatches:
        raise SystemExit("DD operation-order mismatch:\n" + "\n".join(mismatches))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rust = rustfmt(generated_rust(RUST_PATH.read_text(encoding="utf-8")))
    wgsl = generated_wgsl()
    validate_lockstep(rust, wgsl)
    expected = {RUST_PATH: rust, WGSL_PATH: wgsl}
    if args.check:
        stale = [path for path, text in expected.items() if not path.exists() or path.read_text(encoding="utf-8") != text]
        if stale:
            print("stale generated DD files: " + ", ".join(str(path.relative_to(ROOT)) for path in stale))
            return 1
        print("generated DD mirrors are current; lockstep operation order verified")
        return 0
    for path, text in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
