"""TERMINUS panic/unwrap ratchet gate."""

from __future__ import annotations

from pathlib import Path
import re

from _toml_compat import load_toml

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("gis", "vector", "labels", "py_functions", "terrain")
TOKEN_PATTERNS = {
    "panic": re.compile(r"panic!\("),
    "unwrap": re.compile(r"\.unwrap\(\)"),
    "expect": re.compile(r"\.expect\("),
}


def _production_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    skip = False
    depth = 0
    pending_cfg_test = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#[cfg(test)]"):
            pending_cfg_test = True
            continue
        if pending_cfg_test and re.match(r"(pub\s+)?mod\s+tests\b", stripped):
            skip = True
            pending_cfg_test = False
            depth = line.count("{") - line.count("}")
            if depth <= 0:
                depth = 1
            continue
        if pending_cfg_test and stripped and not stripped.startswith("#"):
            pending_cfg_test = False
        if skip:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                skip = False
                depth = 0
            continue
        out.append(line)
    return "\n".join(out)


def _is_test_fixture_path(path: Path) -> bool:
    text = path.as_posix()
    return (
        "/tests/" in text
        or text.endswith("/tests.rs")
        or text.endswith("_tests.rs")
        or "/gsub_tests/" in text
        or "/gpos_tests" in text
        or "/bidi_conformance_tests" in text
        or "/linebreak_conformance_tests" in text
    )


def _counts() -> dict[str, dict[str, int]]:
    result = {module: {"panic": 0, "unwrap": 0, "expect": 0} for module in MODULES}
    for module in MODULES:
        for path in (ROOT / "src" / module).rglob("*.rs"):
            if _is_test_fixture_path(path):
                continue
            text = _production_text(path)
            for token, pattern in TOKEN_PATTERNS.items():
                result[module][token] += len(pattern.findall(text))
    return result


def test_robustness_ratchet_counts_do_not_increase():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    current = _counts()
    entries = {entry["module"]: entry for entry in ratchet["modules"]}
    offenders = []
    for module, counts in current.items():
        expected = entries[module]
        for token, actual in counts.items():
            limit = int(expected[token])
            if actual > limit:
                offenders.append(f"{module}.{token}: {actual} > ratchet {limit}")
    assert offenders == [], "TERMINUS robustness ratchet increased:\n" + "\n".join(offenders)


def test_robustness_ratchet_records_required_burndown():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    meta = ratchet.get("meta", ratchet)
    before = int(meta["step0_reachable_panic_unwrap"])
    after = sum(int(entry["panic"]) + int(entry["unwrap"]) for entry in ratchet["modules"])
    required_pct = float(meta["required_panic_unwrap_reduction_pct"])
    actual_pct = 100.0 * (before - after) / before
    assert actual_pct >= required_pct
