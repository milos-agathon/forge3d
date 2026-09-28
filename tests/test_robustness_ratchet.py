"""TERMINUS panic/unwrap ratchet gate."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

from _toml_compat import load_toml

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("gis", "vector", "labels", "py_functions", "terrain")
TOKEN_PATTERNS = {
    "panic": re.compile(r"panic!\("),
    "unwrap": re.compile(r"\.unwrap\(\)"),
    "expect": re.compile(r"\.expect\("),
}


def _production_text(path: Path) -> str:
    return _production_text_value(path.read_text(encoding="utf-8", errors="ignore"))


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


def _iter_production_sources(
    root: Path = ROOT,
    source_overrides: Mapping[str, str] | None = None,
):
    overrides = source_overrides or {}
    for module in MODULES:
        for path in (root / "src" / module).rglob("*.rs"):
            if _is_test_fixture_path(path):
                continue
            rel = path.relative_to(root).as_posix()
            raw = overrides.get(rel)
            text = _production_text(path) if raw is None else _production_text_value(raw)
            yield module, rel, text


def _production_text_value(raw: str) -> str:
    """Apply the production scanner to source text without touching the tree."""

    lines = raw.splitlines()
    out: list[str] = []
    skip = False
    depth = 0
    opened = False
    pending_cfg_test = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"#\[cfg\((?:test\)|all\(test\s*,)", stripped):
            pending_cfg_test = True
            continue
        if pending_cfg_test and re.match(
            r"(?:pub(?:\([^)]*\))?\s+)?(?:mod|fn)\s+\w+", stripped
        ):
            skip = True
            pending_cfg_test = False
            depth = line.count("{") - line.count("}")
            opened = "{" in line
            if opened and depth == 0:
                skip = False
            continue
        if pending_cfg_test and stripped and not stripped.startswith("#"):
            pending_cfg_test = False
        if skip:
            opened = opened or "{" in line
            depth += line.count("{") - line.count("}")
            if opened and depth <= 0:
                skip = False
                depth = 0
            continue
        out.append(line)
    return "\n".join(out)


def test_production_scanner_excludes_test_only_items_but_keeps_feature_code():
    source = '''
#[cfg(test)]
mod overview_seed_tests { fn helper() { Some(1).unwrap(); } }
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests { fn helper() { panic!("test only"); } }
#[cfg(test)]
pub(super) fn test_helper() { Some(1).expect("test only"); }
#[cfg(test)]
fn multiline(
    value: u32,
) {
    Some(value).unwrap();
}
#[cfg(any(test, feature = "enable-globe"))]
fn production() { Some(1).unwrap(); }
fn ordinary() { panic!("production"); }
'''
    production = _production_text_value(source)
    assert len(TOKEN_PATTERNS["unwrap"].findall(production)) == 1
    assert len(TOKEN_PATTERNS["panic"].findall(production)) == 1
    assert len(TOKEN_PATTERNS["expect"].findall(production)) == 0


def _counts(
    root: Path = ROOT,
    source_overrides: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    result = {module: {"panic": 0, "unwrap": 0, "expect": 0} for module in MODULES}
    for module, _rel, text in _iter_production_sources(root, source_overrides):
        for token, pattern in TOKEN_PATTERNS.items():
            result[module][token] += len(pattern.findall(text))
    return result


def _source_counts(
    root: Path = ROOT,
    source_overrides: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for _module, rel, text in _iter_production_sources(root, source_overrides):
        counts = {token: len(pattern.findall(text)) for token, pattern in TOKEN_PATTERNS.items()}
        if any(counts.values()):
            result[rel] = counts
    return result


def _module_offenders(current, ratchet) -> list[str]:
    entries = {entry["module"]: entry for entry in ratchet["modules"]}
    offenders = []
    for module, counts in current.items():
        expected = entries[module]
        for token, actual in counts.items():
            limit = int(expected[token])
            if actual > limit:
                offenders.append(f"{module}.{token}: {actual} > ratchet {limit}")
    return offenders


def _allowlist_offenders(current, ratchet) -> list[str]:
    entries = {entry["path"]: entry for entry in ratchet["source_allowlist"]}
    offenders = []
    missing = sorted(set(current) - set(entries))
    stale = sorted(set(entries) - set(current))
    if missing or stale:
        offenders.append(f"source allowlist paths differ: missing={missing}, stale={stale}")
    for path, counts in current.items():
        if path not in entries:
            continue
        entry = entries[path]
        if not str(entry.get("reason", "")).strip():
            offenders.append(f"{path}: missing allowlist reason")
        for token, actual in counts.items():
            limit = int(entry[token])
            if actual > limit:
                offenders.append(f"{path}.{token}: {actual} > allowlist {limit}")
    return offenders


def test_robustness_ratchet_counts_do_not_increase():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    current = _counts()
    offenders = _module_offenders(current, ratchet)
    assert offenders == [], "TERMINUS robustness ratchet increased:\n" + "\n".join(offenders)


def test_robustness_ratchet_records_required_burndown():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    meta = ratchet.get("meta", ratchet)
    before = int(meta["step0_reachable_panic_unwrap"])
    after = sum(int(entry["panic"]) + int(entry["unwrap"]) for entry in ratchet["modules"])
    required_pct = float(meta["required_panic_unwrap_reduction_pct"])
    actual_pct = 100.0 * (before - after) / before
    assert actual_pct >= required_pct


def test_robustness_ratchet_records_exact_step0_baseline():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    meta = ratchet.get("meta", ratchet)
    baseline = {
        entry["module"]: {
            token: int(entry[f"baseline_{token}"])
            for token in TOKEN_PATTERNS
        }
        for entry in ratchet["modules"]
    }
    assert set(baseline) == set(MODULES)
    assert sum(counts["panic"] + counts["unwrap"] for counts in baseline.values()) == int(
        meta["step0_reachable_panic_unwrap"]
    )
    assert sum(counts["panic"] for counts in baseline.values()) == 1
    assert sum(counts["unwrap"] for counts in baseline.values()) == 93
    assert sum(counts["expect"] for counts in baseline.values()) == 27


def test_remaining_sources_match_reasoned_allowlist():
    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    current = _source_counts()
    offenders = _allowlist_offenders(current, ratchet)
    assert offenders == [], "TERMINUS source allowlist increased:\n" + "\n".join(offenders)


def test_cog_unwrap_ablation_fails_module_and_source_allowlist_without_rewriting_tree():
    """Red proof for the exact COG failure mode found during the audit."""

    ratchet = load_toml(ROOT / "tests" / "robustness_ratchet.toml")
    rel = "src/terrain/cog/cog_reader.rs"
    original = (ROOT / rel).read_text(encoding="utf-8")
    injection = "\nfn terminus_reachable_unwrap_probe() { let _ = Some(1_u8).unwrap(); }\n"
    overrides = {rel: original + injection}

    assert _module_offenders(_counts(), ratchet) == []
    assert _allowlist_offenders(_source_counts(), ratchet) == []

    source_offenders = _allowlist_offenders(
        _source_counts(source_overrides=overrides), ratchet
    )
    # Exceed the recorded limit even when a production fix has reduced the
    # live count below it; a single new source is still caught independently.
    terrain_limit = next(
        row["unwrap"] for row in ratchet["modules"] if row["module"] == "terrain"
    )
    additional = terrain_limit - _counts()["terrain"]["unwrap"] + 1
    module_overrides = {rel: original + injection * additional}
    assert f"terrain.unwrap: {terrain_limit + 1} > ratchet {terrain_limit}" in _module_offenders(
        _counts(source_overrides=module_overrides), ratchet
    )
    assert any(
        rel in offender and ("missing=" in offender or ".unwrap: 1 > allowlist 0" in offender)
        for offender in source_offenders
    )
    assert (ROOT / rel).read_text(encoding="utf-8") == original
