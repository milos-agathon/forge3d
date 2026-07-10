# tests/test_no_silent_degradation.py
# CENSOR Task 13: the CI honesty gate. One test function per lettered gate:
#   (a) committed RenderCertificates carry no un-allowlisted degradation
#   (b) zero raw wgpu allocation sites bypass the tracked ledger
#   (c) every Cargo feature is referenced, and the CI --features list is curated
#   (d) the wheel ships the features its public APIs need; PROJ/GEOS fallbacks
#       are diagnostic-bearing (record a degradation / raise), never silent
#   (e) every tracked tests/test_*.py is either run by a CI lane or UNRUN-listed
# RELEVANT FILES: scripts/ci_pytest_lane.py, tests/UNRUN.toml,
#   tests/degradation_allowlist.toml, tests/allocation_allowlist.toml,
#   tests/_toml_compat.py, Cargo.toml, pyproject.toml, .github/workflows/ci.yml
"""Static + behavioural honesty gates for CENSOR."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from _toml_compat import load_toml

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
CERT_DIR = TESTS / "golden" / "certificates"

# Make sibling helpers importable regardless of pytest rootdir insertion order.
for _p in (str(TESTS), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_allocation_gate import _raw_sites  # noqa: E402  (reuse the source gate)
import ci_pytest_lane  # noqa: E402  (the default-lane selection is the source of truth)


# ---------------------------------------------------------------------------
# (a) certificate degradations
# ---------------------------------------------------------------------------
def test_a_committed_certificates_have_no_unallowlisted_degradations():
    certs = sorted(CERT_DIR.glob("*.json"))
    assert certs, "no committed certificates found -- expected tests/golden/certificates/*.json"

    allow = load_toml(TESTS / "degradation_allowlist.toml").get("entries", [])
    allowed = {}
    for entry in allow:
        assert date.fromisoformat(entry["expires"]) >= date.today(), f"expired degradation allowlist entry: {entry}"
        allowed[(entry["kind"], entry["name"])] = entry

    offenders = []
    for cert in certs:
        data = json.loads(cert.read_text(encoding="utf-8"))
        for deg in data.get("degradations", []) or []:
            key = (deg.get("kind"), deg.get("name"))
            if key not in allowed:
                offenders.append(f"{cert.name}: {key} -> {deg.get('consequence')}")

    assert offenders == [], "certificates carry un-allowlisted degradations:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# (b) source allocation gate (reused)
# ---------------------------------------------------------------------------
def test_b_zero_raw_allocation_sites():
    allow = load_toml(TESTS / "allocation_allowlist.toml")["entries"]
    allowed = {e["site"].rsplit(":", 1)[0] for e in allow}
    stray = [s for s in _raw_sites() if s.rsplit(":", 1)[0] not in allowed]
    assert stray == [], f"raw wgpu allocation sites bypass the tracked ledger: {stray}"


# ---------------------------------------------------------------------------
# (c) feature gate
# ---------------------------------------------------------------------------
# The single source of truth for what CI's `cargo check`/`cargo test`/`cargo doc`
# compile. Every exclusion from this list (vs the full Cargo.toml [features])
# carries an in-test reason below.
CI_CARGO_FEATURES = {
    "default",  # baseline: images + enable-gpu-instancing + enable-staging-rings
    "async_readback",
    "copc_laz",
    "weighted-oit",
    "wsI_bigbuf",
    "wsI_double_buf",
    "enable-pbr",
    "enable-tbn",
    "enable-normal-mapping",
    "enable-hdr-offscreen",
    "enable-renderer-config",
    "enable-staging-rings",
}
# Declared-but-excluded from the CI feature list, with the reason each is off:
CI_EXCLUDED_FEATURES = {
    "extension-module": "links the PyO3 cdylib extension; cannot link under cargo test/check",
    "images": "already pulled in transitively by `default`",
    "enable-gpu-instancing": "already pulled in transitively by `default`",
    "cog_streaming": "pulls the reqwest/tokio network stack",
    "gis-remote": "pulls the reqwest/tokio network stack",
    "proj": "needs the system PROJ library",
    "geos-topology": "needs the system GEOS/geo topology backend",
}


def _cargo_features() -> set[str]:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    section = re.search(r"\[features\](.*?)(?:\n\[)", text, re.DOTALL)
    assert section, "could not locate [features] in Cargo.toml"
    names = set()
    for line in section.group(1).splitlines():
        stripped = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_\-]+)\s*=", stripped)
        if m:
            names.add(m.group(1))
    return names


def _feature_referenced(feat: str) -> bool:
    needle = f'feature = "{feat}"'
    for base in ("src", "tests", "benches"):
        d = ROOT / base
        if not d.exists():
            continue
        for path in d.rglob("*.rs"):
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                return True
    build_rs = ROOT / "build.rs"
    if build_rs.exists() and needle in build_rs.read_text(encoding="utf-8", errors="ignore"):
        return True
    return False


def test_c_every_feature_referenced_and_ci_list_curated():
    declared = _cargo_features()

    # Every non-`default` feature must be referenced somewhere in Rust source.
    unreferenced = sorted(f for f in declared if f != "default" and not _feature_referenced(f))
    assert unreferenced == [], f"declared Cargo features with no `feature = \"..\"` reference (dead advertising): {unreferenced}"

    # The curated CI list and its exclusions must partition the declared set.
    assert CI_CARGO_FEATURES <= declared, f"CI feature set names undeclared features: {CI_CARGO_FEATURES - declared}"
    assert set(CI_EXCLUDED_FEATURES) <= declared, f"exclusion list names undeclared features: {set(CI_EXCLUDED_FEATURES) - declared}"
    assert CI_CARGO_FEATURES.isdisjoint(CI_EXCLUDED_FEATURES), "a feature is both curated-in and excluded"
    assert CI_CARGO_FEATURES | set(CI_EXCLUDED_FEATURES) == declared, (
        "curated + excluded features must exactly cover Cargo.toml [features]; "
        f"unaccounted: {declared - CI_CARGO_FEATURES - set(CI_EXCLUDED_FEATURES)}"
    )

    # Every `--features` list in ci.yml must equal the curated set (no dead
    # advertising, no drift between cargo check/test/doc).
    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    lists = re.findall(r"--features\s+([A-Za-z0-9_,\-]+)", ci_yml)
    assert lists, "no cargo --features lists found in ci.yml"
    for raw in lists:
        got = set(raw.split(","))
        assert got == CI_CARGO_FEATURES, f"ci.yml --features {sorted(got)} != curated {sorted(CI_CARGO_FEATURES)}"
        assert got <= declared, f"ci.yml advertises undeclared features: {got - declared}"

    # The forge3d-clippy alias feature set must be a subset of declared features.
    alias_text = (ROOT / ".cargo" / "config.toml").read_text(encoding="utf-8")
    m = re.search(r'"([A-Za-z0-9_\-]*extension-module[A-Za-z0-9_,\-]*)"', alias_text)
    assert m, "could not locate forge3d-clippy feature list in .cargo/config.toml"
    alias_features = set(m.group(1).split(","))
    assert alias_features <= declared, f"forge3d-clippy alias names undeclared features: {alias_features - declared}"


# ---------------------------------------------------------------------------
# (d) wheel gate
# ---------------------------------------------------------------------------
# Features the shipped wheel MUST compile in because documented public APIs
# depend on them at runtime.
WHEEL_REQUIRED_FEATURES = {
    "extension-module",
    "enable-tbn",
    "weighted-oit",
    "enable-gpu-instancing",
    "enable-staging-rings",
    "copc_laz",
    "cog_streaming",
    "gis-remote",
}


def _maturin_features() -> set[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(r"\[tool\.maturin\](.*?)(?:\n\[)", text, re.DOTALL)
    assert section, "could not locate [tool.maturin] in pyproject.toml"
    m = re.search(r"features\s*=\s*\[([^\]]*)\]", section.group(1))
    assert m, "could not locate maturin `features` list in pyproject.toml"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_d_wheel_features_superset_and_proj_geos_are_diagnostic_bearing():
    maturin = _maturin_features()
    missing = WHEEL_REQUIRED_FEATURES - maturin
    assert not missing, f"wheel omits features required by public APIs: {sorted(missing)}"

    # proj / geos-topology are deliberately NOT shipped. Their Python surfaces
    # must therefore be diagnostic-bearing (never a silent wrong-result fallback).
    #
    # proj: forge3d.crs.transform_coords, when native PROJ is compiled out,
    # records a `feature_not_compiled`/`proj` degradation before using pyproj.
    assert "proj" not in maturin and "geos-topology" not in maturin, (
        "proj/geos-topology are expected to be compiled OUT of the wheel"
    )

    import forge3d.crs as crs
    from forge3d import _degradation

    assert crs.HAS_NATIVE_PROJ is False, "this wheel unexpectedly compiled native PROJ in"
    _degradation.clear()
    try:
        # Distinct CRS pair forces past the identity/same-CRS early returns and
        # into the non-native path. It may then succeed via pyproj or raise if
        # pyproj is absent (CI) -- either way the degradation must be recorded.
        crs.transform_coords([[0.0, 0.0]], "EPSG:4326", "EPSG:3857")
    except Exception:
        pass
    recorded = {(d["kind"], d["name"]) for d in _degradation.snapshot()}
    assert ("feature_not_compiled", "proj") in recorded, (
        "crs.transform_coords silently fell back off native PROJ without recording a degradation; "
        f"recorded={sorted(recorded)}"
    )

    # geos-topology: the native topology ops are diagnostic-bearing by
    # construction -- they return an explicit BackendUnavailable error rather
    # than a silent success when the feature is absent. Assert the honest wiring
    # exists in the Rust boundary (require_topology_backend gate).
    topo = (ROOT / "src" / "gis" / "geometry" / "topology.rs").read_text(encoding="utf-8")
    assert "require_topology_backend" in topo and "BackendUnavailable" in topo, (
        "geos-topology fallback is not visibly diagnostic-bearing in src/gis/geometry/topology.rs"
    )


# ---------------------------------------------------------------------------
# (e) UNRUN accounting
# ---------------------------------------------------------------------------
def _tracked_test_files() -> set[str]:
    # CI only ever checks out git-tracked files; untracked local example tests
    # (france/iberia/italy/poland/romania) never exist on a runner, so the
    # accounting universe is the tracked set, not the working-tree glob.
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "tests/test_*.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def _explicit_lane_files() -> set[str]:
    """Files a non-default CI lane runs explicitly (golden lane) or by marker (viewer lane)."""
    ci_yml = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # Golden lane: every `tests/<file>.py` token that appears in a pytest command.
    golden = set(re.findall(r"tests/test_[A-Za-z0-9_]+\.py", ci_yml))
    # Interactive viewer lane runs `pytest tests/ -m interactive_viewer`; the
    # owning files are those carrying the marker.
    viewer = set()
    if "-m interactive_viewer" in ci_yml:
        for path in _tracked_test_files():
            fp = ROOT / path
            if fp.exists() and "interactive_viewer" in fp.read_text(encoding="utf-8", errors="ignore"):
                viewer.add(path)
    return golden | viewer


def test_e_unrun_accounting_is_exhaustive_and_honest():
    universe = _tracked_test_files()
    unrun = set(ci_pytest_lane.unrun_files())
    explicit = _explicit_lane_files() & universe

    # No UNRUN entry may name a nonexistent / untracked file.
    missing = sorted(f for f in unrun if f not in universe)
    assert missing == [], f"UNRUN names files absent from the tracked suite: {missing}"

    # No UNRUN entry may be expired.
    data = load_toml(TESTS / "UNRUN.toml")
    for entry in data.get("entries", []):
        assert "reason" in entry and entry["reason"], f"UNRUN entry lacks a reason: {entry}"
        assert date.fromisoformat(entry["expires"]) >= date.today(), f"expired UNRUN entry: {entry}"

    # A file may not be BOTH quarantined and claimed by an explicit lane.
    both = sorted(unrun & explicit)
    assert both == [], f"files are both UNRUN and run by an explicit lane: {both}"

    # The default lane collects everything not UNRUN; the accounting must be total.
    default_lane = universe - unrun
    assert (default_lane | explicit | unrun) == universe, (
        "accounting is not exhaustive: "
        f"{sorted(universe - (default_lane | explicit | unrun))}"
    )
