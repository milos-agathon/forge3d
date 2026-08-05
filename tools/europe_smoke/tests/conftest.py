# tools/europe_smoke/tests/conftest.py
"""Offline by default, and honest about the artifacts a clean checkout lacks.

Two gates live here.

**Network/slow.** Tests marked ``network`` or ``slow`` are skipped unless the
run opts in with ``-m``.

**Artifacts.** §10.2 of the spec says the manifest is the contract for every
input the repository cannot carry: the compiled engine (git-ignored), the OSM
land shapefile (git-ignored), the GHSL raster (off-tree on D:), and the Natural
Earth zip the build downloads for itself. A machine that has never provisioned
those is *un-provisioned*, not *broken* -- but before this file existed the
tests that need them FAILED, so a clean checkout produced 19 red tests and the
suite read as broken. The fixtures below turn exactly that state into a skip
whose reason names the missing file and the manifest entry that declares it.

Three rules keep the gate from becoming a place where failures hide:

1. **It is keyed to the filesystem, never to an opt-out.** The decision comes
   from :func:`provenance.check_artifact` -- the same function gate 16 and
   ``probe --check`` use -- run against the live manifest at call time. There
   is no environment variable, no marker and no config switch that can force a
   skip: provision the artifact and the test runs, delete it and the test
   skips. That is also why a skip cannot silently become permanent.

2. **Only absence skips.** ``missing`` (a required artifact that is not on
   disk) and ``pending`` (an acquired artifact the build has not created yet)
   are the *only* statuses that skip. ``size`` and ``mismatch`` mean the file is
   there and WRONG -- a substituted or corrupted artifact -- and those must
   stay loud, so the test runs and fails. Skipping them would be exactly the
   silent substitution §10.2 exists to prevent.

3. **An unknown name is an error, not a skip.** Asking for an artifact the
   manifest does not declare raises, so a renamed entry cannot quietly disable
   a test.
"""
from __future__ import annotations

import pytest

from tools.europe_smoke import provenance

#: The artifact statuses that mean "this machine simply does not have the file".
#: Deliberately excludes SIZE and MISMATCH -- see rule 2 in the module docstring.
ABSENT_STATUSES = frozenset({provenance.MISSING, provenance.PENDING})

#: Every entry a test may depend on, with the one-line reason it exists. Used
#: only to make the skip message say what the operator has to go and get.
_WHY = {
    "engine": "the compiled smoke engine (.pyc; the source was deleted)",
    "ghsl": "the GHSL global population raster",
    "osm_land": "the OSM simplified land polygons shapefile",
    "natural_earth": "the Natural Earth countries zip",
}


def artifact_finding(name: str) -> provenance.Finding:
    """The live :class:`provenance.Finding` for one manifest entry.

    Reads ``manifest.toml`` through :func:`provenance.load_manifest` on every
    call, so a test that redirects ``provenance.MANIFEST_PATH`` is honoured and
    nothing is cached across a monkeypatch.
    """
    manifest = provenance.load_manifest()
    entry = manifest.get(name)
    if entry is None:
        raise RuntimeError(
            f"no manifest entry named {name!r} in {provenance.MANIFEST_PATH}; "
            f"an artifact gate must name an entry that exists, or renaming an "
            f"entry would silently disable the tests that depend on it"
        )
    return provenance.check_artifact(name, entry)


def require_artifacts(*names: str) -> dict[str, provenance.Finding]:
    """Skip the calling test if any named artifact is absent from this machine.

    Returns the findings for the artifacts that ARE present, so a test can
    assert against them. Raises rather than skips on an unknown name, and does
    NOT skip on ``size``/``mismatch`` -- a wrong artifact must fail loudly.
    """
    findings = {}
    for name in names:
        finding = artifact_finding(name)
        if finding.status in ABSENT_STATUSES:
            pytest.skip(
                f"external artifact [{name}] is not provisioned on this machine: "
                f"{_WHY.get(name, 'declared in tools/europe_smoke/manifest.toml')}. "
                f"{finding.detail} (manifest status: {finding.status}). "
                f"See the [{name}] entry in tools/europe_smoke/manifest.toml for "
                f"its upstream and licence."
            )
        findings[name] = finding
    return findings


@pytest.fixture
def engine_artifact() -> provenance.Finding:
    """The compiled engine .pyc. Skips when it is not on this machine."""
    return require_artifacts("engine")["engine"]


@pytest.fixture
def ghsl_artifact() -> provenance.Finding:
    """The GHSL population raster. Skips when it is not on this machine."""
    return require_artifacts("ghsl")["ghsl"]


@pytest.fixture
def osm_land_artifact() -> provenance.Finding:
    """The verified OSM land-polygon ZIP. Skips when it is not on this machine."""
    return require_artifacts("osm_land")["osm_land"]


@pytest.fixture
def required_artifacts() -> dict[str, provenance.Finding]:
    """Every ``class = "required"`` entry, for tests that run the whole gate.

    ``provenance.check_all()`` and ``probe --check`` are pass/fail over ALL of
    them, so a test that asserts they pass needs all of them present. The set
    is read from the manifest rather than hardcoded, so adding a required
    artifact extends the gate automatically.
    """
    names = [name for name, entry in provenance.load_manifest().items()
             if entry.get("class") == provenance.REQUIRED]
    if not names:
        raise RuntimeError(
            f"{provenance.MANIFEST_PATH} declares no required artifacts; the "
            f"required_artifacts gate would be vacuous"
        )
    return require_artifacts(*names)


def pytest_configure(config):
    config.addinivalue_line("markers", "network: hits a live API; opt in with -m network")
    config.addinivalue_line("markers", "slow: takes minutes; opt in with -m slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="needs -m network or -m slow")
    for item in items:
        if {"network", "slow"} & set(item.keywords):
            item.add_marker(skip)
