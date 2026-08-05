# tools/europe_smoke/tests/test_provenance.py
# Replaces the Task 2 Step 1 test file wholesale. 36 tests.
# Task 2 Step 4 expectation becomes: PASS, 36 passed.
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.europe_smoke import config, provenance


# ------------------------------------------------------------------ manifest
def test_manifest_loads_and_pins_the_engine():
    m = provenance.load_manifest()
    assert m["engine"]["bytes"] == 293372
    assert m["engine"]["sha256"] == (
        "20f302223ed9282965921b80a076ccff208640e8105ae8a9214bf2be816c48bc"
    )


def test_every_manifest_entry_declares_a_known_class():
    m = provenance.load_manifest()
    assert {name: e["class"] for name, e in m.items()} == {
        "engine": "required",
        "ghsl": "required",
        "osm_land": "required",
        "natural_earth": "acquired",
        "cams": "remote",
        "firms": "remote",
    }


def test_acquired_entries_name_who_creates_them():
    m = provenance.load_manifest()
    for name, entry in m.items():
        if entry["class"] == "acquired":
            assert entry.get("acquired_by"), name


def test_remote_entries_name_the_credential_that_gates_them():
    m = provenance.load_manifest()
    creds = {f.name for f in provenance.check_credentials()}
    for name, entry in m.items():
        if entry["class"] == "remote":
            assert entry.get("credential") in creds, name


def test_manifest_rejects_an_entry_with_no_class(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text('[engine]\npath = "x"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="class"):
        provenance.load_manifest(p)


def test_manifest_rejects_a_remote_entry_that_carries_a_path(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text('[cams]\nclass = "remote"\npath = "x"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not carry a path"):
        provenance.load_manifest(p)


def test_manifest_paths_resolve_through_the_config_placeholders():
    m = provenance.load_manifest()
    assert provenance._resolve(m["engine"]["path"]) == config.ENGINE_PYC
    assert provenance._resolve(m["osm_land"]["path"]) == config.OSM_LAND_SHP
    ne = provenance._resolve(m["natural_earth"]["path"])
    assert ne == config.CACHE_DIR / "ne" / "ne_10m_admin_0_countries.zip"
    # the manifest must not hardcode the build drive a second time
    assert "europe_smoke/build" not in m["natural_earth"]["path"]


# ------------------------------------------------------------------ hashing
def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"forge3d")
    assert provenance.sha256_file(p) == hashlib.sha256(b"forge3d").hexdigest()


# ------------------------------------------------------------------ statuses
def test_level_map_is_total_and_is_the_only_severity_opinion():
    for status in (provenance.VERIFIED, provenance.PRESENT, provenance.REMOTE,
                   provenance.UNPINNED, provenance.PENDING, provenance.MISSING,
                   provenance.SIZE, provenance.MISMATCH, provenance.ABSENT):
        assert provenance.level_of(status) in {"ok", "warn", "fail"}
    assert provenance.level_of(provenance.UNPINNED) == "warn"
    assert provenance.level_of(provenance.PENDING) == "warn"
    assert provenance.level_of(provenance.MISMATCH) == "fail"


def test_engine_artifact_verifies_against_the_pinned_hash(engine_artifact):
    m = provenance.load_manifest()
    findings = provenance.check_artifacts(m)
    engine = next(f for f in findings if f.name == "engine")
    assert engine.status == provenance.VERIFIED, engine.detail
    assert engine.ok


def test_missing_required_artifact_reports_not_ok_rather_than_raising():
    m = provenance.load_manifest()
    m["ghsl"]["path"] = "D:/definitely/not/here.tif"
    ghsl = next(f for f in provenance.check_artifacts(m) if f.name == "ghsl")
    assert ghsl.status == provenance.MISSING
    assert ghsl.blocking
    assert "missing" in ghsl.detail.lower()


def test_wrong_hash_reports_not_ok(tmp_path):
    p = tmp_path / "fake.bin"
    p.write_bytes(b"not the engine")
    m = provenance.load_manifest()
    m["engine"]["path"] = str(p)
    m["engine"]["bytes"] = len(b"not the engine")
    engine = next(f for f in provenance.check_artifacts(m) if f.name == "engine")
    assert engine.status == provenance.MISMATCH
    assert engine.blocking
    assert "sha256" in engine.detail.lower()


def test_wrong_size_is_blocking_and_never_reaches_the_hash(tmp_path):
    p = tmp_path / "short.bin"
    p.write_bytes(b"x")
    m = provenance.load_manifest()
    m["engine"]["path"] = str(p)
    engine = next(f for f in provenance.check_artifacts(m) if f.name == "engine")
    assert engine.status == provenance.SIZE
    assert engine.blocking


# --------------------------------------------- the three states, individually
def test_unpinned_is_a_third_state_not_a_silent_pass(tmp_path):
    p = tmp_path / "unpinned.bin"
    p.write_bytes(b"forge3d")
    m = provenance.load_manifest()
    m["engine"].update(path=str(p), sha256="", bytes=len(b"forge3d"))
    f = next(x for x in provenance.check_artifacts(m) if x.name == "engine")
    assert f.status == provenance.UNPINNED
    assert f.level == "warn"            # visible
    assert f.ok                          # does not block the default gate
    assert f.sha256 == hashlib.sha256(b"forge3d").hexdigest()   # recorded
    assert f.sha256 in f.detail          # printed


def test_unpinned_becomes_a_failure_under_strict(monkeypatch, tmp_path,
                                                 ghsl_artifact, osm_land_artifact):
    p = tmp_path / "unpinned.bin"
    p.write_bytes(b"forge3d")
    m = provenance.load_manifest()
    m["engine"].update(path=str(p), sha256="", bytes=len(b"forge3d"))
    monkeypatch.setattr(provenance, "load_manifest", lambda *a, **k: m)
    ok_default, findings = provenance.check_all()
    ok_strict, strict = provenance.check_all(strict=True)
    assert ok_default is True and ok_strict is False
    # the status survives the promotion -- strict raises severity, not the label
    hit = next(f for f in strict if f.name == "engine")
    assert hit.status == provenance.UNPINNED and hit.level == "fail"


def test_pending_acquired_artifact_warns_and_does_not_block():
    """Defect (b): natural_earth does not exist until the derive stage
    downloads it.  The gate must not deadlock the only step that could
    create the file."""
    ne = next(f for f in provenance.check_artifacts(provenance.load_manifest())
              if f.name == "natural_earth")
    if ne.status == provenance.PENDING:
        assert ne.level == "warn"
        assert ne.ok
        assert "boundary.load_countries" in ne.detail
    else:  # already acquired on this machine
        assert ne.status in {provenance.UNPINNED, provenance.VERIFIED}


def test_acquired_artifact_is_verified_like_any_other_once_present(tmp_path,
                                                                   engine_artifact):
    src = config.ENGINE_PYC
    dst = tmp_path / "ne.zip"
    shutil.copyfile(src, dst)
    m = provenance.load_manifest()
    m["natural_earth"].update(path=str(dst), sha256="deadbeef")
    f = next(x for x in provenance.check_artifacts(m) if x.name == "natural_earth")
    assert f.status == provenance.MISMATCH and f.blocking


def test_remote_entries_have_no_local_artifact_and_never_block():
    findings = provenance.check_artifacts(provenance.load_manifest())
    for name in ("cams", "firms"):
        f = next(x for x in findings if x.name == name)
        assert f.status == provenance.REMOTE
        assert f.level == "ok"
        assert f.sha256 is None


# ------------------------------------------------------------- the whole gate
def test_check_all_passes_on_this_machine_with_natural_earth_unacquired(
        required_artifacts):
    ok, findings = provenance.check_all()
    blocking = [f for f in findings if f.blocking]
    assert ok, provenance.format_findings(blocking)


def test_summarise_names_the_unpinned_and_pending_entries():
    _, findings = provenance.check_all()
    s = provenance.summarise(findings)
    assert set(s["counts"]) == {"ok", "warn", "fail"}
    assert set(s["warning"]) == set(s["unpinned"] + s["pending"])
    assert all("status" in d and "level" in d for d in s["findings"])


def test_format_findings_shows_the_status_word_not_just_a_tick(required_artifacts):
    _, findings = provenance.check_all()
    lines = provenance.format_findings(findings)
    assert any(": verified:" in ln for ln in lines)
    for ln in lines:
        assert ln.startswith(("[ok  ]", "[warn]", "[FAIL]"))


def test_credential_check_never_leaks_contents():
    findings = provenance.check_credentials()
    names = {f.name for f in findings}
    assert names == {"cdsapirc", "firms_key"}
    for f in findings:
        assert "key" not in f.detail.lower() or "firms_key" in f.detail
        assert len(f.detail) < 200


# ------------------------------------------------------------------ pinning
def test_propose_pins_returns_only_unpinned_observed_hashes(tmp_path):
    p = tmp_path / "unpinned.bin"
    p.write_bytes(b"forge3d")
    m = provenance.load_manifest()
    m["engine"].update(path=str(p), sha256="", bytes=len(b"forge3d"))
    pins = provenance.propose_pins(provenance.check_artifacts(m))
    assert pins == {"engine": hashlib.sha256(b"forge3d").hexdigest()}


def test_write_proposed_pins_lands_outside_the_tracked_tree(tmp_path):
    out = provenance.write_proposed_pins({"natural_earth": "ab" * 32},
                                         tmp_path / "proposed_pins.toml")
    assert out.name == "proposed_pins.toml"
    assert out.read_text().count("ab" * 32) == 1
    assert provenance.MANIFEST_PATH.parent not in out.parents


def test_apply_pins_writes_only_empty_slots(tmp_path):
    p = tmp_path / "manifest.toml"
    shutil.copyfile(provenance.MANIFEST_PATH, p)
    digest = "cd" * 32
    assert provenance.apply_pins({"natural_earth": digest}, p) == ["natural_earth"]
    assert provenance.load_manifest(p)["natural_earth"]["sha256"] == digest
    # everything else survived unchanged
    assert provenance.load_manifest(p)["engine"]["sha256"] == (
        provenance.load_manifest()["engine"]["sha256"])


def test_apply_pins_refuses_to_rotate_an_existing_pin(tmp_path):
    p = tmp_path / "manifest.toml"
    shutil.copyfile(provenance.MANIFEST_PATH, p)
    with pytest.raises(RuntimeError, match="already pinned"):
        provenance.apply_pins({"engine": "ef" * 32}, p)


def test_apply_pins_refuses_a_remote_entry(tmp_path):
    p = tmp_path / "manifest.toml"
    shutil.copyfile(provenance.MANIFEST_PATH, p)
    with pytest.raises(RuntimeError, match="remote"):
        provenance.apply_pins({"cams": "ef" * 32}, p)


# ----------------------------------------------------- the pinning controls
def _sandbox(tmp_path, ne_bytes=b"pretend-natural-earth", ghsl_bytes=None):
    """A manifest copy whose natural_earth entry is PRESENT and UNPINNED.

    Load-bearing. On a healthy machine every entry is either pinned or
    `pending`, so `propose_pins()` is empty and probe.main never enters the
    branch that could write a hash -- a control test run against that state
    passes no matter what the code does. Arming an unpinned artifact is what
    makes the three tests below mean something. Pass ``ghsl_bytes`` to also
    substitute the GHSL raster and drive a required artifact to `mismatch`.
    """
    p = tmp_path / "manifest.toml"
    shutil.copyfile(provenance.MANIFEST_PATH, p)
    art = tmp_path / "ne.zip"
    art.write_bytes(ne_bytes)
    # MERGE NOTE: the manifest spells this path with {build}, not {cache} (see
    # manifest.toml's merge note). The substitution is asserted rather than
    # assumed -- a silently-missed replace would leave the sandbox unarmed and
    # every control below vacuous, which is the exact failure mode these tests
    # exist to close.
    original = p.read_text(encoding="utf-8")
    needle = 'path = "{build}/cache/ne/ne_10m_admin_0_countries.zip"'
    assert needle in original, "sandbox anchor no longer matches manifest.toml"
    text = original.replace(needle, f'path = "{art.as_posix()}"')
    if ghsl_bytes is not None:
        bad = tmp_path / "substituted.tif"
        bad.write_bytes(ghsl_bytes)
        text = text.replace(
            'path = "D:/ghsl-population/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"',
            f'path = "{bad.as_posix()}"').replace("bytes = 384285897\n", "")
    p.write_text(text, encoding="utf-8")
    return p, art


def test_no_automated_entry_point_mutates_the_manifest(tmp_path, monkeypatch,
                                                       required_artifacts):
    """The control: check_all / probe --check must never self-pin.

    The sandbox leaves an artifact pinnable so the write path is LIVE while
    --check runs; the assertion on propose_pins is what keeps this test from
    being vacuous.
    """
    p, _ = _sandbox(tmp_path)
    before = p.read_bytes()
    monkeypatch.setattr(provenance, "MANIFEST_PATH", p)
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    from tools.europe_smoke import probe

    _, findings = provenance.check_all()
    assert provenance.propose_pins(findings), "sandbox failed to arm the pin path"
    assert probe.main(["--check"]) == 0
    assert p.read_bytes() == before
    # the observed hash went to the proposal file, never into the manifest
    assert (tmp_path / "proposed_pins.toml").exists()


def test_write_pins_refuses_while_a_required_artifact_is_blocking(
        tmp_path, monkeypatch, capsys):
    """A pin must never be installed on a machine whose artifacts are wrong.

    Without the guard, `--write-pins` pins natural_earth and exits 0 while
    GHSL is a substituted file -- an operator command that ignores the very
    gate it belongs to.
    """
    p, _ = _sandbox(tmp_path, ghsl_bytes=b"substituted")
    before = p.read_bytes()
    monkeypatch.setattr(provenance, "MANIFEST_PATH", p)
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    from tools.europe_smoke import probe

    assert probe.main(["--write-pins"]) == 1
    assert p.read_bytes() == before
    assert "refusing to pin" in capsys.readouterr().out


def test_write_pins_survives_its_own_strict_promotion(tmp_path, monkeypatch,
                                                      required_artifacts):
    """--strict turns `unpinned` into a failure; that must not block the one
    command whose job is to clear it. The refusal keys off status, not level."""
    p, art = _sandbox(tmp_path)
    monkeypatch.setattr(provenance, "MANIFEST_PATH", p)
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    from tools.europe_smoke import probe

    assert probe.main(["--write-pins", "--strict"]) == 0
    assert provenance.load_manifest(p)["natural_earth"]["sha256"] == (
        provenance.sha256_file(art))


def test_a_pinned_entry_is_never_offered_for_pinning_again(tmp_path, monkeypatch,
                                                           capsys,
                                                           required_artifacts):
    """Why the CLI cannot rotate a pin: propose_pins stops offering the entry.

    apply_pins() also refuses, but the CLI never reaches it -- so the guard
    that actually runs in the operator flow is this one. The same mechanism is
    why `--write-pins` can never repin a MISMATCHed artifact to make it green.
    """
    p, _ = _sandbox(tmp_path)
    monkeypatch.setattr(provenance, "MANIFEST_PATH", p)
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    from tools.europe_smoke import probe

    assert probe.main(["--write-pins"]) == 0
    pinned = p.read_bytes()
    assert provenance.propose_pins(provenance.check_all()[1]) == {}
    assert probe.main(["--write-pins"]) == 0
    assert "nothing to pin." in capsys.readouterr().out
    assert p.read_bytes() == pinned
    assert probe.main(["--check", "--strict"]) == 0


# ------------------------------------------------------------------ CLI
def test_probe_check_exits_zero_with_a_pending_acquired_artifact(tmp_path, monkeypatch,
                                                                capsys,
                                                                required_artifacts):
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    from tools.europe_smoke import probe
    assert probe.main(["--check"]) == 0
    assert "provenance: PASS" in capsys.readouterr().out


def test_probe_check_strict_exits_one_while_anything_is_unpinned(tmp_path, monkeypatch,
                                                                 capsys):
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    from tools.europe_smoke import probe
    _, findings = provenance.check_all()
    if not [f for f in findings if f.level == "warn"]:
        pytest.skip("nothing unpinned or pending on this machine")
    assert probe.main(["--check", "--strict"]) == 1
    assert "provenance: FAIL" in capsys.readouterr().out


def test_probe_check_exits_one_on_a_missing_required_artifact(tmp_path, monkeypatch,
                                                              capsys):
    m = provenance.load_manifest()
    m["ghsl"]["path"] = "D:/definitely/not/here.tif"
    monkeypatch.setattr(provenance, "load_manifest", lambda *a, **k: m)
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    from tools.europe_smoke import probe
    assert probe.main(["--check"]) == 1
    assert "[FAIL] ghsl" in capsys.readouterr().out


# ------------------------------------------------------------------ report
def test_build_report_roundtrips(tmp_path):
    r = provenance.BuildReport()
    r.set("grid", [136, 241])
    out = tmp_path / "report.json"
    r.write(out)
    assert json.loads(out.read_text())["grid"] == [136, 241]


def test_build_report_carries_the_status_of_every_artifact(tmp_path):
    _, findings = provenance.check_all()
    r = provenance.BuildReport()
    r.set("provenance", provenance.summarise(findings))
    data = json.loads(r.write(tmp_path / "report.json").read_text())
    names = {f["name"] for f in data["provenance"]["findings"]}
    assert {"engine", "ghsl", "osm_land", "natural_earth", "cams", "firms",
            "cdsapirc", "firms_key"} == names


def test_build_token_expands_so_gate16_can_bootstrap(tmp_path, monkeypatch):
    """The literal it replaced duplicated BUILD_DIR, so gate 16 tested a string."""
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    assert provenance._resolve("{build}/cache/ne/x.zip") == tmp_path / "cache" / "ne" / "x.zip"
    assert provenance._resolve("examples/e.pyc") == config.REPO_ROOT / "examples" / "e.pyc"
    assert provenance._resolve("D:/abs/x.tif") == Path("D:/abs/x.tif")


def test_the_manifest_uses_the_token_for_artifacts_the_build_creates(tmp_path,
                                                                     monkeypatch):
    m = provenance.load_manifest()
    assert "{build}" in m["natural_earth"]["path"], (
        "natural_earth is downloaded into BUILD_DIR; hardcoding it makes gate 16 "
        "fail on any machine whose BUILD_DIR differs")
    monkeypatch.setattr(config, "BUILD_DIR", tmp_path)
    p = tmp_path / "cache" / "ne" / "ne_10m_admin_0_countries.zip"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"zip")
    ne = next(f for f in provenance.check_artifacts(m) if f.name == "natural_earth")
    assert ne.ok, ne.detail


# ------------------------------------------------- the artifact gate itself
#
# The tests above that need an external artifact take an `engine_artifact` /
# `ghsl_artifact` / `required_artifacts` fixture from conftest, which skips when
# the file is not on this machine. That gate is itself a control and needs its
# own test: a gate that skipped unconditionally, or one that swallowed a
# CORRUPTED artifact, would turn this suite green while proving nothing.
#
# It is exercised in a subprocess against a one-entry manifest, so all three
# outcomes can be produced on ANY machine -- provisioned or not -- without
# touching the real manifest or the real artifacts:
#
#   entry points at a path that does not exist   -> the test SKIPS
#   entry points at a file that is there         -> the test RUNS
#   entry points at a file with the WRONG hash   -> the test RUNS (and would
#                                                   fail on the artifact, which
#                                                   is the whole point)
#
# A subprocess rather than an in-process pytest run because the outcome we are
# asserting on IS pytest's own skip/pass accounting, and the gate has to be
# reached through real fixture resolution to mean anything.


def _gate_probe(tmp_path, name, artifact_path, sha256=""):
    """Run one throwaway test whose only fixture is the real `engine_artifact`.

    ``artifact_path`` is written into a one-entry manifest that the throwaway
    conftest points ``provenance.MANIFEST_PATH`` at. Returns the CompletedProcess.
    """
    import os
    import subprocess
    import sys
    import textwrap

    d = tmp_path / name
    d.mkdir()
    manifest = d / "manifest.toml"
    manifest.write_text(
        '[engine]\nclass = "required"\n'
        f'path = "{Path(artifact_path).as_posix()}"\n'
        f'sha256 = "{sha256}"\n',
        encoding="utf-8",
    )
    (d / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (d / "conftest.py").write_text(textwrap.dedent(f"""
        from pathlib import Path

        import pytest

        from tools.europe_smoke import provenance
        # the REAL gate, imported rather than reimplemented
        from tools.europe_smoke.tests.conftest import engine_artifact  # noqa: F401

        @pytest.fixture(autouse=True)
        def _one_entry_manifest(monkeypatch):
            monkeypatch.setattr(provenance, "MANIFEST_PATH",
                                Path(r"{manifest}"))
    """), encoding="utf-8")
    (d / "test_gated.py").write_text(textwrap.dedent("""
        def test_needs_the_engine(engine_artifact):
            print("STATUS=" + engine_artifact.status)
    """), encoding="utf-8")

    env = dict(os.environ, PYTHONPATH=str(config.REPO_ROOT))
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(d), "-q", "-rs", "-s",
         "-p", "no:cacheprovider", "-c", str(d / "pytest.ini"),
         "--rootdir", str(d)],
        capture_output=True, text=True, timeout=600, env=env, cwd=str(d),
    )


def test_a_missing_artifact_makes_the_dependent_test_skip_not_fail(tmp_path):
    """Requirement: an un-provisioned machine reports un-provisioned."""
    out = _gate_probe(tmp_path, "absent", tmp_path / "no" / "such" / "engine.pyc")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "1 skipped" in out.stdout, out.stdout
    assert "0 failed" not in out.stdout and "failed" not in out.stdout, out.stdout
    # the reason has to name the entry and the file, or it is not actionable
    assert "external artifact [engine]" in out.stdout, out.stdout
    assert "manifest.toml" in out.stdout, out.stdout
    assert "engine.pyc" in out.stdout, out.stdout


def test_a_present_artifact_lets_the_dependent_test_run(tmp_path):
    """The other half: the gate cannot silently become a permanent skip.

    Nothing but the file's presence is different from the test above, so a gate
    that skipped for any other reason -- a marker, an env var, a stale cached
    verdict -- fails here.
    """
    art = tmp_path / "present_engine.pyc"
    art.write_bytes(b"pretend-bytecode")
    out = _gate_probe(tmp_path, "present", art)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "1 passed" in out.stdout, out.stdout
    assert "skipped" not in out.stdout, out.stdout
    assert f"STATUS={provenance.UNPINNED}" in out.stdout, out.stdout


def test_a_corrupted_artifact_is_not_absence_and_must_not_skip(tmp_path):
    """A substituted artifact is the failure §10.2 exists to catch.

    If the gate skipped on `mismatch` too, a machine with a corrupted GHSL or a
    swapped engine .pyc would report a tidy green run with a skip, which is
    exactly the silent substitution the manifest is supposed to prevent.
    """
    art = tmp_path / "corrupt_engine.pyc"
    art.write_bytes(b"substituted")
    out = _gate_probe(tmp_path, "corrupt", art, sha256="ab" * 32)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "1 passed" in out.stdout, out.stdout
    assert "skipped" not in out.stdout, out.stdout
    assert f"STATUS={provenance.MISMATCH}" in out.stdout, out.stdout


def test_the_gate_refuses_to_name_an_entry_the_manifest_does_not_declare():
    """A renamed manifest entry must break loudly, not disable a test."""
    from tools.europe_smoke.tests import conftest as gate

    with pytest.raises(RuntimeError, match="no manifest entry named"):
        gate.artifact_finding("not_an_entry")


def test_only_absence_is_a_skippable_status():
    """The skip set, stated once and asserted against the status table."""
    from tools.europe_smoke.tests import conftest as gate

    assert gate.ABSENT_STATUSES == {provenance.MISSING, provenance.PENDING}
    for status in (provenance.SIZE, provenance.MISMATCH):
        assert status not in gate.ABSENT_STATUSES
        assert provenance.level_of(status) == provenance.FAIL
