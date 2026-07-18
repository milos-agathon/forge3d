from __future__ import annotations

import hashlib
import importlib.util
import io
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hydrate_lfs_from_media.py"
SPEC = importlib.util.spec_from_file_location("hydrate_lfs_from_media", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pointer_parser_and_media_url_are_commit_pinned() -> None:
    payload = b"fixture bytes"
    digest = hashlib.sha256(payload).hexdigest()
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{digest}\n"
        f"size {len(payload)}\n"
    ).encode()
    assert MODULE.parse_pointer(pointer) == (digest, len(payload))
    commit = "a" * 40
    assert MODULE.media_url("owner/repo", commit, "assets/a b.tif") == (
        "https://media.githubusercontent.com/media/"
        f"owner/repo/{commit}/assets/a%20b.tif"
    )
    with pytest.raises(ValueError, match="canonical Git LFS pointer"):
        MODULE.parse_pointer(b"not a pointer")


def test_ref_must_be_the_full_checked_out_commit() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert MODULE.require_checked_out_commit(head) == head.lower()
    for invalid in ("main", head[:12], "g" * 40):
        with pytest.raises(ValueError, match="full 40- or 64-digit"):
            MODULE.require_checked_out_commit(invalid)
    mismatch = ("0" if head[0] != "0" else "1") + head[1:]
    with pytest.raises(ValueError, match="does not match checked-out HEAD"):
        MODULE.require_checked_out_commit(mismatch)


def test_download_is_atomic_and_refuses_integrity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified fixture"
    expected = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        MODULE.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload)
    )
    target = tmp_path / "fixture.bin"
    MODULE.download_verified("https://example.invalid/fixture", target, expected, len(payload))
    assert target.read_bytes() == payload

    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    bad_target = tmp_path / "bad.bin"
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        MODULE.download_verified(
            "https://example.invalid/bad", bad_target, "0" * 64, len(payload)
        )
    assert not bad_target.exists()


def test_lfs_dependent_ci_lanes_use_verified_hydration() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "lfs: true" not in workflow
    assert workflow.count("scripts/hydrate_lfs_from_media.py") == 4
    assert (
        workflow.count(
            "github.event.pull_request.head.repo.full_name || github.repository"
        )
        == 4
    )
    assert "--path assets/tif/switzerland_dem.tif" in workflow
