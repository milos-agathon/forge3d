from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_lfs_fixtures import (
    LFS_FIXTURE_PATHS,
    verify_manifest,
    write_manifest,
)


def _materialize_fixture_tree(root: Path) -> None:
    for index, relative_path in enumerate(LFS_FIXTURE_PATHS):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode())


def test_lfs_fixture_manifest_round_trip(tmp_path: Path) -> None:
    _materialize_fixture_tree(tmp_path)
    manifest = tmp_path / "target/lfs-fixtures-manifest.json"

    write_manifest(tmp_path, manifest)
    verify_manifest(tmp_path, manifest)

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert [record["path"] for record in data["files"]] == list(LFS_FIXTURE_PATHS)


def test_lfs_fixture_manifest_rejects_pointer(tmp_path: Path) -> None:
    _materialize_fixture_tree(tmp_path)
    (tmp_path / LFS_FIXTURE_PATHS[0]).write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 23\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="still a pointer"):
        write_manifest(tmp_path, tmp_path / "manifest.json")


def test_lfs_fixture_manifest_rejects_corruption(tmp_path: Path) -> None:
    _materialize_fixture_tree(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(tmp_path, manifest)

    (tmp_path / LFS_FIXTURE_PATHS[-1]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="do not match"):
        verify_manifest(tmp_path, manifest)


def test_lfs_fixture_manifest_rejects_missing_file(tmp_path: Path) -> None:
    _materialize_fixture_tree(tmp_path)
    (tmp_path / LFS_FIXTURE_PATHS[3]).unlink()

    with pytest.raises(ValueError, match="missing LFS fixture"):
        write_manifest(tmp_path, tmp_path / "manifest.json")
