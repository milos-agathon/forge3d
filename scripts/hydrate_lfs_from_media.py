#!/usr/bin/env python3
"""Hydrate tracked Git LFS pointers from GitHub's immutable media endpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path


POINTER_RE = re.compile(
    rb"\Aversion https://git-lfs.github.com/spec/v1\r?\n"
    rb"oid sha256:([0-9a-f]{64})\r?\n"
    rb"size ([0-9]+)\r?\n?\Z"
)
COMMIT_OID_RE = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")


def parse_pointer(raw: bytes) -> tuple[str, int]:
    match = POINTER_RE.fullmatch(raw)
    if match is None:
        raise ValueError("index entry is not a canonical Git LFS pointer")
    return match.group(1).decode("ascii"), int(match.group(2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_lfs_paths() -> list[str]:
    result = subprocess.run(
        ["git", "lfs", "ls-files", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def indexed_pointer(path: str) -> tuple[str, int]:
    result = subprocess.run(
        ["git", "show", f":{path}"], check=True, capture_output=True
    )
    return parse_pointer(result.stdout)


def media_url(repository: str, ref: str, path: str) -> str:
    encoded_path = urllib.parse.quote(path.replace("\\", "/"), safe="/")
    encoded_ref = urllib.parse.quote(ref, safe="")
    return (
        "https://media.githubusercontent.com/media/"
        f"{repository}/{encoded_ref}/{encoded_path}"
    )


def require_checked_out_commit(ref: str) -> str:
    if COMMIT_OID_RE.fullmatch(ref) is None:
        raise ValueError("--ref must be a full 40- or 64-digit hexadecimal commit OID")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if ref.lower() != head.lower():
        raise ValueError(f"--ref {ref} does not match checked-out HEAD {head}")
    return head.lower()


def download_verified(url: str, target: Path, expected_sha: str, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        temporary: Path | None = None
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "forge3d-ci-lfs-hydrator/1"}
            )
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=180) as response:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=f".{target.name}.", delete=False
                ) as output:
                    temporary = Path(output.name)
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            actual_sha = digest.hexdigest()
            if size != expected_size or actual_sha != expected_sha:
                raise RuntimeError(
                    f"integrity mismatch for {target}: size {size}/{expected_size}, "
                    f"sha256 {actual_sha}/{expected_sha}"
                )
            os.replace(temporary, target)
            return
        except Exception as error:  # noqa: BLE001 - retry network and integrity failures
            last_error = error
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"failed to hydrate {target} after 3 attempts: {last_error}")


def hydrate(repository: str, ref: str, requested: list[str]) -> None:
    ref = require_checked_out_commit(ref)
    tracked = set(tracked_lfs_paths())
    paths = requested or sorted(tracked)
    unknown = sorted(set(paths) - tracked)
    if unknown:
        raise SystemExit(f"requested paths are not tracked by Git LFS: {unknown}")

    downloaded = 0
    reused = 0
    for relative in paths:
        expected_sha, expected_size = indexed_pointer(relative)
        target = Path(relative)
        if (
            target.is_file()
            and target.stat().st_size == expected_size
            and sha256_file(target) == expected_sha
        ):
            print(f"LFS verified: {relative}")
            reused += 1
            continue
        url = media_url(repository, ref, relative)
        print(f"LFS hydrate: {relative} ({expected_size} bytes)")
        download_verified(url, target, expected_sha, expected_size)
        downloaded += 1
    print(f"LFS hydration complete: {downloaded} downloaded, {reused} reused")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, help="GitHub owner/repository")
    parser.add_argument("--ref", required=True, help="immutable commit SHA")
    parser.add_argument("--path", action="append", default=[], help="hydrate one LFS path")
    args = parser.parse_args()
    hydrate(args.repository, args.ref, args.path)


if __name__ == "__main__":
    main()
