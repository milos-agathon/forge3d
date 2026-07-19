#!/usr/bin/env python3
"""Create and verify the manifest for CI's materialized Git LFS fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LFS_FIXTURE_PATHS = (
    "assets/highres.png",
    "assets/swiss-legend.png",
    "assets/tif/Bryce_Canyon.tif",
    "assets/tif/Gore_Range_Albers_1m.tif",
    "assets/tif/Mount_Fuji_30m.tif",
    "assets/tif/dem_rainier.tif",
    "assets/tif/luxembourg_dem.tif",
    "assets/tif/switzerland_dem.tif",
    "assets/tif/switzerland_land_cover.tif",
    "data/pol_pd_2020_1km_UNadj.tif",
)

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"missing LFS fixture: {relative_path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fixture:
        prefix = fixture.read(len(_LFS_POINTER_PREFIX))
        if prefix == _LFS_POINTER_PREFIX:
            raise ValueError(f"LFS fixture is still a pointer: {relative_path}")
        digest.update(prefix)
        size += len(prefix)
        while chunk := fixture.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": relative_path,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "files": [_file_record(root, path) for path in LFS_FIXTURE_PATHS],
    }


def write_manifest(root: Path, manifest_path: Path) -> None:
    manifest = build_manifest(root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_manifest(root: Path, manifest_path: Path) -> None:
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid LFS fixture manifest: {manifest_path}") from exc

    actual = build_manifest(root)
    if expected != actual:
        raise ValueError("materialized LFS fixtures do not match the staged manifest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing the fixture paths",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-manifest", type=Path)
    action.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    try:
        if args.write_manifest is not None:
            write_manifest(root, args.write_manifest)
            print(f"staged {len(LFS_FIXTURE_PATHS)} materialized LFS fixtures")
        else:
            verify_manifest(root, args.manifest)
            print(f"verified {len(LFS_FIXTURE_PATHS)} materialized LFS fixtures")
    except ValueError as exc:
        print(f"LFS fixture verification failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
