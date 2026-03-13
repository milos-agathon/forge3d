#!/usr/bin/env python3
"""Install the best matching forge3d wheel from a directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Mapping

try:
    from packaging.tags import Tag, sys_tags
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
except ImportError:  # pragma: no cover - exercised in fresh venvs without packaging installed
    from pip._vendor.packaging.tags import Tag, sys_tags
    from pip._vendor.packaging.utils import InvalidWheelFilename, parse_wheel_filename


def _supported_tag_ranks() -> dict[Tag, int]:
    ranks: dict[Tag, int] = {}
    for rank, tag in enumerate(sys_tags()):
        ranks.setdefault(tag, rank)
    return ranks


def _wheel_score(path: Path, supported_tag_ranks: Mapping[Tag, int]) -> int | None:
    try:
        distribution, _version, _build, tags = parse_wheel_filename(path.name)
    except InvalidWheelFilename:
        return None

    if distribution != "forge3d":
        return None

    compatible_ranks = [
        supported_tag_ranks[tag] for tag in tags if tag in supported_tag_ranks
    ]
    if not compatible_ranks:
        return None

    return min(compatible_ranks)


def main() -> int:
    wheel_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = sorted(wheel_dir.glob("forge3d-*.whl"))
    if not wheels:
        raise SystemExit(f"No forge3d wheels found in {wheel_dir}")

    supported_tag_ranks = _supported_tag_ranks()
    compatible: list[tuple[int, Path]] = []
    for wheel in wheels:
        score = _wheel_score(wheel, supported_tag_ranks)
        if score is not None:
            compatible.append((score, wheel))

    if not compatible:
        names = ", ".join(w.name for w in wheels)
        raise SystemExit(
            "No compatible forge3d wheel found for "
            f"{sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor} "
            f"in {wheel_dir}: {names}"
        )

    compatible.sort(key=lambda item: (item[0], item[1].name))
    chosen = compatible[0][1]
    print(f"Installing wheel: {chosen}")
    subprocess.run([sys.executable, "-m", "pip", "install", str(chosen)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
