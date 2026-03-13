#!/usr/bin/env python3
"""Install one compatible wheel from a directory."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _split_wheel_tags(path: Path) -> tuple[str, str, str] | None:
    name = path.name
    if not name.endswith(".whl"):
        return None
    parts = name[:-4].split("-")
    if len(parts) < 5:
        return None
    return parts[-3], parts[-2], parts[-1]


def _cpython_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _pypy_tag() -> str:
    return f"pp{sys.version_info.major}{sys.version_info.minor}"


def _tag_version(tag: str, prefix: str) -> int | None:
    if not tag.startswith(prefix):
        return None
    suffix = tag[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _wheel_score(path: Path) -> tuple[int, int] | None:
    tags = _split_wheel_tags(path)
    if tags is None:
        return None

    py_tag, abi_tag, _platform_tag = tags
    impl = platform.python_implementation()
    py_version = sys.version_info.major * 100 + sys.version_info.minor

    if impl == "CPython":
        if py_tag.startswith("pp"):
            return None

        if py_tag == _cpython_tag() and abi_tag.startswith("cp"):
            return (0, 0)

        cp_floor = _tag_version(py_tag, "cp")
        if cp_floor is not None and abi_tag == "abi3" and py_version >= cp_floor:
            return (1, py_version - cp_floor)

        if py_tag == "py3":
            return (3, 0)

        py_floor = _tag_version(py_tag, "py")
        if py_floor is not None and py_version >= py_floor:
            return (4, py_version - py_floor)

        return None

    if impl == "PyPy":
        if py_tag == _pypy_tag():
            return (0, 0)
        if py_tag == "py3":
            return (3, 0)
        return None

    if py_tag == "py3":
        return (3, 0)
    return None


def main() -> int:
    wheel_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = sorted(wheel_dir.glob("forge3d-*.whl"))
    if not wheels:
        raise SystemExit(f"No forge3d wheels found in {wheel_dir}")

    compatible: list[tuple[tuple[int, int], Path]] = []
    for wheel in wheels:
        score = _wheel_score(wheel)
        if score is not None:
            compatible.append((score, wheel))

    if not compatible:
        names = ", ".join(w.name for w in wheels)
        raise SystemExit(
            f"No compatible forge3d wheel found for {platform.python_implementation()} "
            f"{sys.version_info.major}.{sys.version_info.minor} in {wheel_dir}: {names}"
        )

    compatible.sort(key=lambda item: item[0])
    chosen = compatible[0][1]
    print(f"Installing wheel: {chosen}")
    subprocess.run([sys.executable, "-m", "pip", "install", str(chosen)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
