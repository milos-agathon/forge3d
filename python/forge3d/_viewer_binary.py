from __future__ import annotations

import shutil
import sys
from pathlib import Path


def find_viewer_binary(module_file: str) -> str:
    """Resolve the best available interactive_viewer command/binary."""
    suffix = ".exe" if sys.platform == "win32" else ""

    repo_root = Path(module_file).resolve().parent.parent.parent
    cargo_target = repo_root / "target"

    repo_candidates = []
    for profile in ("release", "debug"):
        binary = cargo_target / profile / f"interactive_viewer{suffix}"
        if binary.exists():
            repo_candidates.append(binary)
    if repo_candidates:
        newest = max(repo_candidates, key=lambda path: path.stat().st_mtime_ns)
        return str(newest)

    installed_script = Path(sys.executable).resolve().parent / f"interactive_viewer{suffix}"
    if installed_script.exists():
        return str(installed_script)

    path_binary = shutil.which("interactive_viewer")
    if path_binary:
        return path_binary

    raise FileNotFoundError(
        "Could not find interactive_viewer. "
        "Install forge3d with pip so the console script is created, "
        "or build with: cargo build --release --bin interactive_viewer"
    )
