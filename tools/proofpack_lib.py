from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from PIL import Image

import hashlib


def collect_repo_state() -> Dict[str, Any]:
    """Return git sha or dirty flag, python version, platform, and package versions."""

    info: Dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    repo = Path(__file__).parent.parent
    try:
        sha = (
            subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = len(status) > 0
        info["git"] = {"sha": sha, "dirty": dirty}
    except Exception:
        info["git"] = {"sha": "unknown", "dirty": True}

    for pkg in ["numpy", "PIL"]:
        try:
            module = __import__(pkg if pkg != "PIL" else "PIL")
            info["packages"][pkg] = getattr(module, "__version__", "unknown")
        except Exception:
            info["packages"][pkg] = "missing"
    return info


def record_png_manifest_entry(manifest: Dict[str, Any], logical_name: str, path: str) -> None:
    """Add sha256, width, height (None if unavailable) for the given PNG path."""

    p = Path(path)
    entry: Dict[str, Any] = {"name": logical_name, "path": str(p), "exists": p.exists(), "width": None, "height": None}
    if p.exists() and p.is_file():
        entry["sha256"] = sha256_file(str(p))
        try:
            with Image.open(p) as im:
                entry["width"], entry["height"] = im.size
        except Exception:
            entry["width"], entry["height"] = None, None
    manifest.setdefault("pngs", []).append(entry)


def sha256_file(path: str) -> str:
    """Return hex sha256 digest for the given file."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
