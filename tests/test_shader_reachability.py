"""Shader reachability guard for renderer consolidation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "shaders"

# BOP-P2-08: known unreferenced/deferred shaders. These must be wired or
# deleted in the consolidation sweep; the guard prevents this list from growing.
ALLOWLIST = {
    "velocity.wgsl": "BOP-P2-04 decision: helper shader is still locked by motion-vector tests",
}


def _source_text() -> str:
    chunks: list[str] = []
    for path in (ROOT / "src").rglob("*"):
        if path.suffix != ".rs":
            continue
        chunks.append(path.read_text(encoding="utf-8", errors="ignore").replace("\\", "/"))
    return "\n".join(chunks)


def test_all_unreferenced_wgsl_files_are_explicitly_allowlisted() -> None:
    source = _source_text()
    unreferenced: list[str] = []

    for path in sorted(SHADER_DIR.rglob("*.wgsl")):
        rel = path.relative_to(SHADER_DIR).as_posix()
        if rel in source or path.name in source:
            continue
        unreferenced.append(rel)

    assert unreferenced == sorted(ALLOWLIST), (
        "Unexpected unreferenced WGSL files. Wire/delete the shader or add a "
        f"reviewed BOP-P2-08 allowlist reason: {unreferenced}"
    )
