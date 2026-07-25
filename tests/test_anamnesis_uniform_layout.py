from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_repr_c_uniform_is_pod_and_zeroable():
    """Compile-time bytemuck contracts forbid padding and initialize all bytes.

    `Pod` rejects any `#[repr(C)]` type with implicit padding, while `Zeroable`
    gives every explicit padding field a deterministic initialization. Hashing
    the exact cast slice therefore cannot observe uninitialized padding bytes.
    This source sweep keeps that contract exhaustive as new uniforms are added.
    """

    declarations: list[tuple[Path, str, str]] = []
    pattern = re.compile(
        r"#\[repr\(C[^\]]*\)\]\s*"
        r"(?P<attrs>(?:#\[[^\]]+\]\s*)*)"
        r"(?:pub(?:\([^)]*\))?\s+)?struct\s+(?P<name>\w*Uniform\w*)",
        re.MULTILINE,
    )
    for path in sorted((ROOT / "src").rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            declarations.append((path, match.group("name"), match.group("attrs")))

    assert declarations, "uniform audit found no #[repr(C)] uniform structs"
    violations = [
        f"{path.relative_to(ROOT)}::{name}"
        for path, name, attrs in declarations
        if "Pod" not in attrs or "Zeroable" not in attrs
    ]
    assert not violations, (
        "all #[repr(C)] uniform structs must derive bytemuck::Pod and Zeroable; "
        f"violations={violations}"
    )
