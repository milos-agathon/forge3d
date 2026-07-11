from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_PIPELINE_CREATION = re.compile(r"\.create_(?:render|compute)_pipeline\s*\(")


def test_every_pipeline_creation_uses_validation_scope() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.rs")):
        if path.as_posix().endswith("src/core/shader_registry.rs"):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if RAW_PIPELINE_CREATION.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")

    assert not violations, (
        "pipeline creation must route through shader_registry's validation-scoped helpers:\n"
        + "\n".join(violations)
    )
