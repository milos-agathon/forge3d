"""Minimal TOML-loading shim shared by repository gate tests.

Uses the stdlib `tomllib` when available (Python >= 3.11). Falls back to a
small hand-rolled parser sufficient for the restricted schemas committed under
`tests/`: scalar assignments, scalar arrays, named tables, and array-of-tables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python >= 3.11
except ImportError:  # pragma: no cover - exercised on Python < 3.11
    tomllib = None  # type: ignore[assignment]


def parse_toml_fallback(text: str) -> dict[str, Any]:
    """Parse the restricted TOML subset without any external dependency."""
    result: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            key = line[2:-2].strip()
            current = {}
            result.setdefault(key, [])
            result[key].append(current)
            continue
        if line.startswith("[") and line.endswith("]"):
            key = line[1:-1].strip()
            current = {}
            result[key] = current
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith(('"', "[")):
            parsed: Any = json.loads(value)
        elif value == "true":
            parsed = True
        elif value == "false":
            parsed = False
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        if current is not None:
            current[key] = parsed
        else:
            result[key] = parsed
    return result


def load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, preferring stdlib `tomllib` when it is available."""
    path = Path(path)
    if tomllib is not None:
        with open(path, "rb") as f:
            return tomllib.load(f)
    return parse_toml_fallback(path.read_text(encoding="utf-8"))
