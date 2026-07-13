# tests/test_reproject_no_silent_suppression.py
# MENSURA M-05 source gate: no CRS-transform error may be silently suppressed
# with `.ok()` / `.unwrap_or*` in the raster or vector reprojection paths. The
# historical "single most dangerous line in the geo stack" was
# `transform_point(...).ok() ... .unwrap_or(fill)`, which turned an unsupported
# CRS into an all-nodata raster with a success status. Every transform result
# must be matched (and counted) or propagated with `?`.
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# Files that perform raster/vector coordinate reprojection through the engine.
REPROJECTION_SOURCES = [
    REPO / "src" / "gis" / "warp.rs",
    REPO / "src" / "gis" / "vector.rs",
]

# A transform_point(...) call whose Result is immediately swallowed.
SUPPRESSION = re.compile(
    r"transform_point\s*\([^;{}]*\)\s*\.\s*(?:ok|unwrap_or|unwrap_or_else|unwrap_or_default)\b",
    re.DOTALL,
)


@pytest.mark.parametrize("path", REPROJECTION_SOURCES, ids=lambda p: p.name)
def test_no_silent_transform_suppression(path):
    assert path.exists(), f"reprojection source missing: {path}"
    text = path.read_text(encoding="utf-8")
    hits = SUPPRESSION.findall(text)
    assert not hits, (
        f"{path.name}: a CRS transform_point() result is suppressed with "
        f".ok()/.unwrap_or* — the MENSURA raise/nodata policy is bypassed. "
        f"Match it (count failures) or propagate with `?`."
    )


def test_transform_point_is_actually_exercised():
    # Guard against the gate passing vacuously if the call sites are renamed.
    joined = "".join(p.read_text(encoding="utf-8") for p in REPROJECTION_SOURCES)
    assert "transform_point(" in joined
