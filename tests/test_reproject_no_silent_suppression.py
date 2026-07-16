# tests/test_reproject_no_silent_suppression.py
# MENSURA M-05 source gate: no coordinate-transform error may be silently
# suppressed with `.ok()` / `.unwrap_or*` in the raster or vector reprojection
# paths. The historical "single most dangerous line in the geo stack" was
# `transform_point(...).ok() ... .unwrap_or(fill)`, which turned an unsupported
# CRS into an all-nodata raster with a success status. The same suppression can
# hide behind the AFFINE INVERSE: `inverse_apply(source_transform, ..).ok()`
# drops a singular source transform to a nodata fill with SUCCESS. Every
# transform/inverse result must be matched (and counted) or propagated with `?`.
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# Files that perform raster/vector coordinate reprojection through the engine.
REPROJECTION_SOURCES = [
    REPO / "src" / "gis" / "warp.rs",
    REPO / "src" / "gis" / "vector.rs",
]

# A CRS transform (`transform_point`/`transform_point3`) OR the source-affine
# inverse (`inverse_apply`) whose Result is immediately swallowed. Both feed the
# per-pixel reproject sampling and both must be counted under the raise/nodata
# policy, never dropped to a default fill.
SUPPRESSION = re.compile(
    r"(?:transform_point3?|inverse_apply)\s*\([^;{}]*\)\s*\.\s*"
    r"(?:ok|unwrap_or|unwrap_or_else|unwrap_or_default)\b",
    re.DOTALL,
)


@pytest.mark.parametrize("path", REPROJECTION_SOURCES, ids=lambda p: p.name)
def test_no_silent_transform_suppression(path):
    assert path.exists(), f"reprojection source missing: {path}"
    text = path.read_text(encoding="utf-8")
    hits = SUPPRESSION.findall(text)
    assert not hits, (
        f"{path.name}: a coordinate-transform result (transform_point/"
        f"inverse_apply) is suppressed with .ok()/.unwrap_or* — the MENSURA "
        f"raise/nodata policy is bypassed. Match it (count failures) or "
        f"propagate with `?`."
    )


def test_transform_primitives_are_actually_exercised():
    # Guard against the gate passing vacuously if the call sites are renamed.
    joined = "".join(p.read_text(encoding="utf-8") for p in REPROJECTION_SOURCES)
    assert "transform_point(" in joined
    assert "inverse_apply(" in joined
