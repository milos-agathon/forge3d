"""Fail-closed source gate for f64 world-coordinate narrowing.

The scanner normalizes comments and whitespace, records every occurrence, and
recognizes cast-before-subtract forms that the original line/vocabulary gate
missed (`as_vec3`, indexed component casts, and multiline expressions).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SANCTIONED = "src/camera/anchor.rs"

WORLD_WORD = (
    r"(?:world(?:_position|_pos|_coord)?|object_(?:origin|translation)|"
    r"origin|target|eye|translation|ecef|longitude|latitude|"
    r"vertex(?:_position|_pos|_coord)?|point(?:_position|_pos|_coord)?|coord|abs)"
)


def _strip_comments_and_strings(text: str) -> str:
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|r#*\".*?\"#*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.S,
    )
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_comments_and_strings(text))


def _forbidden_narrowings(text: str) -> list[str]:
    normalized = _normalize(text)
    patterns = [
        rf"\b{WORLD_WORD}\b(?:\s*\.\s*[xyzw]|\s*\[[^\]]+\])?\s*\.as_vec[234]\s*\(\s*\)",
        rf"\b{WORLD_WORD}\b(?:\s*\.\s*[xyzw]|\s*\[[^\]]+\])?\s+as\s+f32\b",
        r"\b(?:DVec[234]|DMat[234])\b[^;{}]*?\.as_vec[234]\s*\(\s*\)",
    ]
    findings = []
    for pattern in patterns:
        findings.extend(match.group(0) for match in re.finditer(pattern, normalized, re.I))
    return findings


def _rust_files():
    return sorted((ROOT / "src").rglob("*.rs"))


def test_single_narrowing_implementation_lives_only_in_anchor():
    sites = []
    for path in _rust_files():
        text = _strip_comments_and_strings(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"\bas\s+f32\b", text):
            sites.append((path.relative_to(ROOT).as_posix(), match.start()))
    anchor = _strip_comments_and_strings(_read(SANCTIONED))
    assert len(re.findall(r"\bas\s+f32\b", anchor)) == 1
    assert "fn narrow(value: f64) -> f32" in anchor
    assert (SANCTIONED, anchor.index("as f32")) in sites


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_no_world_coordinate_bypasses_anchor_narrowing():
    findings = []
    for path in _rust_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == SANCTIONED:
            continue
        for finding in _forbidden_narrowings(path.read_text(encoding="utf-8")):
            # Physical mouse positions are f64 screen pixels, not world coordinates.
            if rel == "src/viewer/input/viewer_input.rs" and finding in {
                "position.x as f32",
                "position.y as f32",
            }:
                continue
            # Elevation is normalized to a unitless color ramp value here.
            if rel == "src/viewer/pointcloud/load.rs" and "point.z" in finding:
                continue
            findings.append((rel, finding))
    assert not findings, f"world-coordinate f64->f32 bypasses: {findings}"


def test_scanner_rejects_previously_demonstrated_bypasses():
    probes = [
        "let q = world_position.as_vec3();",
        "let q = DVec3::new(1.0, 2.0, 3.0).as_vec3();",
        "let abs = [1.0_f64, 2.0, 3.0]; let q = [abs[0] as f32, abs[1] as f32, abs[2] as f32];",
        "let q = [world_coord[0]\n as f32, world_coord[1] as f32, world_coord[2] as f32];",
    ]
    for probe in probes:
        assert _forbidden_narrowings(probe), f"scanner missed bypass: {probe}"


def test_cityjson_and_viewer_absolute_storage_types_are_explicitly_f64():
    assert "pub positions: Vec<f64>" in _read("src/import/cityjson/types.rs")
    assert "pub(crate) object_translation: glam::DVec3" in _read("src/viewer/viewer_struct.rs")
    assert "pub world_pos: DVec3" in _read("src/labels/types.rs")
    assert "pub position: DVec3" in _read("src/viewer/pointcloud/types.rs")


def test_public_camera_helper_preserves_earth_scale_offset():
    import numpy as np
    from forge3d import _forge3d

    local = np.asarray(
        _forge3d.camera_look_at((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    earth = np.asarray(
        _forge3d.camera_look_at(
            (6_378_137.0, 2_000.0, -3_000.0),
            (6_378_147.0, 2_000.0, -3_000.0),
            (0.0, 1.0, 0.0),
        )
    )
    np.testing.assert_allclose(earth, local, rtol=0.0, atol=1e-6)
