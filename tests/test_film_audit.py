"""ARGUS source-frame acceptance and input/output contracts."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import forge3d
from forge3d._deltae import relative_luminance
from forge3d._film_audit_signals import temporal_amplitudes
from forge3d._png import load_png_rgba, save_png
from forge3d.film_audit import _Grid, _preview, _shift, _triplet, audit
from _film_audit_corpus import (
    CASES,
    CONTROLS,
    COUNT,
    INJECT_AT,
    SIZE,
    declarations,
    frames,
    region_for,
    write,
)


@pytest.mark.parametrize("story", CONTROLS)
def test_clean_controls(tmp_path, story):
    write(tmp_path, story)
    report = audit(tmp_path, **declarations(story))
    assert report["findings"] == []
    assert [row["frame_index"] for row in report["measures"]] == list(range(COUNT))
    if story == "cut":
        for i in (INJECT_AT - 1, INJECT_AT):
            assert report["measures"][i]["flash_luminance"] is None
        assert report["measures"][INJECT_AT]["incoming_mean_code_difference"] is None


@pytest.mark.parametrize(("story", "defect", "expected"), CASES)
def test_injected_defect_recall_and_region(tmp_path, story, defect, expected):
    write(tmp_path, story, defect)
    report = audit(tmp_path, **declarations(story))
    assert [(f["frame_index"], f["class"]) for f in report["findings"]] == [
        (INJECT_AT, expected)
    ]
    (finding,) = report["findings"]
    if expected in {"localized_pop", "fixed_line"} or defect == "localized_reseed":
        left, top, right, bottom = finding["region"]
        x0, y0, x1, y1 = region_for(defect)
        assert min(x1, right) > max(x0, left) and min(y1, bottom) > max(y0, top)
    if expected == "duplicate":
        assert report["measures"][INJECT_AT]["exact_repeat"] is (defect == "duplicate")
    if expected == "periodic_pulse":
        assert finding["pulse_frame_indices"] == [10, 13, 16, 19]
        assert finding["spectral_amplitude"] > 0
    if expected == "fixed_line":
        assert finding["end_frame_index"] == INJECT_AT + 9
    if expected == "dropped_frame":
        assert finding["missing_index_is_explicit"] is (defect == "drop_numbered")
        assert finding["context_roles"]["at"] == INJECT_AT + (defect == "drop_numbered")


def test_declared_one_transition_hold_suppresses_duplicate(tmp_path):
    write(tmp_path, "orbit", "duplicate")
    assert audit(tmp_path)["findings"][0]["class"] == "duplicate"
    assert audit(tmp_path, holds=[(INJECT_AT - 1, INJECT_AT)])["findings"] == []


def test_declared_hold_still_reports_missing_frame_number(tmp_path):
    write(tmp_path, "hold")
    (tmp_path / f"frame_{INJECT_AT}.png").unlink()
    report = audit(tmp_path, **declarations("hold"))
    (finding,) = report["findings"]
    assert (finding["frame_index"], finding["class"]) == (INJECT_AT, "dropped_frame")
    assert finding["missing_index_is_explicit"]
    assert finding["context_roles"]["at"] == INJECT_AT + 1


@pytest.mark.parametrize(("count", "cuts"), [(1, ()), (3, (2,))])
def test_one_frame_shots_keep_their_frame_record(tmp_path, count, cuts):
    for index, frame in enumerate(list(frames("static"))[:count]):
        save_png(tmp_path / f"frame_{index}.png", frame)
    report = audit(tmp_path, cuts=cuts, out_dir=tmp_path / "result")
    assert report["findings"] == []
    assert [row["frame_index"] for row in report["measures"]] == list(range(count))
    assert report["measures"][-1]["incoming_mean_luminance_difference"] is None
    assert report["measures"][-1]["temporal_luminance_second_difference"] is None
    spectrum = report["tile_luminance_spectra"][-1]
    assert spectrum["first_frame_index"] == spectrum["last_frame_index"] == count - 1
    assert spectrum["frequencies_cycles_per_frame"] == [0.0]
    assert json.loads((tmp_path / "result" / "audit.json").read_bytes()) == report


def test_simultaneous_pops_have_independent_regions_scores_and_evidence(tmp_path):
    regions = [(24, 30, 42, 45), region_for("pop")]
    for index, frame in enumerate(frames("orbit")):
        if index == INJECT_AT:
            for x0, y0, x1, y1 in regions:
                frame[y0:y1, x0:x1] = [235, 35, 201]
        save_png(tmp_path / f"frame_{index}.png", frame)
    output = tmp_path / "result"
    findings = audit(tmp_path, out_dir=output)["findings"]
    assert [(f["frame_index"], f["class"]) for f in findings] == [
        (INJECT_AT, "localized_pop"),
        (INJECT_AT, "localized_pop"),
    ]
    for finding, (x0, y0, x1, y1) in zip(findings, regions, strict=True):
        left, top, right, bottom = finding["region"]
        assert min(x1, right) > max(x0, left) and min(y1, bottom) > max(y0, top)
        assert finding["score"] > finding["threshold"]
        assert dict(finding["time_series"]["points"])[INJECT_AT] == finding["score"]
        crop = load_png_rgba(output / finding["evidence"]["at"])
        assert crop.shape[:2] == (bottom - top, right - left)
    assert findings[0]["score"] != findings[1]["score"]
    assert set(findings[0]["evidence"].values()).isdisjoint(
        findings[1]["evidence"].values()
    )


def test_downsampled_pop_maps_back_to_source_pixels(tmp_path):
    size = (641, 359)
    write(tmp_path, "zoom", "edge_pop", size=size)
    result = audit(tmp_path)
    (finding,) = result["findings"]
    assert (finding["frame_index"], finding["class"]) == (INJECT_AT, "localized_pop")
    assert result["resolution"] == list(size)
    assert result["analysis_resolution"] != list(size)
    left, top, right, bottom = finding["region"]
    x0, y0, x1, y1 = region_for("edge_pop", size)
    assert min(x1, right) > max(x0, left) and min(y1, bottom) > max(y0, top)


def test_registration_compensates_known_source_motion():
    rng = np.random.default_rng(1904)
    image = rng.integers(24, 220, (48, 72, 4), dtype=np.uint8)
    image[..., 3] = 255
    grid = _Grid.for_shape(image.shape)
    before, at, after = (
        _preview(np.roll(image, (dy, dx), axis=(0, 1)), grid)
        for dy, dx in ((-2, 3), (0, 0), (2, -3))
    )
    assert tuple(
        round(pixel) for pixel in _shift(before, after, grid.window.shape)
    ) == (4, -6)
    _, registered, _, _ = _triplet(before, at, after, grid)
    _, screen_space, _, _ = _triplet(before, at, after, grid, held=True)
    assert registered.max() < screen_space.max()


def test_manifest_uses_record_indices_and_declared_timeline(tmp_path):
    source = tmp_path / "png"
    source.mkdir()
    inputs = list(frames("cut"))
    records = []
    for i, frame in enumerate(inputs):
        image = (
            f"tile-{COUNT - i}.png"  # File numbers run backwards; manifest order wins.
        )
        save_png(source / image, frame)
        records.append(
            {
                "frame_index": 100 + 2 * i,
                "image": image,
                "width": SIZE[0],
                "height": SIZE[1],
            }
        )
    cut = 100 + 2 * INJECT_AT
    manifest = tmp_path / "flythrough_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "forge3d.chronos.flythrough/1",
                "frames": records,
                "cuts": [cut],
                "holds": [[100, cut - 2]],
            }
        ),
        encoding="utf-8",
    )
    result = audit(
        manifest, frames_dir=source, holds=[(cut, records[-1]["frame_index"])]
    )
    assert result["cuts"] == [cut]
    assert result["holds"] == [[100, cut - 2], [cut, records[-1]["frame_index"]]]
    assert result["findings"] == []
    assert result["measures"][INJECT_AT]["image"] == records[INJECT_AT]["image"]
    assert result["measures"][INJECT_AT]["incoming_mean_code_difference"] is None
    assert [row["frame_index"] for row in result["measures"]] == [
        r["frame_index"] for r in records
    ]


def test_numeric_directory_reports_source_number(tmp_path):
    for index, image in zip((2, 10, 11), list(frames("static", "bright_flash"))[9:12]):
        save_png(tmp_path / f"frame_{index}.png", image)
    result = audit(tmp_path)
    assert [row["frame_index"] for row in result["measures"]] == [2, 10, 11]
    assert [(f["frame_index"], f["class"]) for f in result["findings"]] == [
        (10, "flash")
    ]


def test_report_cannot_replace_input_manifest(tmp_path):
    records = []
    for i, frame in enumerate(list(frames("static"))[:3]):
        name = f"frame_{i}.png"
        save_png(tmp_path / name, frame)
        records.append(
            {"frame_index": i, "image": name, "width": SIZE[0], "height": SIZE[1]}
        )
    manifest = tmp_path / "audit.json"
    original = json.dumps(
        {"schema": "forge3d.chronos.flythrough/1", "frames": records}
    ).encode()
    manifest.write_bytes(original)
    subdir = tmp_path / "folder"
    subdir.mkdir()

    with pytest.raises(
        ValueError, match="report destination overlaps the input manifest"
    ):
        audit(manifest, out_dir=subdir / "..")
    assert manifest.read_bytes() == original
    assert not list(tmp_path.glob(".film-audit-*"))
    assert audit(manifest, out_dir=subdir)["frame_count"] == 3
    assert manifest.read_bytes() == original


def test_cli_status_and_byte_repeatable_evidence(tmp_path):
    source = tmp_path / "src"
    write(source, "orbit", "pop")
    found_output, replay_output, clean_output = (
        tmp_path / part for part in ("one", "two", "clean")
    )

    def run_cli(output: Path):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "forge3d.film_audit",
                str(source),
                "--out-dir",
                str(output),
            ],
            capture_output=True,
            check=False,
            text=True,
        )

    for output in (found_output, replay_output):
        completed = run_cli(output)
        assert completed.returncode == 1, completed.stderr
        assert completed.stderr == ""
    result = json.loads((found_output / "audit.json").read_bytes())
    assert (found_output / "audit.json").read_bytes() == (
        replay_output / "audit.json"
    ).read_bytes()
    (finding,) = result["findings"]
    left, top, right, bottom = finding["region"]
    for label, relpath in finding["evidence"].items():
        png = found_output / relpath
        assert png.read_bytes() == (replay_output / relpath).read_bytes()
        rgba = load_png_rgba(png)
        if label == "contact_sheet":
            assert rgba.shape[0] > bottom - top and rgba.shape[1] >= 4 * (right - left)
            plot = rgba[bottom - top :, :, :3]
            assert np.any(np.all(plot == (118, 230, 191), axis=2))
            assert np.any(np.all(plot == (248, 186, 96), axis=2))
        else:
            assert rgba.shape[:2] == (bottom - top, right - left)
        if label == "heatmap":
            assert np.any(rgba[..., 0] > 0)
        elif label != "contact_sheet":
            index = {"before": INJECT_AT - 1, "at": INJECT_AT, "after": INJECT_AT + 1}[
                label
            ]
            original = load_png_rgba(source / f"frame_{index}.png")
            np.testing.assert_array_equal(rgba, original[top:bottom, left:right])
    write(source, "orbit")
    clean = run_cli(clean_output)
    assert clean.returncode == 0, clean.stderr
    assert json.loads((clean_output / "audit.json").read_bytes())["findings"] == []


def test_duplicate_numbers_and_mislabeled_images_are_rejected(tmp_path):
    frame = next(frames("static"))
    save_png(tmp_path / "before_10.png", frame)
    bad = tmp_path / "after_10.png"
    bad.write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="sorted and unique"):
        audit(tmp_path)
    bad.rename(tmp_path / "after_11.png")
    with pytest.raises(ValueError, match="frame 11.*expected a PNG"):
        audit(tmp_path)


def test_public_module_and_exports():
    assert "film_audit" in forge3d.__all__
    assert forge3d.film_audit.audit is audit


def test_pairwise_luminance_and_second_difference_use_decoded_pixels(tmp_path):
    dark = np.full((4, 4, 3), 56, dtype=np.uint8)
    light = np.full_like(dark, 109)
    halfway = dark.copy()
    halfway[:2] = light[:2]
    for i, image in enumerate((dark, halfway, light)):
        save_png(tmp_path / f"frame_{i}.png", image)
    report = audit(tmp_path)
    one, two, three = report["measures"]
    expected = float((relative_luminance(light) - relative_luminance(dark)).mean() / 2)
    assert one["incoming_mean_luminance_difference"] is None
    for row in (two, three):
        assert row["incoming_mean_luminance_difference"] == pytest.approx(expected)
        assert row["incoming_mean_absolute_luminance_difference"] == pytest.approx(
            expected
        )
        assert row["incoming_tile_luminance_difference"][0] == pytest.approx([expected])
        assert row["incoming_tile_mean_absolute_luminance_difference"][
            0
        ] == pytest.approx([expected])
        assert row["incoming_changed_pixel_fraction"] == 0.5
    assert two["temporal_luminance_second_difference"] == pytest.approx(0.0)
    (spectrum,) = report["tile_luminance_spectra"]
    assert spectrum["frequencies_cycles_per_frame"] == pytest.approx([0, 1 / 3])


def test_temporal_spectrum_resolves_different_frequencies_per_tile():
    time = np.arange(12)
    signals = np.column_stack(
        (
            np.cos(2 * np.pi * time / 3) + 2 * time,
            np.cos(2 * np.pi * time / 4) - time,
        )
    )
    amplitudes = temporal_amplitudes(signals[:, None, :])
    assert amplitudes.shape == (7, 1, 2)
    assert amplitudes[:, 0, :].argmax(axis=0).tolist() == [4, 3]


def test_changed_fraction_counts_source_pixels_after_preview_downsampling(tmp_path):
    rgba = np.zeros((2, 384, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    save_png(tmp_path / "frame_0.png", rgba)
    rgba[0, 0] = [255, 255, 255, 128]
    save_png(tmp_path / "frame_1.png", rgba)
    report = audit(tmp_path)
    assert report["analysis_resolution"] == [192, 1]
    transition = report["measures"][1]
    assert transition["incoming_changed_pixel_fraction"] == 1 / (2 * 384)
    assert transition["incoming_mean_luminance_difference"] == pytest.approx(
        relative_luminance(np.array([[128 / 255] * 3]))[0] / (2 * 384)
    )


def test_line_persistence_requires_motion_beyond_exposure(tmp_path):
    for i, frame in enumerate(frames("daylight")):
        frame[8:100, 84:86] = 255
        save_png(tmp_path / f"frame_{i}.png", frame)
    assert audit(tmp_path)["findings"] == []


def test_all_seven_evidence_types_are_byte_repeatable(tmp_path):
    definitions = [
        ("static", "bright_flash", "flash"),
        ("orbit", "duplicate", "duplicate"),
        ("orbit", "pop", "localized_pop"),
        ("orbit", "drop_numbered", "dropped_frame"),
        ("orbit", "bright_step", "lighting_step"),
        ("orbit", "localized_reseed", "periodic_pulse"),
        ("orbit", "vertical_line", "fixed_line"),
    ]
    source = tmp_path / "source"
    source.mkdir()
    expected = []
    cuts = []
    for shot, (story, defect, label) in enumerate(definitions):
        offset = shot * COUNT
        if shot:
            cuts.append(offset)
        for i, frame in enumerate(frames(story, defect)):
            number = offset + i + (defect == "drop_numbered" and i >= INJECT_AT)
            save_png(source / f"frame_{number}.png", frame)
        expected.append((offset + INJECT_AT, label))

    one, two = tmp_path / "first", tmp_path / "second"
    report = audit(source, out_dir=one, cuts=cuts)
    audit(source, out_dir=two, cuts=cuts)
    assert (one / "audit.json").read_bytes() == (two / "audit.json").read_bytes()
    assert [(f["frame_index"], f["class"]) for f in report["findings"]] == expected
    for finding in report["findings"]:
        assert finding["time_series"]["points"]
        for relative in finding["evidence"].values():
            assert (one / relative).read_bytes() == (two / relative).read_bytes()
        contact = load_png_rgba(one / finding["evidence"]["contact_sheet"])
        assert np.any(np.all(contact[..., :3] == (118, 230, 191), axis=2))
