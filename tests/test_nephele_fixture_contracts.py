"""Routine immutable-fixture, provenance, and lane-accounting contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.nephele_fixture_masks import (
    EXPECTED_RULES,
    MASK_FILES,
    TECHNICAL_CONTRACTS,
    generate_masks,
)
from scripts.generate_media_fixture import (
    DIAGNOSTIC_FIELDS,
    _spatial_tiles,
    _terrain_media_reference_module_source,
    _validated_grid_density,
)
from scripts.record_media_reference_convergence import downstream_metrics
from scripts.nephele_majorant_domain_probe import (
    DENSITY_SAMPLING,
    MAJORANT_QUERY,
    canonical_majorant_cells,
    represented_density,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mutation", ["array", "native", "samples", "sun"])
def test_b6_environment_reference_rejects_unbound_inputs_before_gpu(tmp_path, mutation) -> None:
    from scripts.nephele_b6_environment_diagnostic import environment_reference

    provenance = _object(ROOT / "tests/nephele/fixture/reference-provenance.json")
    crop = _object(ROOT / "tests/nephele/fixture/crop.json")
    array = tmp_path / "reference-environment-rgb.npy"
    np.save(array, np.zeros((crop["height"], crop["width"], 3), dtype=np.uint8), allow_pickle=False)
    record = {
        "schema": "forge3d.nephele.b6_environment_diagnostic/1",
        "sun_intensity": 0.0,
        "fixture_manifest_sha256": _hash(ROOT / "tests/nephele/fixture-manifest.json"),
        "samples_per_pixel": provenance["samples_per_pixel"],
        "imported_native": {"sha256": provenance["native_runtime"]["native_sha256"]},
        "diagnostics": provenance["diagnostics"],
        "artifact_hashes": {array.name: _hash(array)},
    }
    report = tmp_path / "report.json"
    report.write_text(json.dumps(record), encoding="utf-8")
    environment_reference(ROOT, tmp_path)
    if mutation == "array":
        array.write_bytes(b"altered executed array")
    elif mutation == "native":
        record["imported_native"]["sha256"] = "0" * 64
    elif mutation == "samples":
        record["samples_per_pixel"] = 40
    else:
        record["sun_intensity"] = 3.0
    report.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="environment reference"):
        environment_reference(ROOT, tmp_path)


def test_original_b1_audit_hashes_remain_valid() -> None:
    audit = ROOT / "docs/audits/nephele-b1-2026-09-30"
    hashes = _object(audit / "artifact-hashes.json")["files"]
    for name, digest in hashes.items():
        assert _hash(audit / name) == digest, name
    capture = ROOT / "docs/audits/nephele-physical-terrain-2026-09-30"
    assert _hash(capture / "diagnostic-capture.py") == _object(capture / "b1-report.json")["diagnostic_tool"]["sha256"]


def test_b1_registered_control_binds_plan_and_rejects_tampering(tmp_path, monkeypatch) -> None:
    from scripts import nephele_b1_diagnostic as tool
    from scripts.nephele_b1_diagnostic import _require_registered_reference

    fixture = tmp_path / "tests/nephele/fixture"
    fixture.mkdir(parents=True)
    provenance = _object(ROOT / "tests/nephele/fixture/reference-provenance.json")
    (fixture / "reference-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (fixture / "crop.json").write_bytes((ROOT / "tests/nephele/fixture/crop.json").read_bytes())
    manifest = tmp_path / "tests/nephele/fixture-manifest.json"
    manifest.write_bytes((ROOT / "tests/nephele/fixture-manifest.json").read_bytes())
    fixture_manifest = _object(manifest)
    monkeypatch.setattr(tool, "_frozen_fixture", lambda repo: (fixture_manifest, provenance))
    wheel = tmp_path / "reference.whl"
    wheel.write_bytes(b"test wheel; native import is checked separately")
    monkeypatch.setattr(tool, "_reference_wheel", lambda repo, runtime: (wheel, tool._artifact_record(repo, wheel)))
    generator = tmp_path / "scripts/nephele_b1_diagnostic.py"
    generator.parent.mkdir()
    generator.write_bytes((ROOT / "scripts/nephele_b1_diagnostic.py").read_bytes())
    output = tmp_path / "control"
    output.mkdir()
    reference = output / "vacuum-reference-spp320.npy"
    np.save(reference, np.zeros((64, 64, 3), dtype=np.uint8), allow_pickle=False)
    plan = tool._control_plan(tmp_path, output, fixture_manifest, provenance)
    plan_path = tmp_path / "control-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    record = {
        "schema": "forge3d.nephele.vacuum_control/1",
        "plan": {"path": plan_path.name, "artifact": plan_path.name, "sha256": _hash(plan_path)},
        "reference": tool._artifact_record(tmp_path, reference),
        "published_plan_sha256": _hash(plan_path),
        "seed": plan["seed"],
        "medium_coefficients": plan["medium_coefficients"],
        "criterion": plan["criterion"],
        "native_wheel": plan["native_wheel"],
        "imported_native": {"path": str((tmp_path / "_forge3d.pyd").resolve()), "sha256": provenance["native_runtime"]["native_sha256"]},
        "generator": plan["generator"],
        "samples_per_pixel": 320,
        "native_runtime": provenance["native_runtime"],
        "diagnostics": {**provenance["diagnostics"], "sample_count": 320 * 64 * 64},
        "camera_contract": provenance["camera_contract"],
    }
    record["outputs"] = {"reference": record["reference"]}
    registration = output / "vacuum-control-spp320.json"
    registration.write_text(json.dumps(record), encoding="utf-8")
    actual, evidence = _require_registered_reference(tmp_path, reference, registration)
    assert actual == reference and evidence["samples_per_pixel"] == 320
    # A control at another sample count cannot silently replace the planned one.
    record["samples_per_pixel"] = 40
    registration.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="sample count differs"):
        _require_registered_reference(tmp_path, reference, registration)
    record["samples_per_pixel"] = 320
    registration.write_text(json.dumps(record), encoding="utf-8")
    original = reference.read_bytes()
    reference.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="differs from its recorded hash"):
        _require_registered_reference(tmp_path, reference, registration)
    reference.write_bytes(original)
    plan_path.write_text(json.dumps({**plan, "samples_per_pixel": 40}), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its recorded hash"):
        _require_registered_reference(tmp_path, reference, registration)


def test_b1_generator_rejects_changed_plan_before_runtime_validation(tmp_path, monkeypatch) -> None:
    from scripts import nephele_b1_diagnostic as tool

    plan_path = tmp_path / "plan.json"
    plan_path.write_text('{"samples_per_pixel": 320}', encoding="utf-8")
    published = _hash(plan_path)
    plan_path.write_text('{"samples_per_pixel": 40}', encoding="utf-8")
    def forbidden_validation(*args):
        pytest.fail("runtime validation must not run for an unpublished or changed plan")
    monkeypatch.setattr(tool, "_validate_control_plan", forbidden_validation)
    for digest in (None, published):
        with pytest.raises(ValueError, match="digest published before generation"):
            tool._generate_control(tmp_path, plan_path, digest)


def test_b1_generator_rejects_wrong_imported_native(tmp_path) -> None:
    from types import SimpleNamespace
    from scripts.nephele_b1_diagnostic import _verify_imported_native

    extracted = tmp_path / "wheel"
    extracted.mkdir()
    expected = extracted / "_forge3d.pyd"
    expected.write_bytes(b"reference native")
    runtime = {"native_sha256": _hash(expected)}
    native = SimpleNamespace(__file__=str(expected))
    assert _verify_imported_native(native, runtime, extracted)["sha256"] == runtime["native_sha256"]
    expected.write_bytes(b"candidate native")
    with pytest.raises(RuntimeError, match="imported reference native differs"):
        _verify_imported_native(native, runtime, extracted)
    outside = tmp_path / "_forge3d.pyd"
    outside.write_bytes(b"reference native")
    with pytest.raises(RuntimeError, match="imported reference native differs"):
        _verify_imported_native(SimpleNamespace(__file__=str(outside)), runtime, extracted)


def _object(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_tag_preserves_provenance_source_revision() -> None:
    provenance = _object(ROOT / "tests/nephele/fixture/reference-provenance.json")
    tagged_revision = subprocess.check_output(
        ["git", "rev-parse", "nephele-reference-6a4ae50a^{commit}"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert tagged_revision == provenance["source_revision"]


def test_blue_noise_is_generated_rank_data_with_bound_source_and_license() -> None:
    subprocess.run([sys.executable, "scripts/generate_media_blue_noise.py"], cwd=ROOT, check=True)
    tile_path = ROOT / "assets/media/nephele_blue_noise_8x8.txt"
    provenance = _object(ROOT / "assets/media/nephele_blue_noise_provenance.json")
    ranks = [int(value) for line in tile_path.read_text(encoding="ascii").splitlines() if not line.startswith("#") for value in line.split()]
    bayer = [0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21]
    assert sorted(ranks) == list(range(64))
    assert ranks != bayer
    assert provenance["algorithm"] == "deterministic-toroidal-void-and-cluster-rank-v1"
    assert provenance["source"] == "repository-generated; no third-party tile bytes"
    assert provenance["license"] == "MIT" and (ROOT / provenance["license_file"]).is_file()
    assert provenance["tile_sha256"] == _hash(tile_path)
    assert provenance["generator_sha256"] == _hash(ROOT / provenance["generator"])


def test_reference_only_mask_equations_and_shaft_depth_fail_closed(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    output = tmp_path / "masks"
    fixture.mkdir()
    shape = (16, 16)
    terrain_hit = np.zeros(shape, dtype=np.bool_)
    terrain_hit[8:] = True
    transmittance = np.full((*shape, 3), 0.5, dtype=np.float32)
    cloud_shadow = np.full((*shape, 3), 0.5, dtype=np.float32)
    lighting = np.full(shape, 2, dtype=np.uint8)
    lighting[:, 2:14] = 1
    terrain_slice = np.full(shape, 64.0, dtype=np.float32)
    terrain_slice[terrain_hit] = 32.0
    for name, value in {
        "reference-terrain-hit.npy": terrain_hit,
        "reference-transmittance.npy": transmittance,
        "reference-cloud-shadow-aov.npy": cloud_shadow,
        "reference-media-lighting-visibility.npy": lighting,
        "reference-terrain-slice.npy": terrain_slice,
    }.items():
        np.save(fixture / name, value, allow_pickle=False)
    rules = fixture / "mask-rules.json"
    rules.write_text(
        json.dumps({
            "schema": "forge3d.nephele.mask_rules/3",
            "rules": EXPECTED_RULES,
            "technical_contracts": TECHNICAL_CONTRACTS,
        }),
        encoding="utf-8",
    )
    generate_masks(fixture, rules, output)
    shaft = np.load(output / MASK_FILES["shaft_mask"], allow_pickle=False)
    roi = np.load(output / MASK_FILES["godray_roi_mask"], allow_pickle=False)
    assert np.all(terrain_hit[shaft]) and np.all(terrain_slice[shaft] < 64.0)
    assert roi[:, 2:14].all() and not roi[:, :2].any() and not roi[:, 14:].any()

    terrain_slice[shaft] = 64.0
    np.save(fixture / "reference-terrain-slice.npy", terrain_slice, allow_pickle=False)
    with np.testing.assert_raises_regex(ValueError, "finite exact terrain-hit depth"):
        generate_masks(fixture, rules, output)


def test_convergence_uses_actual_gate3_and_approved_gate4_metrics() -> None:
    shape = (12, 12)
    old = np.full((*shape, 3), 128, dtype=np.uint8)
    masks = {
        "sky_cloud_mask": np.ones(shape, dtype=np.bool_),
        "godray_roi_mask": np.ones(shape, dtype=np.bool_),
        "terrain_mask": np.ones(shape, dtype=np.bool_),
        "cloud_shadow_mask": np.ones(shape, dtype=np.bool_),
    }
    metrics = downstream_metrics(old, old.copy(), masks)
    assert metrics == {
        "gate3_sky_cloud_delta_e_below_2_5_fraction": 1.0,
        "gate3_godray_roi_ssim": 1.0,
        "gate4_cloud_shadow_terrain_delta_e_below_2_fraction": 1.0,
    }
    changed = old.copy()
    changed[0, 0] = 255
    metrics = downstream_metrics(old, changed, masks)
    assert metrics["gate4_cloud_shadow_terrain_delta_e_below_2_fraction"] == 143 / 144


def test_spatial_partition_is_exact_and_worker_independent() -> None:
    tiles = _spatial_tiles()
    assert tiles == [
        {"x": 0, "y": y, "width": 64, "height": 1}
        for y in range(64)
    ]
    coverage = np.zeros((64, 64), dtype=np.uint8)
    for tile in tiles:
        coverage[
            tile["y"] : tile["y"] + tile["height"],
            tile["x"] : tile["x"] + tile["width"],
        ] += 1
    assert np.array_equal(coverage, np.ones((64, 64), dtype=np.uint8))
    # executor.map preserves canonical tile order; the worker count changes
    # scheduling only and never sample identity or output assembly.
    expected = np.concatenate([np.full((1, 64), tile["y"]) for tile in tiles])
    for worker_count in (1, 2, 7, 64):
        scheduled = [tile for lane in range(worker_count) for tile in tiles[lane::worker_count]]
        actual = np.concatenate([
            np.full((1, 64), tile["y"])
            for tile in sorted(scheduled, key=lambda tile: tile["y"])
        ])
        assert np.array_equal(actual, expected)


def test_reference_density_shape_and_scale_fail_closed() -> None:
    density = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
    medium = {"domain": {"grid_shape": [2, 3, 4]}, "density_scale": 2.5}
    native_density, density_scale = _validated_grid_density(density, medium)
    assert native_density.shape == (4, 3, 2)
    assert native_density.dtype == np.float32 and native_density.flags.c_contiguous
    assert density_scale == 2.5

    medium["domain"]["grid_shape"] = [4, 3, 2]
    with np.testing.assert_raises_regex(ValueError, "not tracked grid_shape"):
        _validated_grid_density(density, medium)

    medium["domain"]["grid_shape"] = [2, 3, 4]
    for invalid in (0.0, -1.0, float("nan"), True, "1.0"):
        medium["density_scale"] = invalid
        with np.testing.assert_raises_regex(ValueError, "finite and positive"):
            _validated_grid_density(density, medium)


def test_fixture_medium_binds_exact_f16_transport_and_n_cubed_majorants() -> None:
    medium = _object(ROOT / "tests/nephele/fixture/medium.json")
    shape = tuple(medium["domain"]["grid_shape"])
    represented, f16_sha256 = represented_density(medium["density_r16"], shape)
    runtime_sigma_t = max(
        float(np.float32(a) + np.float32(s))
        for a, s in zip(medium["sigma_a"], medium["sigma_s"])
    )
    assert medium["density_transport"] == {
        "schema": "forge3d.nephele.density_transport/1",
        "decode": "unorm16-div-65535-as-f32-then-ieee-f16-rne",
        "storage": "ieee-f16-bits-little-endian",
        "sampling": DENSITY_SAMPLING,
        "f16_sha256": f16_sha256,
    }
    assert medium["transport"] == {
        "sigma_t_spectrum": [0.065, 0.062, 0.06],
        "sigma_t_max_channel": 0.065,
        "extinction_channel": 0,
        "slab_axis": 2,
    }
    assert medium["majorant_transport"]["grid_shape"] == list(shape)
    assert medium["majorant_transport"]["query"] == MAJORANT_QUERY
    assert len(medium["majorant_cells"]) == np.prod(shape)
    assert medium["majorant_cells"] == canonical_majorant_cells(
        represented, medium["density_scale"], runtime_sigma_t
    )


def test_fixture_manifest_hashes_every_input_and_masks_regenerate(tmp_path: Path) -> None:
    manifest = _object(ROOT / "tests/nephele/fixture-manifest.json")
    assert manifest["schema"] == "forge3d.nephele.fixture_manifest/3"
    assert manifest["revision"] == 2
    assert manifest["status"] in {"APPROVED", "UNRESOLVED"}
    assert manifest["color_pipeline"] == {
        "linear_input": "linear-sRGB",
        "tonemap": "aces-fitted",
        "output_encoding": "IEC 61966-2-1 sRGB",
        "quantization": "round-to-nearest uint8",
        "exposure": manifest["scene_inputs"]["exposure"]["sha256"],
        "tonemap_contract": manifest["scene_inputs"]["tonemap"]["sha256"],
        "crop": manifest["scene_inputs"]["crop"]["sha256"],
    }
    assert set(manifest["scene_inputs"]) == {
        "camera", "terrain_dem", "terrain", "medium", "sun", "atmosphere",
        "material", "exposure", "tonemap", "crop",
    }
    assert set(manifest["files"]) == {"reference_rgb", "reference_transmittance", "reference_in_scatter", "reference_cloud_shadow_aov", "reference_optical_depth", "sky_cloud_mask", "godray_roi_mask", "terrain_mask", "cloud_shadow_mask", "shaft_mask", "reference_terrain_slice", "reference_terrain_hit", "reference_media_lighting_visibility"}
    for record in [*manifest["scene_inputs"].values(), *manifest["files"].values()]:
        path = ROOT / record["path"]
        assert record["artifact"] == path.name
        assert record["sha256"] == _hash(path)
    terrain = _object(ROOT / manifest["scene_inputs"]["terrain"]["path"])
    dem_path = ROOT / manifest["scene_inputs"]["terrain_dem"]["path"]
    assert terrain["dem_sha256"] == _hash(dem_path)
    assert Path(terrain["dem"]).name == dem_path.name
    camera = _object(ROOT / manifest["scene_inputs"]["camera"]["path"])
    assert set(camera) == {"schema", "fov_y", "terrain_camera"}
    assert camera["terrain_camera"] == {
        "mode": "mesh:yup", "radius": 50.990195, "phi_deg": 90.0,
        "theta_deg": 78.690068, "target": [0.0, 15.0, 0.0],
    }
    assert camera["fov_y"] == 45.0
    material = _object(ROOT / manifest["scene_inputs"]["material"]["path"])
    assert material == {
        "schema": "forge3d.nephele.material/1",
        "albedo": [0.6, 0.6, 0.6],
        "metallic": 0.0,
        "roughness": 1.0,
        "triplanar_scale": 1.0,
        "normal_strength": 0.0,
        "blend_sharpness": 1.0,
        "colormap_strength": 0.0,
        "albedo_mode": "material",
    }
    crop = _object(ROOT / manifest["scene_inputs"]["crop"]["path"])
    assert crop == {
        "schema": "forge3d.nephele.crop/1", "x": 0, "y": 0,
        "width": 64, "height": 64, "full_viewport": [64, 64],
    }
    provenance = _object(ROOT / "tests/nephele/fixture/reference-provenance.json")
    assert provenance["schema"] == "forge3d.nephele.reference_provenance/2"
    assert provenance["algorithm"] == "integrated-hybrid-terrain-spectral-ratio-delta-tracking-surface-env-mis-v3"
    assert provenance["samples_per_pixel"] > 0
    assert len(provenance["source_revision"]) == 40
    assert provenance["assembled_sources"] == {
        "terrain_media_reference_module_sha256": hashlib.sha256(
            _terrain_media_reference_module_source()
        ).hexdigest()
    }
    diagnostics = provenance["diagnostics"]
    assert set(diagnostics) == DIAGNOSTIC_FIELDS
    assert diagnostics["majorant_proof"] == "TrilinearConvexHull"
    assert diagnostics["majorant_valid"] is True
    assert diagnostics["executed_multi_scatter"] is True
    assert diagnostics["sample_count"] == 64 * 64 * provenance["samples_per_pixel"]
    assert diagnostics["temporal_history_decision"] == "not_applicable"
    assert diagnostics["single_scatter_luminance"] is None
    assert diagnostics["multiple_scatter_luminance"] is None
    assert diagnostics["energy_accounting_residual"] is None
    if provenance["acceptance_eligible"]:
        assert provenance["diagnostic_reason"] is None
        assert provenance["sample_identity"] == {
            "algorithm": "per-pixel-absolute-sample-index-v1",
            "seed": provenance["seed"],
            "range": [0, provenance["samples_per_pixel"]],
        }
        assert provenance["spatial_tiles"] == _spatial_tiles()
        assert provenance["runtime_partition"]["kind"] == "process_pool_max_workers"
        assert provenance["runtime_partition"]["value"] >= 1
        assert diagnostics["host_visible_bytes"] == 64 * 64 * (5 * 3 * 4 + 4 + 2)
        assert set(provenance["scene_inputs"]) == set(manifest["scene_inputs"])
        for role, digest in provenance["scene_inputs"].items():
            assert digest == manifest["scene_inputs"][role]["sha256"]
        for relative, expected in provenance["source_inputs"].items():
            assert expected == _hash(ROOT / relative)
        assert set(provenance["native_runtime"]) == {
            "source_revision", "wheel_filename", "wheel_sha256",
            "wheel_native_member", "native_sha256",
        }
        assert provenance["native_runtime"]["source_revision"] == provenance["diagnostics"]["source_revision"]
        assert set(provenance["camera_contract"]) == {
            "origin", "look_at", "up", "right", "forward", "fov_y",
        }
        for key in ("origin", "look_at", "up", "right", "forward"):
            vector = provenance["camera_contract"][key]
            assert np.asarray(vector, dtype=np.float32).shape == (3,)
        assert np.asarray(provenance["camera_contract"]["fov_y"], dtype=np.float32).shape == ()
    else:
        assert provenance["diagnostic_reason"]
        if provenance["sample_identity"] is None:
            assert provenance["native_runtime"] is None
            assert provenance["spatial_tiles"] is None
        else:
            assert provenance["sample_identity"] == {
                "algorithm": "per-pixel-absolute-sample-index-v1",
                "seed": provenance["seed"],
                "range": [0, provenance["samples_per_pixel"]],
            }
            assert provenance["spatial_tiles"] == _spatial_tiles()
            if provenance["native_runtime"] is not None:
                assert set(provenance["native_runtime"]) == {
                    "source_revision", "wheel_filename", "wheel_sha256",
                    "wheel_native_member", "native_sha256",
                }
    convergence = _object(ROOT / "tests/nephele/fixture/reference-convergence.json")
    assert convergence["status"] in {"CONVERGED", "UNRESOLVED"}
    assert manifest["status"] == ("APPROVED" if convergence["status"] == "CONVERGED" else "UNRESOLVED")
    assert convergence["generator_sha256"] == _hash(ROOT / convergence["generator"])
    assert convergence["final"]["rgb_sha256"] == _hash(ROOT / convergence["final"]["rgb_path"])
    assert convergence["final"]["provenance_sha256"] == _hash(ROOT / convergence["final"]["provenance_path"])
    final_provenance = _object(ROOT / convergence["final"]["provenance_path"])
    assert final_provenance["samples_per_pixel"] == convergence["final"]["samples_per_pixel"]
    if convergence["status"] == "CONVERGED":
        assert convergence["nested_prefix"] is True
        previous_provenance = _object(ROOT / convergence["previous"]["provenance_path"])
        assert previous_provenance["sample_identity"]["range"] == [0, convergence["previous"]["samples_per_pixel"]]
        assert final_provenance["sample_identity"]["range"] == [0, convergence["final"]["samples_per_pixel"]]
        assert convergence["spatial_tiles_identity"] is True
        assert all(convergence["mask_identity"].values())
        assert convergence["metrics"]["gate3_sky_cloud_delta_e_below_2_5_fraction"] >= 0.95
        assert convergence["metrics"]["gate3_godray_roi_ssim"] > 0.95
        assert convergence["metrics"]["gate4_cloud_shadow_terrain_delta_e_below_2_fraction"] >= 0.95
    else:
        assert convergence["nested_prefix"] is False
        expected_next = (
            2 * convergence["final"]["samples_per_pixel"]
            if final_provenance["acceptance_eligible"] else None
        )
        assert convergence["next_samples_per_pixel_if_unresolved"] == expected_next
    rules = ROOT / manifest["mask_rules"]["path"]
    assert _object(rules) == {
        "schema": "forge3d.nephele.mask_rules/3",
        "rules": EXPECTED_RULES,
        "technical_contracts": TECHNICAL_CONTRACTS,
    }
    assert TECHNICAL_CONTRACTS["godray_roi_minimum_shape"] == {
        "source": "tests/_ssim.py",
        "parameter": "ssim.win_size",
        "value": 11,
    }
    generate_masks(ROOT / "tests/nephele/fixture", rules, tmp_path)
    for role, filename in MASK_FILES.items():
        assert _hash(tmp_path / filename) == manifest["files"][role]["sha256"]
        mask = np.load(tmp_path / filename, allow_pickle=False)
        assert mask.any() and not mask.all()
    terrain_hit = np.load(ROOT / manifest["files"]["reference_terrain_hit"]["path"], allow_pickle=False)
    lighting = np.load(ROOT / manifest["files"]["reference_media_lighting_visibility"]["path"], allow_pickle=False)
    terrain_slice = np.load(ROOT / manifest["files"]["reference_terrain_slice"]["path"], allow_pickle=False)
    shaft = np.load(tmp_path / MASK_FILES["shaft_mask"], allow_pickle=False)
    assert terrain_hit.dtype == np.bool_ and terrain_hit.any() and (~terrain_hit).any()
    assert lighting.dtype == np.uint8 and set(np.unique(lighting)) <= {0, 1, 2}
    assert np.all(terrain_hit[shaft])
    assert np.isfinite(terrain_slice[shaft]).all() and np.all(terrain_slice[shaft] < 64.0)
    assert len({_hash(tmp_path / filename) for filename in MASK_FILES.values()}) > 1
    assert not (ROOT / "tests/nephele/fixture/reference-rgb-spp4096.npy").exists()
    assert not (ROOT / "tests/nephele/fixture/reference-provenance-spp4096.json").exists()


def test_gate4_and_exhaustive_lane_inventory_are_literal() -> None:
    policy = _object(ROOT / "tests/nephele/gate4-policy.json")
    assert policy["status"] == "APPROVED"
    assert policy["aggregation"] == {"kind": "per_pixel_pass_fraction", "minimum_fraction": 0.95}
    inventory = _object(ROOT / "tests/nephele/evidence-inventory.json")
    assert inventory["gate4_aggregation"] == "per_pixel_pass_fraction"
    assert inventory["required_junit_cases"] == [
        "gate1_estimator_majorant_rr", "gate2_energy", "gate3_realtime_reference",
        "gate4_terrain_coupling", "gate5_compute_shadow_ridgeline", "gate6_determinism_memory",
    ]
    required = set(inventory["required_raw_artifacts"])
    assert {"heterogeneous-comparator.samples.bin", "majorant-domain-probe.pairs.bin", "rr-evidence.json", "rr-contributions.bin"} <= required
    assert inventory["transport_representation"] == {
        "comparator_algorithm": "independent-f16-texel-center-column-bernoulli-v3",
        "comparator_schema": "forge3d.nephele.heterogeneous_comparator/4",
        "density_sampling": DENSITY_SAMPLING,
        "density_storage": "ieee-f16-bits-little-endian",
        "majorant_cells_topology": "N^3",
        "majorant_probe_mapping": "domain-boundaries-texel-extrema-majorant-boundaries-then-irrational-lattice-v3",
        "majorant_probe_schema": "forge3d.nephele.majorant_probe/3",
        "majorant_query": MAJORANT_QUERY,
    }
    assert "tests/test_nephele_physical.py" not in (ROOT / "tests/UNRUN.toml").read_text(encoding="utf-8")
