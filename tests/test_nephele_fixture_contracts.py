"""Tracked NEPHELE fixture contracts: scene inputs, masks, medium and blue noise."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.nephele_acceptance import canonical_majorant_cells, represented_density
from scripts.nephele_fixture_masks import (
    EXPECTED_RULES,
    MASK_FILES,
    TECHNICAL_CONTRACTS,
    generate_masks,
)
from scripts.generate_media_fixture import _spatial_tiles, _validated_grid_density


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/nephele/fixture"


def _object(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    medium = _object(FIXTURE / "medium.json")
    shape = medium["domain"]["grid_shape"]
    f16 = represented_density(medium).astype(np.float16)
    assert medium["density_transport"]["f16_sha256"] == hashlib.sha256(f16.astype("<f2").tobytes()).hexdigest()
    assert medium["transport"] == {
        "sigma_t_spectrum": [0.065, 0.062, 0.06],
        "sigma_t_max_channel": 0.065,
        "extinction_channel": 0,
        "slab_axis": 2,
    }
    assert medium["majorant_transport"]["grid_shape"] == shape
    assert len(medium["majorant_cells"]) == np.prod(shape)
    assert medium["majorant_cells"] == canonical_majorant_cells(medium)


def test_fixture_scene_inputs_and_masks_regenerate(tmp_path: Path) -> None:
    terrain = _object(FIXTURE / "terrain.json")
    assert terrain["dem_sha256"] == _hash(FIXTURE / terrain["dem"])
    assert _object(FIXTURE / "camera.json")["terrain_camera"] == {
        "mode": "mesh:yup", "radius": 50.990195, "phi_deg": 90.0,
        "theta_deg": 78.690068, "target": [0.0, 15.0, 0.0],
    }
    assert _object(FIXTURE / "crop.json") == {
        "schema": "forge3d.nephele.crop/1", "x": 0, "y": 0,
        "width": 64, "height": 64, "full_viewport": [64, 64],
    }
    provenance = _object(FIXTURE / "reference-provenance.json")
    assert provenance["samples_per_pixel"] > 0
    assert set(provenance["camera_contract"]) == {"origin", "look_at", "up", "right", "forward", "fov_y"}
    diagnostics = provenance["diagnostics"]
    assert diagnostics["majorant_valid"] is True and diagnostics["executed_multi_scatter"] is True
    assert diagnostics["sample_count"] == 64 * 64 * provenance["samples_per_pixel"]
    rules = FIXTURE / "mask-rules.json"
    assert _object(rules) == {
        "schema": "forge3d.nephele.mask_rules/3",
        "rules": EXPECTED_RULES,
        "technical_contracts": TECHNICAL_CONTRACTS,
    }
    generate_masks(FIXTURE, rules, tmp_path)
    for filename in MASK_FILES.values():
        assert _hash(tmp_path / filename) == _hash(FIXTURE / filename), filename
        mask = np.load(tmp_path / filename, allow_pickle=False)
        assert mask.any() and not mask.all()
    terrain_hit = np.load(FIXTURE / "reference-terrain-hit.npy", allow_pickle=False)
    terrain_slice = np.load(FIXTURE / "reference-terrain-slice.npy", allow_pickle=False)
    shaft = np.load(tmp_path / MASK_FILES["shaft_mask"], allow_pickle=False)
    assert np.all(terrain_hit[shaft])
    assert np.isfinite(terrain_slice[shaft]).all() and np.all(terrain_slice[shaft] < 64.0)
