from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import forge3d
from forge3d.codec import compress_dem, decompress_dem, verify_dem


CORPUS = Path(__file__).parent / "data" / "codec_corpus"
EPSILONS = (0.05, 0.1, 0.5, 1.0)
TILES = ("alpine", "rolling", "coastal_flat_nodata", "canyon")


def _source(name: str) -> np.ndarray:
    return np.load(CORPUS / f"{name}.npy", allow_pickle=False)


@lru_cache(maxsize=None)
def _case(name: str, eps: float, progressive: bool) -> tuple[bytes, dict]:
    source = _source(name)
    encoded = compress_dem(source, eps, progressive=progressive)
    return encoded, verify_dem(encoded, source)


def _finite_max_error(source: np.ndarray, decoded: np.ndarray) -> float:
    source_nan = np.isnan(source)
    assert np.array_equal(source_nan, np.isnan(decoded))
    finite = ~source_nan
    return float(np.max(np.abs(decoded[finite] - source[finite]), initial=0.0))


def test_committed_real_corpus_manifest_and_hashes() -> None:
    import tomllib

    manifest = tomllib.loads((CORPUS / "MANIFEST.toml").read_text())
    assert manifest["format"] == "forge3d-f3dz-corpus/1"
    assert manifest["license"].startswith("Public domain")
    assert [tile["name"] for tile in manifest["tiles"]] == list(TILES)
    for tile in manifest["tiles"]:
        payload = (CORPUS / tile["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == tile["output_sha256"]
        source = _source(tile["name"])
        assert source.shape == (256, 256)
        assert source.dtype == np.dtype("<f4")
    coastal = _source("coastal_flat_nodata")
    assert 0 < np.count_nonzero(np.isnan(coastal)) < coastal.size
    assert all(not np.isnan(_source(name)).any() for name in TILES if name != "coastal_flat_nodata")


@pytest.mark.parametrize("eps", EPSILONS)
@pytest.mark.parametrize("name", TILES)
def test_error_bound_stored_page_error_nan_and_determinism(name: str, eps: float) -> None:
    source = _source(name)
    encoded, report = _case(name, eps, True)
    second = compress_dem(source, eps, progressive=True)
    decoded, info = decompress_dem(encoded)

    assert encoded == second
    assert hashlib.sha256(encoded).digest() == hashlib.sha256(second).digest()
    assert info["codec"] == "f3dz/1"
    assert info["eps"] == float(np.float32(eps))
    assert not info["base_quality"]
    assert _finite_max_error(source, decoded) <= eps
    assert report["valid"] and report["crc_ok"] and report["header_consistent"]
    assert report["all_within_epsilon"]
    assert report["stored_errors_match"]
    assert all(page["within_epsilon"] for page in report["pages"])
    assert all(page["stored_error_matches"] for page in report["pages"])


def test_flate2_compression_win_and_predictor_ablation() -> None:
    rows: list[str] = []
    for eps in EPSILONS:
        f3dz_total = 0
        baseline_total = 0
        ablation_total = 0
        for name in TILES:
            encoded, report = _case(name, eps, False)
            f3dz = len(encoded)
            baseline = report["baseline_flate2_stream_size"]
            ablation = report["ablation_stream_size"]
            assert baseline is not None and ablation is not None
            assert f3dz < baseline, f"{name} eps={eps}: {f3dz=} must beat {baseline=}"
            f3dz_total += f3dz
            baseline_total += baseline
            ablation_total += ablation
            rows.append(
                f"{name:24s} eps={eps:>4} f3dz={f3dz:6d} "
                f"flate2={baseline:6d} order0={ablation:6d} "
                f"saving={1.0 - f3dz / baseline:7.2%}"
            )
        ratio = f3dz_total / baseline_total
        ablation_ratio = ablation_total / baseline_total
        assert ratio <= 0.70, f"eps={eps}: corpus ratio {ratio:.6f} exceeds 0.70"
        assert ablation_ratio > 0.70, (
            f"eps={eps}: order-0 ratio {ablation_ratio:.6f} unexpectedly retains the 30% win"
        )
        rows.append(
            f"{'CORPUS MEAN':24s} eps={eps:>4} f3dz={f3dz_total:6d} "
            f"flate2={baseline_total:6d} order0={ablation_total:6d} "
            f"saving={1.0 - ratio:7.2%}"
        )
    print("\n".join(rows))


@pytest.mark.parametrize("eps", EPSILONS)
@pytest.mark.parametrize("name", TILES)
def test_progressive_base_layer_is_bounded_and_exactly_declared(name: str, eps: float) -> None:
    _, report = _case(name, eps, True)
    base = report["base"]
    assert base is not None
    assert base["base_quality"]
    assert base["within_4epsilon"]
    assert base["max_abs_err"] <= 4.0 * eps
    assert base["stored_errors_match"]
    assert all(page["stored_error_matches"] for page in base["pages"])
    assert base["degradation_kind"] == "base_quality"
    assert base["degradation_name"] == "f3dz_unrefined_pages"


def test_base_quality_render_capture_declares_degradation_and_refined_does_not() -> None:
    source = _source("alpine")
    encoded = compress_dem(source, 0.1, progressive=True)

    forge3d.clear_native_degradations()
    forge3d.begin_render_execution_capture("f3dz.base-quality.test")
    report = verify_dem(encoded, source)
    assert report["base"]["base_quality"]
    forge3d.finish_render_execution_capture("f3dz.decode", 1)
    base_capture = json.loads(forge3d.render_execution_report())
    assert base_capture["codec"] == {
        "codec": "f3dz/1",
        "eps": 0.1,
        "pages_base_quality": 16,
    }
    assert any(
        entry["kind"] == "base_quality" and entry["name"] == "f3dz_unrefined_pages"
        for entry in base_capture["degradations"]
    )

    forge3d.clear_native_degradations()
    forge3d.begin_render_execution_capture("f3dz.refined.test")
    decompress_dem(encoded)
    forge3d.finish_render_execution_capture("f3dz.decode", 1)
    refined_capture = json.loads(forge3d.render_execution_report())
    assert refined_capture["codec"] == {
        "codec": "f3dz/1",
        "eps": 0.1,
        "pages_base_quality": 0,
    }
    assert not any(
        entry["kind"] == "base_quality" and entry["name"] == "f3dz_unrefined_pages"
        for entry in refined_capture["degradations"]
    )


def test_gpu_matches_cpu_for_every_corpus_page() -> None:
    unavailable: list[str] = []
    for name in TILES:
        _, report = _case(name, 0.1, True)
        gpu = report["gpu"]
        if not gpu["available"]:
            unavailable.append(f"{name}: {gpu['error']}")
            continue
        assert gpu["bit_identical"]
        assert gpu["gpu_sha256"] == gpu["cpu_sha256"]
        assert gpu["gpu_page_sha256"] == gpu["cpu_page_sha256"]
        assert len(gpu["gpu_page_sha256"]) == 16
    if unavailable:
        if os.getenv("FORGE3D_REQUIRE_F3DZ_GPU") == "1":
            pytest.fail("physical GPU gate required but unavailable:\n" + "\n".join(unavailable))
        pytest.skip("no GPU adapter available for F3DZ identity gate")


def test_cross_platform_determinism_hashes() -> None:
    import tomllib

    expected = tomllib.loads((CORPUS / "DETERMINISM.toml").read_text())
    for name in TILES:
        source = _source(name)
        for eps in EPSILONS:
            encoded = compress_dem(source, eps, progressive=True)
            key = f"eps_{str(eps).replace('.', '_')}"
            assert hashlib.sha256(encoded).hexdigest() == expected[name][key]
