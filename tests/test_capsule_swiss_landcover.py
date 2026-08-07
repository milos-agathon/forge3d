import json
from pathlib import Path

CAPSULE = Path(__file__).resolve().parents[1] / "examples" / "capsules" / "swiss-landcover"


def _registry():
    from forge3d import datasets as ds

    return ds._REMOTE_DATASETS


def test_manifest_matches_registry_hashes():
    manifest = json.loads((CAPSULE / "data-manifest.json").read_text(encoding="utf-8"))
    reg = _registry()
    by_name = {d["name"]: d for d in manifest["datasets"]}
    assert set(by_name) == {"swiss", "swiss-land-cover"}
    for name, entry in by_name.items():
        assert entry["sha256"] == reg[name].known_hash.removeprefix("sha256:")
        assert entry["file"] == reg[name].filename


def test_provenance_complete_no_tbd():
    prov = json.loads((CAPSULE / "provenance.json").read_text(encoding="utf-8"))
    assert {r["dataset"] for r in prov["records"]} == {"swiss", "swiss-land-cover"}
    required = {
        "dataset",
        "distributed_file",
        "distributed_sha256",
        "upstream_provider",
        "upstream_product",
        "upstream_url",
        "license_basis",
        "required_attribution",
        "acquired_date",
        "transformations",
        "notes",
    }
    for record in prov["records"]:
        assert required <= set(record), f"missing keys in {record['dataset']}"
        for key in required:
            assert "TBD" not in json.dumps(record[key]), f"TBD left in {record['dataset']}.{key}"


def test_landcover_attribution_line_exact():
    prov = json.loads((CAPSULE / "provenance.json").read_text(encoding="utf-8"))
    lc = next(r for r in prov["records"] if r["dataset"] == "swiss-land-cover")
    assert lc["required_attribution"] == (
        "Data: Sentinel-2 10m Land Use/Land Cover – Esri, Impact Observatory, and Microsoft"
    )


def _load_recipe_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("swiss_capsule_recipe", CAPSULE / "recipe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_recipe_imports_without_gpu_and_declares_contract():
    mod = _load_recipe_module()
    # Proven Swiss PT register, verbatim (spec: do not re-derive).
    assert (mod.GRID_MAX, mod.FRAME, mod.TILES, mod.MAX_FRAMES, mod.SEED) == (1536, 1120, 2, 2048, 7)
    assert mod.SNAPSHOT.name == "swiss_landcover.png"
    assert mod.CERT_PATH.name == "local.certificate.json"
    assert mod.EXIT_NO_GPU == 2 and mod.EXIT_NO_HITS == 3 and mod.EXIT_NO_CERT == 4


def test_recipe_pins_vendored_pt_register_literals():
    # The upstream examples/swiss_landcover_pt_3d.py is gitignored and untracked,
    # so it cannot serve as a drift guard. These literals are the guard.
    mod = _load_recipe_module()
    assert mod.SUN_AZIMUTH == 135.0
    assert mod.SUN_ELEVATION == 28.0
    assert mod.SUN_INTENSITY == 2.6
    assert mod.ENV_INTENSITY == 0.82
    assert mod.RELIEF_WORLD == 3.2
    assert mod.CAMERA_FOV_Y == 8.0
    assert mod.CAMERA_MARGIN == 1.06
    assert mod.SPAN_X == 100.0
    assert mod.PT_ALBEDO == (0.62, 0.62, 0.62)


def test_recipe_result_writer_binds_recipe_inputs_and_image(tmp_path):
    import datetime
    import hashlib

    mod = _load_recipe_module()
    assert mod.RESULT_PATH.name == "recipe_result.json"

    png = tmp_path / "swiss_landcover.png"
    png.write_bytes(b"png-bytes")
    cert = tmp_path / "local.certificate.json"
    cert.write_bytes(b"{}")
    dem = tmp_path / "switzerland_dem.tif"
    dem.write_bytes(b"dem-bytes")
    lc = tmp_path / "switzerland_land_cover.tif"
    lc.write_bytes(b"lc-bytes")
    out = tmp_path / "recipe_result.json"

    returned = mod.write_recipe_result(
        out, snapshot=png, certificate=cert,
        inputs=[("swiss", dem), ("swiss-land-cover", lc)],
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == returned
    assert set(payload) == {
        "png_sha256", "inputs", "recipe_sha256", "certificate_sha256",
        "forge3d_version", "created",
    }
    assert payload["png_sha256"] == hashlib.sha256(b"png-bytes").hexdigest()
    assert payload["certificate_sha256"] == hashlib.sha256(b"{}").hexdigest()
    # Defaults to hashing recipe.py itself, so the manifest binds the code that ran.
    # Text artifacts are hashed newline-normalized so LF/CRLF checkouts agree.
    assert payload["recipe_sha256"] == hashlib.sha256(
        (CAPSULE / "recipe.py").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()

    assert [entry["name"] for entry in payload["inputs"]] == ["swiss", "swiss-land-cover"]
    assert [entry["path_name"] for entry in payload["inputs"]] == [
        "switzerland_dem.tif", "switzerland_land_cover.tif",
    ]
    assert [entry["sha256"] for entry in payload["inputs"]] == [
        hashlib.sha256(b"dem-bytes").hexdigest(), hashlib.sha256(b"lc-bytes").hexdigest(),
    ]
    for entry in payload["inputs"]:
        assert set(entry) == {"name", "path_name", "sha256"}

    from forge3d import __version__ as forge3d_version

    assert payload["forge3d_version"] == forge3d_version
    datetime.date.fromisoformat(payload["created"])  # raises if not an ISO date


def test_recipe_has_no_pro_imports_and_no_telemetry():
    text = (CAPSULE / "recipe.py").read_text(encoding="utf-8")
    for banned in ("map_plate", "export_svg", "export_pdf", "add_buildings", "set_license_key",
                   "requests.", "urllib.request", "httpx"):
        assert banned not in text, f"banned reference in recipe.py: {banned}"


# ---------------------------------------------------------------------------
# verify.py — the single acceptance command
# ---------------------------------------------------------------------------
import hashlib  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402


def _load_verify_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("swiss_capsule_verify", CAPSULE / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha_raw(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_text(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _synthetic_cert(*, version="1.34.0", wgsl=None, degradations=()):
    return {
        "schema": "forge3d.render_certificate/1",
        "engine": {
            "version": version,
            "git_sha": "0123456789ab",
            "wgsl_module_hashes": dict(wgsl or {"hybrid-pt-kernel": "aa" * 32}),
        },
        "adapter": {"backend": "vulkan", "device": "Synthetic Adapter", "vendor": "4318"},
        "passes": [{"label": "hybrid_pt.terrain_gbuffer", "gpu_ms": 1.25, "draw_calls": 1}],
        "degradations": list(degradations),
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _provenance_record(dataset, file_name, sha):
    return {
        "dataset": dataset,
        "distributed_file": file_name,
        "distributed_sha256": sha,
        "upstream_provider": "provider",
        "upstream_product": "product",
        "upstream_url": "https://example.invalid/{z}/{x}/{y}.png",
        "license_basis": "CC BY 4.0",
        "required_attribution": "Data: synthetic",
        "acquired_date": "2026-01-01",
        "transformations": ["none"],
        "notes": "synthetic fixture",
    }


def _build_synthetic_capsule(
    tmp_path,
    *,
    local_pixel=128,
    reference_pixel=128,
    local_cert=None,
    reference_cert=None,
    root_binding=True,
):
    """A fully consistent capsule on disk, with no GPU and no network involved.

    Every hash written into the manifests is computed here with plain hashlib,
    so the cross-checks verify.py performs are never tautological.
    """
    capsule = tmp_path / "capsule"
    (capsule / "out").mkdir(parents=True, exist_ok=True)
    (capsule / "expected").mkdir(parents=True, exist_ok=True)

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    dem = data_dir / "switzerland_dem.tif"
    dem.write_bytes(b"dem-bytes")
    lc = data_dir / "switzerland_land_cover.tif"
    lc.write_bytes(b"lc-bytes")
    dataset_paths = {"swiss": dem, "swiss-land-cover": lc}

    (capsule / "recipe.py").write_text("# synthetic recipe\nX = 1\n", encoding="utf-8")

    reference_png = capsule / "expected" / "reference.png"
    Image.fromarray(np.full((24, 32, 3), reference_pixel, dtype=np.uint8)).save(reference_png)
    local_png = capsule / "out" / "swiss_landcover.png"
    if local_pixel == reference_pixel:
        shutil.copyfile(reference_png, local_png)
    else:
        Image.fromarray(np.full((24, 32, 3), local_pixel, dtype=np.uint8)).save(local_png)

    cert = local_cert if local_cert is not None else _synthetic_cert()
    ref_cert = reference_cert if reference_cert is not None else _synthetic_cert()
    _write_json(capsule / "out" / "local.certificate.json", cert)
    _write_json(capsule / "expected" / "reference.certificate.json", ref_cert)

    _write_json(
        capsule / "data-manifest.json",
        {
            "capsule": "swiss-landcover",
            "version": "0.1.0",
            "datasets": [
                {
                    "name": "swiss",
                    "file": dem.name,
                    "sha256": _sha_raw(dem),
                    "fetch": "forge3d.datasets.fetch_dem('swiss')",
                },
                {
                    "name": "swiss-land-cover",
                    "file": lc.name,
                    "sha256": _sha_raw(lc),
                    "fetch": "forge3d.datasets.fetch('swiss-land-cover')",
                },
            ],
        },
    )
    _write_json(
        capsule / "provenance.json",
        {
            "records": [
                _provenance_record("swiss", dem.name, _sha_raw(dem)),
                _provenance_record("swiss-land-cover", lc.name, _sha_raw(lc)),
            ]
        },
    )

    def _binding(png_path):
        return {
            "png_sha256": _sha_raw(png_path),
            "inputs": [
                {"name": "swiss", "path_name": dem.name, "sha256": _sha_raw(dem)},
                {"name": "swiss-land-cover", "path_name": lc.name, "sha256": _sha_raw(lc)},
            ],
            "recipe_sha256": _sha_text(capsule / "recipe.py"),
            "certificate_sha256": _sha_text(capsule / "out" / "local.certificate.json"),
            "forge3d_version": "1.34.0",
            "created": "2026-08-07",
        }

    _write_json(capsule / "out" / "recipe_result.json", _binding(local_png))
    if root_binding:
        _write_json(capsule / "recipe_result.json", _binding(reference_png))

    return capsule, dataset_paths


def test_verify_ssim_matches_repo_reference_impl():
    from tests._ssim import ssim as repo_ssim

    v = _load_verify_module()
    rng = np.random.default_rng(11)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8).astype(np.float64)
    b = np.clip(a + rng.normal(0, 6, a.shape), 0, 255)
    assert abs(v.ssim(a, b, data_range=255.0) - repo_ssim(a, b, data_range=255.0)) < 1e-9


def test_verify_compare_passes_and_fails_correctly():
    v = _load_verify_module()
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    cert = {"engine": {"version": "1.30.0", "wgsl_module_hashes": {"a": "h1"}}, "degradations": []}
    cfg = {"ssim_min": 0.99, "mean_abs_max": 2.0}
    ok = v.compare(img, img.copy(), cert, cert, cfg)
    assert ok["passed"] and ok["ssim"] == 1.0 and ok["engine_match"] and ok["wgsl_match"]
    bad_cert = {
        "engine": {"version": "1.30.0", "wgsl_module_hashes": {"a": "CHANGED"}},
        "degradations": [],
    }
    bad = v.compare(img, img.copy(), bad_cert, cert, cfg)
    assert not bad["passed"] and not bad["wgsl_match"]
    noisy = np.clip(img.astype(int) + 40, 0, 255).astype(np.uint8)
    assert not v.compare(noisy, img, cert, cert, cfg)["passed"]


def test_verify_compare_gates_engine_version_and_degradations():
    v = _load_verify_module()
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    ref = {"engine": {"version": "1.34.0", "wgsl_module_hashes": {"a": "h1"}}, "degradations": []}
    cfg = {"ssim_min": 0.99, "mean_abs_max": 2.0}

    older = {"engine": {"version": "1.30.0", "wgsl_module_hashes": {"a": "h1"}}, "degradations": []}
    out = v.compare(img, img.copy(), older, ref, cfg)
    assert not out["engine_match"] and not out["passed"]

    degraded = {
        "engine": {"version": "1.34.0", "wgsl_module_hashes": {"a": "h1"}},
        "degradations": [{"consequence": "fell back"}],
    }
    out = v.compare(img, img.copy(), degraded, ref, cfg)
    assert not out["degradations_empty"] and not out["passed"]

    # Adapter identity is informational only and must never gate.
    other_adapter = dict(ref, adapter={"device": "Some Other GPU", "backend": "dx12"})
    assert v.compare(img, img.copy(), other_adapter, ref, cfg)["passed"]


def test_verify_compare_result_is_json_serializable_and_shape_guarded():
    v = _load_verify_module()
    cert = {"engine": {"version": "1.34.0", "wgsl_module_hashes": {}}, "degradations": []}
    cfg = {"ssim_min": 0.99, "mean_abs_max": 2.0}
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    json.dumps(v.compare(img, img.copy(), cert, cert, cfg))  # raises on numpy scalars

    small = np.full((16, 16, 3), 128, dtype=np.uint8)
    out = v.compare(small, img, cert, cert, cfg)
    assert out["passed"] is False and "error" in out


def test_verify_text_hashes_are_newline_normalized_and_binaries_are_raw(tmp_path):
    v = _load_verify_module()
    lf = tmp_path / "lf.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert v.sha256_file(lf) == v.sha256_file(crlf)
    assert v.sha256_file(lf) == hashlib.sha256(b"a = 1\nb = 2\n").hexdigest()

    lf_json = tmp_path / "lf.json"
    lf_json.write_bytes(b'{\n "a": 1\n}\n')
    crlf_json = tmp_path / "crlf.json"
    crlf_json.write_bytes(b'{\r\n "a": 1\r\n}\r\n')
    assert v.sha256_file(lf_json) == v.sha256_file(crlf_json)

    # Rasters are byte streams: CRLF sequences in them are data, never newlines.
    raw_a = tmp_path / "a.png"
    raw_a.write_bytes(b"\x89PNG\r\n\x1a\n")
    raw_b = tmp_path / "b.png"
    raw_b.write_bytes(b"\x89PNG\n\x1a\n")
    assert v.sha256_file(raw_a) != v.sha256_file(raw_b)
    assert v.sha256_file(raw_a) == hashlib.sha256(b"\x89PNG\r\n\x1a\n").hexdigest()

    tif_a = tmp_path / "a.tif"
    tif_a.write_bytes(b"II*\x00\r\n")
    assert v.sha256_file(tif_a) == hashlib.sha256(b"II*\x00\r\n").hexdigest()


def test_recipe_and_verify_hash_helpers_agree(tmp_path):
    recipe = _load_recipe_module()
    v = _load_verify_module()
    for name, payload in (
        ("sample.py", b"x = 1\r\ny = 2\r\n"),
        ("sample.json", b"{\r\n}\r\n"),
        ("sample.png", b"\x89PNG\r\n\x1a\nbody"),
        ("sample.tif", b"II*\x00\r\nbody"),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        assert recipe._sha256(path) == v.sha256_file(path), name


def test_recipe_result_recipe_hash_is_crlf_insensitive(tmp_path):
    recipe = _load_recipe_module()
    png = tmp_path / "swiss_landcover.png"
    png.write_bytes(b"png-bytes")
    cert = tmp_path / "local.certificate.json"
    cert.write_bytes(b"{}")
    lf_recipe = tmp_path / "recipe_lf.py"
    lf_recipe.write_bytes(b"import os\nprint(os)\n")
    crlf_recipe = tmp_path / "recipe_crlf.py"
    crlf_recipe.write_bytes(b"import os\r\nprint(os)\r\n")

    def _run(recipe_path, out_name):
        return recipe.write_recipe_result(
            tmp_path / out_name, snapshot=png, certificate=cert, inputs=[],
            recipe_path=recipe_path,
        )["recipe_sha256"]

    assert _run(lf_recipe, "lf.json") == _run(crlf_recipe, "crlf.json")


def test_verify_manifest_validation_requires_keys():
    v = _load_verify_module()
    manifest = json.loads((CAPSULE / "data-manifest.json").read_text(encoding="utf-8"))
    prov = json.loads((CAPSULE / "provenance.json").read_text(encoding="utf-8"))
    assert v.validate_data_manifest(manifest) == []
    assert v.validate_provenance(prov) == []

    broken = json.loads(json.dumps(manifest))
    del broken["datasets"][0]["sha256"]
    assert v.validate_data_manifest(broken)

    broken = json.loads(json.dumps(manifest))
    del broken["version"]
    assert v.validate_data_manifest(broken)

    broken = json.loads(json.dumps(prov))
    del broken["records"][1]["required_attribution"]
    assert v.validate_provenance(broken)

    assert v.validate_data_manifest([])
    assert v.validate_provenance({"records": []})

    result = json.loads((CAPSULE / "out" / "recipe_result.json").read_text(encoding="utf-8")) \
        if (CAPSULE / "out" / "recipe_result.json").is_file() else None
    if result is not None:
        assert v.validate_recipe_result(result) == []
    broken_result = {"png_sha256": "aa", "inputs": []}
    assert v.validate_recipe_result(broken_result)


def test_verify_passes_on_consistent_synthetic_capsule(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    assert payload["passed"] is True, payload["failed_checks"]
    for name in (
        "data_manifest_valid",
        "provenance_valid",
        "recipe_result_valid",
        "input_hashes_match",
        "png_hash_matches",
        "recipe_hash_matches",
        "certificate_hash_matches",
        "reference_binding_matches",
        "image_ssim",
        "image_mean_abs",
        "engine_version_match",
        "wgsl_module_hashes_match",
        "degradations_empty",
    ):
        assert payload["checks"][name]["ok"] is True, name
    assert payload["failed_checks"] == []
    assert payload["image"]["ssim"] == 1.0

    written = json.loads((capsule / "out" / "verify_result.json").read_text(encoding="utf-8"))
    assert written == payload
    assert written["trust"] == {
        "reference_certificate": "development-signed — not production-signed (v1)",
        "local_report": "locally produced",
    }
    # Adapter and timings are reported but never gate.
    assert "adapter" in written["informational"]["local"]
    assert written["informational"]["local"]["total_gpu_ms"] > 0
    v._print_report(payload, capsule)  # the human report must render both verdicts


def test_verify_flags_manifest_datasets_it_cannot_resolve(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    manifest = json.loads((capsule / "data-manifest.json").read_text(encoding="utf-8"))
    manifest["datasets"].append(
        {"name": "extra", "file": "extra.tif", "sha256": "ab" * 32, "fetch": "?"}
    )
    _write_json(capsule / "data-manifest.json", manifest)
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    assert payload["passed"] is False
    assert payload["checks"]["input_hashes_match"]["ok"] is False
    assert "extra" in payload["checks"]["input_hashes_match"]["detail"]


def test_verify_does_not_let_an_emptied_binding_silence_the_input_check(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    binding = json.loads((capsule / "out" / "recipe_result.json").read_text(encoding="utf-8"))
    binding["inputs"] = []
    _write_json(capsule / "out" / "recipe_result.json", binding)
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    assert payload["passed"] is False
    assert payload["checks"]["input_hashes_match"]["ok"] is False


def test_verify_marks_unhashable_inputs_unrun_not_passed(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    paths["swiss"].unlink()
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    assert payload["passed"] is False
    assert payload["checks"]["input_hashes_match"]["status"] == "not-evaluated"
    assert "input_hashes_match" in payload["failed_checks"]


def test_verify_refuses_an_uncalibrated_config(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    with pytest.raises(ValueError, match="not yet calibrated"):
        v.verify(capsule, config={"ssim_min": None, "mean_abs_max": None}, dataset_paths=paths)
    with pytest.raises(ValueError, match="not yet calibrated"):
        v.compare(
            np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), {}, {},
            {"ssim_min": None, "mean_abs_max": 2.0},
        )
    assert not (capsule / "out" / "verify_result.json").is_file()


def test_verify_fails_and_reports_on_png_hash_mismatch(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    # Repaint the local render after the binding manifest was written.
    Image.fromarray(np.full((24, 32, 3), 200, dtype=np.uint8)).save(
        capsule / "out" / "swiss_landcover.png"
    )
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    assert payload["passed"] is False
    assert payload["checks"]["png_hash_matches"]["ok"] is False
    assert payload["checks"]["image_mean_abs"]["ok"] is False
    assert payload["checks"]["input_hashes_match"]["ok"] is True
    assert "png_hash_matches" in payload["failed_checks"]

    written = json.loads((capsule / "out" / "verify_result.json").read_text(encoding="utf-8"))
    assert written["passed"] is False
    assert written["checks"]["png_hash_matches"]["ok"] is False


def test_verify_fails_on_input_hash_mismatch(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    paths["swiss"].write_bytes(b"tampered-dem-bytes")
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    assert payload["passed"] is False
    assert payload["checks"]["input_hashes_match"]["ok"] is False
    assert payload["checks"]["image_ssim"]["ok"] is True


def test_verify_fails_on_manifest_missing_key(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    manifest = json.loads((capsule / "data-manifest.json").read_text(encoding="utf-8"))
    del manifest["datasets"][0]["sha256"]
    _write_json(capsule / "data-manifest.json", manifest)
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    assert payload["passed"] is False
    assert payload["checks"]["data_manifest_valid"]["ok"] is False
    assert (capsule / "out" / "verify_result.json").is_file()


def test_verify_fails_on_reference_binding_mismatch_and_skips_when_absent(tmp_path):
    v = _load_verify_module()
    cfg = {"ssim_min": 0.99, "mean_abs_max": 2.0}

    capsule, paths = _build_synthetic_capsule(tmp_path)
    binding = json.loads((capsule / "recipe_result.json").read_text(encoding="utf-8"))
    binding["png_sha256"] = "0" * 64
    _write_json(capsule / "recipe_result.json", binding)
    payload = v.verify(capsule, config=cfg, dataset_paths=paths)
    assert payload["passed"] is False
    assert payload["checks"]["reference_binding_matches"]["ok"] is False

    other = tmp_path / "no-binding"
    other.mkdir()
    capsule2, paths2 = _build_synthetic_capsule(other, root_binding=False)
    payload2 = v.verify(capsule2, config=cfg, dataset_paths=paths2)
    assert payload2["passed"] is True
    assert payload2["checks"]["reference_binding_matches"]["ok"] is None
    assert payload2["checks"]["reference_binding_matches"]["status"] == "absent"


def test_verify_fails_on_certificate_gates(tmp_path):
    v = _load_verify_module()
    cfg = {"ssim_min": 0.99, "mean_abs_max": 2.0}
    capsule, paths = _build_synthetic_capsule(
        tmp_path, local_cert=_synthetic_cert(degradations=[{"consequence": "fallback"}])
    )
    payload = v.verify(capsule, config=cfg, dataset_paths=paths)
    assert payload["passed"] is False
    assert payload["checks"]["degradations_empty"]["ok"] is False
    assert payload["checks"]["engine_version_match"]["ok"] is True


def test_verify_reports_missing_artifacts_rather_than_crashing(tmp_path):
    v = _load_verify_module()
    capsule, paths = _build_synthetic_capsule(tmp_path)
    (capsule / "out" / "swiss_landcover.png").unlink()
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    assert payload["passed"] is False
    assert payload["checks"]["png_hash_matches"]["ok"] is None
    assert payload["checks"]["png_hash_matches"]["status"] == "not-evaluated"
    assert payload["checks"]["image_ssim"]["status"] == "not-evaluated"
    assert "png_hash_matches" in payload["failed_checks"]
    # An unreadable image must not silence the certificate gates.
    assert payload["checks"]["engine_version_match"]["ok"] is True
    assert payload["checks"]["degradations_empty"]["ok"] is True
    v._print_report(payload, capsule)

    # ...and a missing certificate must not silence them either: it fails them.
    (capsule / "out" / "local.certificate.json").unlink()
    payload = v.verify(
        capsule, config={"ssim_min": 0.99, "mean_abs_max": 2.0}, dataset_paths=paths
    )
    for name in ("engine_version_match", "wgsl_module_hashes_match", "degradations_empty"):
        assert payload["checks"][name]["status"] == "not-evaluated"
        assert name in payload["failed_checks"]


def test_verify_main_refuses_until_thresholds_are_calibrated(capsys):
    v = _load_verify_module()
    config = json.loads((CAPSULE / "verify_config.json").read_text(encoding="utf-8"))
    assert set(config) >= {"ssim_min", "mean_abs_max"}
    if not v.thresholds_calibrated(config):
        assert v.main() == 2
        captured = capsys.readouterr()
        assert "thresholds not yet calibrated" in (captured.err + captured.out)
        assert not (CAPSULE / "out" / "verify_result.json").is_file()
    else:
        assert 0.0 < config["ssim_min"] <= 1.0
        assert config["mean_abs_max"] >= 0.0
    assert v.thresholds_calibrated({"ssim_min": None, "mean_abs_max": 2.0}) is False
    assert v.thresholds_calibrated({"ssim_min": 0.99, "mean_abs_max": None}) is False
    assert v.thresholds_calibrated({"ssim_min": 0.99, "mean_abs_max": 2.0}) is True


def test_verify_is_standalone_and_has_no_telemetry():
    import re

    text = (CAPSULE / "verify.py").read_text(encoding="utf-8")
    # No runtime dependency on the repo's test package: the capsule must work
    # from a bare copy of its own directory.
    for pattern in (r"^\s*from\s+tests\b", r"^\s*import\s+tests\b", r"^\s*import\s+scipy\b"):
        assert not re.search(pattern, text, re.M), f"non-standalone import in verify.py: {pattern}"
    for banned in ("requests.", "urllib.request", "httpx", "socket.", "map_plate",
                   "set_license_key"):
        assert banned not in text, f"banned reference in verify.py: {banned}"
    assert "tests/_ssim.py" in text, "SSIM source must be cited"
    # Only the reference PNG is ever compared — never a lossy web derivative.
    for lossy in ("preview.webp", "preview.avif", "thumb.webp", ".avif", ".webp"):
        assert lossy not in text, f"verify.py must not reference {lossy}"
