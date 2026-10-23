from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import forge3d as f3d
import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "check_anamnesis_portability.py"


def test_ci_portability_jobs_use_available_physical_gpu_runner():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    seed_job = ci.split("  test-anamnesis-portability-seed:", 1)[1].split(
        "\n  test-anamnesis-portability:", 1
    )[0]
    consumer_job = ci.split("  test-anamnesis-portability:", 1)[1].split(
        "\n  test-anamnesis-production:", 1
    )[0]
    assert (
        "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]" in seed_job
    )
    assert (
        "runs-on: [self-hosted, Windows, X64, forge3d-gpu, gpu-nvidia]"
        in consumer_job
    )
    assert "anamnesis-producer" not in seed_job
    assert "anamnesis-consumer" not in consumer_job
    assert "WGPU_BACKENDS: vulkan" in seed_job
    assert "WGPU_BACKENDS: vulkan" in consumer_job
    # Both real graph renders must use the committed canonical TERRA mode.
    assert "FORGE3D_DETERMINISTIC: '1'" in seed_job
    assert "FORGE3D_DETERMINISTIC: '1'" in consumer_job
    assert "name: wheels-windows" in seed_job
    assert "--producer-backend vulkan --consumer-backend vulkan" in seed_job
    assert "--machine-id-file" in seed_job
    assert "terra_determinata_v1.sha256" in consumer_job
    assert "--runner-name '${{ runner.name }}'" in seed_job
    assert "--require-distinct-machine" not in consumer_job
    assert "metal" not in seed_job.lower()


def test_portability_driver_preserves_diagnostics_and_requires_independent_acceptance():
    source = DRIVER.read_text(encoding="utf-8")
    assert '"distinct_machine": "ABSENT"' in source
    assert 'runner_name == record.get("producer_runner_name")' not in source
    assert '"forge3d.anamnesis.native-portability/1"' in source
    assert 'report["hits"] != _PASS_LABELS' in source
    assert 'report["graph_command_submissions"] != 0' in source
    assert "png_to_numpy" not in source
    assert "render_sequence" not in source


def _run(*arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(DRIVER), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])


@pytest.fixture
def portability_check(tmp_path, monkeypatch):
    """Unit fixtures for rejection paths; they provide no physical GPU evidence."""
    monkeypatch.setattr(
        f3d, "anamnesis_engine_fingerprint", lambda: '{"unit_test_engine":true}', raising=False
    )
    driver = runpy.run_path(str(DRIVER))
    main = driver["main"]
    pixels = bytes((1, 2, 3, 255))
    image = tmp_path / "image.png"
    Image.frombytes("RGBA", (1, 1), pixels).save(image)
    golden = tmp_path / "image.sha256"
    golden.write_text(hashlib.sha256(image.read_bytes()).hexdigest(), encoding="utf-8")
    adapter = tmp_path / "adapter.jsonl"
    adapter.write_text(json.dumps({"adapter": {"backend": "vulkan", "software_fallback": False}}), encoding="utf-8")
    machine = tmp_path / "machine.txt"
    machine.write_text("consumer", encoding="utf-8")
    record = {
        "schema": "forge3d.anamnesis.native-portability/1",
        "compatibility_profile": driver["_PORTABLE_PROFILE"],
        "engine_fingerprint": json.loads(f3d.anamnesis_engine_fingerprint()),
        "producer_machine_id": "producer",
        "producer_adapter": {"backend": "vulkan"},
        "rgba_sha256": hashlib.sha256(pixels).hexdigest(),
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        str(DRIVER), "check",
        "--cache", str(tmp_path / "cache"), "--record", str(record_path),
        "--consumer-frame-blob", str(image), "--consumer-golden", str(golden),
        "--consumer-adapter-record", str(adapter), "--machine-id-file", str(machine),
        "--runner-name", "unit-test-runner",
    ])
    labels = driver["_PASS_LABELS"]
    hit = {"hits": labels, "misses": [], "graph_command_submissions": 0, "hit_rate": 1.0, "bytes_read": len(pixels)}
    miss = {"hits": [], "misses": labels, "graph_command_submissions": 5}
    return main, machine, pixels, hit, miss


def test_same_machine_reports_absent_but_still_checks_restoration(portability_check, monkeypatch, capsys):
    main, machine, pixels, hit, miss = portability_check
    machine.write_text("producer", encoding="utf-8")
    outputs = iter([(pixels, hit), (pixels, miss)])
    monkeypatch.setitem(main.__globals__, "_native_render", lambda *args: next(outputs))
    assert main() == 0
    captured = capsys.readouterr()
    # stdout remains one JSON document for the existing CI artifact.
    json.loads(captured.out)
    reports = [json.loads(line) for line in (captured.err + captured.out).splitlines()]
    assert all(report["distinct_machine"] == "ABSENT" for report in reports)
    assert all(report["producer_machine_id"] == report["consumer_machine_id"] == "producer" for report in reports)
    assert reports[-1]["fresh_native_matches"] is True


def test_same_machine_absence_does_not_excuse_stale_pixels(portability_check, monkeypatch):
    main, machine, _, hit, _ = portability_check
    machine.write_text("producer", encoding="utf-8")
    monkeypatch.setitem(main.__globals__, "_native_render", lambda *args: (b"stale", hit))
    with pytest.raises(SystemExit, match="differ from the producer"):
        main()


def test_portability_rejects_restored_pixels_that_differ_from_producer(portability_check, monkeypatch):
    main, _, _, hit, _ = portability_check
    monkeypatch.setitem(main.__globals__, "_native_render", lambda *args: (b"stale", hit))
    with pytest.raises(SystemExit, match="differ from the producer"):
        main()


@pytest.mark.parametrize("wrong_pixels", [True, False])
def test_portability_requires_independent_fresh_native_render(portability_check, monkeypatch, wrong_pixels):
    main, _, pixels, hit, miss = portability_check
    outputs = iter([(pixels, hit), (b"different" if wrong_pixels else pixels, miss if wrong_pixels else hit)])
    monkeypatch.setitem(main.__globals__, "_native_render", lambda *args: next(outputs))
    with pytest.raises(SystemExit, match="independent fresh render"):
        main()


def test_independent_portability_check_reports_fresh_native_equality(portability_check, monkeypatch, capsys):
    main, _, pixels, hit, miss = portability_check
    outputs = iter([(pixels, hit), (pixels, miss)])
    monkeypatch.setitem(main.__globals__, "_native_render", lambda *args: next(outputs))
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["distinct_machine"] is True
    assert report["fresh_native_matches"] is True


def test_required_portability_ties_native_pixels_to_golden_checked_png(portability_check, monkeypatch):
    main, _, _, hit, miss = portability_check
    # Make the raw seed/restoration agree with each other while retaining the
    # original golden-checked PNG. That unrelated PNG must not qualify reuse.
    record_path = Path(sys.argv[sys.argv.index("--record") + 1])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    different = bytes((4, 5, 6, 255))
    record["rgba_sha256"] = hashlib.sha256(different).hexdigest()
    record_path.write_text(json.dumps(record), encoding="utf-8")
    outputs = iter([(different, hit), (different, miss)])
    monkeypatch.setitem(main.__globals__, "_native_render", lambda *args: next(outputs))
    with pytest.raises(SystemExit, match="golden-checked consumer PNG"):
        main()


def test_engine_wgsl_fingerprint_uses_portable_relative_paths():
    digest = hashlib.sha256()
    shader_root = ROOT / "src" / "shaders"
    for path in sorted(shader_root.rglob("*.wgsl")):
        relative = path.relative_to(shader_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    native = json.loads(f3d.anamnesis_engine_fingerprint())
    assert native["wgsl_tree_sha256"] == digest.hexdigest()


def test_portable_store_hits_and_capability_mismatch_misses(tmp_path):
    if os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1":
        pytest.skip(
            "native portability is exercised by the protected producer/consumer jobs"
        )
    if not f3d.has_gpu():
        pytest.skip("portable resource rehydration requires a GPU adapter")
    frame_blob = tmp_path / "terra.png"
    rendered = subprocess.run(
        [sys.executable, "-m", "forge3d.determinism", "--scene", "terra_determinata_v1",
         "--width", "512", "--height", "512", "--out-png", str(frame_blob)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    golden = ROOT / "tests/goldens/determinism/terra_determinata_v1.sha256"
    adapter_record = tmp_path / "adapter.jsonl"
    adapter_record.write_text(rendered.stdout, encoding="utf-8")
    cache = tmp_path / "cache"
    record = tmp_path / "record.json"
    consumer_blob = tmp_path / "consumer.png"
    consumer_blob.write_bytes(frame_blob.read_bytes())
    consumer_adapter = tmp_path / "consumer-adapter.jsonl"
    consumer_adapter.write_bytes(adapter_record.read_bytes())
    producer_machine = tmp_path / "producer-machine.txt"
    producer_machine.write_text(
        "11111111-1111-1111-1111-111111111111\n", encoding="utf-8"
    )
    consumer_machine = tmp_path / "consumer-machine.txt"
    consumer_machine.write_text(
        "11111111-1111-1111-1111-111111111111\n", encoding="utf-8"
    )

    seeded = _run(
        "seed",
        "--cache",
        str(cache),
        "--record",
        str(record),
        "--frame-blob",
        str(frame_blob),
        "--golden",
        str(golden),
        "--adapter-record",
        str(adapter_record),
        "--machine-id-file",
        str(producer_machine),
        "--runner-name",
        "producer-runner",
    )
    checked = _run(
        "check",
        "--cache",
        str(cache),
        "--record",
        str(record),
        "--consumer-frame-blob",
        str(consumer_blob),
        "--consumer-golden",
        str(golden),
        "--consumer-adapter-record",
        str(consumer_adapter),
        "--machine-id-file",
        str(consumer_machine),
        "--runner-name",
        "producer-runner",
    )
    mismatched = _run("mismatch", "--cache", str(cache), "--record", str(record))

    assert seeded["misses"] > 0
    assert checked["hit_rate"] == 1.0
    assert checked["backend_goldens_match"] is True
    assert checked["distinct_machine"] == "ABSENT"
    assert checked["fresh_native_matches"] is True
    assert mismatched["hit_rate"] == 0.0
    assert mismatched["cache_isolated"] is True
    assert set(mismatched["dimensions"]) == {"backend", "granted_features", "limits", "dx12_compiler", "naga_capabilities"}
    assert all(row["hit_rate"] == 0.0 for row in mismatched["dimensions"].values())


def test_seed_rejects_blob_that_differs_from_committed_golden(tmp_path):
    frame_blob = tmp_path / "terra.png"
    frame_blob.write_bytes(b"wrong pixels")
    golden = tmp_path / "terra.sha256"
    golden.write_text(hashlib.sha256(b"golden pixels").hexdigest() + "\n", encoding="utf-8")
    adapter_record = tmp_path / "adapter.jsonl"
    adapter_record.write_text(
        json.dumps({"adapter": {"software_fallback": False}}) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "seed",
            "--cache",
            str(tmp_path / "cache"),
            "--record",
            str(tmp_path / "record.json"),
            "--frame-blob",
            str(frame_blob),
            "--golden",
            str(golden),
            "--adapter-record",
            str(adapter_record),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "differs from committed golden" in completed.stderr


def test_portability_cli_requires_golden_and_keeps_machine_identity_diagnostic():
    completed = subprocess.run([sys.executable, str(DRIVER), "--help"],
                               capture_output=True, text=True, check=True)
    help_text = " ".join(completed.stdout.split())
    assert "--require-distinct-machine" not in help_text
    assert "--consumer-golden" in help_text
    assert "--machine-id-file" in help_text
    source = DRIVER.read_text(encoding="utf-8")
    assert 'if fresh_rgba != golden_rgba:' in source
    assert 'FORGE3D_ANAMNESIS_COMPATIBILITY_PROFILE"] =' not in source


def test_native_store_mismatch_control_checks_all_five_dimensions(tmp_path, monkeypatch, capsys):
    """CPU key/store boundary only; this does not claim a physical render."""
    from forge3d.anamnesis import _Store, _pass_material, capability_fingerprint, engine_fingerprint, pass_key
    driver = runpy.run_path(str(DRIVER))
    cache = tmp_path / "store"
    store = _Store(cache, 512 * 1024 * 1024, True)
    caps = capability_fingerprint({"granted": [], "limits": {"max_texture_dimension_2d": 8192}},
                                  backend="vulkan", dx12_compiler="unit", naga_capabilities=[])
    engine = engine_fingerprint()
    labels = [label for label in driver["_PASS_LABELS"] if label != "terrain.shadow"]
    for label in labels:
        material = _pass_material(label, b"unit-pipeline", b"unit-uniforms", [], caps, engine)
        key = pass_key(label, b"unit-pipeline", b"unit-uniforms", [], caps, engine)
        store.put_pass(key, b"unit-payload", material)
        assert f3d.anamnesis_store_get(cache, key, store.max_bytes) == b"unit-payload"
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"schema": "forge3d.anamnesis.native-portability/1",
        "compatibility_profile": driver["_PORTABLE_PROFILE"],
        "engine_fingerprint": json.loads(f3d.anamnesis_engine_fingerprint())}))
    monkeypatch.setattr(sys, "argv", [str(DRIVER), "mismatch", "--cache", str(cache), "--record", str(record)])
    assert driver["main"]() == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["dimensions"]) == {"backend", "granted_features", "limits", "dx12_compiler", "naga_capabilities"}
    assert all(row == {"hits": 0, "misses": len(labels), "hit_rate": 0.0}
               for row in result["dimensions"].values())
