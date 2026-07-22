from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import forge3d as f3d


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "check_anamnesis_portability.py"


def _run(*arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(DRIVER), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])


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
    frame_blob = tmp_path / "terra.png"
    frame_blob.write_bytes(b"actual rendered TERRA golden bytes")
    golden = tmp_path / "terra.sha256"
    golden.write_text(
        hashlib.sha256(frame_blob.read_bytes()).hexdigest() + "  terra.png\n",
        encoding="utf-8",
    )
    adapter_record = tmp_path / "adapter.jsonl"
    adapter_record.write_text(
        json.dumps(
            {
                "adapter": {
                    "backend": "vulkan",
                    "name": "physical-test-adapter",
                    "software_fallback": False,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    record = tmp_path / "record.json"

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
    )
    checked = _run("check", "--cache", str(cache), "--record", str(record))
    mismatched = _run("mismatch", "--cache", str(cache), "--record", str(record))

    assert seeded["misses"] > 0
    assert checked["hit_rate"] == 1.0
    assert checked["hashes_match"] is True
    assert mismatched["hit_rate"] == 0.0
    assert mismatched["hashes_match"] is False


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
