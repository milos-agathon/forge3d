# tests/test_certificate_verifier.py
# CENSOR Task 10: offline verifier tests for forge3d.certificate. These run
# WITHOUT the native _forge3d module — certificate.py is pure Python — so a
# third party can re-verify a signed RenderCertificate with a stock CPython.
# RELEVANT FILES: python/forge3d/certificate.py, python/forge3d/_ed25519.py

import copy
import json
import os
import sys
import subprocess
from pathlib import Path

import pytest

from forge3d import certificate

FIXTURE = {
    "schema": "forge3d.render_certificate/1",
    "engine": {"version": "1.30.1", "git_sha": "deadbeef", "wgsl_module_hashes": {"terrain.main": "ab" * 32}},
    "adapter": {"vendor": "v", "device": "d", "backend": "dx12", "driver_info": "i"},
    "capabilities": {"requested": ["timestamp_query"], "granted": [], "limits": {"max_bind_groups": 4}},
    "passes": [{"label": "terrain.main", "gpu_ms": 1.25, "draw_calls": 7}],
    "allocations": {"peak_host_visible_bytes": 1, "peak_device_local_bytes": 2, "by_label": {"x": 1}},
    "degradations": [],
}


def test_sign_then_verify_roundtrip(tmp_path):
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    p = tmp_path / "cert.json"
    certificate.write_certificate(cert, p)
    assert certificate.verify(p, cert["signature"]["pubkey"]) is True


def test_gpu_ms_is_not_signed(tmp_path):
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    cert["passes"][0]["gpu_ms"] = 99.0
    p = tmp_path / "cert.json"; certificate.write_certificate(cert, p)
    assert certificate.verify(p, cert["signature"]["pubkey"]) is True


def test_any_signed_byte_tamper_fails(tmp_path):
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    cert["allocations"]["peak_host_visible_bytes"] += 1
    p = tmp_path / "cert.json"; certificate.write_certificate(cert, p)
    assert certificate.verify(p, cert["signature"]["pubkey"]) is False


def test_payload_bytes_are_deterministic():
    a = certificate.canonical_payload_bytes(copy.deepcopy(FIXTURE))
    b = certificate.canonical_payload_bytes(json.loads(json.dumps(FIXTURE)))
    assert a == b


def test_payload_normalizes_negative_zero():
    fixture = copy.deepcopy(FIXTURE)
    fixture["value"] = -0.0
    assert b'"value":0.0' in certificate.canonical_payload_bytes(fixture)


def test_payload_rejects_non_finite_float():
    fixture = copy.deepcopy(FIXTURE)
    fixture["value"] = float("nan")
    with pytest.raises(ValueError):
        certificate.canonical_payload_bytes(fixture)


def test_signing_is_deterministic():
    # Ed25519 signing is deterministic (RFC 8032): the same certificate and
    # seed must yield identical payload digests AND identical signature bytes.
    first = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    second = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    assert certificate.payload_sha256(first) == certificate.payload_sha256(second)
    assert first["signature"]["sig"] == second["signature"]["sig"]


def test_signing_uses_native_ed25519_dalek(monkeypatch, tmp_path):
    def reject_python_signing(*_args, **_kwargs):
        raise AssertionError("certificate signing must use native ed25519-dalek")

    monkeypatch.setattr(certificate._ed25519, "sign", reject_python_signing)
    monkeypatch.setattr(
        certificate._ed25519, "public_key_from_private", reject_python_signing
    )

    signed = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    path = tmp_path / "native-signed.json"
    certificate.write_certificate(signed, path)
    assert certificate.verify(path, signed["signature"]["pubkey"])


def test_cli_verify(tmp_path):
    cert = certificate.sign_certificate(copy.deepcopy(FIXTURE))
    p = tmp_path / "cert.json"; certificate.write_certificate(cert, p)
    k = tmp_path / "k.pub"; k.write_text(cert["signature"]["pubkey"])
    # Resolve forge3d for the subprocess from the SAME location this test
    # imported it, so the CLI is exercised against the code under test rather
    # than whatever ambient install `sys.path` happens to expose (the dev venv
    # may point its `.pth` at a sibling worktree).
    pkg_parent = str(Path(certificate.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    r = subprocess.run([sys.executable, "-m", "forge3d.certificate", "verify", str(p), "--pubkey", str(k)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "VALID" in r.stdout, r.stderr
