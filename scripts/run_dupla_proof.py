#!/usr/bin/env python3
"""Run and archive the complete pinned-backend DUPLA acceptance proof."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any


OPERATIONS = ("add", "mul", "div", "sqrt")
GENERATED_COUNT = 100_000_000


def _certificate(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("certificate_json")
    if not isinstance(value, str):
        raise RuntimeError("native DUPLA report omitted certificate_json")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise RuntimeError("native DUPLA certificate is not a JSON object")
    return parsed


def run_proof() -> dict[str, Any]:
    from forge3d import precision

    selftest = precision.dd_selftest()
    if selftest.get("passed") is not True or selftest.get("mismatch_count") != 0:
        raise RuntimeError(f"DUPLA exactness self-test failed: {selftest!r}")

    harness: dict[str, dict[str, Any]] = {}
    certificates: dict[str, dict[str, Any]] = {}
    for operation in OPERATIONS:
        report = precision.dd_harness(operation, n=GENERATED_COUNT)
        if report["generated_count"] < GENERATED_COUNT:
            raise RuntimeError(f"{operation}: generated-vector floor was not met")
        if report["adversarial_count"] < 1_000_000:
            raise RuntimeError(f"{operation}: adversarial-vector floor was not met")
        if report["mismatch_count"] != 0:
            raise RuntimeError(f"{operation}: Rust/WGSL mirror mismatch")
        if report["max_err_u2"] > report["cited_bound_u2"]:
            raise RuntimeError(f"{operation}: published error bound exceeded")
        harness[operation] = report
        certificates[operation] = _certificate(report)

    jitter = precision.dd_jitter_demo(frames=1_000)
    if not jitter["dd_max_error_px"] < 0.01:
        raise RuntimeError("DD jitter exceeded 0.01 px")
    if not jitter["raw_over_one_px"] >= 100:
        raise RuntimeError("raw-f32 ablation did not exceed 1 px on 10% of frames")
    if not jitter["dd_hash_a"] == jitter["dd_hash_b"]:
        raise RuntimeError("DD render was not byte-identical across two executions")
    certificates["jitter"] = _certificate(jitter)

    return {
        "schema": "forge3d.dupla-proof.v1",
        "backend": selftest["backend"],
        "adapter": selftest["adapter"],
        "two_prod_variant": selftest["two_prod_variant"],
        "selftest": selftest,
        "harness": harness,
        "jitter": jitter,
        "certificates": certificates,
    }


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _hardware_absence() -> str | None:
    import forge3d

    backend = os.environ.get("WGPU_BACKENDS") or os.environ.get("WGPU_BACKEND")
    probe = forge3d.device_probe(backend)
    status = probe.get("status") if isinstance(probe, dict) else None
    if status == "no_adapter":
        return str(probe.get("reason", "no adapter"))
    if status != "ok":
        raise RuntimeError(f"DUPLA hardware probe failed: {probe!r}")
    name = str(probe.get("name", ""))
    virtualized = any(
        marker in name.lower()
        for marker in ("paravirtual", "virtio", "vmware", "virtualbox", "qxl")
    )
    if probe.get("software_fallback") or virtualized:
        kind = "virtualized" if virtualized else "software"
        return f"{kind} adapter is not a physical backend proof target: {name}"
    return None


def _validate_proof(value: dict[str, Any]) -> None:
    if value.get("schema") != "forge3d.dupla-proof.v1":
        raise ValueError("wrong or missing DUPLA proof schema")
    selftest = value.get("selftest", {})
    if selftest.get("passed") is not True or selftest.get("mismatch_count") != 0:
        raise ValueError("invalid exactness self-test evidence")
    harness = value.get("harness", {})
    if set(harness) != set(OPERATIONS):
        raise ValueError("proof does not contain all four DD operations")
    for operation, report in harness.items():
        if report.get("generated_count", 0) < GENERATED_COUNT:
            raise ValueError(f"{operation}: generated count below acceptance floor")
        if report.get("adversarial_count", 0) < 1_000_000:
            raise ValueError(f"{operation}: adversarial count below acceptance floor")
        if report.get("mismatch_count") != 0:
            raise ValueError(f"{operation}: mirror mismatch")
        if report.get("max_err_u2", float("inf")) > report.get("cited_bound_u2", 0):
            raise ValueError(f"{operation}: cited bound exceeded")
    jitter = value.get("jitter", {})
    if jitter.get("dd_max_error_px", float("inf")) >= 0.01:
        raise ValueError("DD jitter threshold exceeded")
    if jitter.get("raw_over_one_px", 0) < 100:
        raise ValueError("raw-f32 ablation threshold not demonstrated")
    if jitter.get("dd_hash_a") != jitter.get("dd_hash_b"):
        raise ValueError("DD hashes differ")


def validate_artifacts(root: Path, expected_legs: list[str]) -> None:
    for leg in expected_legs:
        matches = {
            suffix: list(root.rglob(f"dupla-proof-{leg}.{suffix}"))
            for suffix in ("json", "ABSENT", "FAILED")
        }
        present = [(suffix, path) for suffix, paths in matches.items() for path in paths]
        if len(present) != 1:
            raise ValueError(f"{leg}: expected exactly one DUPLA result, found {present!r}")
        suffix, path = present[0]
        if suffix == "FAILED":
            raise ValueError(f"{leg}: DUPLA proof failed: {path.read_text(encoding='utf-8')}")
        if suffix == "ABSENT":
            print(f"DUPLA {leg}: ABSENT — {path.read_text(encoding='utf-8').strip()}")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        _validate_proof(value)
        print(f"DUPLA {leg}: VERIFIED ({value['backend']}, {value['adapter']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--validate-artifacts", type=Path)
    parser.add_argument("--absence-output", type=Path)
    parser.add_argument("--failure-output", type=Path)
    parser.add_argument("--expected-legs", nargs="+")
    args = parser.parse_args()
    if args.validate_artifacts:
        if not args.expected_legs:
            parser.error("--validate-artifacts requires --expected-legs")
        validate_artifacts(args.validate_artifacts, args.expected_legs)
        return 0
    if args.absence_output is None or args.failure_output is None:
        parser.error("proof mode requires --absence-output and --failure-output")
    try:
        absence = _hardware_absence()
        if absence is not None:
            _write(args.absence_output, absence)
            print(f"DUPLA proof absent: {absence}")
            return 0
        proof = run_proof()
        _validate_proof(proof)
        _write(args.output, json.dumps(proof, indent=2, sort_keys=True, allow_nan=False))
        print(json.dumps({key: proof[key] for key in ("backend", "adapter", "two_prod_variant")}))
        return 0
    except Exception:
        details = traceback.format_exc()
        _write(args.failure_output, details)
        print(details, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
