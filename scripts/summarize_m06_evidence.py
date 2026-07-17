"""Create a compact public summary for the M-06 hardware evidence artifact."""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

NVIDIA_VENDOR_ID = 0x10DE


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_kv_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _junit_counts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    root = ET.parse(path).getroot()
    counts = {"exists": True, "tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for testcase in root.iter("testcase"):
        counts["tests"] += 1
        child_tags = {child.tag for child in testcase}
        counts["failures"] += int("failure" in child_tags)
        counts["errors"] += int("error" in child_tags)
        counts["skipped"] += int("skipped" in child_tags)
    counts["zero_skip_clean"] = (
        counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    )
    return counts


def _adapter_identity(artifact_dir: Path) -> dict[str, Any]:
    probe_record = _read_json(artifact_dir / "adapter-probe.json") or {}
    probe = probe_record.get("probe", probe_record)
    viewer_record = _read_json(artifact_dir / "viewer-adapter.json") or {}
    identity = viewer_record.get("validated_identity") or probe or {}
    return {
        "name": identity.get("name"),
        "vendor": identity.get("vendor"),
        "vendor_hex": f"0x{int(identity.get('vendor', -1)):04x}"
        if identity.get("vendor") is not None
        else None,
        "device": identity.get("device"),
        "backend": str(identity.get("backend", "")).lower() or None,
        "device_type": str(identity.get("device_type", "")).lower() or None,
        "driver": identity.get("driver"),
        "driver_info": identity.get("driver_info"),
        "source": "viewer-adapter.json"
        if viewer_record.get("validated_identity")
        else "adapter-probe.json",
    }


def build_summary(artifact_dir: Path) -> dict[str, Any]:
    run_context = _read_json(artifact_dir / "run-context.json") or {}
    checked_out = None
    checked_out_path = artifact_dir / "checked-out-head.txt"
    if checked_out_path.is_file():
        checked_out = checked_out_path.read_text(encoding="utf-8", errors="replace").strip()
    adapter = _adapter_identity(artifact_dir)
    junit = _junit_counts(artifact_dir / "junit.xml")
    exit_codes = {
        **_read_kv_file(artifact_dir / "install-exit-codes.txt"),
        **_read_kv_file(artifact_dir / "build-exit-code.txt"),
        **_read_kv_file(artifact_dir / "rust-gates-exit-code.txt"),
        **_read_kv_file(artifact_dir / "adapter-probe-exit-code.txt"),
        **_read_kv_file(artifact_dir / "test-exit-codes.txt"),
    }
    adapter_ok = (
        adapter.get("vendor") == NVIDIA_VENDOR_ID
        and adapter.get("backend") == "vulkan"
        and adapter.get("device_type") == "discretegpu"
    )
    summary = {
        "schema_version": 1,
        "repository": run_context.get("repository") or os.environ.get("GITHUB_REPOSITORY"),
        "head_sha": run_context.get("head_sha") or os.environ.get("GITHUB_SHA"),
        "checked_out_head": checked_out,
        "run_id": run_context.get("run_id") or os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": run_context.get("run_attempt") or os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner": {
            "name": run_context.get("runner_name") or os.environ.get("RUNNER_NAME"),
            "os": run_context.get("runner_os") or os.environ.get("RUNNER_OS"),
            "arch": run_context.get("runner_arch") or os.environ.get("RUNNER_ARCH"),
            "required_labels": run_context.get("required_labels", []),
        },
        "required": {
            "vendor": "NVIDIA",
            "vendor_hex": "0x10de",
            "backend": "vulkan",
            "device_type": "discretegpu",
            "zero_skip_junit": True,
        },
        "adapter": adapter,
        "junit": junit,
        "exit_codes": exit_codes,
        "status": "pass" if adapter_ok and junit.get("zero_skip_clean") else "incomplete",
    }
    return summary


def markdown_summary(summary: dict[str, Any]) -> str:
    adapter = summary["adapter"]
    junit = summary["junit"]
    runner = summary["runner"]
    lines = [
        "# M-06 Public Evidence Summary",
        "",
        f"- status: {summary['status']}",
        f"- head_sha: {summary.get('head_sha')}",
        f"- checked_out_head: {summary.get('checked_out_head')}",
        f"- runner: {runner.get('name')} ({runner.get('os')}, {runner.get('arch')})",
        (
            "- adapter: "
            f"{adapter.get('name')} vendor={adapter.get('vendor_hex')} "
            f"backend={adapter.get('backend')} device_type={adapter.get('device_type')}"
        ),
        (
            "- junit: "
            f"tests={junit.get('tests')} failures={junit.get('failures')} "
            f"errors={junit.get('errors')} skipped={junit.get('skipped')}"
        ),
        f"- evidence_source: {adapter.get('source')}",
        "",
    ]
    return "\n".join(lines)


def github_notice(summary: dict[str, Any]) -> str:
    """Return an unauthenticated-check-visible GitHub annotation."""
    adapter = summary["adapter"]
    junit = summary["junit"]
    message = (
        f"status={summary['status']} "
        f"head_sha={summary.get('head_sha')} "
        f"checked_out_head={summary.get('checked_out_head')} "
        f"adapter={adapter.get('name')} "
        f"vendor={adapter.get('vendor_hex')} "
        f"backend={adapter.get('backend')} "
        f"device_type={adapter.get('device_type')} "
        f"tests={junit.get('tests')} "
        f"failures={junit.get('failures')} "
        f"errors={junit.get('errors')} "
        f"skipped={junit.get('skipped')}"
    )
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::notice title=M-06 exact-head evidence::{escaped}"


def write_summary(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(artifact_dir)
    (artifact_dir / "m06-public-evidence-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "m06-public-evidence-summary.md").write_text(
        markdown_summary(summary),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    summary = write_summary(args.artifact_dir)
    print(markdown_summary(summary))
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(github_notice(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
