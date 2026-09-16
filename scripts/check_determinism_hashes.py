"""Validate attributable TERRA-DETERMINATA hash artifacts.

Artifact model (per ``determinism-hash-<leg>/`` directory):

- ``<scene>.sha256`` + ``<scene>.json``: a produced native render leg. The JSON
  record must carry attributable physical-hardware adapter metadata
  (``adapter.name/backend/device_type`` present, ``software_fallback`` false,
  and no hypervisor/virtual marker in the adapter name) or the leg is rejected.
  A leg that also ran the arithmetic canary carries ``probe.sha256`` and
  ``probe.raster_sha256`` in the same record.
- ``<scene>.ABSENT``: a documented informational absence (hosted runner with no
  qualifying adapter). Absences are reported loudly and never count as evidence.
- ``<scene>.FAILED``: a loud infrastructure failure. For non-required legs it is
  recorded as ``GATED-FAILURE`` (documented absence); for ``--require`` legs it
  fails the gate. A FAILED artifact carries no hardware hash and can never be
  counted as produced evidence.
- Browser leg (``determinism-hash-browser/``): ``probe.sha256``,
  ``raster.sha256`` and ``browser.json``. The browser does not render the
  canonical PNG; it contributes the executed probe/raster hashes, which must
  equal the probe/raster goldens (and hence the native legs' canary hashes).

``--require LEG`` names legs that must produce qualifying artifacts; an ABSENT
or FAILED artifact (or a missing artifact entirely) on a required leg fails the
gate. ``--probe-golden``/``--raster-golden`` pin the canary hashes so a leg whose
arithmetic diverges fails even when its PNG happens to match.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Mirrors src/core/gpu.rs::is_virtualized_adapter_name: a hypervisor VM GPU is
# not the physical hardware a determinism leg claims to measure.
_VIRTUAL_ADAPTER_MARKERS = ("paravirtual", "virtio", "vmware", "virtualbox", "qxl")

_REQUIRED_ADAPTER_KEYS = ("name", "backend", "device_type", "software_fallback")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _adapter_failure(adapter: object) -> str | None:
    """Return a rejection reason, or None if the adapter qualifies."""
    if not isinstance(adapter, dict):
        return "adapter metadata is not an object"
    if not all(key in adapter for key in _REQUIRED_ADAPTER_KEYS):
        return "incomplete adapter metadata"
    if not all(
        isinstance(adapter[key], str) and adapter[key].strip()
        for key in ("name", "backend", "device_type")
    ):
        return "incomplete adapter metadata"
    if adapter["software_fallback"] is not False:
        return "software fallback adapter"
    name = adapter["name"].strip()
    device_type = adapter["device_type"].strip()
    # device_type comes back as wgpu's Debug spelling: DiscreteGpu,
    # IntegratedGpu, VirtualGpu, Cpu, Other. Only real GPU hardware classes
    # qualify — Cpu is a software rasterizer and VirtualGpu is a VM device.
    if device_type.lower() not in ("discretegpu", "integratedgpu"):
        return f"unqualified adapter device_type {device_type!r}"
    lowered_name = name.lower()
    for marker in _VIRTUAL_ADAPTER_MARKERS:
        if marker in lowered_name:
            return f"virtualized adapter {adapter['name']!r} is not physical hardware"
    return None


def _browser_adapter_failure(adapter: object) -> str | None:
    if not isinstance(adapter, dict):
        return "browser adapter metadata is not an object"
    if not all(key in adapter for key in _REQUIRED_ADAPTER_KEYS):
        return "incomplete browser adapter metadata"
    if not all(
        isinstance(adapter[key], str) and adapter[key].strip()
        for key in ("name", "backend", "device_type")
    ):
        return "incomplete browser adapter metadata"
    if type(adapter["software_fallback"]) is not bool:
        return "browser software_fallback is not boolean"
    if adapter["backend"].lower() != "browserwebgpu":
        return f"browser adapter backend is {adapter['backend']!r}, not BrowserWebGpu"
    if adapter["device_type"].lower() != "webgpu":
        return f"browser adapter device_type is {adapter['device_type']!r}, not webgpu"
    return None


def _read_sha256(path: Path) -> str:
    values = path.read_text().split()
    if len(values) != 1 or _SHA256_PATTERN.fullmatch(values[0]) is None:
        raise ValueError(f"{path} does not contain exactly one lowercase SHA-256 digest")
    return values[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="LEG",
        help="Leg that must produce qualifying artifacts; repeatable.",
    )
    parser.add_argument(
        "--probe-golden",
        type=Path,
        default=None,
        help="Committed probe_sha256 canary golden (optional).",
    )
    parser.add_argument(
        "--raster-golden",
        type=Path,
        default=None,
        help="Committed raster_sha256 canary golden (optional).",
    )
    args = parser.parse_args(argv)

    if not args.golden.exists():
        print(f"FAIL: --golden {args.golden} does not exist", file=sys.stderr)
        return 2
    # A passed-but-missing canary golden must not silently disable the
    # comparison — that would let an absent file weaken the gate.
    if args.probe_golden and not args.probe_golden.exists():
        print(
            f"FAIL: --probe-golden {args.probe_golden} does not exist",
            file=sys.stderr,
        )
        return 2
    if args.raster_golden and not args.raster_golden.exists():
        print(
            f"FAIL: --raster-golden {args.raster_golden} does not exist",
            file=sys.stderr,
        )
        return 2
    try:
        golden = _read_sha256(args.golden)
        probe_golden = _read_sha256(args.probe_golden) if args.probe_golden else None
        raster_golden = _read_sha256(args.raster_golden) if args.raster_golden else None
    except (OSError, ValueError) as exc:
        print(f"FAIL: invalid golden configuration ({exc})", file=sys.stderr)
        return 2

    produced = {}
    probe_hashes = {}
    raster_hashes = {}
    absent = {}
    adapters = {}
    failures = []
    gated_failure = False

    for artifact_dir in sorted(args.hashes.glob("determinism-hash-*")):
        leg = artifact_dir.name.removeprefix("determinism-hash-")
        sha_file = artifact_dir / f"{args.scene}.sha256"
        absent_file = artifact_dir / f"{args.scene}.ABSENT"
        failed_file = artifact_dir / f"{args.scene}.FAILED"
        meta_file = artifact_dir / f"{args.scene}.json"
        browser_meta = artifact_dir / "browser.json"
        probe_file = artifact_dir / "probe.sha256"
        raster_file = artifact_dir / "raster.sha256"

        native_states = [
            path.name
            for path in (sha_file, absent_file, failed_file)
            if path.exists()
        ]
        if len(native_states) > 1:
            failures.append(f"{leg}: conflicting result artifacts {native_states}")
            continue

        if sha_file.exists():
            try:
                pending_sha = _read_sha256(sha_file)
                record = json.loads(meta_file.read_text())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"{leg}: invalid render evidence ({exc})")
                continue
            try:
                reason = _adapter_failure(record["adapter"])
            except (KeyError, TypeError) as exc:
                reason = f"incomplete adapter metadata ({exc})"
            if reason is not None:
                failures.append(f"{leg}: missing attributable adapter metadata ({reason})")
                continue
            if record.get("scene") != args.scene or record.get("sha256") != pending_sha:
                failures.append(f"{leg}: render record does not match its scene/hash sidecar")
                continue
            probe = record.get("probe") or {}
            if not isinstance(probe, dict):
                failures.append(f"{leg}: probe metadata is not an object")
                continue
            adapters[leg] = record["adapter"]
            produced[leg] = pending_sha
            probe_sha = probe.get("sha256")
            if probe.get("status") is not None and probe.get("status") != "ok":
                failures.append(
                    f"{leg}: probe status {probe.get('status')!r} is not 'ok'"
                    f" ({probe.get('reason', 'no reason recorded')})"
                )
            raster_sha = probe.get("raster_sha256")
            if args.probe_golden is not None and not probe_sha:
                failures.append(f"{leg}: produced a PNG hash but no probe_sha256 canary")
            if args.raster_golden is not None and not raster_sha:
                failures.append(f"{leg}: produced a PNG hash but no raster_sha256 canary")
            if probe_sha:
                # A probe that ran on a software/virtual adapter while the
                # render claims physical hardware is contradictory evidence.
                if probe.get("software_fallback") is not False:
                    failures.append(f"{leg}: probe ran on a software adapter")
                elif not isinstance(probe_sha, str) or _SHA256_PATTERN.fullmatch(probe_sha) is None:
                    failures.append(f"{leg}: probe_sha256 is not a SHA-256 digest")
                else:
                    probe_hashes[leg] = probe_sha
            if raster_sha:
                if not isinstance(raster_sha, str) or _SHA256_PATTERN.fullmatch(raster_sha) is None:
                    failures.append(f"{leg}: raster_sha256 is not a SHA-256 digest")
                else:
                    raster_hashes[leg] = raster_sha
        elif absent_file.exists():
            lines = absent_file.read_text().splitlines()
            if not lines:
                failures.append(f"{leg}: empty ABSENT marker")
                continue
            absent[leg] = lines[0]
        elif failed_file.exists():
            # Documented, loud infrastructure failure: never evidence, and for
            # required legs it is a gate failure — except the single documented
            # hosted-macOS absence (the runner's GPU is an Apple Paravirtual VM
            # device, which deterministic mode refuses), which stays a loud
            # documented absence even when the leg is required.
            reason_text = failed_file.read_text()
            lines = reason_text.splitlines()
            if not lines:
                failures.append(f"{leg}: empty FAILED marker")
                continue
            if leg == "apple" and "hypervisor-virtualized GPU" in reason_text:
                absent[leg] = "DOCUMENTED-ABSENCE: " + lines[0]
            else:
                absent[leg] = "GATED-FAILURE: " + lines[0]
            gated_failure = True
        elif browser_meta.exists() or probe_file.exists() or raster_file.exists():
            # Browser leg: executed probe/raster canaries only (no canonical PNG).
            if leg != "browser" or not all(
                path.exists() for path in (browser_meta, probe_file, raster_file)
            ):
                failures.append(f"{leg}: incomplete browser canary artifact")
                continue
            try:
                browser_record = json.loads(browser_meta.read_text())
                browser_probe_sha = _read_sha256(probe_file)
                browser_raster_sha = _read_sha256(raster_file)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"{leg}: invalid browser canary evidence ({exc})")
                continue
            if not isinstance(browser_record, dict):
                failures.append(f"{leg}: browser metadata is not an object")
                continue
            browser_reason = _browser_adapter_failure(browser_record.get("adapter"))
            if browser_record.get("status") != "ok":
                failures.append(
                    f"{leg}: browser probe status {browser_record.get('status')!r} is not 'ok'"
                )
                continue
            if browser_reason is not None:
                failures.append(f"{leg}: {browser_reason}")
                continue
            if browser_record.get("raster_format") != "rgba32float":
                failures.append(
                    f"{leg}: browser raster format {browser_record.get('raster_format')!r} "
                    "is not byte-comparable rgba32float"
                )
                continue
            if (
                browser_record.get("probe_sha256") != browser_probe_sha
                or browser_record.get("raster_sha256") != browser_raster_sha
            ):
                failures.append(f"{leg}: browser record does not match its canary sidecars")
                continue
            adapters[leg] = browser_record["adapter"]
            probe_hashes[leg] = browser_probe_sha
            raster_hashes[leg] = browser_raster_sha

    print("produced hashes:")
    for leg, sha in sorted(produced.items()):
        adapter = adapters.get(leg)
        ident = (
            f"{adapter['name']} ({adapter['backend']}, {adapter['device_type']})"
            if adapter
            else "UNATTRIBUTED"
        )
        print(f"  {leg:8s} {sha}  adapter: {ident}")
    for leg, why in sorted(absent.items()):
        print(f"informational/absent: {leg}: {why}")
    print(f"committed golden: {golden}" if golden else "no committed golden")
    if probe_hashes:
        print("probe_sha256 canaries:")
        for leg, sha in sorted(probe_hashes.items()):
            print(f"  {leg:8s} {sha}")
    if raster_hashes:
        print("raster_sha256 canaries:")
        for leg, sha in sorted(raster_hashes.items()):
            print(f"  {leg:8s} {sha}")

    # Required-leg enforcement: a required leg must produce evidence; an ABSENT
    # or FAILED marker on a required leg fails closed. The only tolerated
    # absence on a required leg is the DOCUMENTED-ABSENCE marker (the hosted
    # macOS Paravirtual VM diagnostic), which is loud and specific.
    for leg in args.require:
        if leg == "browser":
            if leg not in probe_hashes or leg not in raster_hashes:
                failures.append(
                    f"required leg 'browser' did not produce probe/raster canary hashes"
                )
        elif leg not in produced and not absent.get(leg, "").startswith("DOCUMENTED-ABSENCE"):
            reason = absent.get(leg, "no artifact at all")
            failures.append(f"required leg '{leg}' produced no hash ({reason})")

    if not produced and not gated_failure:
        failures.append("no hardware-backed leg produced a hash")
    values = set(produced.values())
    if len(values) > 1:
        failures.append(f"pairwise mismatch across legs: {produced}")
    if any(sha != golden for sha in produced.values()):
        failures.append(f"mismatch against committed golden {golden}: {produced}")

    # Canary hashes: pairwise equality across every leg that ran them (native
    # AND browser), plus equality with the committed goldens when provided.
    if len(set(probe_hashes.values())) > 1:
        failures.append(f"probe canary mismatch across legs: {probe_hashes}")
    if len(set(raster_hashes.values())) > 1:
        failures.append(f"raster canary mismatch across legs: {raster_hashes}")
    if probe_golden is not None:
        for leg, sha in sorted(probe_hashes.items()):
            if sha != probe_golden:
                failures.append(f"{leg}: probe_sha256 {sha} != probe golden {probe_golden}")
    if raster_golden is not None:
        for leg, sha in sorted(raster_hashes.items()):
            if sha != raster_golden:
                failures.append(f"{leg}: raster_sha256 {sha} != raster golden {raster_golden}")

    # Honest coverage summary: a green aggregate with absent devices is NOT a
    # five-way portability proof — say exactly which legs produced evidence.
    evidence_legs = set(produced) | set(probe_hashes) | set(raster_hashes)
    total_legs = sorted(evidence_legs | set(absent))
    print(
        f"legs with produced evidence: {len(evidence_legs)}/{len(total_legs)}"
        f" (png: {sorted(produced)}; canary-only: {sorted(set(probe_hashes) - set(produced))})"
    )
    if absent:
        print(
            f"legs absent/failed: {sorted(absent)} — this run is NOT a"
            f" {len(total_legs)}-way portability proof"
        )

    if failures:
        print("DETERMINISM FAILURE (zero-byte tolerance):", file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1
    print("determinism diff: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
