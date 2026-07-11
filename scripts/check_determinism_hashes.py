"""Validate attributable TERRA-DETERMINATA hash artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)

    golden = (
        args.golden.read_text().split()[0].strip() if args.golden.exists() else None
    )
    produced = {}
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
        if sha_file.exists():
            produced[leg] = sha_file.read_text().split()[0].strip()
            try:
                adapter = json.loads(meta_file.read_text())["adapter"]
                required = ("name", "backend", "device_type", "software_fallback")
                if not all(key in adapter for key in required) or adapter[
                    "software_fallback"
                ]:
                    raise ValueError("incomplete or software adapter")
                adapters[leg] = adapter
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                failures.append(f"{leg}: missing attributable adapter metadata")
        elif absent_file.exists():
            absent[leg] = absent_file.read_text().splitlines()[0]
        elif failed_file.exists():
            # 04b explicitly permits a documented, loud Apple infrastructure
            # failure; its render job remains red and this table preserves why.
            absent[leg] = "GATED-FAILURE: " + failed_file.read_text().splitlines()[0]
            gated_failure = True

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

    if not produced and not gated_failure:
        failures.append("no hardware-backed leg produced a hash")
    values = set(produced.values())
    if len(values) > 1:
        failures.append(f"pairwise mismatch across legs: {produced}")
    if golden is None:
        failures.append("no committed golden found")
    elif any(sha != golden for sha in produced.values()):
        failures.append(f"mismatch against committed golden {golden}: {produced}")

    if failures:
        print("DETERMINISM FAILURE (zero-byte tolerance):", file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1
    print("determinism diff: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
