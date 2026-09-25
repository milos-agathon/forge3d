"""Validate attributable TERRA-DETERMINATA hash artifacts.

Each leg's hash is compared with the committed golden of the leg's own wgpu
backend (tests/goldens/determinism/<scene>.json). Legs sharing a backend must
also agree pairwise. A backend without a committed golden is reported ABSENT;
cross-backend identity is enforced only when the golden records it PROVEN.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _determinism_golden import backend_key, golden_sha256, load_golden  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args(argv)

    failures = []
    golden = None
    if not args.golden.exists():
        failures.append("no committed golden found")
    else:
        try:
            golden = load_golden(args.golden, args.scene)
        except (ValueError, json.JSONDecodeError) as error:
            failures.append(f"invalid committed golden: {error}")
    produced = {}
    absent = {}
    adapters = {}
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
    by_backend: dict[str, dict[str, str]] = {}
    for leg, sha in sorted(produced.items()):
        adapter = adapters.get(leg)
        ident = (
            f"{adapter['name']} ({adapter['backend']}, {adapter['device_type']})"
            if adapter
            else "UNATTRIBUTED"
        )
        print(f"  {leg:8s} {sha}  adapter: {ident}")
        if adapter:
            by_backend.setdefault(backend_key(adapter["backend"]), {})[leg] = sha
    for leg, why in sorted(absent.items()):
        print(f"informational/absent: {leg}: {why}")

    if not produced and not gated_failure:
        failures.append("no hardware-backed leg produced a hash")
    for backend, legs in sorted(by_backend.items()):
        if len(set(legs.values())) > 1:
            failures.append(f"pairwise mismatch across {backend} legs: {legs}")
        if golden is None:
            continue
        expected = golden_sha256(golden, backend)
        if expected is None:
            print(f"ABSENT: no committed {backend} golden; {sorted(legs)} not compared")
            continue
        print(f"committed {backend} golden: {expected}")
        mismatched = {leg: sha for leg, sha in legs.items() if sha != expected}
        if mismatched:
            failures.append(f"mismatch against committed {backend} golden {expected}: {mismatched}")
    if golden is not None:
        identity = golden["cross_backend_identity"]
        if identity["status"] == "PROVEN":
            if len({sha for legs in by_backend.values() for sha in legs.values()}) > 1:
                failures.append(f"pairwise mismatch across backends: {produced}")
        else:
            print(
                f"cross-backend identity: ABSENT ({identity.get('tracking', 'untracked')}); "
                "backends are compared only with their own golden"
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
