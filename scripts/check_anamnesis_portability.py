"""CI driver for ANAMNESIS store portability and mismatch isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forge3d.anamnesis import render_sequence


_PORTABLE_PROFILE = "terra-determinata-portable-v1"
_RENDERER_FINGERPRINT = hashlib.sha256(
    b"forge3d.anamnesis.terra-determinata-golden/v1"
).digest()


def recipe(profile: str = _PORTABLE_PROFILE) -> dict:
    return {
        "terrain": {"dem_sha256": "42" * 32, "z_scale": 2.0},
        "atmosphere": {"model": "clear"},
        "lighting": {"azimuth": 210.0, "elevation": 33.0},
        "camera": {"path": "terra-determinata-portability-v1"},
        # This is an effective deterministic compatibility backend, not an
        # omitted backend. A physical backend may enter this equivalence class
        # only after the TERRA matrix proves its output equals the committed
        # golden. Changing the profile is therefore a mandatory key miss.
        "anamnesis_state": {"backend": profile},
        "layers": [],
        "output": {"width": 512, "height": 512, "samples": 4},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seed", "check", "mismatch"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--frame-blob")
    parser.add_argument("--golden")
    parser.add_argument("--adapter-record")
    args = parser.parse_args()
    record_path = Path(args.record)

    if args.mode == "seed":
        if not args.frame_blob or not args.golden or not args.adapter_record:
            parser.error("seed requires --frame-blob, --golden, and --adapter-record")
        frame_blob = Path(args.frame_blob).read_bytes()
        actual_sha256 = hashlib.sha256(frame_blob).hexdigest()
        golden_sha256 = Path(args.golden).read_text(encoding="utf-8").split()[0]
        if actual_sha256 != golden_sha256:
            raise SystemExit(
                f"seed render differs from committed golden: actual={actual_sha256} "
                f"golden={golden_sha256}"
            )
        adapter_lines = Path(args.adapter_record).read_text(encoding="utf-8").splitlines()
        adapter = json.loads(adapter_lines[-1]).get("adapter")
        if not adapter or adapter.get("software_fallback") is not False:
            raise SystemExit("seed render lacks attributable physical-adapter metadata")
        result = render_sequence(
            recipe(),
            frames=[0],
            cache=args.cache,
            render_frame=lambda _recipe, _frame: frame_blob,
            render_frame_fingerprint=_RENDERER_FINGERPRINT,
            capabilities={},
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "frame_hashes": result.frame_hashes,
                    "golden_sha256": golden_sha256,
                    "producer_adapter": adapter,
                    "compatibility_profile": _PORTABLE_PROFILE,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "mode": "seed",
                    "hits": len(result.cache_report.hits),
                    "misses": len(result.cache_report.misses),
                    "hit_rate": result.cache_report.hit_rate,
                    "bytes_written": result.cache_report.bytes_written,
                },
                sort_keys=True,
            )
        )
        return 0

    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = record["frame_hashes"]
    golden_sha256 = record["golden_sha256"]
    active_recipe = (
        recipe(_PORTABLE_PROFILE + "-mismatch") if args.mode == "mismatch" else recipe()
    )

    def should_not_render(_recipe, _frame):
        if args.mode == "check":
            raise RuntimeError("portable cache unexpectedly invoked the renderer")
        return b"capability-mismatch-recomputed"

    result = render_sequence(
        active_recipe,
        frames=[0],
        cache=args.cache,
        render_frame=should_not_render,
        render_frame_fingerprint=_RENDERER_FINGERPRINT,
        capabilities={},
    )
    if args.mode == "check":
        if result.cache_report.hit_rate < 0.99:
            raise SystemExit(f"portability hit rate {result.cache_report.hit_rate:.6f} < 0.99")
        if result.frame_hashes != expected:
            raise SystemExit("portable cache frame hashes differ from seed host")
        if any(frame_hash != golden_sha256 for frame_hash in result.frame_hashes):
            raise SystemExit("portable cache output does not match committed TERRA golden")
    elif result.cache_report.hit_rate != 0.0:
        raise SystemExit(
            f"capability/backend mismatch served hits: {result.cache_report.hit_rate:.6f}"
        )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "hashes_match": result.frame_hashes == expected,
                "golden_sha256": golden_sha256,
                "hits": len(result.cache_report.hits),
                "misses": len(result.cache_report.misses),
                "hit_rate": result.cache_report.hit_rate,
                "bytes_read": result.cache_report.bytes_read,
                "bytes_written": result.cache_report.bytes_written,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
