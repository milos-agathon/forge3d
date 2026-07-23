"""CI driver for ANAMNESIS store portability and mismatch isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import forge3d as f3d
from forge3d.anamnesis import render_sequence


_PORTABLE_PROFILE = "terra-determinata-portable-v1"
_RENDERER_FINGERPRINT = hashlib.sha256(
    b"forge3d.anamnesis.terra-determinata-golden/v1"
).digest()
_PASS_FINGERPRINTS = {
    label: hashlib.sha256(_RENDERER_FINGERPRINT + label.encode("utf-8")).digest()
    for label in ("terrain.shade", "accumulation", "frame.output")
}


def _identity(_state, _frame, inputs):
    return inputs[0]


def _pass_contexts(golden_sha256: str) -> dict[str, bytes]:
    return {
        "terrain.shade": (
            b"terra-determinata-production-rgba/v1\0"
            + golden_sha256.encode("ascii")
        ),
        "accumulation": b"identity/no-hidden-inputs/v1",
        "frame.output": b"identity/no-hidden-inputs/v1",
    }


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
    parser.add_argument("--consumer-frame-blob")
    parser.add_argument("--consumer-adapter-record")
    parser.add_argument("--producer-backend", default="vulkan")
    parser.add_argument("--consumer-backend", default="dx12")
    args = parser.parse_args()
    record_path = Path(args.record)

    if args.mode == "seed":
        if not args.frame_blob or not args.golden or not args.adapter_record:
            parser.error("seed requires --frame-blob, --golden, and --adapter-record")
        png_blob = Path(args.frame_blob).read_bytes()
        actual_sha256 = hashlib.sha256(png_blob).hexdigest()
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
        rgba = np.ascontiguousarray(f3d.png_to_numpy(args.frame_blob), dtype=np.uint8)
        if rgba.shape != (512, 512, 4):
            raise SystemExit(f"unexpected seed RGBA shape: {rgba.shape!r}")
        frame_blob = rgba.tobytes()
        rgba_sha256 = hashlib.sha256(frame_blob).hexdigest()
        result = render_sequence(
            recipe(),
            frames=[0],
            cache=args.cache,
            pass_executors={
                "terrain.shade": lambda _state, _frame, _inputs: frame_blob,
                "accumulation": _identity,
                "frame.output": _identity,
            },
            pass_executor_fingerprints=_PASS_FINGERPRINTS,
            pass_executor_contexts=_pass_contexts(golden_sha256),
            capabilities={},
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "frame_hashes": result.frame_hashes,
                    "golden_sha256": golden_sha256,
                    "rgba_sha256": rgba_sha256,
                    "width": 512,
                    "height": 512,
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
    if args.mode == "check":
        if not args.consumer_frame_blob or not args.consumer_adapter_record:
            parser.error(
                "check requires --consumer-frame-blob and --consumer-adapter-record"
            )
        consumer_sha256 = hashlib.sha256(
            Path(args.consumer_frame_blob).read_bytes()
        ).hexdigest()
        if consumer_sha256 != golden_sha256:
            raise SystemExit(
                "consumer render differs from committed golden: "
                f"actual={consumer_sha256} golden={golden_sha256}"
            )
        consumer_lines = Path(args.consumer_adapter_record).read_text(
            encoding="utf-8"
        ).splitlines()
        consumer_adapter = json.loads(consumer_lines[-1]).get("adapter")
        if not consumer_adapter or consumer_adapter.get("software_fallback") is not False:
            raise SystemExit("consumer render lacks attributable physical-adapter metadata")
        producer_backend = str(record["producer_adapter"].get("backend", "")).lower()
        consumer_backend = str(consumer_adapter.get("backend", "")).lower()
        expected_producer = str(args.producer_backend).lower()
        expected_consumer = str(args.consumer_backend).lower()
        if (
            producer_backend != expected_producer
            or consumer_backend != expected_consumer
        ):
            raise SystemExit(
                "portability check physical backend mismatch; "
                f"expected producer={expected_producer!r} consumer={expected_consumer!r}, "
                f"got producer={producer_backend!r} consumer={consumer_backend!r}"
            )
        consumer_rgba = np.ascontiguousarray(
            f3d.png_to_numpy(args.consumer_frame_blob), dtype=np.uint8
        )
        consumer_raw = consumer_rgba.tobytes()
        if hashlib.sha256(consumer_raw).hexdigest() != record["rgba_sha256"]:
            raise SystemExit(
                "consumer production RGBA differs from the seeded production pass"
            )
    active_recipe = (
        recipe(_PORTABLE_PROFILE + "-mismatch") if args.mode == "mismatch" else recipe()
    )

    def should_not_render(_state, _frame, _inputs):
        if args.mode == "check":
            raise RuntimeError("portable cache unexpectedly invoked the renderer")
        return b"capability-mismatch-recomputed"

    result = render_sequence(
        active_recipe,
        frames=[0],
        cache=args.cache,
        pass_executors={
            "terrain.shade": should_not_render,
            "accumulation": _identity,
            "frame.output": _identity,
        },
        pass_executor_fingerprints=_PASS_FINGERPRINTS,
        pass_executor_contexts=_pass_contexts(golden_sha256),
        capabilities={},
    )
    if args.mode == "check":
        if result.cache_report.hit_rate < 0.99:
            raise SystemExit(f"portability hit rate {result.cache_report.hit_rate:.6f} < 0.99")
        if result.frame_hashes != expected:
            raise SystemExit("portable cache frame hashes differ from seed host")
        if any(
            frame_hash != record["rgba_sha256"] for frame_hash in result.frame_hashes
        ):
            raise SystemExit(
                "portable cached production RGBA differs from the seed host"
            )
        restored = bytes(
            f3d.anamnesis_restore_rgba8(
                result.frame_blobs[0],
                int(record["width"]),
                int(record["height"]),
            )
        )
        if restored != result.frame_blobs[0] or restored != consumer_raw:
            raise SystemExit(
                "portable cached pass did not rehydrate byte-identically as a GPU texture"
            )
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
