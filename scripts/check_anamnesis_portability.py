"""Fail-closed native-terrain cache portability and mismatch controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile

import numpy as np

import forge3d as f3d
from forge3d.determinism import (
    _canonical_params_config,
    canonical_heightmap,
    write_canonical_hdr,
)


_PORTABLE_PROFILE = "terra-determinata-native-portable-v1"
_PASS_LABELS = [
    "terrain.prepare",
    "terrain.shadow",
    "terrain.forward",
    "terrain.resolve",
]


def _adapter(path: str) -> dict:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    adapter = json.loads(lines[-1]).get("adapter")
    if not adapter or adapter.get("software_fallback") is not False:
        raise SystemExit("render lacks attributable physical-adapter metadata")
    return dict(adapter)


def _machine_id(path: str | None) -> str:
    if not path:
        raise SystemExit("native portability requires --machine-id-file")
    value = Path(path).read_text(encoding="utf-8").strip().lower()
    if not value or value in {"none", "unknown", "00000000-0000-0000-0000-000000000000"}:
        raise SystemExit("native portability machine identity is absent or invalid")
    return value


def _runner_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise SystemExit("native portability requires an attributable runner name")
    return value


def _native_render(
    cache: str
) -> tuple[bytes, dict]:
    heightmap = np.ascontiguousarray(canonical_heightmap(), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="forge3d-anamnesis-portable-") as root:
        hdr_path = Path(root) / "environment.hdr"
        write_canonical_hdr(str(hdr_path))
        renderer = f3d.TerrainRenderer(f3d.Session(window=False))
        frame = renderer.render_terrain_pbr_pom(
            f3d.MaterialSet.terrain_default(),
            f3d.IBL.from_hdr(str(hdr_path), intensity=1.0),
            f3d.TerrainRenderParams(_canonical_params_config()(512, 512)),
            heightmap,
            time_seconds=0.0,
            cache=cache,
        )
        rgba = np.ascontiguousarray(frame.to_numpy(), dtype=np.uint8).tobytes()
        return rgba, dict(renderer.last_anamnesis_cache_report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seed", "check", "mismatch"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--frame-blob")
    parser.add_argument("--golden")
    parser.add_argument("--adapter-record")
    parser.add_argument("--consumer-frame-blob")
    parser.add_argument("--consumer-golden")
    parser.add_argument("--consumer-adapter-record")
    parser.add_argument("--machine-id-file")
    parser.add_argument("--runner-name", default=os.environ.get("RUNNER_NAME", ""))
    parser.add_argument("--producer-backend", default="vulkan")
    parser.add_argument("--consumer-backend", default="vulkan")
    args = parser.parse_args()
    if os.environ.get("FORGE3D_ANAMNESIS_COMPATIBILITY_PROFILE"):
        raise SystemExit("compatibility acceptance requires the actual device profile, without an override")
    if args.producer_backend.lower() != args.consumer_backend.lower():
        raise SystemExit("cross-backend cache reuse is outside the compatibility contract")
    record_path = Path(args.record)

    if args.mode == "seed":
        if not args.frame_blob or not args.golden or not args.adapter_record:
            parser.error("seed requires --frame-blob, --golden, and --adapter-record")
        png_sha256 = hashlib.sha256(Path(args.frame_blob).read_bytes()).hexdigest()
        golden_sha256 = Path(args.golden).read_text(encoding="utf-8").split()[0]
        if png_sha256 != golden_sha256:
            raise SystemExit(
                f"seed render differs from committed golden: actual={png_sha256} "
                f"golden={golden_sha256}"
            )
        adapter = _adapter(args.adapter_record)
        machine_id = _machine_id(args.machine_id_file)
        runner_name = _runner_name(args.runner_name)
        rgba, report = _native_render(args.cache)
        if (
            report["hits"]
            or report["misses"] != _PASS_LABELS
            or report["graph_command_submissions"] != 5
        ):
            raise SystemExit(f"native producer did not seed every terrain pass: {report}")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "forge3d.anamnesis.native-portability/1",
            "golden_sha256": golden_sha256,
            "rgba_sha256": hashlib.sha256(rgba).hexdigest(),
            "producer_adapter": adapter,
            "producer_machine_id": machine_id,
            "producer_runner_name": runner_name,
            "compatibility_profile": _PORTABLE_PROFILE,
            "engine_fingerprint": json.loads(f3d.anamnesis_engine_fingerprint()),
            "native_passes": _PASS_LABELS,
            "seed_report": report,
        }
        record_path.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "mode": "seed",
                    "machine_id": machine_id,
                    "runner_name": runner_name,
                    "hits": len(report["hits"]),
                    "misses": len(report["misses"]),
                    "bytes_written": report["bytes_written"],
                },
                sort_keys=True,
            )
        )
        return 0

    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema") != "forge3d.anamnesis.native-portability/1":
        raise SystemExit("portability record is not a native terrain graph record")
    if record.get("compatibility_profile") != _PORTABLE_PROFILE:
        raise SystemExit("portability record compatibility profile mismatch")
    if record.get("engine_fingerprint") != json.loads(
        f3d.anamnesis_engine_fingerprint()
    ):
        raise SystemExit("portability record was not produced by this exact engine head")

    if args.mode == "check":
        if not args.consumer_frame_blob or not args.consumer_golden or not args.consumer_adapter_record:
            parser.error(
                "check requires --consumer-frame-blob, --consumer-golden, and --consumer-adapter-record"
            )
        machine_id = _machine_id(args.machine_id_file)
        runner_name = _runner_name(args.runner_name)
        distinct_machine = machine_id != record["producer_machine_id"]
        if not distinct_machine:
            # Machine identity is diagnostic. Separate producer/consumer
            # processes with the same full device profile qualify reuse.
            print(json.dumps({
                "check": "distinct_machine", "distinct_machine": "ABSENT",
                "producer_machine_id": record["producer_machine_id"],
                "consumer_machine_id": machine_id,
            }, sort_keys=True), file=sys.stderr)
        consumer_png_sha = hashlib.sha256(
            Path(args.consumer_frame_blob).read_bytes()
        ).hexdigest()
        consumer_golden_sha = Path(args.consumer_golden).read_text(encoding="utf-8").split()[0]
        if consumer_png_sha != consumer_golden_sha:
            raise SystemExit(
                "consumer render differs from committed golden: "
                f"actual={consumer_png_sha} golden={consumer_golden_sha}"
            )
        consumer_adapter = _adapter(args.consumer_adapter_record)
        producer_backend = str(record["producer_adapter"].get("backend", "")).lower()
        consumer_backend = str(consumer_adapter.get("backend", "")).lower()
        if (
            producer_backend != args.producer_backend.lower()
            or consumer_backend != args.consumer_backend.lower()
        ):
            raise SystemExit(
                "physical backend mismatch: "
                f"producer={producer_backend!r}, consumer={consumer_backend!r}"
            )
        rgba, report = _native_render(args.cache)
        if (
            report["hits"] != _PASS_LABELS
            or report["misses"]
            or report["graph_command_submissions"] != 0
        ):
            raise SystemExit(f"native consumer did not restore every terrain pass: {report}")
        if hashlib.sha256(rgba).hexdigest() != record["rgba_sha256"]:
            raise SystemExit("restored native terrain pixels differ from the producer")
        # A cached frame cannot qualify the consumer's compatibility profile.
        # Independently encode the same production graph before accepting reuse.
        with tempfile.TemporaryDirectory(prefix="forge3d-anamnesis-fresh-") as fresh_cache:
            fresh_rgba, fresh_report = _native_render(fresh_cache)
        if (
            fresh_report["hits"]
            or fresh_report["misses"] != _PASS_LABELS
            or fresh_report["graph_command_submissions"] != 5
            or fresh_rgba != rgba
        ):
            raise SystemExit("restored native terrain differs from an independent fresh render")
        from PIL import Image

        with Image.open(args.consumer_frame_blob) as image:
            golden_rgba = image.convert("RGBA").tobytes()
        if fresh_rgba != golden_rgba:
            raise SystemExit("fresh native pixels differ from the golden-checked consumer PNG")
        result = {
            "mode": "check",
            "distinct_machine": True if distinct_machine else "ABSENT",
            "producer_machine_id": record["producer_machine_id"],
            "consumer_machine_id": machine_id,
            "producer_runner_name": record.get("producer_runner_name"),
            "consumer_runner_name": runner_name,
            "producer_adapter": record["producer_adapter"],
            "consumer_adapter": consumer_adapter,
            "backend_goldens_match": True,
            "fresh_native_matches": True,
            "hits": len(report["hits"]),
            "misses": len(report["misses"]),
            "hit_rate": report["hit_rate"],
            "bytes_read": report["bytes_read"],
        }
        print(json.dumps(result, sort_keys=True))
        return 0

    # Reconstruct the actual production keys from their persisted derivation.
    # Vary one negotiated dimension at a time, then ask the native store for
    # each new key. This probes key/store isolation without pretending to have
    # rendered another backend or weakening the physical profile.
    from copy import deepcopy

    packed_index = json.loads((Path(args.cache) / "index.v2.json").read_text(encoding="utf-8"))
    if packed_index.get("schema") != "forge3d.anamnesis.packed-index/2":
        raise SystemExit("mismatch control requires the current packed store schema")
    entries = [record["metadata"] for record in packed_index["entries"].values()
               if record["metadata"].get("pass_label") in _PASS_LABELS]
    if sorted(meta["pass_label"] for meta in entries) != sorted(label for label in _PASS_LABELS if label != "terrain.shadow"):
        raise SystemExit("mismatch control requires all stored production terrain passes (shadow is keyed on demand)")
    results = {}
    for dimension in ("backend", "granted_features", "limits", "dx12_compiler", "naga_capabilities"):
        hits = 0
        for meta in entries:
            material = meta["derivation"]["material"]
            original = bytes.fromhex(material["capability_fingerprint_hex"])
            capabilities = json.loads(original)
            changed = deepcopy(capabilities)
            if dimension in {"backend", "dx12_compiler"}:
                changed[dimension] += "-mismatch"
            elif dimension == "limits":
                name = sorted(changed[dimension])[0]
                changed[dimension][name] += 1
            else:
                changed[dimension].append("mismatch-control")
            def key(capability_bytes):
                return f3d.anamnesis_pass_key(
                    material["label"], bytes.fromhex(material["pipeline_descriptor_hex"]),
                    bytes.fromhex(material["uniform_hex"]),
                    [(item["binding"], item["key"]) for item in material["input_keys"]],
                    capability_bytes, bytes.fromhex(material["engine_fingerprint_hex"]),
                )
            if key(original) != meta["key"]:
                raise SystemExit("production key cannot be reconstructed")
            revised = key(json.dumps(changed, sort_keys=True, separators=(",", ":")).encode())
            if revised == meta["key"]:
                raise SystemExit(f"{dimension} did not affect the production key")
            if f3d.anamnesis_store_get(args.cache, revised, 512 * 1024 * 1024) is not None:
                hits += 1
        results[dimension] = {"hits": hits, "misses": len(entries) - hits, "hit_rate": hits / len(entries)}
        if hits:
            raise SystemExit(f"{dimension} mismatch served cached terrain: {results[dimension]}")
    print(json.dumps({"mode": "mismatch", "hits": 0, "misses": len(entries) * len(results),
                      "hit_rate": 0.0, "cache_isolated": True, "dimensions": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
