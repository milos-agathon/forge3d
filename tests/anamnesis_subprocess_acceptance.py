"""The public 64x64/600-frame workload, restored by a new interpreter."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _run_phase(phase: str, root: Path) -> None:
    import forge3d as f3d
    import forge3d.map_scene as ms
    from forge3d.anamnesis import _Store, render_sequence
    import forge3d.anamnesis as anamnesis
    from forge3d._native import get_native_module
    from test_anamnesis_scene import _scenes
    from test_chronos_flythrough import _synthetic_scene

    # Keep this identical to test_public_scene_600_frame_acceptance. Scene
    # construction and interpreter imports precede SequenceResult's timer.
    scenes = {frame: _synthetic_scene() for frame in range(600)}
    for frame, scene in scenes.items():
        scene.recipe = replace(scene.recipe, camera=replace(scene.recipe.camera, azimuth_deg=20.0 + frame * 0.25))
    scenes[250].recipe = replace(scenes[250].recipe, layers=_scenes("Peak" if phase == "original" else "Hill")[1].recipe.layers)
    native = get_native_module()
    native.clear_native_degradations()
    native_calls, first_read = [0], [None]
    native_render, read_entry = ms._render_native_offscreen_result, _Store._read_entry
    resident_manifest = _Store._resident_manifest
    residency_seconds = []
    refresh_residency = _Store._refresh_residency
    inventory_seconds = []
    preload_snapshot = anamnesis._load_packed_snapshot
    preload_seconds = []
    verify_packed_reads = anamnesis._verify_packed_reads
    native_verify_seconds = []
    write_batch = _Store.put_pass_batch
    write_batch_seconds = []
    first_packed_open = [None]
    open_path = Path.open

    def observe(*args, **kwargs):
        native_calls[0] += 1
        return native_render(*args, **kwargs)

    def observe_read(*args, **kwargs):
        if first_read[0] is None:
            first_read[0] = time.time_ns()
        return read_entry(*args, **kwargs)

    def observe_residency(*args, **kwargs):
        before = time.perf_counter_ns()
        try:
            return resident_manifest(*args, **kwargs)
        finally:
            residency_seconds.append((time.perf_counter_ns() - before) / 1e9)

    def observe_inventory(*args, **kwargs):
        before = time.perf_counter_ns()
        try:
            return refresh_residency(*args, **kwargs)
        finally:
            inventory_seconds.append((time.perf_counter_ns() - before) / 1e9)

    def observe_preload(*args, **kwargs):
        before = time.perf_counter_ns()
        try:
            return preload_snapshot(*args, **kwargs)
        finally:
            preload_seconds.append((time.perf_counter_ns() - before) / 1e9)

    def observe_native_verify(*args, **kwargs):
        if first_read[0] is None:
            first_read[0] = time.time_ns()
        before = time.perf_counter_ns()
        try:
            return verify_packed_reads(*args, **kwargs)
        finally:
            native_verify_seconds.append((time.perf_counter_ns() - before) / 1e9)

    def observe_write_batch(*args, **kwargs):
        before = time.perf_counter_ns()
        try:
            return write_batch(*args, **kwargs)
        finally:
            write_batch_seconds.append((time.perf_counter_ns() - before) / 1e9)

    def observe_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if (first_packed_open[0] is None and mode == "rb" and path.parent == root / "store"
                and path.name in {_Store.DATA, _Store.INDEX}):
            first_packed_open[0] = time.time_ns()
        return open_path(path, *args, **kwargs)

    ms._render_native_offscreen_result = observe
    _Store._read_entry = observe_read
    _Store._resident_manifest = observe_residency
    _Store._refresh_residency = observe_inventory
    anamnesis._load_packed_snapshot = observe_preload
    anamnesis._verify_packed_reads = observe_native_verify
    Path.open = observe_open
    _Store.put_pass_batch = observe_write_batch
    started = time.time_ns()
    try:
        result = render_sequence(scenes, cache=None if phase == "cold" else root / "store")
    finally:
        ms._render_native_offscreen_result = native_render
        _Store._read_entry = read_entry
        _Store._resident_manifest = resident_manifest
        _Store._refresh_residency = refresh_residency
        anamnesis._load_packed_snapshot = preload_snapshot
        anamnesis._verify_packed_reads = verify_packed_reads
        Path.open = open_path
        _Store.put_pass_batch = write_batch
    finished = time.time_ns()
    record = dict(
        phase=phase, pid=os.getpid(), elapsed_seconds=result.elapsed_seconds,
        resident_manifest_seconds=sum(residency_seconds),
        resident_manifest_calls=residency_seconds,
        resident_inventory_seconds=sum(inventory_seconds),
        resident_inventory_calls=inventory_seconds,
        resident_preload_seconds=sum(preload_seconds), resident_preload_calls=preload_seconds,
        first_packed_file_open_unix_ns=first_packed_open[0],
        native_verify_seconds=sum(native_verify_seconds), native_verify_calls=native_verify_seconds,
        packed_write_batch_seconds=sum(write_batch_seconds), packed_write_batch_calls=write_batch_seconds,
        render_started_unix_ns=started, render_finished_unix_ns=finished,
        first_persisted_read_unix_ns=first_read[0], native_calls=native_calls[0],
        frame_hashes=result.frame_hashes,
        # Hash the returned blobs independently of the driver's hash list.
        blob_hashes=[hashlib.sha256(blob).hexdigest() for blob in result.frame_blobs],
        predicted=sorted(result.predicted_recompute), observed=sorted(result.observed_recompute),
        hits=len(result.cache_report.hits), bytes_written=result.cache_report.bytes_written,
        adapter=f3d.device_probe(), degradations=native.native_degradations(),
    )
    (root / f"{phase}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def run_acceptance(root: Path) -> dict:
    """Retain child failures and metrics before the caller asserts the gate."""
    root.mkdir(parents=True, exist_ok=True)
    receipt = {"metric": "SequenceResult.elapsed_seconds", "phases": {}}

    def save():
        payload = json.dumps(receipt, indent=2)
        (root / "fresh-process-600.json").write_text(payload, encoding="utf-8")
        if destination := os.environ.get("FORGE3D_ANAMNESIS_FRESH_RESULT_PATH"):
            Path(destination).write_text(payload, encoding="utf-8")

    try:
        for phase in ("original", "incremental", "cold"):
            print(f"ANAMNESIS fresh-process {phase}: started", flush=True)
            started = time.perf_counter()
            child = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), phase, str(root.resolve())],
                capture_output=True, text=True,
            )
            wall = time.perf_counter() - started
            (root / f"{phase}.stdout.log").write_text(child.stdout, encoding="utf-8")
            (root / f"{phase}.stderr.log").write_text(child.stderr, encoding="utf-8")
            record = {"exit_code": child.returncode, "process_wall_seconds": wall}
            if (root / f"{phase}.json").exists():
                record.update(json.loads((root / f"{phase}.json").read_text(encoding="utf-8")))
            record["device_lost"] = (any(item.get("kind") == "device_lost" for item in record.get("degradations", ()))
                                     or any(token in child.stderr.lower() for token in ("device lost", "devicelost", "device_lost", "device reset")))
            receipt["phases"][phase] = record
            save()
            if record["device_lost"]:
                raise RuntimeError(f"Device lost in {phase}: later GPU results in this process are invalid; start a new process")
            print(f"ANAMNESIS fresh-process {phase}: checkpoint exit={child.returncode}, elapsed={record.get('elapsed_seconds')} s", flush=True)
            if child.returncode:
                raise RuntimeError(f"ANAMNESIS {phase} child exited {child.returncode}; see {root / (phase + '.stderr.log')}")
        original, incremental, cold = (receipt["phases"][name] for name in ("original", "incremental", "cold"))
        receipt.update(
            matching_frames=sum(a == b for a, b in zip(incremental["blob_hashes"], cold["blob_hashes"], strict=True)),
            cold_seconds=cold["elapsed_seconds"], incremental_seconds=incremental["elapsed_seconds"],
            speedup=cold["elapsed_seconds"] / incremental["elapsed_seconds"],
            original_process_wall_seconds=original["process_wall_seconds"],
            store_age_at_incremental_start_seconds=(incremental["render_started_unix_ns"] - original["render_finished_unix_ns"]) / 1e9,
            store_age_at_first_read_seconds=(incremental["first_persisted_read_unix_ns"] - original["render_finished_unix_ns"]) / 1e9,
            store_age_at_first_packed_open_seconds=(incremental["first_packed_file_open_unix_ns"] - original["render_finished_unix_ns"]) / 1e9,
        )
        return receipt
    finally:
        save()


if __name__ == "__main__":
    _run_phase(sys.argv[1], Path(sys.argv[2]))
