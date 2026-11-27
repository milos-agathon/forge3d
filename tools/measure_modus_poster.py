"""Run one MODUS qualification job per fresh interpreter. See the poster guide."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from forge3d import path_tracing as pt


DEM_SHAPES = {
    "dem-1024": (1024, 1024),
    "dem-rect": (960, 1536),
    "dem-2048": (2048, 2048),
    "dem-2048-nodata": (2048, 2048),
    "dem-2048-many": (2048, 2048),
    "refuse-dem": (2048, 2048),
    "empty-dem": (2048, 2048),
}


def peak():
    # Independent test observer. Windows psutil exposes PeakPagefileUsage.
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_pagefile
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def assert_tracked_cpu_peak(report):
    samples = report["ledger_samples"]
    baseline = samples["baseline"]
    at_cpu_peak = samples["at_cpu_peak"]
    rust_growth = at_cpu_peak["rust_heap"] - baseline["rust_heap"]
    python_growth = at_cpu_peak["python_arrays"] - baseline["python_arrays"]
    observed = rust_growth + python_growth
    at_process_peak = samples["at_peak"]
    assert observed >= max(0, at_process_peak["rust_heap"] + at_process_peak["python_arrays"]
                           - baseline["rust_heap"] - baseline["python_arrays"])
    assert observed == report["sampled_cpu_peak_growth_bytes"]
    assert observed <= report["cpu_peak_bytes"], (
        f"tracked CPU growth {observed:,} exceeds estimate {report['cpu_peak_bytes']:,} "
        f"(Rust {rust_growth:,}, Python arrays {python_growth:,})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("calibration", "compact", "large", "plain", "material", "masked", "empty",
                                        "refuse", "cancel", "resume", *DEM_SHAPES))
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--control", action="store_true", help="run the pre-MODUS implementation")
    parser.add_argument("--exact-limit", action="store_true", help="admit at the public preview's exact prediction")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    material = args.job == "material"
    y, x = np.mgrid[:192, :192]
    dem = (4 * np.sin(x / 12) * np.cos(y / 16)).astype("f4")
    width, height, tile = (513, 257, 256) if not material else (256, 256, 128)
    if args.job == "large":
        width, height, tile = 1024, 1024, 512
    if args.job == "calibration":
        width, height, tile = 4, 4, 2
    if args.job in DEM_SHAPES:
        dem = np.ones(DEM_SHAPES[args.job], dtype=np.float32)
        if args.job == "dem-2048-nodata":
            dem[1024, 1024] = np.nan
        width = height = 8 if args.job == "dem-2048-many" else 4
        tile = 2
    if args.job == "masked":
        mask = (x - 96) ** 2 + (y - 96) ** 2 < 24 ** 2
        dem = np.ma.array(np.asfortranarray(dem, dtype=np.float64), mask=mask)
    opts = dict(tile=tile, min_frames=32, max_frames=32, variance_threshold=1e9, seed=7,
                camera=dict(model="orthographic", origin=(0, 300, 100), look_at=(0, 0, 0),
                            up=(0, 0, -1), half_height=32))
    if args.job == "compact":
        dem = np.zeros((32, 32), dtype=np.float32)
        width = height = opts["tile"] = 32
        opts["camera"] = dict(model="off_axis", origin=(0, 63, 44), look_at=(0, 0.25, 0),
                              up=(0, 0.574, -0.819), fov_y=35.0)
    if material:
        dem = dem.T.astype("f8")
        opts["albedo_map"] = np.full((*dem.shape, 4), (0.6, 0.4, 0.2, 1), dtype="f8")
    if args.job in ("cancel", "resume"):
        if args.checkpoint is None:
            parser.error("recovery requires --checkpoint")
        opts.update(checkpoint_path=args.checkpoint, checkpoint_every=8, resume=args.job == "resume")
        if args.job == "cancel":
            opts["on_progress"] = lambda e: not (e.tile == (1, 0) and e.completed_frames == 8)
    if args.job in ("empty", "refuse"):
        # Present a real, already resident caller input requiring a float32 copy.
        dem = np.ones((2048, 2048), dtype=np.float64)
        width = height = 16384
        opts["process_limit_bytes"] = width * height * 4

    def empty(*args, **kwargs):
        pass

    if args.job in ("refuse-dem", "empty-dem"):
        opts["process_limit_bytes"] = 1_146_987_452  # original admitted overrun
    empty_call = args.job in ("empty", "empty-dem")
    refusal = args.job in ("refuse", "refuse-dem")
    render = not (empty_call or refusal or args.job == "cancel")
    certificate = args.output / (args.job + ".cert.json")
    if render and not args.control and not args.exact_limit:
        opts["certificate"] = certificate
    call = empty if empty_call else pt.render_terrain_poster
    before = peak()
    if args.exact_limit:
        if not render or args.control:
            parser.error("--exact-limit requires a candidate render")
        from forge3d import mem
        opts["process_limit_bytes"] = mem.estimate(dict(entry_point="render_terrain_poster",
            heightmap=dem, width=width, height=height, **opts))["predicted_peak_bytes"]
    try:
        result = call(dem, width, height, **opts)
    except (MemoryError, pt.TerrainRenderCancelled) as exc:
        after = peak()
        expected = {"refuse": "memory_admission_refused", "refuse-dem": "memory_admission_refused",
                    "cancel": "terrain_render_cancelled"}
        assert exc.code == expected[args.job]
        result = {"memory": getattr(exc, "estimate", {}), "error_code": exc.code,
                  "limit_bytes": getattr(exc, "limit_bytes", None)}
    else:
        after = peak()
        assert not refusal and args.job != "cancel"
    report = dict(job=args.job, pid=os.getpid(), baseline=before, peak=after, growth=after-before)
    if render:
        assert isinstance(result, dict) and "rgba" in result
        if not args.exact_limit:
            expected_keys = {"rgba", "tiles", "certificate_digest"}
            if args.job == "resume":
                expected_keys.update({"resumed", "recovered_tiles", "frames_dispatched"})
            assert set(result) == expected_keys
    if result:
        report.update(memory=result.get("memory"), recovered_tiles=result.get("recovered_tiles"),
                      frames_dispatched=result.get("frames_dispatched"))
        if "error_code" in result:
            report.update(error_code=result["error_code"], limit_bytes=result["limit_bytes"])
        if "rgba" in result:
            rgba = result["rgba"]
            report["sha256"] = hashlib.sha256(memoryview(rgba)).hexdigest()
            Image.fromarray(rgba).save(args.output / (args.job + ".png"))
            if args.reference:
                with Image.open(args.reference) as ref:
                    np.testing.assert_array_equal(rgba, np.asarray(ref))
            for k in ("gpu_resource_bytes", "peak_host_visible_bytes", "peak_device_local_bytes"):
                values = [t[k] for t in result["tiles"] if k in t]
                if values:
                    report[k] = max(values)
            # Value storage in a tile receipt, including its traversal dictionaries.
            def retained(value):
                size = sys.getsizeof(value)
                if isinstance(value, dict):
                    size += sum(retained(v) for v in value.values())
                elif isinstance(value, (list, tuple)):
                    size += sum(retained(v) for v in value)
                return size
            report["tile_record_bytes"] = max(retained(t) for t in result["tiles"])
    if render and not args.control:
        if args.exact_limit:
            report["limit_bytes"] = opts["process_limit_bytes"]
            memory = result["memory"]
        else:
            assert result["certificate_digest"]
            memory = json.loads(certificate.read_text(encoding="utf8"))["process_memory"]
        report["memory"] = memory
    (args.output / (args.job + ".json")).write_text(json.dumps(report, indent=2), encoding="utf8")
    print({k: v for k, v in report.items() if k != "memory"})
    if render and not args.control:
        assert memory["measured_peak_bytes"] == after, (memory["measured_peak_bytes"], after)
        assert before <= after <= memory["predicted_peak_bytes"]
        assert memory["prediction_ratio"] >= 1 and memory["unexplained_bytes"] is not None
        assert sum(memory["ledger_samples"]["residual_by_phase_at_peak"].values()) == memory["sampled_unexplained_bytes"]
        if args.exact_limit:
            assert after <= report["limit_bytes"]
        assert report["peak_host_visible_bytes"] <= memory["host_visible_peak_bytes"]
        assert report["peak_device_local_bytes"] <= memory["device_local_peak_bytes"]
        assert_tracked_cpu_peak(memory)
    if refusal:
        empty_job = "empty-dem" if args.job == "refuse-dem" else "empty"
        noise = json.loads((args.output / (empty_job + ".json")).read_text())["growth"]
        assert after - before <= noise, (after - before, noise)
    if args.job == "resume":
        assert result["recovered_tiles"] == 1 and result["tiles"][1]["resume_frame"] == 8


if __name__ == "__main__":
    main()
