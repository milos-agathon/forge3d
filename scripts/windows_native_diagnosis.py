"""CI-only observation of the existing Windows failures; preserves pytest results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SELECTORS = [
    "tests/test_hdr_fallback_frame.py",
    "tests/test_mapscene_native_pipeline.py::test_height_atmosphere_and_froxel_stage_combinations",
    "tests/test_mapscene_render_cache.py::test_nan_hole_terrain_compiles_and_renders",
    "tests/test_mapscene_screen_framing.py::test_screen_framing_preserves_feature_location",
    "tests/test_mapscene_terrain_native_cache.py::test_physical_terrain_mutation_cannot_cross_hit",
    "tests/test_material_resident_cache.py::test_cache_tracks_resident_material_bytes_and_dimensions",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "win32":
        raise SystemExit("This diagnostic may run only in Windows GitHub Actions; no local GPU execution.")
    if os.environ.get("FORGE3D_NO_BOOTSTRAP") != "1" or os.environ.get("FORGE3D_TEST_INSTALLED_WHEEL") != "1":
        raise SystemExit("Requires the exact-head installed wheel and disabled bootstrap.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise SystemExit(f"Checkout mismatch: {head} != {args.expected_head}")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import pytest
    import forge3d as f3d
    from forge3d._native import get_native_module
    import forge3d.map_scene as ms

    native = Path(get_native_module().__file__).resolve()
    package = Path(f3d.__file__).resolve()
    source = Path("python/forge3d").resolve()
    if package.is_relative_to(source) or native.is_relative_to(source):
        raise SystemExit("Refusing source-tree bindings; install the exact-head wheel first.")
    metadata = {
        "head": head,
        "python": sys.version,
        "package": str(package),
        "native": str(native),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "adapter": f3d.device_probe(),
        "environment": {name: os.environ.get(name) for name in (
            "GITHUB_ACTIONS", "RUNNER_OS", "RUNNER_NAME", "WGPU_BACKEND", "WGPU_BACKENDS",
            "FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN", "FORGE3D_DETERMINISTIC",
            "FORGE3D_NO_BOOTSTRAP", "FORGE3D_TEST_INSTALLED_WHEEL",
        )},
        "selectors": SELECTORS,
        "startup_degradations": f3d.native_degradations(),
    }
    (args.output / "identity.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    class Observe:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            outcome = yield
            report = outcome.get_result()
            if report.when != "call":
                return
            record = {
                "nodeid": item.nodeid, "outcome": report.outcome,
                "duration": report.duration,
                "degradations_cumulative": f3d.native_degradations(),
                "terrain_availability_cache": ms._terrain_renderer_runtime_available.cache_info()._asdict(),
                "failed_frame_locals": [],
            }
            if call.excinfo is not None:
                record["exception"] = str(call.excinfo.value)
                for entry in call.excinfo.traceback:
                    local = entry.frame.f_locals
                    details = {"path": str(entry.path), "line": entry.lineno + 1}
                    for name in ("pixels", "original", "changed", "reused", "uncached", "rgba"):
                        value = local.get(name)
                        if isinstance(value, np.ndarray):
                            finite = np.isfinite(value)
                            details[name] = {
                                "shape": list(value.shape), "dtype": str(value.dtype),
                                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                                "finite": bool(finite.all()),
                                "min": float(value[finite].min()) if finite.any() else None,
                                "max": float(value[finite].max()) if finite.any() else None,
                                "nonzero_rgb": int(np.count_nonzero(value[..., :3])) if value.ndim >= 3 else None,
                            }
                    renderer = local.get("renderer")
                    if renderer is not None and hasattr(renderer, "last_anamnesis_cache_report"):
                        details["cache_report"] = dict(renderer.last_anamnesis_cache_report)
                    if len(details) > 2:
                        record["failed_frame_locals"].append(details)
            with (args.output / "cases.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=str) + "\n")

    result = pytest.main([
        *SELECTORS, "-v", "--tb=long", "--capture=tee-sys",
        f"--junitxml={args.output / 'junit.xml'}",
        f"--basetemp={args.output / 'pytest-temp'}",
    ], plugins=[Observe()])
    (args.output / "result.json").write_text(json.dumps({"pytest_exit": int(result)}), encoding="utf-8")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
