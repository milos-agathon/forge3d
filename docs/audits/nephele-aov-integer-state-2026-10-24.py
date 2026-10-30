"""Reproduce the recorded G6 failure from downloaded evidence; CPU only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["FORGE3D_NO_BOOTSTRAP"] = "1"
from scripts.nephele_acceptance import GATES

HEAD = "bbda8eeb0f77f386126a15edd1268b74f48f537b"
RUN = "40031822131"


def counts(path):
    cases = list(ET.parse(path).iter("testcase"))
    result = dict(tests=len(cases), failed=0, errors=0, skipped=0)
    for tag, key in (("failure", "failed"), ("error", "errors"), ("skipped", "skipped")):
        result[key] = sum(case.find(tag) is not None for case in cases)
    result["passed"] = result["tests"] - result["failed"] - result["errors"] - result["skipped"]
    return result, cases


def verify(artifact):
    def read(name):
        return json.loads((artifact / name).read_text(encoding="utf-8"))

    assert (artifact / "checked-out-head.txt").read_text(encoding="utf-8-sig").strip() == HEAD
    probe = read("adapter-probe-raw.json")["probe"]
    assert probe["backend"].lower() == "vulkan" and "NVIDIA" in probe["name"].upper()
    assert probe["device_type"] == "DiscreteGpu" and probe["software_fallback"] is False
    fast, fast_cases = counts(artifact / "fast-junit.xml")
    contracts, cases = counts(artifact / "contracts-environment-junit.xml")
    physical, _ = counts(artifact / "junit.xml")
    assert fast == dict(tests=661, failed=0, errors=0, skipped=0, passed=661)
    assert contracts == dict(tests=10, failed=0, errors=0, skipped=0, passed=10)
    assert physical == dict(tests=6, failed=1, errors=0, skipped=0, passed=5)
    environment = [dict(case.attrib) for case in cases if case.get("classname") == "tests.test_nephele_environment_defects"]
    assert {case["name"] for case in environment} == {
        "test_camera_visible_uniform_environment_readback", "test_shadow_scattering_component_readback"}
    budget = [case for case in fast_cases if case.get("name") == "test_600mib_host_visible_raises_named_budget_error"]
    assert len(budget) == 1 and len(budget[0]) == 0
    trace = (artifact / "directional-trace.log").read_text(encoding="utf-8-sig")
    iterations = [dict(order=int(m[1]), progress=[int(v) for v in m[2].split(",")], elapsed_ms=int(m[3]))
        for m in re.finditer(r"order=(\d+) progress=\[([^]]+)\] elapsed_ms=(\d+)", trace)]
    assert iterations and iterations[-1]["progress"] == [0, 0, 0, 0]
    assert all(item["progress"][3] == 0 for item in iterations)
    diagnostics = {name: read(f"run-{name}.json")["diagnostics"] for name in ("a", "b")}
    for d in diagnostics.values():
        assert d["source_revision"] == HEAD
        assert d["backend"].lower() == "vulkan" and "NVIDIA" in d["adapter"].upper()
        convergence = d["directional_transport"]
        assert convergence["converged"]
        assert convergence["maximum_source_change"] == convergence["maximum_surface_change"] == 0
    differences = []
    def compare(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                compare(a.get(key), b.get(key), path + "." + key)
        elif a != b:
            differences.append(dict(path=path, a=a, b=b))
    compare(diagnostics["a"], diagnostics["b"], "diagnostics")
    assert len(differences) == 1
    assert differences[0]["path"] == "diagnostics.directional_transport.iteration_wall_milliseconds"
    gates = {name: gate(artifact) for name, gate in GATES.items()}
    assert all(not gates[name]["failures"] for name in ("gate1", "gate2", "gate3", "gate4", "gate5"))
    assert gates["gate6"]["failures"] == ["diagnostics identical across runs"]
    assert not (artifact / "realistic-dem-frame-time.json").exists()
    qualification = json.loads((ROOT / "docs/audits/nephele-directional-qualification-2026-10-22.json").read_text(encoding="utf-8"))
    measured = dict(
        b1_fraction=gates["gate3"]["metrics"]["sky_cloud_delta_e_pass_fraction"],
        shaft_ssim=gates["gate3"]["metrics"]["godray_roi_ssim"],
        g4a_fraction=gates["gate4"]["metrics"]["shadow_delta_e_pass_fraction"],
        medium_changed_fraction=gates["gate4"]["metrics"]["medium_ablation_changed_fraction"],
        g4b_response=gates["gate4"]["metrics"]["terrain_occlusion_mean_linear_luminance_change"],
        g4b_relative_error=gates["gate4"]["metrics"]["terrain_occlusion_response_relative_error"])
    forecast = qualification["candidate"]["metrics"]
    return dict(status="FAILED_GATE6_DIAGNOSTIC_WALL_TIME; NOT RELEASE READY", head_sha=HEAD,
        run_url=f"https://github.com/milos-agathon/forge3d/actions/runs/{RUN}", adapter=probe,
        fast_junit=fast, contracts_environment_junit=contracts, environment_cases=environment,
        physical_junit=physical, trace_convergence=iterations[-1], capture_diagnostics=diagnostics,
        diagnostic_differences=differences, gates=gates, measured_image_metrics=measured,
        forecast=forecast, measured_minus_forecast={k: v-forecast[k] for k, v in measured.items()},
        prior_physical=qualification["baselines"]["68cc04e9"],
        dem_1500_frame_time="ABSENT: CI benchmark step skipped after Gate 6 failed",
        evidence_sha256={name: hashlib.sha256((artifact / name).read_bytes()).hexdigest() for name in (
            "checked-out-head.txt", "fast-junit.xml", "contracts-environment-junit.xml",
            "junit.xml", "directional-trace.log", "run-a.json", "run-b.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / f".release-evidence/2026-10-24/ci-{RUN}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.artifact.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "head_sha", "trace_convergence", "physical_junit", "measured_image_metrics", "diagnostic_differences")}))


if __name__ == "__main__":
    main()
