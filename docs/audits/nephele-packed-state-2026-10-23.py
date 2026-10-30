"""Verify the downloaded 95b66c50 trace; CPU only, never constructs GPU objects."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["FORGE3D_NO_BOOTSTRAP"] = "1"
from scripts.nephele_acceptance import GATES

HEAD = "95b66c50566e38006aa2e7ac5ecb2065e9253731"
RUN = "40023229060"


def verify(artifact):
    artifact = artifact.resolve()
    assert (artifact / "checked-out-head.txt").read_text(encoding="utf-8-sig").strip() == HEAD
    probe = json.loads((artifact / "adapter-probe-raw.json").read_text())["probe"]
    assert probe["status"] == "ok" and probe["backend"].lower() == "vulkan"
    assert "NVIDIA" in probe["name"].upper()
    assert probe["device_type"] == "DiscreteGpu" and probe["software_fallback"] is False
    xml = ET.parse(artifact / "fast-junit.xml")
    cases = list(xml.iter("testcase"))
    counts = dict(tests=len(cases), failures=0, errors=0, skipped=0)
    for name in ("failure", "error", "skipped"):
        counts[{"failure": "failures", "error": "errors", "skipped": "skipped"}[name]] = sum(
            case.find(name) is not None for case in cases
        )
    assert counts == dict(tests=661, failures=0, errors=0, skipped=0)
    budget = [case for case in cases if case.get("name") == "test_600mib_host_visible_raises_named_budget_error"]
    assert len(budget) == 1 and len(budget[0]) == 0

    trace = (artifact / "directional-trace.log").read_text(encoding="utf-8-sig")
    passes = []
    for match in re.finditer(r"\[NEPHELE directional\] order=(\d+) progress=\[([^]]+)\] elapsed_ms=(\d+)", trace):
        passes.append(dict(order=int(match[1]), progress=[int(x) for x in match[2].split(",")], elapsed_ms=int(match[3])))
    assert [entry["order"] for entry in passes] == list(range(1, 23))
    assert all(entry["progress"][0] == 1 for entry in passes[:-1])
    assert passes[-1]["progress"] == [0, 931111685, 956246357, 0]
    assert all(entry["progress"][3] == 0 for entry in passes)
    stages = {}
    for match in re.finditer(r"order=(\d+) stage=(\w+) elapsed_ms=(\d+)", trace):
        stages.setdefault(int(match[1]), {})[match[2]] = int(match[3])
    waits = [stages[order]["after_wait_for_device_idle"] - stages[order]["before_wait_for_device_idle"] for order in range(1, 23)]
    receives = [stages[order]["after_recv"] - stages[order]["before_recv"] for order in range(1, 23)]
    error = 'renderer pass order mismatch: expected "nephele.media.inject", got "terrain.forward_aov"'
    assert error in trace
    absent = ["contracts-environment-junit.xml", "junit.xml", "run-a.json", "run-b.json", "realistic-dem-frame-time.json"]
    assert all(not (artifact / name).exists() for name in absent)
    cpu_gates = {name: GATES[name](artifact) for name in ("gate1", "gate2")}
    assert all(not result["failures"] for result in cpu_gates.values())
    static = json.loads((artifact / "gate5-static.json").read_text())
    shader_static = {key: static[key] for key in (
        "naga_validated_assemblies", "compute_entry_texture_sample_compare_calls",
        "unresolved_source_expressions", "violating_compute_entries",
    )}
    assert shader_static == dict(naga_validated_assemblies=356,
        compute_entry_texture_sample_compare_calls=0, unresolved_source_expressions=[], violating_compute_entries=[])
    forecast = json.loads((ROOT / "docs/audits/nephele-directional-qualification-2026-10-22.json").read_text())["candidate"]["metrics"]
    assert forecast["all_five_pass"]
    files = ["checked-out-head.txt", "adapter-probe-raw.json", "fast-junit.xml", "fast-zero-skip.log", "directional-trace.log", "gate5-static.json"]
    return dict(
        head_sha=HEAD,
        run_url=f"https://github.com/milos-agathon/forge3d/actions/runs/{RUN}",
        adapter=probe,
        fast_junit=counts,
        budget_test=dict(status="passed", classname=budget[0].get("classname"), seconds=float(budget[0].get("time"))),
        convergence=dict(
            orders=passes[-1]["order"], elapsed_ms=passes[-1]["elapsed_ms"],
            final_progress=passes[-1]["progress"],
            reported_residual_floats=[struct.unpack("<f", struct.pack("<I", value))[0] for value in passes[-1]["progress"][1:3]],
            wait_ms=dict(min=min(waits), max=max(waits)), recv_ms=dict(min=min(receives), max=max(receives)),
            all_waits_and_receives_returned=True, order_cap_added=False,
            limitation="Loop elapsed time is not a complete-frame or realistic-DEM timing. Nonzero float residual fields are not the approved packed-bit stopping predicate.",
        ),
        capture_error=error,
        unavailable_outputs=absent,
        cpu_gates=cpu_gates,
        shader_static=shader_static,
        complete_gate5_result="ABSENT: realtime-termination-slice.npy was not captured",
        unchanged_nominal_forecast=forecast,
        physical_gate_result="ABSENT: render rejected by frame-graph order check before capture",
        dem_1500_frame_time="ABSENT: benchmark step not executed after trace failure",
        evidence_sha256={name: hashlib.sha256((artifact / name).read_bytes()).hexdigest() for name in files},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / f".release-evidence/2026-10-23/ci-{RUN}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.artifact)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("head_sha", "fast_junit", "budget_test", "convergence", "capture_error")}))


if __name__ == "__main__":
    main()
