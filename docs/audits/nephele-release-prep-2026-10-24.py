"""Verify downloaded final-head NEPHELE evidence without executing GPU work.

Run with --artifact <download directory> --head <full SHA> --output <local JSON>.
The unchanged acceptance functions rescore all six gates. This verifier also
requires the existing zero-skip suites and clean real-DEM frame certificates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["FORGE3D_NO_BOOTSTRAP"] = "1"
from scripts.nephele_acceptance import GATES, HOST_VISIBLE_LIMIT


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(artifact, head):
    assert (artifact / "checked-out-head.txt").read_text(encoding="utf-8-sig").strip() == head
    probe = read(artifact / "adapter-probe-raw.json")["probe"]
    assert probe["status"] == "ok" and probe["backend"].lower() == "vulkan"
    assert "nvidia" in probe["name"].lower()
    assert probe["software_fallback"] is False and probe["device_type"] == "DiscreteGpu"
    for arm in ("a", "b"):
        assert read(artifact / f"run-{arm}.json")["diagnostics"]["source_revision"] == head
    gates = {name: gate(artifact) for name, gate in GATES.items()}
    assert all(not gate["failures"] for gate in gates.values()), gates
    suites = {}
    for name, minimum in (("junit.xml", 6), ("fast-junit.xml", 661), ("contracts-environment-junit.xml", 10)):
        cases = list(ET.parse(artifact / name).iter("testcase"))
        failed = [c.attrib["name"] for c in cases if any(c.find(tag) is not None for tag in ("failure", "error", "skipped"))]
        assert not failed, (name, failed)
        assert len(cases) >= minimum, (name, len(cases))
        if name != "fast-junit.xml":
            assert len(cases) == minimum, (name, len(cases))
        suites[name] = {"passed": len(cases), "failed": 0, "errors": 0, "skipped": 0}
    benchmark = read(artifact / "realistic-dem-frame-time.json")
    assert benchmark["status"] == "measured" and benchmark["head_sha"] == head
    assert benchmark["diagnostics"]["source_revision"] == head
    assert benchmark["source"]["shape"] == [1500, 1500] and benchmark["viewport"] == [1920, 1080]
    assert benchmark["source"]["sha256"] == read(ROOT / "tests/data/prometheus/gore_range_dem.json")["source_sha256"]
    observations = benchmark["observations"]
    assert [o["state"] for o in observations] == ["first_frame", "unchanged_frame"]
    for observation in observations:
        assert observation["certificate"]["engine"]["git_sha"] == head[:12]
        assert observation["certificate"]["degradations"] == [], observation
        for field in ("completed_frame_wall_ms", "render_call_ms", "output_readback_ms"):
            assert math.isfinite(observation[field]) and observation[field] > 0
    convergence = benchmark["diagnostics"]["directional_transport"]
    assert convergence["converged"] and convergence["cache_reused"]
    assert convergence["maximum_source_change"] == 0 and convergence["maximum_surface_change"] == 0
    assert benchmark["memory"]["peak_host_visible_bytes"] < HOST_VISIBLE_LIMIT
    return {"head": head, "adapter": probe, "gates": gates, "junit": suites,
            "benchmark": benchmark, "GPU_executed_by_verifier": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.artifact, args.head)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"head": args.head, "junit": result["junit"], "all_gates_pass": True}))


if __name__ == "__main__":
    main()
