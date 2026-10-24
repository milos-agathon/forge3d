"""CPU-only verification of the recorded NEPHELE scheduler and DEM evidence.

Run from the release worktree. This reads saved arrays/JSON/JUnit and calls the
unchanged acceptance functions; it never constructs a GPU renderer or device.
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

HEAD = "024c7803d17814c447637208cb4535112c9ac166"
RUN = 40075114543
OUT = ROOT / ".release-evidence/2026-10-24/media-leaf-keys"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def junit(artifact, name):
    cases = list(ET.parse(artifact / name).iter("testcase"))
    counts = {
        "passed": sum(not any(c.find(t) is not None for t in ("failure", "error", "skipped")) for c in cases),
        "failed": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped": sum(c.find("skipped") is not None for c in cases),
    }
    assert not any(counts[k] for k in ("failed", "errors", "skipped")), (name, counts)
    return counts, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / f".release-evidence/2026-10-24/ci-{RUN}")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    assert (artifact / "checked-out-head.txt").read_text(encoding="utf-8-sig").strip() == HEAD
    probe = read(artifact / "adapter-probe-raw.json")["probe"]
    assert probe["status"] == "ok" and probe["backend"].lower() == "vulkan"
    assert "nvidia" in (probe["name"] + probe["driver"]).lower()
    assert probe["software_fallback"] is False and probe["device_type"] == "DiscreteGpu"
    captures = {arm: read(artifact / f"run-{arm}.json") for arm in ("a", "b")}
    for capture in captures.values():
        assert capture["diagnostics"]["source_revision"] == HEAD
        assert "iteration_wall_milliseconds" not in capture["diagnostics"]["directional_transport"]
    assert captures["a"]["diagnostics"] == captures["b"]["diagnostics"]
    gates = {name: gate(artifact) for name, gate in GATES.items()}
    assert all(not gate["failures"] for gate in gates.values())
    counts, cases = {}, {}
    for name in ("fast-junit.xml", "contracts-environment-junit.xml", "junit.xml"):
        counts[name], cases[name] = junit(artifact, name)
    assert counts["junit.xml"] == dict(passed=6, failed=0, errors=0, skipped=0)
    required_cases = {}
    for name in (
        "test_600mib_host_visible_raises_named_budget_error",
        "test_camera_visible_uniform_environment_readback",
        "test_shadow_scattering_component_readback",
    ):
        found = [c for group in cases.values() for c in group if c.attrib["name"] == name]
        assert len(found) == 1 and not list(found[0])
        required_cases[name] = {"result": "passed", "seconds": float(found[0].attrib["time"])}
    benchmark = read(artifact / "realistic-dem-frame-time.json")
    assert benchmark["status"] == "measured" and benchmark["head_sha"] == HEAD
    assert benchmark["diagnostics"]["source_revision"] == HEAD
    provenance = read(ROOT / "tests/data/prometheus/gore_range_dem.json")
    assert benchmark["source"]["sha256"] == provenance["source_sha256"]
    assert benchmark["source"]["shape"] == [1500, 1500]
    assert benchmark["viewport"] == [1920, 1080]
    observations = benchmark["observations"]
    assert [o["state"] for o in observations] == ["first_frame", "unchanged_frame"]
    for observation in observations:
        assert observation["certificate"]["engine"]["git_sha"] == HEAD[:12]
        for field in ("render_call_ms", "completed_frame_wall_ms", "output_readback_ms"):
            assert math.isfinite(observation[field]) and observation[field] > 0
    convergence = benchmark["diagnostics"]["directional_transport"]
    assert convergence["converged"] and convergence["cache_reused"]
    assert convergence["maximum_source_change"] == convergence["maximum_surface_change"] == 0
    assert benchmark["memory"]["peak_host_visible_bytes"] < HOST_VISIBLE_LIMIT
    workflow = read(OUT / "final-workflow-result.json")
    assert workflow["headSha"] == HEAD and workflow["conclusion"] == "success"
    baseline = read(ROOT / ".release-evidence/2026-10-24/ci-40067871680/local-exact-head-verification.json")
    assert gates["gate6"]["metrics"]["frame_hashes"] == baseline["gates"]["gate6"]["metrics"]["frame_hashes"]
    unchanged_metrics = {g: gates[g]["metrics"] == baseline["gates"][g]["metrics"] for g in ("gate1", "gate2", "gate3", "gate4", "gate5")}
    assert all(unchanged_metrics.values())
    local_checks = {name: read(OUT / f"{name}.json") for name in (
        "upload-identity-unit-final", "media-framegraph-unit", "media-shader-unit",
        "resolve-unit", "resolve-fmt", "resolve-clippy", "resolve-commit",
    )}
    for check in local_checks.values():
        assert check.get("exit_code", check.get("result")) == 0
    certificates = {o["state"]: o["certificate"] for o in observations}
    report = {
        "date": "2026-10-24", "head_sha": HEAD,
        "run": f"https://github.com/milos-agathon/forge3d/actions/runs/{RUN}",
        "workflow_result": "success", "JUnit": counts, "required_cases": required_cases,
        "adapter": probe, "diagnostics_identical": True,
        "physical_directional_transport": captures["a"]["diagnostics"]["directional_transport"],
        "metrics_unchanged_from_98cc1df1": unchanged_metrics,
        "gates_before": baseline["gates"], "gates_after": gates,
        "image_margins": {
            "B1_above_0.95": gates["gate3"]["metrics"]["sky_cloud_delta_e_pass_fraction"] - 0.95,
            "SSIM_above_0.95": gates["gate3"]["metrics"]["godray_roi_ssim"] - 0.95,
            "G4a_above_0.95": gates["gate4"]["metrics"]["shadow_delta_e_pass_fraction"] - 0.95,
            "medium_change_above_0.10": gates["gate4"]["metrics"]["medium_ablation_changed_fraction"] - 0.10,
            "G4b_relative_error_below_0.20": 0.20 - gates["gate4"]["metrics"]["terrain_occlusion_response_relative_error"],
        },
        "DEM_benchmark": benchmark,
        "convergence_scope": "The convergence/cache-reuse fields are from the explicit untimed capture after the two timing observations, not separate per-frame diagnostic samples.",
        "certificate_degradations": {state: c["degradations"] for state, c in certificates.items()},
        "local_checks": local_checks,
        "protected_values": read(OUT / "protected-values.json"),
        "commits": [read(OUT / "commit-verification.json"), read(OUT / "resolve-commit-verification.json")],
        "changed_files": {
            "src/terrain/renderer/media.rs": "Hash exact uploaded media bytes, canonical medium/resource versions, sun input and previous GPU producer identities without readback.",
            "src/terrain/renderer/draw/mod.rs": "Supply every initial graph read key; validate the actual shadow transition at its first reader and resolve transitions according to the compiled input type.",
            "src/terrain/renderer/draw/leaf_keys.rs": "Shared production key map and CPU regression coverage for both transport modes, invalidation and resolve validation.",
        },
        "zero_skip_environment": {
            "FORGE3D_ALLOW_HOSTED_WINDOWS_TERRAIN": "1", "FORGE3D_NO_BOOTSTRAP": "1",
            "FORGE3D_TEST_INSTALLED_WHEEL": "1", "FORGE3D_NEPHELE_LANE": "windows-nvidia-vulkan",
            "FORGE3D_NEPHELE_CANDIDATE_SHA": HEAD, "WGPU_BACKEND": "vulkan", "WGPU_BACKENDS": "vulkan",
        },
        "owner_authorization_words": "Commit, push, dispatch `scope=nephele`, and report the physical JUnit (expect 6/0/0) and the 1500² timing: first and unchanged frames, with the convergence and cache-reuse fields.",
        "new_protected_value_approval": "Not needed: pins, shader contracts, f32 inventory/ledger, committed goldens/certificates and existing assertions unchanged.",
        "checks_not_run": ["Local GPU execution (prohibited)", "D3D12 ABSENT", "Metal ABSENT", "WASM ABSENT", "scope=full", "merge/tag/publish"],
        "limitations": benchmark["limitations"] + [
            "The unsigned runtime certificates report missing NEPHELE shader provenance; registration/certificate remediation was not applied.",
            "First and unchanged frame hashes differ; this benchmark does not establish warm-frame bitwise identity or large-DEM reference image accuracy.",
            "There was no valid before-fix frame timing: the API raised in the scheduler, then in the resolve guard.",
        ],
        "steering_gap_proposed_not_applied": "The f32 scanner can pair a Rust lifetime with an apostrophe in a new test comment and hide a cfg(test) marker. Compare site tuples before requesting a ledger update; an owner-reviewed lexer correction is separate work.",
        "reference_RNG_overlap": "Unchanged, as required.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"head_sha": HEAD, "workflow": "success", "JUnit": counts,
        "DEM_timings": [{k: v for k, v in o.items() if k != "certificate"} for o in observations],
        "final_untimed_convergence": convergence,
        "certificate_degradation_counts": {state: len(c["degradations"]) for state, c in certificates.items()},
        "all_six_gates_pass": True}, indent=2))


if __name__ == "__main__":
    main()
