"""The certificate is evidence of the encoded frame, not requested metadata."""
from __future__ import annotations
import json
from _concordia_scene import capture, scene


def test_certificate_locks_native_pass_order_sizes_and_effects():
    result, _ = capture(scene(atmosphere=True))
    pipeline = result.metadata["native_pipeline"]
    report = pipeline["execution_report"]
    labels = [item["label"] for item in report["passes"] if item["label"].startswith("concordia.")]
    expected = ["concordia.resources","concordia.cloud_shadows","concordia.shadows",
        "concordia.geometry","concordia.cartography_depth","concordia.depth_pyramid","concordia.ao","concordia.ssgi",
        "concordia.ssr","concordia.atmosphere","concordia.froxels","concordia.temporal",
        "concordia.output_transform","concordia.vectors","concordia.labels"]
    if not pipeline["label_batches"]:
        expected.remove("concordia.labels")
    assert labels == expected
    assert set(pipeline["stage_sizes"]) == set(labels)
    assert pipeline["stage_sizes"]["concordia.labels" if pipeline["label_batches"] else "concordia.output_transform"]["output"] == [256,192]
    assert pipeline["fallbacks"]["ssgi_miss"] == "shared_ibl"
    assert "concordia.pipeline" in json.dumps(report)
