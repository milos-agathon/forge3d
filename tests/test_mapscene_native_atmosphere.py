"""Shared native cloud/atmosphere execution and ablation evidence."""
from __future__ import annotations

import numpy as np
from _concordia_scene import capture, scene


def test_cloud_atmosphere_and_froxels_execute_before_temporal_resolve():
    result, data = capture(scene(atmosphere=True))
    pipeline = result.metadata["native_pipeline"]
    labels = [entry["label"] for entry in pipeline["execution_report"]["passes"]]
    assert labels.index("concordia.cloud_shadows") < labels.index("concordia.geometry")
    assert labels.index("concordia.ssr") < labels.index("concordia.atmosphere")
    assert labels.index("concordia.atmosphere") < labels.index("concordia.froxels") < labels.index("concordia.temporal")
    assert pipeline["effects"]["cloud_shadows"] and pipeline["effects"]["atmosphere"]
    assert pipeline["effects"]["volumetrics"]
    assert data["cloud_field"].shape[-1] == 4
    assert np.isfinite(data["geometry_hdr"]).all()
    assert not pipeline["execution_report"]["degradations"]


def test_cloud_ablation_changes_at_least_five_percent_of_lit_terrain():
    enabled, data = capture(scene(effects=("taa","cloud_shadows"),cartography=False,height_ao=False))
    disabled, clear = capture(scene(effects=("taa",),cartography=False,height_ao=False))
    np.testing.assert_array_equal(data["depth"],clear["depth"])
    terrain = (data["surface_kind"] == 1) & np.any(clear["geometry_hdr"][...,:3]>0,axis=-1)
    assert terrain.any()
    changed = np.any(enabled.rgba[...,:3] != disabled.rgba[...,:3],axis=-1)
    assert np.mean(changed[terrain]) >= 0.05
    assert enabled.metadata["native_pipeline"]["effects"]["cloud_shadows"]
    assert not disabled.metadata["native_pipeline"]["effects"]["cloud_shadows"]


def test_terrain_and_building_cloud_coverage_agree_within_one_pixel():
    import os
    from _concordia_measurements import cloud_coupling,write_cloud_evidence
    metrics,masks,frames=cloud_coupling()
    directory=os.environ.get("FORGE3D_CONCORDIA_ARTIFACT_DIR")
    if directory:
        write_cloud_evidence(metrics,masks,frames,directory)
    assert metrics["changed_terrain_fraction"] >= 0.05
    for name in ("terrain","buildings"):
        receiver=metrics["surfaces"][name]
        assert receiver["lit_pixels"] > 0 and receiver["expected_covered"] > 0
        assert receiver["max_displacement_px"] <= 1.0
    reports=metrics["ablation_certificates"]
    assert reports["cloud_True_sun_1.0"]["effects"]["cloud_shadows"]
    assert not reports["cloud_False_sun_1.0"]["effects"]["cloud_shadows"]
