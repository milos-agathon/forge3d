"""Physical analytic and bounded-composition gates for the existing GTAO."""
from __future__ import annotations

import numpy as np
from _concordia_scene import capture, scene, world_positions


def test_unobstructed_plane_has_unit_gtao_visibility():
    model = scene(effects=("ssao",), cartography=False, height_ao=False)
    model.recipe.layers = []
    model.recipe.terrain.data = np.zeros((32,32), dtype=np.float32)
    _, data = capture(model)
    world = world_positions(data)
    # Interior is one AO radius away from each finite plane edge.
    roi = ((data["surface_kind"] == 1) & (np.abs(world[...,0]) < 14)
           & (np.abs(world[...,1]) < 14))
    assert roi.any()
    # R32Float reconstruction and half-float normals bound round-off here.
    np.testing.assert_allclose(data["gtao_raw"][roi], 1.0, atol=1e-4)
    np.testing.assert_allclose(data["gtao_visibility"][roi], 1.0, atol=1e-4)


def test_partitioned_visibility_is_bounded_and_preserves_clear_pixels():
    _, data = capture(scene(effects=("ssao",), cartography=False))
    gtao = data["gtao_visibility"]
    horizon = data["height_visibility"]
    assert np.isfinite(gtao).all() and np.isfinite(horizon).all()
    assert np.all((gtao >= 0) & (gtao <= 1))
    assert np.all((horizon >= 0) & (horizon <= 1))
    # Analytic independent azimuth / cosine-CDF intervals are resolved by the
    # actual production combine_ao GPU pipeline on unit radiance inputs.
    azimuth,cdf=np.meshgrid((np.arange(16)+0.5)/16,(np.arange(16)+0.5)/16)
    reference=np.mean((azimuth>=0.25)&(cdf>=0.25))
    expected=np.asarray([reference,1.0,0.5,0.5,0.75])
    np.testing.assert_array_equal(data["analytic_ao_resolve"],expected)
    # Also test identical blocked directions, where exact joint visibility is
    # 0.75. Scalar marginal visibilities cannot identify their correlation.
    # The actual GPU result must remain inside the Frechet joint-visibility
    # bounds; the mixed PT gate separately measures approximation quality.
    correlated_reference=np.mean(azimuth>=0.25)
    resolved=float(data["analytic_ao_resolve"][0])
    assert max(0.0,0.75+0.75-1.0)<=resolved<=correlated_reference
    assert resolved < correlated_reference  # disclose the overlap bias

    # The same physical mixed scene has nontrivial height and building terms
    # at the same receivers. Check the GPU resolve applies each exactly once.
    from _concordia_scene import height_visibility_at_pixels
    height=height_visibility_at_pixels(data)
    overlap=(data["surface_kind"]==1)&(height<1)&(gtao<1)
    assert overlap.any()
    radiance=data["geometry_hdr"][...,:3].astype(np.float32)
    expected_hdr=(radiance*(height*gtao)[...,None]).astype(np.float16)
    # F16 storage and independent CPU/GPU f32 projection can differ by one
    # adjacent storage value; compare in that storage precision envelope.
    observed=data["last_composition_hdr"][...,:3].astype(np.float32)
    spacing=np.abs(np.spacing(expected_hdr)).astype(np.float32)
    assert np.all(np.abs(observed[overlap]-expected_hdr[overlap].astype(np.float32))<=spacing[overlap])


def test_unobstructed_sloped_plane_has_unit_height_visibility():
    model=scene(effects=(),cartography=False,height_ao=True)
    model.recipe.layers=[]
    x,y=np.meshgrid(np.linspace(-16,16,32,dtype=np.float32),np.linspace(-16,16,32,dtype=np.float32))
    model.recipe.terrain.data=np.ascontiguousarray(0.2*x+0.1*y)
    model.recipe.lighting.settings["height_ao"].update(directions=16,steps=64,max_distance=32.0)
    _,data=capture(model)
    # The finite plane has no horizon above its own tangent hemisphere.
    np.testing.assert_allclose(data["height_visibility"][1:-1,1:-1],1.0,atol=1e-4)


def test_nonoccluded_luminance_changes_by_less_than_two_percent():
    from _concordia_ao_reference import ao_reference_metrics
    metrics,_=ao_reference_metrics()
    assert metrics["nonoccluded_pixels"]>0
    assert metrics["nonoccluded_luminance_change"]<0.02


def test_combined_ao_beats_both_ablations_on_prometheus_contact_and_horizon_reference(tmp_path):
    import os
    from _concordia_ao_reference import ao_reference_metrics,write_ao_evidence
    metrics,data=ao_reference_metrics()
    write_ao_evidence(metrics,data,os.environ.get("FORGE3D_CONCORDIA_ARTIFACT_DIR",tmp_path))
    scores=metrics["scores"]
    assert scores["combined"]["ssim"]-scores["gtao_only"]["ssim"] >= 0.03
    assert scores["combined"]["ssim"]-scores["height_only"]["ssim"] >= 0.03
    assert metrics["contact_pixels"] > 0
    assert metrics["contact_depth_reference_agreement"] >= 0.98
