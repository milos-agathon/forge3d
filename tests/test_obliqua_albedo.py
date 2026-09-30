import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import forge3d.path_tracing as pt

from _obliqua_gpu_guard import require_qualifying_gpu


class _Native:
    def hybrid_render_terrain_reference(self, *args, **kwargs):
        return kwargs


def _dem():
    return np.zeros((4, 5), dtype=np.float32)


def test_albedo_map_is_grid_aligned_and_forwarded(monkeypatch):
    monkeypatch.setattr(pt, "_NATIVE", _Native())
    albedo_map = np.ones((4, 5, 4), dtype=np.float64)
    result = pt.hybrid_render_terrain_reference(
        _dem(), 8, 8, albedo_map=albedo_map, albedo_sampling="bilinear"
    )
    assert result["albedo_map"].dtype == np.float32
    assert result["albedo_map"].flags.c_contiguous
    assert result["albedo_sampling"] == "bilinear"


def test_rgb_albedo_map_is_accepted_with_form_default_sampling(monkeypatch):
    # (H, W, 3) linear RGB is the alpha-one form of the RGBA map; sampling is
    # left to the native seam, which defaults RGB maps to bilinear and RGBA
    # maps to nearest.
    monkeypatch.setattr(pt, "_NATIVE", _Native())
    result = pt.hybrid_render_terrain_reference(
        _dem(), 8, 8, albedo_map=np.ones((4, 5, 3), dtype=np.float64)
    )
    assert result["albedo_map"].shape == (4, 5, 3)
    assert result["albedo_map"].dtype == np.float32
    assert result["albedo_sampling"] is None


@pytest.mark.parametrize(
    "value",
    [
        np.ones((5, 4, 4), dtype=np.float32),
        np.ones((4, 5, 2), dtype=np.float32),
        np.ones((20, 4), dtype=np.float32),
    ],
)
def test_albedo_map_shape_mismatch_is_an_error(monkeypatch, value):
    monkeypatch.setattr(pt, "_NATIVE", _Native())
    with pytest.raises(ValueError, match=r"albedo_map must have shape \(4, 5, 4\)"):
        pt.hybrid_render_terrain_reference(_dem(), 8, 8, albedo_map=value)


def test_albedo_map_rejects_nonfinite_and_bad_sampling(monkeypatch):
    monkeypatch.setattr(pt, "_NATIVE", _Native())
    albedo_map = np.ones((4, 5, 4), dtype=np.float32)
    albedo_map[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pt.hybrid_render_terrain_reference(_dem(), 8, 8, albedo_map=albedo_map)
    with pytest.raises(ValueError, match="nearest.*bilinear"):
        pt.hybrid_render_terrain_reference(
            _dem(), 8, 8, albedo_sampling="cubic"
        )


def test_albedo_certificate_inputs_are_recorded(tmp_path):
    require_qualifying_gpu("OBLIQUA albedo certificate proof")
    dem = np.zeros((4, 4), dtype=np.float32)
    albedo_map = np.full((4, 4, 4), (0.62, 0.62, 0.62, 1.0), dtype=np.float32)
    certificate = tmp_path / "obliqua.json"
    pt.hybrid_render_terrain_reference(
        dem,
        8,
        8,
        {
            "origin": (0.0, 8.0, 8.0),
            "look_at": (0.0, 0.0, 0.0),
            "up": (0.0, 1.0, -1.0),
        },
        albedo_map=albedo_map,
        albedo_sampling="nearest",
        sun_intensity=2.5,
        max_frames=32,
        min_frames=32,
        variance_threshold=1e9,
        certificate=str(certificate),
    )
    report = json.loads(certificate.read_text(encoding="utf-8"))
    assert report["inputs"]["camera_model"] == "pinhole"
    assert report["inputs"]["sensor_rect"] == "0,0,1,1"
    assert report["inputs"]["albedo"] == "0.6,0.6,0.6"
    assert len(report["inputs"]["albedo_map_sha256"]) == 64
    assert report["inputs"]["albedo_sampling"] == "nearest"
    labels = {record["label"] for record in report["passes"]}
    assert {"hybrid_pt.restir_temporal", "hybrid_pt.restir_spatial"} <= labels


def test_default_flag_off_matches_pre_obliqua_sha():
    require_qualifying_gpu("pre-OBLIQUA determinism proof")
    out = pt.hybrid_render_terrain_reference(
        np.zeros((4, 4), dtype=np.float32),
        16,
        16,
        {
            "origin": (0.0, 5.0, 5.0),
            "look_at": (0.0, 0.0, 0.0),
            "up": (0.0, 1.0, -1.0),
            "fov_y": 45.0,
        },
        albedo=(0.62, 0.62, 0.62),
        sun_intensity=2.5,
        env_intensity=0.0,
        max_frames=32,
        min_frames=32,
        variance_threshold=1e9,
        seed=7,
    )
    # Captured from the same scene on main da2fd15c, the renderer without
    # OBLIQUA (the 1.36.0 pin predates #174's seeding and convergence
    # changes). The merged renderer reproduces it byte for byte.
    assert hashlib.sha256(out["rgba"].tobytes()).hexdigest() == (
        "7097703b062a41009f2f4f3b42a5fc4f001c9be4087e360acc3eef6e0cada01a"
    )


def _render(albedo_map=None, albedo_sampling="nearest"):
    require_qualifying_gpu("OBLIQUA terrain albedo proof")
    return pt.hybrid_render_terrain_reference(
        np.zeros((4, 4), dtype=np.float32),
        16,
        16,
        {
            "model": "orthographic",
            "origin": (0.0, 5.0, 0.0),
            "look_at": (0.0, 0.0, 0.0),
            "up": (0.0, 0.0, -1.0),
            "half_height": 2.0,
        },
        albedo=(0.62, 0.62, 0.62),
        albedo_map=albedo_map,
        albedo_sampling=albedo_sampling,
        sun_intensity=2.5,
        env_intensity=0.0,
        max_frames=32,
        min_frames=32,
        variance_threshold=1e9,
    )


@pytest.mark.parametrize("sampling", ["nearest", "bilinear"])
def test_constant_albedo_map_is_byte_identical(sampling):
    material = np.full((4, 4, 4), (0.62, 0.62, 0.62, 1.0), dtype=np.float32)
    constant = _render()
    textured = _render(material, sampling)
    assert np.array_equal(textured["rgba"], constant["rgba"])
    assert np.array_equal(textured["albedo"], constant["albedo"])


def test_alpha_below_one_falls_back_to_constant():
    material = np.full((4, 4, 4), (1.0, 0.0, 0.0, 0.5), dtype=np.float32)
    assert np.array_equal(_render(material)["albedo"], _render()["albedo"])


def test_nearest_and_bilinear_feed_the_albedo_aov():
    material = np.ones((4, 4, 4), dtype=np.float32)
    material[:, :2, :3] = (1.0, 0.0, 0.0)
    material[:, 2:, :3] = (0.0, 0.0, 1.0)
    nearest = _render(material, "nearest")["albedo"]
    bilinear = _render(material, "bilinear")["albedo"]
    assert np.any(np.all(nearest == (1.0, 0.0, 0.0), axis=-1))
    assert np.any(np.all(nearest == (0.0, 0.0, 1.0), axis=-1))
    assert np.any((bilinear[..., 0] > 0.0) & (bilinear[..., 2] > 0.0))


def test_bilinear_falls_back_each_masked_texel_before_interpolation():
    material = np.full((4, 4, 4), (1.0, 0.0, 0.0, 1.0), dtype=np.float32)
    material[1, 1] = (0.0, 1.0, 0.0, 0.5)
    albedo = _render(material, "bilinear")["albedo"]
    fallback_mix = (
        (albedo[..., 0] > 0.62)
        & (albedo[..., 0] < 1.0)
        & (albedo[..., 1] > 0.0)
        & (albedo[..., 1] < 0.62)
        & (albedo[..., 2] > 0.0)
        & (albedo[..., 2] < 0.62)
    )
    assert np.any(fallback_mix)


def test_offscreen_material_edit_does_not_change_visible_pixels():
    require_qualifying_gpu("OBLIQUA offscreen material visibility proof")
    constant = np.full((4, 4, 4), (0.62, 0.62, 0.62, 1.0), dtype=np.float32)
    edited = constant.copy()
    edited[0, 0, :3] = (1.0, 0.0, 0.0)
    baseline = pt.hybrid_render_terrain_reference(
        np.zeros((4, 4), dtype=np.float32),
        16,
        16,
        {
            "model": "orthographic",
            "origin": (0.0, 5.0, 0.0),
            "look_at": (0.0, 0.0, 0.0),
            "up": (0.0, 0.0, -1.0),
            "half_height": 0.5,
        },
        albedo=(0.62, 0.62, 0.62),
        albedo_map=constant,
        sun_intensity=2.5,
        env_intensity=0.0,
        max_frames=32,
        min_frames=32,
        variance_threshold=1e9,
    )
    changed = pt.hybrid_render_terrain_reference(
        np.zeros((4, 4), dtype=np.float32),
        16,
        16,
        {
            "model": "orthographic",
            "origin": (0.0, 5.0, 0.0),
            "look_at": (0.0, 0.0, 0.0),
            "up": (0.0, 0.0, -1.0),
            "half_height": 0.5,
        },
        albedo=(0.62, 0.62, 0.62),
        albedo_map=edited,
        sun_intensity=2.5,
        env_intensity=0.0,
        max_frames=32,
        min_frames=32,
        variance_threshold=1e9,
    )
    assert np.array_equal(changed["albedo"], baseline["albedo"])
    assert np.array_equal(changed["rgba"], baseline["rgba"])


def test_restir_mapped_material_and_seamless_camera_guards():
    # Mapped terrain is shaded through the spectral-residual ReSTIR path (the
    # receiver material enters the estimator, not a G-buffer tag); texels with
    # alpha < 1 fall back to the constant albedo; seamless/global-camera
    # renders skip reservoir reweighting and spatial neighbours so tiles match
    # the monolithic frame.
    shaders = Path(__file__).parents[1] / "src" / "shaders"
    terrain = (shaders / "hybrid_terrain_traversal.wgsl").read_text()
    spatial = (shaders / "pt_restir_spatial.wgsl").read_text()
    assert "terrain_spectral_restir_enabled()" in terrain
    assert "select(terrain.albedo_pad.rgb, c.rgb, mapped)" in terrain
    assert "if (uniforms.camera_flags == 0u && prev_valid)" in terrain
    assert "reuse_w = clamp(prev_r.weight, 0.0, TERRAIN_RESTIR_W_CAP)" in terrain
    assert "if (uniforms.camera_flags == 0u)" in spatial
